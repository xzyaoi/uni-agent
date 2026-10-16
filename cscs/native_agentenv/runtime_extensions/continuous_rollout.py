"""Bounded prompt-group refill independent of the separate-async backward loop."""
import contextlib
import json
import logging
import math
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def positive_int(config, key, default):
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"continuous_refill.{key} must be a positive integer")
    return value


class ContinuousRolloutMixin:
    """One producer owns dataloader fetch/dispatch while the sampler stays on the main thread."""

    def __init__(self, config):
        self._refill_thread = None
        self._refill_stop = threading.Event()
        self._refill_lock = threading.RLock()
        self._refill_error = None
        self._refill_stats = {}
        super().__init__(config)

    def _continuous_refill_enabled(self):
        return bool(self.config.trainer.v1.separate_async.get("continuous_refill", {}).get("enabled", False))

    def _check_refill_health(self):
        if self._refill_error is not None:
            raise RuntimeError("Continuous rollout producer failed") from self._refill_error

    def _start_continuous_refill(self):
        if self._refill_thread is not None:
            raise RuntimeError("Continuous rollout producer already started")
        if self.config.data.get("dataloader_num_workers", 0) != 0:
            raise ValueError("Continuous refill requires data.dataloader_num_workers=0 to avoid forking from its producer thread")
        settings = self.config.trainer.v1.separate_async.continuous_refill
        self._refill_target = positive_int(settings, "target_inflight_prompts", 128)
        self._refill_ready = settings.get("max_ready_prompts", 64)
        if isinstance(self._refill_ready, bool) or not isinstance(self._refill_ready, int) or self._refill_ready < 0:
            raise ValueError("continuous_refill.max_ready_prompts must be a nonnegative integer (0 disables the backlog cap)")
        self._refill_chunk = positive_int(settings, "submit_prompts", 16)
        self._refill_interval = float(settings.get("poll_interval", 1.0))
        if not math.isfinite(self._refill_interval) or self._refill_interval <= 0:
            raise ValueError("continuous_refill.poll_interval must be finite and positive")
        sampler = self.config.trainer.v1.sampler.get("custom_sampler")
        if sampler and sampler.get("path") and sampler.get("name"):
            raise ValueError("Continuous refill requires the built-in async replay sampler")
        fetch_size = self.config.data.get("gen_batch_size") or self.config.data.train_batch_size
        if any(value % fetch_size for value in (self._refill_target, self._refill_ready, self._refill_chunk)):
            raise ValueError("Continuous refill prompt counts must be multiples of data.gen_batch_size")
        self._previous_progress_check = getattr(self.replay_buffer, "progress_check", None)
        self.replay_buffer.progress_check = self._check_refill_health
        self._refill_stop.clear()
        self._refill_thread = threading.Thread(target=self._refill_loop, name="continuous-rollout", daemon=True)
        self._refill_thread.start()
        logger.info("Continuous refill enabled: target=%s groups, ready threshold=%s, chunk=%s",
                    self._refill_target, self._refill_ready, self._refill_chunk)

    def _refill_loop(self):
        submitted = 0
        try:
            from verl.utils import tensordict_utils as tu
            from verl.utils.transferqueue_utils import tq

            while not self._refill_stop.is_set():
                with self._refill_lock:
                    if self._refill_stop.is_set():
                        break
                    # Independent metadata only: never mutate replay_buffer's sampling sets.
                    groups = (tq.kv_list("train") or {}).get("train", {})
                    active = ready = failed = 0
                    for tag in groups.values():
                        if not tag.get("is_prompt", False):
                            continue
                        status = tag.get("status")
                        if status in ("pending", "running"):
                            active += 1
                        elif status in ("finished", "failure"):
                            ready += 1
                            failed += int(status == "failure")
                        else:
                            raise ValueError(f"Unknown prompt status in continuous refill: {status}")
                    count = 0
                    backlog_paused = self._refill_ready > 0 and ready >= self._refill_ready
                    if not backlog_paused and self.global_steps <= self.total_training_steps:
                        count = min(self._refill_target - active, self._refill_chunk)
                        if self._refill_ready > 0:
                            count = min(count, self._refill_target + self._refill_ready - active - ready)
                        fetch_size = self.config.data.get("gen_batch_size") or self.config.data.train_batch_size
                        count = max(0, count // fetch_size * fetch_size)
                    if count:
                        dispatch_step = self.global_steps
                        batch = self._next_train_batch(num_prompts=count)
                        tu.assign_non_tensor_data(batch, "global_steps", dispatch_step)
                        self._submit_batch_to_rollout(batch)
                        submitted += count
                    self._refill_stats = {
                        "inflight_prompts": active + count, "ready_prompts": ready,
                        "failed_prompts": failed, "submitted_prompts": submitted,
                        "target_prompts": self._refill_target, "ready_threshold": self._refill_ready,
                        "backlog_paused": backlog_paused,
                    }
                    self._record_refill_status(count)
                self._refill_stop.wait(self._refill_interval)
        except BaseException as exc:
            self._refill_error = exc
            logger.exception("Continuous rollout producer failed")

    def _record_refill_status(self, submitted_now):
        directory = os.environ.get("UNI_AGENT_TRACE_DIR")
        if not directory:
            return
        try:
            path = Path(directory) / f"feeder-{os.getpid()}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {"schema_version": 1, "kind": "pipeline", "event": "feeder_status",
                      "timestamp": time.time(), "global_steps": self.global_steps,
                      "submitted_now": submitted_now, **self._refill_stats}
            descriptor = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
            with os.fdopen(descriptor, "a") as stream:
                stream.write(json.dumps(record) + "\n")
        except Exception:
            logger.warning("Unable to record continuous refill status", exc_info=True)

    def _stop_continuous_refill(self):
        if self._refill_thread is None:
            return
        self._refill_stop.set()
        self._refill_thread.join(timeout=30)
        if self._refill_thread.is_alive():
            raise RuntimeError("Continuous rollout producer did not stop within 30 seconds")
        self.replay_buffer.progress_check = self._previous_progress_check
        self._refill_thread = None

    @contextlib.contextmanager
    def _pause_continuous_refill(self):
        if not self._refill_lock.acquire(timeout=30):
            raise RuntimeError("Continuous rollout producer did not pause within 30 seconds")
        try:
            self._check_refill_health()
            yield
        finally:
            self._refill_lock.release()

    def fit(self, agent_loop_manager):
        try:
            return super().fit(agent_loop_manager)
        finally:
            self._stop_continuous_refill()
            self._check_refill_health()

    def step(self, metrics, timing_raw):
        self._check_refill_health()
        result = super().step(metrics, timing_raw)
        self._check_refill_health()
        metrics.update({"training/refill/" + key: value for key, value in self._refill_stats.items()})
        return result

    def _add_batch_to_generate(self, *args, **kwargs):
        if self._refill_thread is not None:
            self._check_refill_health()
            return
        return super()._add_batch_to_generate(*args, **kwargs)

    def _add_prompts_to_generate(self, num_prompts):
        if self._refill_thread is not None:
            # Existing sampler still evicts stale/failed groups; the producer owns replacement.
            self._check_refill_health()
            return 0
        return super()._add_prompts_to_generate(num_prompts)

    def _save_checkpoint(self):
        with self._pause_continuous_refill():
            return super()._save_checkpoint()

    def _validate(self):
        with self._pause_continuous_refill():
            return super()._validate()
