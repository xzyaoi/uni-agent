"""Run one finite sandbox-concurrency point through the training rollout stack.

No PPO trainer or optimizer is constructed. Model initialization is timed
separately from the finite rollout workload; failed episodes are not replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import random
import signal
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from uuid import uuid4


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def harbor_counts(run: Path) -> dict:
    """Keep reward-file completion distinct from framework trajectory success."""
    counts = Counter()
    for path in (run / "harbor").glob("*/result.json"):
        try:
            trial = json.loads(path.read_text())
        except (OSError, ValueError):
            counts["unreadable_results"] += 1
            continue
        counts["terminal_results"] += int(bool(trial.get("finished_at")))
        rewards = (trial.get("verifier_result") or {}).get("rewards")
        if trial.get("exception_info") or not isinstance(rewards, dict) or rewards.get("reward") is None:
            counts["evaluation_incomplete"] += 1
        else:
            counts["evaluation_completed"] += 1
            counts["resolved"] += int(rewards["reward"] == 1)
    return dict(counts)


class Progress:
    """Independent metadata polling; never fetch trajectory tensors while polling."""

    def __init__(self, run: Path, uids: list[str], responses: int, tq):
        self.run, self.uids, self.responses, self.tq = run, set(uids), responses, tq
        self.offsets: dict[Path, int] = {}
        self.terminals: dict[str, dict] = {}
        self.batch_end: dict | None = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, name="sweep-progress", daemon=True)
        self.started = time.monotonic()
        self.latest: dict = {}

    def events(self) -> None:
        for path in (self.run / "telemetry" / "model-calls").glob("pipeline-*.jsonl"):
            with path.open() as stream:
                stream.seek(self.offsets.get(path, 0))
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line or not line.endswith("\n"):
                        self.offsets[path] = offset
                        break
                    row = json.loads(line)
                    if row.get("event") == "batch_end":
                        self.batch_end = row
                    elif row.get("uid") in self.uids and row.get("event") in {"episode_end", "episode_error"}:
                        self.terminals[row["session_id"]] = row

    def sample(self) -> dict:
        listing = (self.tq.kv_list("train") or {}).get("train", {}) or {}
        groups = Counter((listing.get(uid) or {}).get("status", "pending") for uid in self.uids)
        # One selected trajectory per session; metadata still excludes synthetic
        # failure materializations, even if a runtime marks them status=success.
        sessions: dict[tuple[str, str], dict] = {}
        for key, tag in listing.items():
            parts = key.rsplit("_", 2)
            if len(parts) != 3 or parts[0] not in self.uids or (tag or {}).get("status") != "success":
                continue
            if not parts[2].isdigit():
                continue
            tag = tag or {}
            identity = (parts[0], parts[1])
            previous = sessions.get(identity)
            if previous is None or int(parts[2]) > previous["index"]:
                sessions[identity] = {**tag, "index": int(parts[2])}
        usable = sum(
            int(tag.get("response_len", 0)) > 0
            and not any(word in str(tag.get("materialization_reason", "")).lower() for word in ("pad_fail", "padding", "failure"))
            for tag in sessions.values()
        )
        self.events()
        errors = sum(row["event"] == "episode_error" for row in self.terminals.values())
        harbor = harbor_counts(self.run)
        row = {
            "timestamp": time.time(),
            "rollout_elapsed_seconds": time.monotonic() - self.started,
            "groups_expected": len(self.uids),
            "episodes_expected": len(self.uids) * self.responses,
            "groups_by_status": dict(groups),
            "groups_terminal": groups["finished"] + groups["failure"],
            "groups_failed": groups["failure"],
            "episodes_terminal_observed": len(self.terminals),
            "episodes_error_observed": errors,
            "episodes_with_tq_trajectory": len(sessions),
            "episodes_with_usable_trajectory": usable,
            "episodes_with_unusable_trajectory": len(sessions) - usable,
            "harbor": harbor,
            "episode_event_capture_available": bool(self.offsets),
            "batch_end": self.batch_end,
        }
        # eval_completed cannot be inferred from reward=0 or TQ success. Harbor
        # result files provide a separate verifier-completion count below.
        self.latest = row
        return row

    def persist(self, row: dict) -> None:
        with (self.run / "status.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        atomic_json(self.run / "progress.json", row)

    def loop(self) -> None:
        while not self.stop.is_set():
            try:
                self.persist(self.sample())
            except Exception as exc:
                with (self.run / "status.jsonl").open("a") as stream:
                    stream.write(json.dumps({"timestamp": time.time(), "poll_error_class": type(exc).__name__}) + "\n")
            self.stop.wait(5)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("data-path", "model-path", "task-config", "run-dir"):
        result.add_argument("--" + name, required=True)
    result.add_argument("--concurrency", type=int, required=True, help="Maximum active prompt groups; each group owns --responses episodes")
    result.add_argument("--groups", type=int, default=128)
    result.add_argument("--responses", type=int, default=8)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--benchmark", default="swebench-pro-v2")
    result.add_argument("--served-model-name", default="Qwen3.5-9B")
    result.add_argument("--tensor-parallel-size", type=int, default=4)
    result.add_argument("--n-gpus-per-node", type=int, default=4)
    result.add_argument("--nnodes", type=int, default=1)
    result.add_argument("--prompt-length", type=int, default=4096)
    result.add_argument("--response-length", type=int, default=8192)
    result.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    result.add_argument("--gateway-count", type=int, default=8)
    result.add_argument("--temperature", type=float, default=1.0)
    result.add_argument("--top-p", type=float, default=1.0)
    result.add_argument("--top-k", type=int, default=-1)
    result.add_argument("--tool-parser", default="qwen3_coder")
    result.add_argument("--kv-cache-dtype", default="auto")
    return result


def main() -> None:
    args = parser().parse_args()
    if min(args.concurrency, args.groups, args.responses, args.prompt_length, args.response_length) <= 0:
        raise ValueError("Concurrency, groups, responses and token lengths must be positive")
    run = Path(args.run_dir).expanduser().resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run / "manifest.json").exists():
        raise FileExistsError("Use a fresh run directory for each point")
    trace = run / "telemetry" / "model-calls"
    trace.mkdir(parents=True, exist_ok=True)
    os.environ["UNI_AGENT_TRACE_DIR"] = str(trace)
    os.environ["AGENTENV_TRACE_DIR"] = str(run / "native-traces")
    # Ray forwards these to actors started by this driver, including gateways.
    repository = Path(os.environ.get("REPO_DIR", Path(__file__).resolve().parents[2])).resolve()
    sys.path.insert(0, str(repository))
    import ray
    from datasets import load_dataset
    from omegaconf import OmegaConf
    from examples.inference.parallel_infer_verl import (
        AgentFrameworkRolloutAdapter, LLMServerManager, TaskConfigResolver,
        _build_prompts, _read_rm_scores, init_config, tq, tu,
    )

    # Fail on unsupported framework APIs before allocating model workers.
    from cscs.sandbox_sweep.group_framework import GroupLimitedGatewayAgentFramework

    initialized = time.monotonic()
    progress = None
    outcome = {"state": "initializing", "started_at": time.time(),
               "benchmark": args.benchmark,
               "concurrency_groups": args.concurrency,
               "max_concurrent_episodes": args.concurrency * args.responses}
    atomic_json(run / "result.json", outcome)

    def interrupt(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        dataset = load_dataset("parquet", data_files=str(Path(args.data_path).expanduser()), split="train")
        if len(dataset) < args.groups:
            raise ValueError(f"Dataset has {len(dataset)} rows, need {args.groups}")
        indices = list(range(len(dataset)))
        random.Random(args.seed).shuffle(indices)
        indices = indices[:args.groups]
        samples = [dataset[index] for index in indices]
        identities = []
        for index, sample in zip(indices, samples, strict=True):
            metadata = sample["extra_info"]["tools_kwargs"]["task"]["metadata"]
            task_file = Path(metadata["task_path"]) / "task.toml"
            identities.append({"dataset_index": index, "instance_id": metadata["instance_id"],
                               "task_toml_sha256": digest(task_file)})
        source_config = Path(args.task_config).expanduser().resolve()
        source_config_sha256 = digest(source_config)
        task_config = OmegaConf.load(source_config)
        entries = task_config if OmegaConf.is_list(task_config) else [task_config]
        for entry in entries:
            if entry.get("name") == "harbor":
                entry.artifacts_dir = str(run / "harbor")
        point_config = run / "task_config.yaml"
        OmegaConf.save(task_config, point_config)
        point_config.chmod(0o600)
        args.task_config = str(point_config)
        args.log_dir = str(run / "sessions")
        args.n = args.responses
        args.engine = "vllm"
        args.enable_rollout_routing_replay = False
        args.allowed_request_sampling_param_keys = ["temperature", "top_p", "top_k"]
        resolver = TaskConfigResolver.from_file(args.task_config)
        manifest = {"seed": args.seed, "groups": args.groups, "responses": args.responses,
                    "selected_tasks": identities, "source_task_config_sha256": source_config_sha256,
                    "run_task_config_sha256": digest(point_config),
                    "model_config_sha256": digest(Path(args.model_path).expanduser() / "config.json")}
        manifest["selection_sha256"] = hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()
        atomic_json(run / "manifest.json", manifest)
        # Only explicit public CLI settings; never serialize environment or the
        # entire framework config, which may contain authentication settings.
        atomic_json(run / "settings.json", {**vars(args), "concurrency_unit": "groups",
                                            "max_concurrent_episodes": args.concurrency * args.responses})
        engine_args = argparse.Namespace(**vars(args))
        engine_args.concurrency = args.concurrency * args.responses
        init_kwargs = {"served_model_name": args.served_model_name}
        # The deployed inference helper predates explicit sampling/length CLI
        # controls and requires resolved task defaults. Detect its declared API;
        # never retry after configuration construction has already started.
        if "task_configs" in inspect.signature(init_config).parameters:
            init_kwargs["task_configs"] = list(resolver.defaults_by_name.values())
        config = init_config(engine_args, **init_kwargs)
        for key, value in {
            "actor_rollout_ref.rollout.temperature": args.temperature,
            "actor_rollout_ref.rollout.top_p": args.top_p,
            "actor_rollout_ref.rollout.top_k": args.top_k,
            "actor_rollout_ref.rollout.val_kwargs.temperature": args.temperature,
            "actor_rollout_ref.rollout.val_kwargs.top_p": args.top_p,
            "actor_rollout_ref.rollout.val_kwargs.top_k": args.top_k,
            "actor_rollout_ref.rollout.val_kwargs.do_sample": True,
            "actor_rollout_ref.rollout.prompt_length": args.prompt_length,
            "actor_rollout_ref.rollout.response_length": args.response_length,
            "actor_rollout_ref.rollout.max_model_len": args.prompt_length + args.response_length,
            "actor_rollout_ref.rollout.max_num_batched_tokens": args.prompt_length + args.response_length,
            "actor_rollout_ref.rollout.load_format": "auto",
            "actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype": args.kv_cache_dtype,
            "data.max_prompt_length": args.prompt_length,
            "data.max_response_length": args.response_length,
            "actor_rollout_ref.rollout.custom.agent_framework.allowed_request_sampling_param_keys": args.allowed_request_sampling_param_keys,
            "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode": "inline_async",
            "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.trajectory_selection": "longest",
            "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.report_reward": True,
            "actor_rollout_ref.rollout.custom.agent_framework.use_reward_loop_worker": False,
            "actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode": False,
            "actor_rollout_ref.rollout.custom.agent_framework.framework_class_fqn": "cscs.sandbox_sweep.group_framework.GroupLimitedGatewayAgentFramework",
            "actor_rollout_ref.rollout.custom.agent_framework.max_concurrent_groups": args.concurrency,
        }.items():
            OmegaConf.update(config, key, value, force_add=True)
        ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), runtime_env={"env_vars": {
            "UNI_AGENT_TRACE_DIR": str(trace), "AGENTENV_TRACE_DIR": os.environ["AGENTENV_TRACE_DIR"],
        }})
        tq.init(config.transfer_queue)
        manager = LLMServerManager.create(config=config)
        adapter = AgentFrameworkRolloutAdapter.create(config=config, llm_client=manager.get_client())
        # Synchronous actor construction may be lazy: force actor initialization
        # before starting the workload clock, without executing any episode.
        ray.get(adapter.framework_worker.__ray_ready__.remote())
        outcome.update(state="running", model_ready_at=time.time(),
                       initialization_seconds=time.monotonic() - initialized)
        uids = [str(uuid4()) for _ in samples]
        atomic_json(run / "uids.json", {"groups": [dict(uid=uid, **identity) for uid, identity in zip(uids, identities, strict=True)]})
        prompts = _build_prompts(samples, uids)
        tu.assign_non_tensor_data(prompts, "validate", False)
        tu.assign_non_tensor_data(prompts, "global_steps", 1)
        tq.kv_batch_put(keys=uids, partition_id="train", tags=[{"is_prompt": True, "status": "pending", "global_steps": 1} for _ in uids])
        progress = Progress(run, uids, args.responses, tq)
        progress.thread.start()
        atomic_json(run / "result.json", outcome)
        adapter.generate_sequences_and_wait(prompts)
        rollout_wall = time.monotonic() - progress.started
        progress.stop.set()
        progress.thread.join(timeout=15)
        if progress.thread.is_alive():
            raise RuntimeError("Progress poll did not stop; refusing concurrent final snapshot")
        last = progress.sample()
        progress.persist(last)
        read = _read_rm_scores(uids, partition_id="train")
        expected = args.groups * args.responses
        terminal = last["groups_terminal"] == args.groups
        outcome.update(state="completed" if terminal else "incomplete", finished_at=time.time(),
                       rollout_wall_seconds=rollout_wall, progress=last,
                       all_episodes_usable=last["episodes_with_usable_trajectory"] == expected,
                       harbor=harbor_counts(run),
                       mean_rm_score=sum(read["scores"]) / len(read["scores"]) if read["scores"] else None,
                       scored_sessions=len(read["scores"]), scores_by_uid=read["per_uid"])
    except BaseException as exc:
        outcome.update(state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                       finished_at=time.time(), error_class=type(exc).__name__,
                       progress=progress.latest if progress else None,
                       harbor=harbor_counts(run))
        if progress:
            outcome["rollout_wall_seconds"] = time.monotonic() - progress.started
        raise
    finally:
        if progress:
            progress.stop.set()
            progress.thread.join(timeout=15)
        atomic_json(run / "result.json", outcome)
        ray.shutdown()


if __name__ == "__main__":
    main()
