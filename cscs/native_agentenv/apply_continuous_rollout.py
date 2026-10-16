#!/usr/bin/env python3
"""Install bounded continuous refill into the pinned separate-async runtime."""
import argparse
import ast
import shutil
from pathlib import Path


def replace(path, old, new):
    source = path.read_text()
    if new in source:
        return
    if source.count(old) != 1:
        raise RuntimeError(f"Unexpected pinned runtime: cannot patch {path}: {old[:70]}")
    updated = source.replace(old, new, 1)
    ast.parse(updated)
    path.write_text(updated)


def apply(runtime):
    target = runtime / "verl/verl/trainer/ppo/v1"
    module = Path(__file__).parent / "runtime_extensions/continuous_rollout.py"
    shutil.copyfile(module, target / "continuous_rollout.py")
    separate = target / "trainer_separate_async.py"
    replace(separate, "from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer",
            "from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer\n"
            "from verl.trainer.ppo.v1.continuous_rollout import ContinuousRolloutMixin")
    replace(separate, "class PPOTrainerSeparateAsync(PPOTrainer):", "class PPOTrainerSeparateAsync(ContinuousRolloutMixin, PPOTrainer):")
    replace(separate, "        num_warmup_batches = self.config.trainer.v1.separate_async.num_warmup_batches",
            "        if self._continuous_refill_enabled():\n"
            "            self._start_continuous_refill()\n            return\n"
            "        num_warmup_batches = self.config.trainer.v1.separate_async.num_warmup_batches")
    replace(separate, "    def on_sample_begin(self):\n", "    def on_sample_begin(self):\n        self._check_refill_health()\n")
    replace(separate, "    def on_step_end(self):\n", "    def on_step_end(self):\n        self._check_refill_health()\n")
    base = target / "trainer_base.py"
    replace(base, 'tags = [{"is_prompt": True, "status": "pending", "global_steps": self.global_steps} for _ in range(len(batch))]',
            'tags = [{"is_prompt": True, "status": "pending", "global_steps": tu.get(batch, "global_steps")} for _ in range(len(batch))]')
    sampler = target / "replay_buffer.py"
    replace(sampler, "        self.poll_interval = poll_interval\n", "        self.poll_interval = poll_interval\n        self.progress_check = None\n")
    replace(sampler, "    def _wait_for_next_poll(self, partition_id: str, last_debug_time: float) -> float:\n",
            "    def _wait_for_next_poll(self, partition_id: str, last_debug_time: float) -> float:\n"
            "        if self.progress_check is not None:\n            self.progress_check()\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path)
    args = parser.parse_args()
    apply(args.runtime)
    print("Bounded continuous rollout support prepared")
