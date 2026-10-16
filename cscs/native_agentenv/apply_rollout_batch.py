#!/usr/bin/env python3
"""Add an independent separate_async rollout prompt batch to a pinned runtime."""
import argparse
import ast
import re
from pathlib import Path


def apply(runtime):
    path = runtime / "verl/verl/trainer/ppo/v1/trainer_base.py"
    source = path.read_text()
    marker = "# Independent rollout dispatch size; optimizer batch size is unchanged."
    if marker in source:
        return
    method = re.search(r"(?m)^    def _add_batch_to_generate\([^\n]*\):\n", source)
    if method is None:
        raise RuntimeError("Cannot locate _add_batch_to_generate in pinned runtime")
    end = source.find("\n    def ", method.end())
    if end < 0:
        end = len(source)
    block = source[method.start():end]
    call = re.search(r"(?m)^        batch = self\._next_train_batch\(([^\n]*)\)$", block)
    if call is None or "num_prompts=" in call[1]:
        raise RuntimeError("Unexpected _next_train_batch call in pinned runtime")
    kwargs = ", " + call[1] if call[1] else ""
    replacement = '''        # Independent rollout dispatch size; optimizer batch size is unchanged.
        num_prompts = None
        if self.trainer_mode == "separate_async":
            num_prompts = self.config.trainer.v1.separate_async.get("rollout_prompt_batch_size")
            if num_prompts is not None and (
                isinstance(num_prompts, bool) or not isinstance(num_prompts, int) or num_prompts <= 0
            ):
                raise ValueError("rollout_prompt_batch_size must be a positive integer or null")
        batch = self._next_train_batch(num_prompts=num_prompts''' + kwargs + ")"
    block = block[:call.start()] + replacement + block[call.end():]
    updated = source[:method.start()] + block + source[end:]
    ast.parse(updated)
    path.write_text(updated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path)
    args = parser.parse_args()
    apply(args.runtime)
    print("Independent rollout prompt batch support prepared")
