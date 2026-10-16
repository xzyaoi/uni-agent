"""End-to-end BrowserGym smoke run against flash-sandbox (no trainer needed).

Boots one flash sandbox from the BrowserGym image, starts the bridge daemon,
runs one react episode against a single MiniWoB++ task, and prints the env's
own verdict (reward / success / steps). Use this to validate image + daemon +
tool wiring before wiring RL.

Usage::

    export OTELA_API_KEY=...                # flash-sandbox API key
    export BROWSERGYM_IMAGE=ghcr.io/you/browsergym:latest
    python smoke_flash.py \\
        --task browsergym.miniwob.miniwob_click-test \\
        --base-url http://localhost:8000/v1 --model-name Qwen3-Coder-30B \\
        [--seed 0] [--limit-steps 10]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from uni_agent.tasks import get_task  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(description="One BrowserGym episode over flash-sandbox.")
    parser.add_argument("--task", default="browsergym.miniwob.miniwob_click-test")
    parser.add_argument("--benchmark", default="browsergym.miniwob")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-steps", type=int, default=10, help="Cap the agent loop (smoke).")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible endpoint the agent talks to.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=32768,
        help="Episode token budget (some endpoints cap per-request max_tokens below this).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    image = os.environ.get("BROWSERGYM_IMAGE")
    task = get_task(
        {
            "name": "browsergym",
            "sandbox": {"provider": "flash_sandbox", **({"image": image} if image else {})},
            "agent": {
                "name": "react",
                "max_steps": args.max_steps,
                "tools": [
                    {"name": "browser", "command_timeout": 90, "obs_max_chars": 30000},
                    {"name": "submit"},
                ],
                "model": {
                    "base_url": args.base_url,
                    "api_key": args.api_key,
                    "model_name": args.model_name,
                    "temperature": 1.0,
                    "max_total_tokens": args.max_total_tokens,
                },
            },
            "metadata": {
                "instance_id": f"smoke/{args.task.rsplit('.', 1)[-1]}",
                "task_id": args.task,
                "benchmark": args.benchmark,
                "seed": args.seed,
                "timeout_s": args.timeout,
                "obs_max_chars": 30000,
                "env_kwargs_json": json.dumps({"headless": True}),
            },
        }
    )
    result = await task.run()
    print("\n===== TaskResult =====")
    print(json.dumps(result.extra_info, indent=2, default=str))
    print(f"reward={result.reward} finished={result.finished}")
    return 0 if result.extra_info.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
