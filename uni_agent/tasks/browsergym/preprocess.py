"""Build Uni-Agent parquet rows for a BrowserGym benchmark (MiniWoB++ first).

Enumerates task envs through browsergym's own registry and emits the same row
shape Terminal-Bench uses (``data_source`` / ``prompt`` /
``extra_info.tools_kwargs.task``), so the standard inference + RL pipelines
consume it unchanged. Run on a host with the benchmark packages installed::

    pip install browsergym-core browsergym-miniwob
    python -m uni_agent.tasks.browsergym.preprocess \\
        --benchmark browsergym.miniwob \\
        --image ghcr.io/<owner>/browsergym:latest \\
        --output browsergym_miniwob.parquet

Only the *listing* needs browsergym on the host; episodes themselves run inside
the sandbox image (browsergym + headless Chromium live there, not here).
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

TASK_NAME = "browsergym"

#: Known benchmark registry names -> (pip extras, default episode timeout).
BENCHMARKS: dict[str, dict[str, Any]] = {
    "browsergym.miniwob": {"pip": "browsergym-miniwob", "timeout_s": 300.0},
    # Add e.g. "browsergym.webarena" when you host the WebArena stack; its
    # sites are external, so also route their base URLs through the row's
    # env_kwargs_json (the sandbox needs them at env-construction time).
}


def list_env_tasks(benchmark: str) -> list[str]:
    """Enumerate the benchmark's env names via browsergym's registry.

    browsergym >= 0.4 registers tasks with gymnasium under ids like
    ``browsergym/miniwob.click-test`` (listing = registry scan after importing
    the benchmark module); older versions expose ``get_env_task_list``.
    """
    import importlib

    importlib.import_module(benchmark)  # e.g. browsergym.miniwob -> registers tasks

    try:  # legacy helper (browsergym < 0.4)
        from browsergym.utils import get_env_task_list

        tasks = list(get_env_task_list(benchmark))
    except ImportError:
        import gymnasium as gym

        family = benchmark.split(".", 1)[1]  # "browsergym.miniwob" -> "miniwob"
        tasks = sorted(i for i in gym.envs.registry if i.startswith(f"browsergym/{family}."))

    if not tasks:
        raise ValueError(f"browsergym listed no tasks for {benchmark!r} (is the package installed?)")
    return tasks


def build_row(
    env_name: str,
    benchmark: str,
    *,
    seed: int,
    image: str,
    timeout_s: float,
    obs_max_chars: int,
) -> dict[str, Any]:
    """Build one provider-agnostic Uni-Agent sample."""
    short_name = env_name.replace("/", ".").rsplit(".", 1)[-1]
    metadata = {
        "instance_id": f"{benchmark}/{short_name}",
        "task_id": env_name,
        "benchmark": benchmark,
        "seed": int(seed),
        "timeout_s": float(timeout_s),
        "obs_max_chars": int(obs_max_chars),
        # Headless-only in sandboxes; slow_mo/viewport can ride along here too.
        "env_kwargs_json": json.dumps({"headless": True}, sort_keys=True),
    }
    task_config = {
        "name": TASK_NAME,
        "sandbox": {"image": image},
        "metadata": metadata,
    }
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                goal="(set at episode start)", obs="(the initial page observation)", url="(the initial page url)"
            ),
        },
    ]
    return {
        "data_source": f"{TASK_NAME}/{benchmark}",
        "prompt": prompt,
        "extra_info": {"tools_kwargs": {"task": task_config}},
    }


def build_browsergym(
    benchmark: str,
    *,
    image: str,
    seed_base: int = 0,
    max_instances: int | None = None,
    timeout_s: float | None = None,
    obs_max_chars: int = 30000,
):
    """Build a Hugging Face Dataset of BrowserGym rows for one benchmark."""
    from datasets import Dataset

    timeout = float(timeout_s if timeout_s is not None else BENCHMARKS.get(benchmark, {}).get("timeout_s", 300.0))
    env_names = list_env_tasks(benchmark)
    if max_instances is not None and max_instances >= 0:
        env_names = env_names[:max_instances]

    rows = []
    for index, env_name in enumerate(env_names):
        print(f"[{index + 1}/{len(env_names)}] {env_name}", flush=True)
        rows.append(
            build_row(
                env_name,
                benchmark,
                seed=seed_base + index,
                image=image,
                timeout_s=timeout,
                obs_max_chars=obs_max_chars,
            )
        )
    return Dataset.from_list(rows)


def main() -> None:
    known = sorted(BENCHMARKS)
    parser = argparse.ArgumentParser(description="Preprocess BrowserGym benchmarks for Uni-Agent.")
    parser.add_argument(
        "--benchmark",
        default="browsergym.miniwob",
        help=f"browsergym registry name (known: {known}; any installed benchmark works)",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="BrowserGym sandbox image the rows should run in (e.g. ghcr.io/you/browsergym:latest)",
    )
    parser.add_argument("--output", required=True, help="Output parquet path.")
    parser.add_argument("--seed-base", type=int, default=0, help="First episode seed; row i uses seed_base + i.")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of tasks (smoke tests).")
    parser.add_argument("--timeout", type=float, default=None, help="Per-episode agent timeout (s).")
    parser.add_argument("--obs-max-chars", type=int, default=30000, help="Observation text cap surfaced to the model.")
    args = parser.parse_args()

    dataset = build_browsergym(
        args.benchmark,
        image=args.image,
        seed_base=args.seed_base,
        max_instances=args.limit,
        timeout_s=args.timeout,
        obs_max_chars=args.obs_max_chars,
    )
    dataset.to_parquet(args.output)
    print(f"wrote {len(dataset)} rows to {args.output}")


if __name__ == "__main__":
    main()
