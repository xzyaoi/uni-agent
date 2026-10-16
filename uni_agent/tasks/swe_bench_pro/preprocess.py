"""Preprocess SWE-bench Pro: native v1 tasks or official v2 Harbor tasks."""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
from typing import Any

from datasets import Dataset, DownloadManager, load_dataset

DATA_SOURCE = "ScaleAI/SWE-bench_Pro"
DATASET_CONFIG = "v1"
EXPECTED_INSTANCES = 731
# The upstream repository has no v1 tag. Pin the final commit that changed the
# legacy v1 run_scripts so preprocessing cannot drift with the main branch.
RUN_SCRIPTS_COMMIT = "a19d238c0eff0d8585da673760cf46d8c20d3520"
RUN_SCRIPTS_ARCHIVE = f"https://github.com/scaleapi/SWE-bench_Pro-os/archive/{RUN_SCRIPTS_COMMIT}.tar.gz"
V2_TASKS_COMMIT = "66f92766bba642462d4bbe5479e83f91f9211862"
V2_TASKS_ARCHIVE = f"https://github.com/scaleapi/SWE-bench_Pro-os/archive/{V2_TASKS_COMMIT}.tar.gz"
V2_DATASET_REVISION = "2d52cb3df914a3fcf80c7f66738b3a88ae37fc50"
EXPECTED_V2_INSTANCES = {"default": 642, "hard": 51}


def _build_v2(dataset_config: str, max_instances: int | None, tasks_dir: str | None) -> Dataset:
    """Match Hugging Face rows to the release's self-contained Harbor tasks."""
    dataset = load_dataset(DATA_SOURCE, dataset_config, split="test", revision=V2_DATASET_REVISION)
    expected = EXPECTED_V2_INSTANCES[dataset_config]
    if len(dataset) != expected:
        raise RuntimeError(
            f"Expected SWE-bench Pro {dataset_config} to contain {expected} instances, got {len(dataset)}"
        )
    if max_instances is not None:
        dataset = dataset.select(range(min(max_instances, len(dataset))))

    if tasks_dir:
        root = Path(tasks_dir).expanduser().resolve()
    else:
        archive = Path(DownloadManager().download_and_extract(V2_TASKS_ARCHIVE))
        candidates = [archive / "v2/tasks", *archive.glob("*/v2/tasks")]
        root = next((path for path in candidates if path.is_dir()), archive / "v2/tasks").resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"SWE-bench Pro v2 tasks directory not found: {root}")

    rows = []
    for example in dataset:
        instance_id = example["instance_id"]
        # Only resolve immediate children of the task root.
        if Path(instance_id).name != instance_id or instance_id in {".", ".."}:
            raise ValueError(f"Invalid SWE-bench Pro instance_id: {instance_id!r}")
        task_dir = root / instance_id
        for filename in ("task.toml", "instruction.md", "tests/test.sh"):
            if not (task_dir / filename).is_file():
                raise FileNotFoundError(f"Missing SWE-bench Pro v2 asset: {task_dir / filename}")
        instruction = (task_dir / "instruction.md").read_text(encoding="utf-8").strip()
        rows.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": [{"role": "user", "content": instruction}],
                "extra_info": {
                    "tools_kwargs": {
                        "task": {
                            "name": "harbor",
                            "metadata": {
                                "instance_id": instance_id,
                                "dataset": DATA_SOURCE,
                                "dataset_config": dataset_config,
                                "dataset_revision": V2_DATASET_REVISION,
                                "tasks_revision": V2_TASKS_COMMIT,
                                "task_path": str(task_dir),
                            },
                        }
                    }
                },
            }
        )
    return Dataset.from_list(rows)


def _parse_string_list(value: object, field_name: str) -> list[str]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"{field_name} must be a serialized list of strings") from exc
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    return value


def _resolve_run_scripts_dir(scripts_dir: str | None) -> Path:
    if scripts_dir:
        run_scripts_dir = Path(scripts_dir).expanduser()
    else:
        archive = Path(DownloadManager().download_and_extract(RUN_SCRIPTS_ARCHIVE))
        candidates = [archive / "run_scripts", *archive.glob("*/run_scripts")]
        run_scripts_dir = next((path for path in candidates if path.is_dir()), archive / "run_scripts")

    if not run_scripts_dir.is_dir():
        raise FileNotFoundError(f"SWE-bench Pro run_scripts directory not found: {run_scripts_dir}")
    return run_scripts_dir


def build_swe_bench_pro(
    max_instances: int | None = None,
    *,
    scripts_dir: str | None = None,
    dataset_config: str = DATASET_CONFIG,
    tasks_dir: str | None = None,
):
    """Convert v1 natively, or select default/hard for official v2 Harbor tasks.

    The Python API keeps v1 as its default for existing callers.
    V2 task assets must remain accessible at the same paths on every worker.
    """
    if max_instances is not None and max_instances < 0:
        raise ValueError("max_instances must be non-negative")
    if dataset_config not in {"v1", *EXPECTED_V2_INSTANCES}:
        raise ValueError(f"Unknown SWE-bench Pro config: {dataset_config!r}")
    if dataset_config != "v1":
        if scripts_dir:
            raise ValueError("scripts_dir is only supported for v1; use tasks_dir for v2")
        return _build_v2(dataset_config, max_instances, tasks_dir)
    if tasks_dir:
        raise ValueError("tasks_dir is only supported for v2; use scripts_dir for v1")
    run_scripts_dir = _resolve_run_scripts_dir(scripts_dir)

    def process(example: dict[str, Any]) -> dict[str, Any]:
        requirements = example.get("requirements") or ""
        interface = example.get("interface") or ""
        problem_statement = (
            f"{example['problem_statement']}\n\n"
            f"Requirements:\n{requirements}\n\n"
            f"New interfaces introduced:\n{interface}"
        )
        fail_to_pass = _parse_string_list(example["fail_to_pass"], "fail_to_pass")
        pass_to_pass = _parse_string_list(example["pass_to_pass"], "pass_to_pass")
        selected_tests = _parse_string_list(
            example["selected_test_files_to_run"],
            "selected_test_files_to_run",
        )
        instance_scripts_dir = run_scripts_dir / example["instance_id"]

        metadata = {
            "repo": example["repo"],
            "instance_id": example["instance_id"],
            "base_commit": example["base_commit"],
            "patch": example["patch"],
            "test_patch": example["test_patch"],
            "problem_statement": problem_statement,
            "repo_language": example["repo_language"],
            "fail_to_pass": fail_to_pass,
            "pass_to_pass": pass_to_pass,
            "selected_test_files_to_run": ",".join(selected_tests),
            "run_script": (instance_scripts_dir / "run_script.sh").read_text(encoding="utf-8"),
            "parser": (instance_scripts_dir / "parser.py").read_text(encoding="utf-8"),
        }
        task_config = {
            "name": "swe_bench_pro",
            "sandbox": {"image": f"jefzda/sweap-images:{example['dockerhub_tag']}"},
            "metadata": metadata,
        }
        return {
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": problem_statement}],
            "extra_info": {"tools_kwargs": {"task": task_config}},
        }

    print(f"Loading {DATA_SOURCE} (config={DATASET_CONFIG}) from Hugging Face...", flush=True)
    dataset = load_dataset(DATA_SOURCE, DATASET_CONFIG, split="test")
    print(f"Loaded {len(dataset)} raw instances", flush=True)
    if len(dataset) != EXPECTED_INSTANCES:
        raise RuntimeError(
            f"Expected SWE-bench Pro {DATASET_CONFIG} to contain {EXPECTED_INSTANCES} instances, got {len(dataset)}"
        )

    if max_instances is not None and max_instances >= 0:
        dataset = dataset.select(range(min(max_instances, len(dataset))))
        print(f"Capped to {len(dataset)} instances", flush=True)

    return dataset.map(process, remove_columns=dataset.column_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess SWE-bench Pro for Uni-Agent.")
    parser.add_argument("--local-save-dir", default="~/data/swe_agent")
    parser.add_argument(
        "--dataset-config",
        choices=["default", "hard", "v1"],
        default="default",
        help="Hugging Face config: default (v2), hard (v2 HARD-51), or legacy v1.",
    )
    parser.add_argument("--tasks-dir", default=None, help="Local official v2/tasks directory (shared across workers).")
    parser.add_argument("--scripts-dir", default=None, help="Optional local official run_scripts directory.")
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Optional cap on the number of instances kept (smoke testing).",
    )
    args = parser.parse_args()

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    dataset = build_swe_bench_pro(
        max_instances=args.max_instances,
        scripts_dir=args.scripts_dir,
        dataset_config=args.dataset_config,
        tasks_dir=args.tasks_dir,
    )
    suffix = {"default": "v2", "hard": "v2_hard", "v1": "v1"}[args.dataset_config]
    output_path = os.path.join(save_dir, f"swe_bench_pro_{suffix}.parquet")
    dataset.to_parquet(output_path)
    print(f"Wrote {len(dataset)} instances to {output_path}", flush=True)
