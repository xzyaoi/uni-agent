"""Copy official V2 Harbor tasks for online Flash Sandbox training."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from datasets import Dataset

from .preprocess import build_swe_bench_pro


def _online_agent_config(text: str) -> str:
    """Change only the offline policy value in the exact [agent] section."""
    sections = list(re.finditer(r"^\s*\[([^\]\r\n]+)\][ \t]*(?:#.*)?$", text, re.MULTILINE))
    agents = [index for index, section in enumerate(sections) if section.group(1) == "agent"]
    if len(agents) != 1:
        raise ValueError("Expected exactly one [agent] section in task.toml")
    index = agents[0]
    start = sections[index].end()
    end = sections[index + 1].start() if index + 1 < len(sections) else len(text)
    section = text[start:end]
    pattern = r"^([ \t]*network_mode[ \t]*=[ \t]*)([\"'])no-network\2([ \t]*(?:#.*)?\r?$)"
    section, count = re.subn(pattern, r"\g<1>\g<2>public\g<2>\g<3>", section, flags=re.MULTILINE)
    if count != 1:
        raise ValueError('Expected exactly one agent network_mode = "no-network" in task.toml')
    return text[:start] + section + text[end:]


def prepare_training(
    local_save_dir: str,
    *,
    dataset_config: str = "default",
    max_instances: int | None = None,
    tasks_dir: str | None = None,
) -> Path:
    """Create distinct online task copies; leave official source assets untouched."""
    if dataset_config not in {"default", "hard"}:
        raise ValueError("Training preparation supports only V2 default and hard configurations")
    destination = Path(local_save_dir).expanduser().resolve()
    suffix = "v2_hard" if dataset_config == "hard" else "v2"
    output = destination / f"swe_bench_pro_{suffix}_train.parquet"
    if output.exists():
        raise FileExistsError(f"Training dataset already exists; choose a fresh destination: {output}")
    dataset = build_swe_bench_pro(dataset_config=dataset_config, max_instances=max_instances, tasks_dir=tasks_dir)
    if not len(dataset):
        raise ValueError("Training preparation requires at least one selected task")
    copies = []
    rows = []
    # Check every selected task before making any copies.
    for row in dataset:
        metadata = row["extra_info"]["tools_kwargs"]["task"]["metadata"]
        source = Path(metadata["task_path"])
        target = destination / "tasks" / metadata["instance_id"]
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Task copy already exists; choose a fresh destination: {target}")
        config = _online_agent_config((source / "task.toml").read_bytes().decode("utf-8"))
        copies.append((source, target, config))
        metadata["task_path"] = str(target)
        metadata["training_network_mode"] = "public"
        rows.append(row)

    destination.mkdir(parents=True, exist_ok=True)
    for source, target, config in copies:
        shutil.copytree(source, target)
        (target / "task.toml").write_bytes(config.encode("utf-8"))
    Dataset.from_list(rows).to_parquet(output)
    print(f"Wrote {len(rows)} online training tasks to {output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-save-dir", required=True, help="Fresh training destination shared by all workers.")
    parser.add_argument("--dataset-config", choices=["default", "hard"], default="default")
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--tasks-dir", default=None, help="Optional local official V2 tasks directory.")
    args = parser.parse_args()
    prepare_training(**vars(args))


if __name__ == "__main__":
    main()
