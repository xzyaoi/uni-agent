"""Prepare WebArena rows from the registry manifest exported by its browser image."""

import json
import os
from pathlib import Path

from datasets import Dataset

from uni_agent.tasks.browsergym.preprocess import build_row

root = Path(os.environ["WORK_ROOT"])
manifest = Path(os.environ.get("WEBARENA_TASK_IDS", str(root / "data/webarena-task-ids.json")))
task_ids = json.loads(manifest.read_text())
if not isinstance(task_ids, list) or not task_ids or len(set(task_ids)) != len(task_ids):
    raise ValueError("Expected a nonempty JSON list of unique BrowserGym WebArena registry IDs")
if any(not isinstance(task, str) or not task.startswith("browsergym/webarena.") for task in task_ids):
    raise ValueError("Task manifest must contain only browsergym/webarena.* IDs")
rows = [
    build_row(
        task, "browsergym.webarena", seed=index, image=os.environ["WEBARENA_IMAGE"], timeout_s=3600, obs_max_chars=30000
    )
    for index, task in enumerate(task_ids)
]
output = root / "data/webarena.parquet"
Dataset.from_list(rows).to_parquet(output)
print(f"Prepared {len(rows)} WebArena tasks: {output}")
