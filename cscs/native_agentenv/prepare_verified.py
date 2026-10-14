"""Prepare all Verified tasks, using GHCR mirror images on native AgentENV."""

import os
from pathlib import Path

from datasets import Dataset

from uni_agent.tasks.swe_bench.preprocess import build_swe_bench_verified

root = Path(os.environ["WORK_ROOT"])
rows = list(build_swe_bench_verified())
for row in rows:
    sandbox = row["extra_info"]["tools_kwargs"]["task"]["sandbox"]
    canonical = sandbox["image"]
    if not canonical.startswith("swebench/sweb.eval.x86_64."):
        raise ValueError(f"Unexpected Verified image: {canonical}")
    sandbox["image"] = "ghcr.io/swe-images/" + canonical.split("/", 1)[1]
output = root / "data/swebench-verified.parquet"
Dataset.from_list(rows).to_parquet(output)
print(f"Prepared {len(rows)} SWE-bench Verified tasks: {output}")
