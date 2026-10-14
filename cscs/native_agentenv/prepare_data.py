"""Relocate bundled Harbor task paths to the receiving account's scratch."""

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

root = Path(os.environ["WORK_ROOT"]).resolve()
source = root / "data/swe_bench_pro_v2_agentenv_train.parquet"
table = pq.read_table(source)
rows = table.to_pylist()
if len(rows) != 642:
    raise RuntimeError(f"Expected all 642 V2 tasks, found {len(rows)}")
for row in rows:
    metadata = row["extra_info"]["tools_kwargs"]["task"]["metadata"]
    task = root / "data/tasks" / metadata["instance_id"]
    if not (task / "task.toml").is_file():
        raise FileNotFoundError(task / "task.toml")
    metadata["task_path"] = str(task)
target = root / "data/train.parquet"
pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), target)
print(f"Prepared {len(rows)} task rows: {target}")
