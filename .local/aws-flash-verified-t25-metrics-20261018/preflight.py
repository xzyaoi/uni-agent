import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import urllib.request

source = Path(os.environ["REPO_DIR"])
for name in ("flash_sandbox", "uni_agent.sandbox.flash_sandbox", "verl", "uni_agent.tasks.harbor.verified_flash_environment"):
    module = importlib.import_module(name)
    location = Path(module.__file__).resolve()
    if not location.is_relative_to(source):
        raise RuntimeError(f"Unexpected {name} import: {location}")
    print(f"[preflight] {socket.gethostname()} {name}={location}", flush=True)
from flash_sandbox import AsyncHTTPClient
from uni_agent.tasks.harbor.verified_flash_environment import VerifiedFlashEnvironment
archive = Path(os.environ["FLASH_SANDBOX_UV_ARCHIVE"])
digest = hashlib.sha256()
with archive.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != os.environ["FLASH_SANDBOX_UV_ARCHIVE_SHA256"]:
    raise RuntimeError("uv archive SHA256 mismatch")
import pyarrow.parquet as pq
rows = pq.read_table(os.environ["TRAIN_FILE"], columns=["extra_info"]).to_pylist()
if len(rows) != 500:
    raise RuntimeError(f"Expected 500 Verified tasks, got {len(rows)}")
for row in rows:
    task = row["extra_info"]["tools_kwargs"]["task"]
    task_path = Path(task["metadata"]["task_path"])
    if task.get("name") != "harbor" or not task_path.is_absolute() or not (task_path / "task.toml").is_file():
        raise RuntimeError(f"Invalid Verified Harbor task: {task_path}")
print(f"[preflight] {socket.gethostname()} uv hash and 500 Verified task paths valid", flush=True)
request = urllib.request.Request(
    os.environ["FLASH_SANDBOX_ENDPOINT"].rstrip("/") + "/status",
    headers={"Authorization": "Bearer " + os.environ["OTELA_API_KEY"]},
)
with urllib.request.urlopen(request, timeout=20) as response:
    if response.status != 200:
        raise RuntimeError(f"Scheduler status HTTP {response.status}")
    status = json.load(response)
healthy = [node for node in status.get("nodes", [])
           if node.get("status") == "healthy" and "firecracker" in node.get("backend_types", [])]
if not healthy:
    raise RuntimeError("Scheduler has no healthy firecracker node")
print(f"[preflight] {socket.gethostname()} scheduler reachable", flush=True)

import subprocess
import yaml
task_config = json.loads(Path(os.environ["TASK_CONFIG"]).read_text())
assert task_config[0]["agent"]["kwargs"]["max_steps"] == 25
assert os.environ["VLLM_LOGGING_LEVEL"] == "INFO"
assert Path(os.environ["RUN_DIR"], "observability", "collect_usage.py").is_file()
check = subprocess.run(
    ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
if check.returncode or len(check.stdout.strip().splitlines()) != 4:
    raise RuntimeError("Expected four readable GPUs for persistent telemetry")
print(f"[preflight] {socket.gethostname()} 25-turn cap and GPU telemetry prerequisites valid", flush=True)
