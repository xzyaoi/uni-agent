#!/usr/bin/env python3
"""Freeze the existing Clariden launcher for this authorized Flash training run."""
from __future__ import annotations

import json
from pathlib import Path
import re
import shlex
import subprocess
import tomllib

OUT = Path(__file__).resolve().parent
BASE = OUT / "base"
RUN = "/iopsstor/scratch/cscs/xyao/uni-agent/runs/qwen35_27b_verified_flash_2n45m_c512_20261018"
SOURCE = f"{RUN}/source"
PYTHONPATH = f"{SOURCE}/sdk/python:{SOURCE}:{SOURCE}/verl"
PROJECT = "Uni-Agent-Qwen3.5-27B-SWE-bench-Verified-Flash"


def replace_once(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise ValueError(f"Expected one occurrence: {before!r}")
    return text.replace(before, after, 1)


settings = {
    "RUN_DIR": RUN,
    "SCRATCH": "/iopsstor/scratch/cscs/xyao",
    "REPO_DIR": SOURCE,
    "PYTHONPATH": PYTHONPATH,
    "ENV_NAME": "flash-training",
    "EXP_NAME": "qwen35_27b_verified_flash_2n45m_c512_20261018",
    "PROJECT_NAME": PROJECT,
    "MODEL_PATH": "/iopsstor/scratch/cscs/xyao/data/models/Qwen3.5-27B",
    "TRAIN_FILE": "/iopsstor/scratch/cscs/xyao/data/uni_agent/harbor_verified_native_20261016_v2/swebench_verified_harbor.parquet",
    "TEST_FILE": "/iopsstor/scratch/cscs/xyao/data/uni_agent/harbor_verified_native_20261016_v2/swebench_verified_harbor.parquet",
    "TASK_CONFIG": f"{RUN}/task_config.yaml",
    "RUNTIME_ENV": f"{RUN}/runtime_env.json",
    "CKPTS_DIR": f"{RUN}/checkpoints",
    "AGENT_LOG_DIR": f"{RUN}/agent-logs",
    "TRAINING_STEPS": "5",
    "TRAINER_MODE": "colocate_async",
    "NNODES_ROLLOUT": "0",
    "GEN_TP": "4", "TP": "4", "PP": "2", "CP": "1",
    "TRAIN_PROMPT_BSZ": "64", "N_RESP_PER_PROMPT": "8", "PPO_MINI_BATCH_SIZE": "16",
    "CONCURRENCY": "512", "GATEWAY_COUNT": "2", "NUM_WARMUP_BATCHES": "1",
    "MAX_PROMPT_LENGTH": "8192", "MAX_RESPONSE_LENGTH": "8192",
    "MAX_NUM_BATCHED_TOKENS": "8192",
    "VLLM_GPU_MEM_UTIL": "0.45", "OFFLOAD": "True", "OFFLOAD_FRACTION": "1.0",
    "SAVE_FREQ": "1", "TEST_FREQ": "-1", "LOGGER": "['console']",
    "FLASH_SANDBOX_ENDPOINT": "http://172.28.38.18:28090/",
    "FLASH_SANDBOX_TYPE": "firecracker", "FLASH_SANDBOX_IMAGE_REGISTRY": "",
    "FLASH_SANDBOX_START_TIMEOUT_S": "900",
    "FLASH_SANDBOX_UV_ARCHIVE": "/iopsstor/scratch/cscs/xyao/data/uni_agent/assets/uv-0.7.13-uv-x86_64-unknown-linux-gnu.tar.gz",
    "FLASH_SANDBOX_UV_ARCHIVE_SHA256": "909278eb197c5ed0e9b5f16317d1255270d1f9ea4196e7179ce934d48c4c2545",
    "GPUS_PER_NODE": "4", "NGPUS_PER_NODE": "4", "CPUS_PER_NODE": "288",
}
(OUT / "run.env").write_text("".join(f"export {key}={shlex.quote(value)}\n" for key, value in settings.items()))

launcher = (BASE / "run_train_qwen3p5_clariden.sh").read_text()
# Credentials are explicitly supplied by batch.sh; keep launch logs console-only.
start = launcher.index('# ---- wandb:')
end = launcher.index('REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"', start)
launcher = launcher[:start] + 'unset WANDB_API_KEY WANDB_ENTITY\nLOGGER="[\'console\']"\n\n' + launcher[end:]
launcher = replace_once(launcher, 'REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"', f'REPO_DIR="{SOURCE}"')
launcher = replace_once(launcher, 'HERE="${REPO_DIR}"', f'HERE="{RUN}"')
launcher = replace_once(launcher, '    TRAIN_SCRIPT="examples/quickstart/training/train_qwen3p5_dense.sh"', f'    TRAIN_SCRIPT="{RUN}/train.sh"')
launcher = replace_once(launcher, '    bash examples/quickstart/training/run_ray_node.sh', f'    bash "{RUN}/run_ray_node.sh"')
launcher = replace_once(launcher, 'FLASH_SANDBOX_IMAGE_REGISTRY="${FLASH_SANDBOX_IMAGE_REGISTRY:-ghcr.io/swe-images}"', 'FLASH_SANDBOX_IMAGE_REGISTRY="${FLASH_SANDBOX_IMAGE_REGISTRY-}"')
extra = ["TRAINING_STEPS", "PROJECT_NAME", "CKPTS_DIR", "AGENT_LOG_DIR", "PYTHONPATH", "TEST_FREQ", "OFFLOAD", "OFFLOAD_FRACTION", "FLASH_SANDBOX_START_TIMEOUT_S", "FLASH_SANDBOX_UV_ARCHIVE", "FLASH_SANDBOX_UV_ARCHIVE_SHA256"]
launcher = replace_once(launcher, "RUN_ENV=(\n", "RUN_ENV=(\n" + "".join(f'    {key}="${{{key}}}"\n' for key in extra))
(OUT / "launcher.sh").write_text(launcher)

train = (BASE / "train_qwen3p5_dense.sh").read_text()
if "trainer.total_training_steps=" in train:
    raise ValueError("Review existing total_training_steps override")
train = replace_once(train, "    trainer.total_epochs=${total_epochs} \\\n", "    trainer.total_epochs=${total_epochs} \\\n    trainer.total_training_steps=${TRAINING_STEPS} \\\n")
(OUT / "train.sh").write_text(train)
(OUT / "run_ray_node.sh").write_text((BASE / "run_ray_node.sh").read_text())

runtime = {
    "env_vars": {
        "PYTHONPATH": PYTHONPATH, "PYTHONNOUSERSITE": "1",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "MEGATRON_CONFIG_LOCK_DIR": "/tmp/megatron-config-locks",
        "XDG_CACHE_HOME": "/tmp/xdg-cache", "TRITON_CACHE_DIR": "/tmp/triton",
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/torchinductor", "HF_HOME": "/tmp/hf",
        "ROCR_VISIBLE_DEVICES": "", "FLASH_SANDBOX_ENDPOINT": settings["FLASH_SANDBOX_ENDPOINT"],
        "FLASH_SANDBOX_TYPE": "firecracker", "FLASH_SANDBOX_IMAGE_REGISTRY": "",
        "FLASH_SANDBOX_START_TIMEOUT_S": settings["FLASH_SANDBOX_START_TIMEOUT_S"],
        "FLASH_SANDBOX_UV_ARCHIVE": settings["FLASH_SANDBOX_UV_ARCHIVE"],
        "FLASH_SANDBOX_UV_ARCHIVE_SHA256": settings["FLASH_SANDBOX_UV_ARCHIVE_SHA256"],
    }
}
(OUT / "runtime_env.json").write_text(json.dumps(runtime, indent=2) + "\n")
task = [{
    "name": "harbor",
    "harbor_executable": "/iopsstor/scratch/cscs/xyao/venvs/uni-harbor-v2/bin/harbor",
    "harbor_env": "uni_agent.tasks.harbor.verified_flash_environment:VerifiedFlashEnvironment",
    "artifacts_dir": f"{RUN}/harbor", "override_cpus": 4, "override_memory_mb": 8192,
    "agent": {"name": "uni_agent.tasks.harbor.react_agent:UniAgentReAct", "timeout_sec": 900,
              "kwargs": {"max_steps": 64, "temperature": 1.0, "top_p": 1.0}},
}]
# JSON is also YAML, avoiding a generator dependency on PyYAML.
(OUT / "task_config.yaml").write_text(json.dumps(task, indent=2) + "\n")
edf = (BASE / "harbor-swe-megatron-clariden.toml").read_text()
edf = replace_once(edf, 'workdir = "${SCRATCH}/uni-agent"', f'workdir = "{SOURCE}"')
edf = replace_once(edf, 'PYTHONPATH = "${SCRATCH}/uni-agent/verl"', f'PYTHONPATH = "{PYTHONPATH}"')
(OUT / "flash-training.toml").write_text(edf)
tomllib.loads(edf)

batch = f'''#!/usr/bin/env bash
#SBATCH --job-name=qwen35-27b-vfy-c512
#SBATCH --partition=preemptable
#SBATCH --account=infra02
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --mem=850000M
#SBATCH --time=00:45:00
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --output={RUN}/training.out
set -euo pipefail
set +x
source "{RUN}/run.env"
set -a
source "{RUN}/credentials.env"
set +a
: "${{OTELA_API_KEY:?Missing Flash scheduler credential}}"
unset WANDB_API_KEY WANDB_ENTITY
export SLURM_EXPORT_ENV=ALL
export EDF_PATH="{RUN}:${{EDF_PATH:-}}"
export TMPDIR=/tmp
export NO_PROXY="172.28.38.18,${{NO_PROXY:-}}"
export no_proxy="${{NO_PROXY}}"
export RAY_PORT=$((10000 + SLURM_JOB_ID % 40000))
mkdir -p "{RUN}/checkpoints" "{RUN}/agent-logs" "{RUN}/harbor"
echo '[preflight] Checking frozen imports and scheduler connectivity on both allocated nodes'
srun --export=ALL,TMPDIR=/tmp --nodes=2 --ntasks=2 --ntasks-per-node=1 \\
    --environment=flash-training python "{RUN}/preflight.py"
exec bash "{RUN}/launcher.sh"
'''
(OUT / "batch.sh").write_text(batch)
preflight = '''import hashlib
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
'''
(OUT / "preflight.py").write_text(preflight)
for name in ("launcher.sh", "train.sh", "run_ray_node.sh", "batch.sh"):
    (OUT / name).chmod(0o750)
    subprocess.run(["bash", "-n", str(OUT / name)], check=True)
(OUT / "submit.json").write_text(json.dumps({"command": ["sbatch", "--parsable", "--partition=preemptable", "--export=ALL", f"--chdir={RUN}", f"{RUN}/batch.sh"], "run_dir": RUN}, indent=2) + "\n")
print(f"Prepared frozen run files in {OUT}")
