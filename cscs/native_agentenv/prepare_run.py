"""Generate an immutable run directory using only the receiving account's paths."""

import json
import os
import shlex
import shutil
from pathlib import Path

bundle = Path(os.environ["BUNDLE_ROOT"]).resolve()
root = Path(os.environ["WORK_ROOT"]).resolve()
run = Path(os.environ["RUN_DIR"]).resolve()
image = Path(os.environ["TRAINING_IMAGE"]).resolve()
model = Path(os.environ["MODEL_PATH"]).resolve()
credentials = Path(os.environ["AGENTENV_CREDENTIALS"]).resolve()
runtime = root / "runtime"
run.mkdir()  # Refuse accidental reuse of a previous run.
settings = {
    "REPO_DIR": str(runtime),
    "WORK_ROOT": str(root),
    "RUN_DIR": str(run),
    "SCRATCH": str(Path(os.environ["SCRATCH"]).resolve()),
    "AGENTENV_CREDENTIALS": str(credentials),
    "AGENTENV_ENDPOINT": os.environ["AGENTENV_ENDPOINT"],
    "AGENTENV_TRACE_DIR": str(run / "native-traces"),
    "NATIVE_SHELL_ROOT": str(root / "native-shell"),
    "HARBOR_VENV": str(root / "venv"),
    "TRAINING_IMAGE": str(image),
    "MODEL_PATH": str(model),
    "TRAIN_FILE": str(root / "data/train.parquet"),
    "TEST_FILE": str(root / "data/train.parquet"),
    "TASK_CONFIG": str(run / "task_config.yaml"),
    "TRAIN_SCRIPT": str(run / "train.sh"),
    "RUNTIME_ENV": str(run / "runtime_env.yaml"),
    "RUNTIME_DIR": str(root),
    "DATA_DIR": str(root / "data"),
    "CKPTS_DIR": str(run / "checkpoints"),
    "AGENT_LOG_DIR": str(run / "agent-logs"),
    "PROJECT_NAME": "Uni-Agent-Qwen3.5-27B-SWE-bench-Pro-V2-AgentENV",
    "EXP_NAME": os.environ["EXP_NAME"],
    "TRAINER_MODE": "separate_async",
    "NNODES_ROLLOUT": "2",
    "PARAMETER_SYNC_STEP": "4",
    "GEN_TP": "4",
    "TP": "4",
    "PP": "2",
    "CP": "1",
    "CONCURRENCY": os.environ["CONCURRENCY"],
    "TRAINING_STEPS": os.environ["TRAINING_STEPS"],
    "SAVE_FREQ": os.environ["SAVE_FREQ"],
    "TRAIN_PROMPT_BSZ": "64",
    "N_RESP_PER_PROMPT": "8",
    "PPO_MINI_BATCH_SIZE": "16",
    "VLLM_GPU_MEM_UTIL": "0.45",
    "MAX_PROMPT_LENGTH": "8192",
    "MAX_RESPONSE_LENGTH": "32768",
    "SANDBOX_START_ATTEMPT_TIMEOUT": "600",
    "SANDBOX_START_RETRIES": "1",
    "SANDBOX_START_RETRY_BACKOFF": "0",
    "GPUS_PER_NODE": "4",
    "NGPUS_PER_NODE": "4",
    "CPUS_PER_NODE": "288",
    "ENV_NAME": "native-agentenv",
}
(run / "launch.env").write_text("".join(f"export {key}={shlex.quote(value)}\n" for key, value in settings.items()))
for name in ("train.sh", "run_ray_node.sh", "harbor-native"):
    shutil.copyfile(bundle / "templates" / name, run / name)
    (run / name).chmod(0o750)
shutil.copyfile(bundle / "templates/batch.sh", run / "launcher.sh")
(run / "launcher.sh").chmod(0o750)
task_config = [
    {
        "name": "harbor",
        "harbor_executable": str(run / "harbor-native"),
        "harbor_env": "uni_agent.tasks.harbor.agentenv_environment:AgentEnvEnvironment",
        "artifacts_dir": str(run / "harbor"),
        "override_cpus": 4,
        "override_memory_mb": 4096,
        "agent": {
            "name": "uni_agent.tasks.harbor.react_agent:UniAgentReAct",
            "timeout_sec": 3600,
            "kwargs": {"max_steps": 200, "temperature": 1.0, "top_p": 1.0},
        },
    }
]
(run / "task_config.yaml").write_text(json.dumps(task_config, indent=2) + "\n")
pythonpath = f"{runtime}:{runtime}/verl:{root}/native-shell"
runtime_env = {
    "env_vars": {
        "PYTHONPATH": pythonpath,
        "PYTHONNOUSERSITE": "1",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "MEGATRON_CONFIG_LOCK_DIR": "/tmp/megatron-config-locks",
        "XDG_CACHE_HOME": "/tmp/xdg-cache",
        "AGENTENV_TRACE_DIR": str(run / "native-traces"),
    }
}
(run / "runtime_env.yaml").write_text(json.dumps(runtime_env, indent=2) + "\n")
mount_paths = sorted({root, run, runtime, model.parent, image.parent, Path(settings["SCRATCH"])})
for path in mount_paths:
    if any(char in str(path) for char in (":", "\n", ",")):
        raise ValueError(f"Unsupported mount path: {path}")
env = {
    "PYTHONPATH": pythonpath,
    "PYTHONNOUSERSITE": "1",
    "HOME": str(root / "home"),
    "HF_HOME": str(root / "hf"),
    "RLLM_HOME": str(root / "rllm_home"),
    "PIP_CACHE_DIR": str(root / "pip-cache"),
    "WANDB_DIR": str(root / "wandb"),
    "MEGATRON_CONFIG_LOCK_DIR": "/tmp/megatron-config-locks",
    "XDG_CACHE_HOME": "/tmp/xdg-cache",
    "TOKENIZERS_PARALLELISM": "true",
    "VLLM_LOGGING_LEVEL": "WARN",
    "NCCL_DEBUG": "WARN",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}
edf = (
    f"image = {json.dumps(str(image))}\n"
    f"mounts = {json.dumps([f'{path}:{path}' for path in mount_paths])}\n"
    f"workdir = {json.dumps(str(runtime))}\n"
    "writable = true\n\n[env]\n"
    + "".join(f"{key} = {json.dumps(value)}\n" for key, value in env.items())
    + '\n[annotations]\ncom.hooks.aws_ofi_nccl.enabled = "true"\n'
    + 'com.hooks.aws_ofi_nccl.variant = "cuda13"\n'
)
(run / "native-agentenv.toml").write_text(edf)
print(f"Prepared run: {run}")
