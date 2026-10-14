"""Generate an immutable run directory using only the receiving account's paths."""

import json
import os
import shlex
import shutil
from pathlib import Path

import pyarrow.parquet as pq

bundle = Path(os.environ["BUNDLE_ROOT"]).resolve()
root = Path(os.environ["WORK_ROOT"]).resolve()
run = Path(os.environ["RUN_DIR"]).resolve()
image = Path(os.environ["TRAINING_IMAGE"]).resolve()
model = Path(os.environ["MODEL_PATH"]).resolve()
credentials = Path(os.environ["AGENTENV_CREDENTIALS"]).resolve()
runtime = root / "runtime"
benchmark = os.environ["BENCHMARK"]
model_size = os.environ["MODEL_SIZE"]
train_file = Path(os.environ["TRAIN_FILE"]).resolve()
expected_task = {"swebench-pro": "harbor", "swebench-verified": "swe_bench", "webarena": "browsergym"}[benchmark]
rows = pq.read_table(train_file).to_pylist()
if not rows:
    raise ValueError(f"Empty training dataset: {train_file}")
for row in rows:
    task = row["extra_info"]["tools_kwargs"]["task"]
    if task["name"] != expected_task:
        raise ValueError(f"Expected {expected_task} task rows in {train_file}")
    if benchmark == "webarena" and task["metadata"]["benchmark"] != "browsergym.webarena":
        raise ValueError("WebArena preset requires browsergym.webarena rows, not MiniWoB")
run.mkdir()  # Refuse accidental reuse of a previous run.
settings = {
    "REPO_DIR": str(runtime),
    "WORK_ROOT": str(root),
    "RUN_DIR": str(run),
    "SCRATCH": str(Path(os.environ["SCRATCH"]).resolve()),
    "AGENTENV_CREDENTIALS": str(credentials),
    "AGENTENV_ENDPOINT": os.environ["AGENTENV_ENDPOINT"],
    "AGENTENV_TRACE_DIR": str(run / "native-traces"),
    "UNI_AGENT_TRACE_DIR": str(run / "telemetry/model-calls"),
    "METRICS_DIR": str(run / "telemetry/gpu"),
    "METRICS_COLLECTOR": str(run / "collect.py"),
    "NATIVE_SHELL_ROOT": str(root / "native-shell"),
    "HARBOR_VENV": str(root / "venv"),
    "TRAINING_IMAGE": str(image),
    "MODEL_PATH": str(model),
    "TRAIN_FILE": str(train_file),
    "TEST_FILE": str(train_file),
    "TASK_CONFIG": str(run / "task_config.yaml"),
    "TRAIN_SCRIPT": str(run / "train.sh"),
    "RUNTIME_ENV": str(run / "runtime_env.yaml"),
    "RUNTIME_DIR": str(root),
    "DATA_DIR": str(root / "data"),
    "CKPTS_DIR": str(run / "checkpoints"),
    "AGENT_LOG_DIR": str(run / "agent-logs"),
    "PROJECT_NAME": f"Uni-Agent-Qwen3.5-{model_size}-{benchmark}-AgentENV",
    "MODEL_SIZE": model_size,
    "BENCHMARK": benchmark,
    "WEBARENA_ENV_FILE": os.environ["WEBARENA_ENV_FILE"],
    "BROWSERGYM_FORCE_IMAGE": os.environ["WEBARENA_IMAGE"] if benchmark == "webarena" else "",
    "EXP_NAME": os.environ["EXP_NAME"],
    "TRAINER_MODE": "separate_async",
    "NNODES_ROLLOUT": os.environ["NNODES_ROLLOUT"],
    "PARAMETER_SYNC_STEP": "4",
    "GEN_TP": os.environ["GEN_TP"],
    "TP": os.environ["TP"],
    "PP": os.environ["PP"],
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
for name in ("train.sh", "run_ray_node.sh", "harbor-native", "configure_webarena.py"):
    shutil.copyfile(bundle / "templates" / name, run / name)
    (run / name).chmod(0o750)
shutil.copyfile(runtime / "cscs/observability/collect.py", run / "collect.py")
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
if benchmark != "swebench-pro":
    task_config = [
        {
            "name": expected_task,
            "sandbox": {"provider": "agentenv", "runtime_timeout": 7200, "sandbox_kwargs": {"start_timeout": 600}},
            "agent": {
                "name": "react",
                "max_steps": 200,
                "tools": [
                    {"name": "str_replace_editor"},
                    {"name": "stateful_shell", "command_timeout": 120},
                    {"name": "submit"},
                ],
                "model": {"temperature": 1.0, "top_p": 1.0},
            },
        }
    ]
if benchmark == "webarena":
    task_config[0]["sandbox"]["image"] = os.environ["WEBARENA_IMAGE"]
    task_config[0]["agent"]["tools"] = [{"name": "browser", "command_timeout": 120}, {"name": "submit"}]
    task_config[0]["agent"]["max_steps"] = 100
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
if benchmark == "webarena":
    runtime_env["env_vars"]["BROWSERGYM_FORCE_IMAGE"] = os.environ["WEBARENA_IMAGE"]
(run / "runtime_env.yaml").write_text(json.dumps(runtime_env, indent=2) + "\n")
mount_paths = sorted({root, run, runtime, train_file.parent, model.parent, image.parent, Path(settings["SCRATCH"])})
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
