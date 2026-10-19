"""Prepare an isolated 25-turn run with persistent replica and GPU statistics."""

import json
import subprocess
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "base"
OLD = "qwen35_27b_verified_flash_2n45m_c512_20261018"
NAME = "qwen35_27b_verified_flash_2n45m_c512_t25_metrics_20261018"
RUN = "/iopsstor/scratch/cscs/xyao/uni-agent/runs/" + NAME


def once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after, 1)


names = (
    "batch.sh",
    "launcher.sh",
    "train.sh",
    "run_ray_node.sh",
    "run.env",
    "runtime_env.json",
    "task_config.yaml",
    "flash-training.toml",
    "preflight.py",
    "preflight_verified.py",
    "verified_flash_environment.py",
    "verify_run.py",
)
files = {name: (BASE / name).read_text().replace(OLD, NAME) for name in names}
task = json.loads(files["task_config.yaml"])
task[0]["agent"]["kwargs"]["max_steps"] = 25
files["task_config.yaml"] = json.dumps(task, indent=2) + "\n"

files["run.env"] += (
    f"export TELEMETRY_DIR={RUN}/telemetry\n"
    f"export VLLM_PROMETHEUS_CONFIG={RUN}/telemetry/prometheus.yml\n"
    "export VLLM_LOGGING_LEVEL=INFO\n"
)
runtime = json.loads(files["runtime_env.json"])
runtime["env_vars"]["VLLM_LOGGING_LEVEL"] = "INFO"
files["runtime_env.json"] = json.dumps(runtime, indent=2) + "\n"
files["flash-training.toml"] = once(
    files["flash-training.toml"], 'VLLM_LOGGING_LEVEL = "WARN"', 'VLLM_LOGGING_LEVEL = "INFO"'
)
files["launcher.sh"] = once(
    files["launcher.sh"],
    "RUN_ENV=(\n",
    'RUN_ENV=(\n    RUN_DIR="${RUN_DIR}"\n'
    '    TELEMETRY_DIR="${TELEMETRY_DIR}"\n'
    '    VLLM_PROMETHEUS_CONFIG="${VLLM_PROMETHEUS_CONFIG}"\n'
    '    VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL}"\n',
)
anchor = "    actor_rollout_ref.rollout.name=${rollout_name} \\\n"
files["train.sh"] = once(
    files["train.sh"],
    anchor,
    anchor + "    actor_rollout_ref.rollout.disable_log_stats=False \\\n"
    "    actor_rollout_ref.rollout.prometheus.enable=True \\\n"
    "    actor_rollout_ref.rollout.prometheus.file=${VLLM_PROMETHEUS_CONFIG} \\\n",
)
files["batch.sh"] = once(
    files["batch.sh"], "#SBATCH --job-name=qwen35-27b-vfy-c512", "#SBATCH --job-name=qwen35-27b-vfy-t25-metrics"
)

# These processes live inside the same exclusive allocation/container as Ray.
# Every sample is written immediately to shared scratch, independent of Ray logs.
hook = """
: "${RUN_DIR:?Missing run directory}"
: "${TELEMETRY_DIR:?Missing telemetry directory}"
: "${VLLM_PROMETHEUS_CONFIG:?Missing vLLM discovery path}"
mkdir -p "${TELEMETRY_DIR}/gpu" "${TELEMETRY_DIR}/vllm"
COLLECTOR_PIDS=()
stop_collectors() {
    local result=$?
    trap - EXIT
    for pid in "${COLLECTOR_PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
    for pid in "${COLLECTOR_PIDS[@]}"; do wait "${pid}" 2>/dev/null || true; done
    exit "${result}"
}
trap stop_collectors EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
python "${RUN_DIR}/observability/collect_usage.py" \\
    --mode gpu --output-dir "${TELEMETRY_DIR}/gpu" --interval-seconds 5 \\
    >"${TELEMETRY_DIR}/gpu/collector-${SLURM_JOB_ID}-${SLURM_NODEID}.log" 2>&1 &
COLLECTOR_PIDS+=("$!")
if [ "${SLURM_NODEID:-0}" = "0" ]; then
    python "${RUN_DIR}/observability/collect_usage.py" \\
        --mode prometheus --discovery-format prometheus-yaml \\
        --discovery-file "${VLLM_PROMETHEUS_CONFIG}" \\
        --output-dir "${TELEMETRY_DIR}/vllm" --interval-seconds 5 \\
        >"${TELEMETRY_DIR}/vllm/collector-${SLURM_JOB_ID}.log" 2>&1 &
    COLLECTOR_PIDS+=("$!")
fi
sleep 1
for pid in "${COLLECTOR_PIDS[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
        echo "[metrics] Collector failed during startup; inspect ${TELEMETRY_DIR}" >&2
        exit 1
    fi
done
echo "[metrics] Persistent collectors started on $(hostname), interval=5s"
"""
files["run_ray_node.sh"] = once(
    files["run_ray_node.sh"],
    'if [ "${SLURM_NODEID:-0}" = "0" ]; then\n',
    hook + '\nif [ "${SLURM_NODEID:-0}" = "0" ]; then\n',
)

# Assert the requested settings on the actual allocated nodes before Ray starts.
files["preflight.py"] += """
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
"""
files["verify_run.py"] = once(
    files["verify_run.py"],
    "required = [\n",
    "required = [\n"
    '    "actor_rollout_ref.rollout.disable_log_stats=False",\n'
    '    "actor_rollout_ref.rollout.prometheus.enable=True",\n'
    f'    "actor_rollout_ref.rollout.prometheus.file={RUN}/telemetry/prometheus.yml",\n',
)
files["verify_run.py"] += """
assert task["agent"]["kwargs"]["max_steps"] == 25
assert runtime["env_vars"]["VLLM_LOGGING_LEVEL"] == "INFO"
assert tomllib.loads((ROOT / "flash-training.toml").read_text())["env"]["VLLM_LOGGING_LEVEL"] == "INFO"
for key in ("RUN_DIR", "TELEMETRY_DIR", "VLLM_PROMETHEUS_CONFIG", "VLLM_LOGGING_LEVEL"):
    assert f'{key}="${{{key}}}"' in (ROOT / "launcher.sh").read_text()
evidence.update(max_agent_turns=25, persistent_replica_statistics=True, gpu_samples_per_node=True,
                metrics_interval_seconds=5, metrics_logging_level="INFO")
(ROOT / "validation.json").write_text(json.dumps(evidence, indent=2) + "\\n")
print("Verified: 25-turn agent cap and persistent replica/GPU metrics configuration")
"""
for name, text in files.items():
    (ROOT / name).write_text(text)
    if name.endswith(".sh"):
        (ROOT / name).chmod(0o750)
        subprocess.run(["bash", "-n", str(ROOT / name)], check=True)
    if name.endswith(".py"):
        compile(text, name, "exec")
tomllib.loads(files["flash-training.toml"])
(ROOT / "submit.json").write_text(
    json.dumps(
        dict(
            command=[
                "sbatch",
                "--parsable",
                "--partition=preemptable",
                "--export=ALL",
                "--chdir=" + RUN,
                RUN + "/batch.sh",
            ],
            run_dir=RUN,
        ),
        indent=2,
    )
    + "\n"
)
print("Prepared " + RUN)
