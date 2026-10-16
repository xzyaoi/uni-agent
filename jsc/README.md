# JUPITER continuous rollout experiment

Run Qwen3.5-27B on SWE-bench Pro V2 using native AgentENV. This site adapter reuses the runtime archive, dataset, native SDK helper, installation patchers, training script, Ray bootstrap, and telemetry from [`cscs/native_agentenv`](../cscs/native_agentenv/README.md). The site-specific launcher uses Apptainer rather than CSCS EDF/Enroot.

## Experiment

| Setting | Default |
| --- | --- |
| Account / partition | `reformo` / `booster` |
| Allocation | 10 exclusive nodes, 4 GH200 GPUs and 288 CPUs per node |
| Memory / time | All node memory (`--mem=0`), 4 hours |
| Trainer / rollout | 4 trainer nodes (16 GPUs), 6 rollout nodes (24 GPUs) |
| Trainer parallelism | TP4, PP2, DP2 |
| Rollout parallelism | TP4, six replicas |
| Continuous refill | 128 in-flight prompt groups, eight responses each |
| Refill / ready backlog | 16 groups per refill, pause at 64 ready groups |
| Agent concurrency | Uncapped runner (`CONCURRENCY=0`) |
| Optimizer batch | 64 prompts × eight responses; unchanged by rollout batch |
| Duration | Five training steps; checkpoint at step five; validation disabled |

The continuous producer uses the existing staleness checks. The target of 128 prompt groups means up to 1,024 agent episodes; it is not a guarantee of 1,024 simultaneous vLLM requests. Agents also spend time executing commands and verifying results.

## Prepare another account

Clone this repository on JUPITER (SSH alias `jsc`) and retain both `jsc/` and `cscs/`. Copy the example configuration:

```bash
cp jsc/config.env.example jsc/config.env
# Edit SCRATCH, WORK_ROOT, SLURM_ACCOUNT, image/model paths and AgentENV endpoint.
bash jsc/setup.sh
```

`setup.sh` installs a user-local `uv` under `$WORK_ROOT/tools` if needed, then invokes the shared setup. Python 3.12, the native Harbor environment, source and all task assets live beneath `WORK_ROOT`. Run setup on ARM64 so that the client environment can run inside the ARM64 training container. Internet/package access is required during setup.

Provision the following independently before submission:

```text
$WORK_ROOT/images/harbor-swe-megatron-lean.aarch64.sqsh
$WORK_ROOT/models/Qwen3.5-27B/config.json
$WORK_ROOT/models/Qwen3.5-27B/<all tokenizer and model weight files>
$WORK_ROOT/secrets/agentenv.env
```

The image is the same ARM64 training squashfs used on CSCS. Image/model files are not included in Git. The bundled Pro V2 task manifests select `ghcr.io/scaleapi/swe-bench_pro-v2:<instance_id>` sandbox images; these execute on AgentENV, not on the GPU nodes.

Create the credential file with mode `0600` and a shell assignment `AGENTENV_API_KEY=...`. Never commit it. Confirm the configured endpoint is reachable from a JUPITER compute node before launch. For this deployment, CloudLab port 80 redirects in the kernel to the existing native AgentENV port 8000. Login-host health and authenticated API requests succeeded. The queued one-node readiness job checks compute-container reachability before the ten-node run. This experiment uses no scheduler or SSH tunnel in the request path.

The host needs `apptainer`, `dcgmi`, a matching `libdcgm`, and access to the existing DCGM hostengine. Load the site's required modules before invoking `submit.sh`; `APPTAINER_BIN` can name an absolute binary path. The current preparation step bundles the login host's DCGM client/library; confirm the compute hostengine is compatible.

## Submit and inspect

```bash
bash jsc/submit.sh
# Optional Slurm flags, e.g. hold a prepared job for inspection:
# bash jsc/submit.sh --hold
squeue -u "$USER"
```

Runs are created at `$WORK_ROOT/runs/<experiment-name>/`. `job_id`, `launch.env`, the task config and scripts capture the selected settings. A generated CSCS `native-agentenv.toml` is retained as preparation metadata; JUPITER does not use it.

The launcher sets `CUDA_VISIBLE_DEVICES=0,1,2,3` before `srun`, then sets it explicitly inside Apptainer. All experiment settings are propagated with `APPTAINERENV_*` overrides. Apptainer uses `--nv`, host networking and shared absolute path binds for the work root, model/image parents and native SDK helpers. It does not assume a NIC or HCA name. Native Harbor's interpreter and library paths therefore remain accessible inside the container.

## Measurements and replay

| Artifact | Path inside the run |
| --- | --- |
| Slurm/training logs | `training.out` |
| In-container GPU/DCGM/vLLM telemetry | `telemetry/gpu/` |
| Independent host collector | `telemetry/host/` |
| Incremental model calls | `telemetry/model-calls/` |
| Native sandbox lifecycle/commands/files | `native-traces/` |
| Per-trial Harbor artifacts | `harbor/` |
| Training checkpoints | `checkpoints/` |

The shared Ray bootstrap enables the bundled in-container DCGM client; collectors record SM active, SM occupancy, tensor and DRAM activity when supported by the hostengine. Availability and permissions must be checked on actual allocated GPUs. The host collector is an independent fallback and exits with the node launcher. Hard termination may truncate final samples or Ray log copies.

For sandbox-host CPU, memory, PSI and I/O measurements, run the existing [`cscs/observability/collect.py`](../cscs/observability/collect.py) collector on the AgentENV server into a directory associated with the JUPITER job. Cluster-side traces alone cannot establish sandbox-host saturation. If the CSCS experiment is running against that same server, record both job identities: their combined load confounds comparisons.

Use the shared [observability tools](../cscs/observability/README.md) to fetch and replay traces. Traces can contain source code, commands and task output; keep them private. Launching this experiment does not execute benchmark validation or a replay workload.

## Current deployment (2026-10-15)

- Remote root: `/e/scratch/reformo/yao4/coding-agent-trainer`.
- Training job **2336862**; readiness dependency **2336875**.
- Ordering was updated to **CSCS first**: job **3668373** is released; JSC training job **2336862** is held until CSCS terminates. The one-node readiness job can proceed independently.
- Training image SHA256: `0a854beb4e42b7b689a42827ce8e75d0b7bc90b1028b884e81c107228da286fb` (matches CSCS).
- Qwen config, weight index, and tokenizer SHA256 match CSCS; HF downloaded all 11 weight shards.
- Login DCGM is 4.6.1; compute hostengine/counter access awaits allocation.
- JSC scheduler currently reports nodes reserved for maintenance; no performance result is available yet.

`platform_ready.sh` discovers GPU/CPU memory and profiling access, starts the
exact image, and makes an authenticated native API request from that container.
It runs no model inference or benchmark validation. The training job depends on
its success. `release_when_ready.sh` verified image bytes and checksum before
releasing both queued jobs. Slurm `--mem=0` requests all node memory; this avoids
assuming that published host-RAM capacity and Slurm's reported memory are equal.

The direct HTTP port alias was created on CloudLab with:

```bash
sudo iptables -t nat -A PREROUTING -p tcp --dport 80 \
  -m comment --comment agentenv-jsc-http -j REDIRECT --to-ports 8000
```

The existing native API-key authentication still applies, and CSCS continues to
use port 8000. No scheduler, user-space proxy, or SSH tunnel was added. This
rule is runtime configuration and must be reapplied after a host reboot. Undo
uses the same command with `-D` in place of `-A`. Check `iptables -C` before
adding it again to avoid duplicate rules.

## Site references

- [JUPITER batch system and network access](https://apps.fz-juelich.de/jsc/hps/jupiter/batchsystem.html)
- [Explicit four-GPU visibility](https://apps.fz-juelich.de/jsc/hps/jupiter/gpu-computing.html)
- [JUPITER container runtime](https://apps.fz-juelich.de/jsc/hps/jupiter/container-runtime.html)
- [Apptainer image formats](https://apptainer.org/docs/user/main/cli/apptainer_exec.html)
