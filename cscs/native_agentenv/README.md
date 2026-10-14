# Portable SWE-bench Pro V2 native AgentENV training

Repository location: `cscs/native_agentenv/`. See [CSCS reproduction guide](../README.md)
for the baseline settings and account setup overview.

Copy this entire directory to another **Clariden GH200 account**. It includes
the Uni-Agent/verl source snapshot used by job 3652372, the native persistent
shell helper, all **642 prepared V2 tasks**, launch scripts, and a pinned native
Harbor dependency list. No checkout of the original repository is needed.
Archives have checksums in `assets/SHA256SUMS`.

## Inputs the receiving account supplies

- A valid Slurm allocation/account on Clariden with access to 10 GH200 nodes.
- A readable `harbor-swe-megatron-lean.aarch64.sqsh` training image, matching the
  original image's CUDA 13/Megatron/Bridge/vLLM dependencies. The large image is
  supplied separately; it is not included in this script package.
- Readable Qwen3.5-27B weights, supplied separately.
- `uv` on the login node's PATH; setup creates its own Python 3.12 environment.
- Credentials for the running native AgentENV service and direct network access
  to `http://clnode286.clemson.cloudlab.us:8000` from compute nodes.

Use paths owned by, or explicitly readable by, the receiving account. All writes
go under its `WORK_ROOT` on shared scratch. Setup downloads Python dependencies;
AgentENV pulls the per-task `ghcr.io/scaleapi/swe-bench_pro-v2:<instance_id>`
images on CloudLab. This package uses the existing service; it does not deploy it.

## Start

```bash
cp config.env.example config.env
# Edit config.env: Slurm account, scratch root, model, training image, credentials.
bash setup.sh

# At the AGENTENV_CREDENTIALS path configured above, create a private shell file:
# export AGENTENV_API_KEY='your-key'
# Keep its directory private (0700) and the file private (0600).

bash submit.sh
```

`setup.sh` checks archive hashes, extracts the source/task snapshots, installs
the pinned Harbor client environment, and rewrites every parquet task path to
the new account's task directory. It performs no model evaluation or training.
`submit.sh` creates a fresh run directory and submits exactly one job. It does
not cancel or modify any existing job. It checks required local inputs; it does
not run benchmark validation.

Defaults reproduce the native run: **10 nodes** (8 trainer + 2 rollout), trainer
TP4/PP2/CP1, rollout TP4, batch 64 × 8, mini-batch 16, parameter sync every 4
steps, sandbox concurrency **64**, prompt/response limits 8192/32768, vLLM memory
utilization 0.45, **5 training steps**, checkpoint at step 5, **no validation**.
The default partition is `preemptable`, wall time 11:30:00, automatic requeue
disabled. Concurrency, step count, save frequency, partition and time are in
`config.env`; endpoint and all input/output paths are configurable too.

The launcher sources the credential file inside the batch allocation. The key
is not copied into the package or `launch.env`. W&B is optional: export
`WANDB_API_KEY` and optionally `WANDB_ENTITY` before submission; otherwise the
run uses console logging. Each run freezes its public settings, scripts and
container/Ray configuration. The large runtime snapshot lives under WORK_ROOT.

## Progress and traces

```bash
bash status.sh "$SCRATCH/swe-pro-native/runs/<experiment>"
bash collect_traces.sh "$SCRATCH/swe-pro-native/runs/<experiment>" "$SCRATCH/replay-traces.tar.gz"
```

Each run contains:

```text
training.out       Slurm / Ray logs
agent-logs/        framework rollout logs
harbor/            trial results, verifier output and agent transcripts
native-traces/     per-sandbox incremental JSONL workload traces
checkpoints/       model/optimizer/extra state at the configured save step
launch.env         public launch settings, including credential FILE PATH only
```

Native traces preserve lifecycle timestamps, durations, commands, cwd/env,
persistent shell IDs, output/exit status, errors and file bytes (base64). Harbor
trial metadata links to the trace file. They support later workload replay, but
the existing Flash Sandbox `bench_replay.py` needs an importer for this native
JSONL schema; this package does not run replay. Collect after the run ends for
a complete archive. Network and model behavior are not guaranteed deterministic.

## Provenance and limits

The source and task archives were exported from the running Clariden setup on
2026-10-14; dependency pins came from its isolated native Harbor environment.
`templates/train.sh` keeps the existing training recipe. The portable batch
launcher replaces account-specific paths and forwards the native SDK settings;
the sandbox workload does not pass through a scheduler, proxy or SSH tunnel.
The service's fixed VM shape is 4 CPUs / 4096 MiB. This package is for Clariden's
GH200/Slurm/enroot environment, not a generic GPU cluster deployment.

Checked shell/Python syntax and archive integrity. Setup completed in a separate
Clariden scratch directory using the bundled sources and pinned dependencies;
all 642 task paths were relocated. Generated container, Ray and Harbor configs
parsed successfully. No new training job or benchmark validation was run. A
training run from a second account has not been executed as part of packaging.
