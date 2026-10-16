# Portable Qwen3.5 native AgentENV training

Repository location: `cscs/native_agentenv/`. See [CSCS reproduction guide](../README.md)
for the baseline settings and account setup overview.

Separate launchers for 4B, 9B and 27B across Verified, Pro V2 and WebArena live
in `cscs/qwen3.5-{4b,9b,27b}/`. This standalone package also accepts:

```bash
bash submit_preset.sh 4B swebench-verified
bash submit_preset.sh 9B swebench-pro
bash submit_preset.sh 27B webarena
```

Run setup once. Pro V2 data is included; prepare Verified with
`bash prepare_verified.sh`. For WebArena, follow [the browser/site setup guide](webarena/README.md)
and run `bash prepare_webarena.sh`. MODEL_PATH_4B, MODEL_PATH_9B and MODEL_PATH_27B
override each preset's weights. All sizes retain the original conservative
10-node topology; the smaller sizes have not been tuned or trained here.
WebArena defaults to concurrency 1 for a shared website deployment.

Entropy diagnostics use the existing Megatron chunked implementation with
1,024-token chunks. This targets the large temporary allocation seen during
old-log-prob evaluation in run 3661417. Entropy remains available as a metric;
its loss coefficient remains zero. The required old-log-prob recomputation
continues to run. Runtime memory improvement must be confirmed by the next
instrumented training run.

To match the latest 27B full-memory run, use
`config.fullmem_uncapped.env.example`: 4 trainer nodes / 6 rollout nodes,
CONCURRENCY=0, and 850,000 MiB per exclusive node. Each portable run now records
GPU/DCGM/vLLM telemetry and incremental full model calls alongside sandbox
traces. See [observability](../observability/README.md) for CloudLab host
collection and replay. When upgrading an older prepared runtime, choose a
fresh WORK_ROOT so setup extracts the updated source snapshot.

Copy this entire directory to another **Clariden GH200 account**. It includes
the Uni-Agent/verl source snapshot used by job 3652372, the native persistent
shell helper, all **642 prepared V2 tasks**, launch scripts, and a pinned native
Harbor dependency list. No checkout of the original repository is needed.
Archives have checksums in `assets/SHA256SUMS`.

## Inputs the receiving account supplies

- A valid Slurm allocation/account on Clariden with access to 10 GH200 nodes.
- Access to the shared training image described below. It includes the original
  run's CUDA 13/Megatron/Bridge/vLLM dependencies and is not bundled in the scripts.
- Readable Qwen3.5-27B weights, supplied separately.
- `uv` on the login node's PATH; setup creates its own Python 3.12 environment.
- Credentials for the running native AgentENV service and direct network access
  to `http://clnode286.clemson.cloudlab.us:8000` from compute nodes.

Use paths owned by, or explicitly readable by, the receiving account. All writes
go under its `WORK_ROOT` on shared scratch. Setup downloads Python dependencies;
AgentENV pulls the per-task `ghcr.io/scaleapi/swe-bench_pro-v2:<instance_id>`
images on CloudLab. This package uses the existing service; it does not deploy it.

### Shared training image

`config.env.example` already selects the exact GH200/aarch64 image used by the run:

```bash
TRAINING_IMAGE="/iopsstor/scratch/cscs/xyao/harbor-swe-megatron-lean.aarch64.sqsh"
```

This file is readable by other cluster accounts and read-only (`0444`). The
parent scratch directory permits traversal without directory listing. You can
use this path directly on Clariden; copying or rebuilding the image is not
required. Scratch storage is not permanent, so set TRAINING_IMAGE to another
compatible readable image if this path becomes unavailable. Model weights and
AgentENV credentials still need to be supplied by the receiving account.

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
`submit.sh` defaults to 27B/Pro V2. It creates a fresh run directory and submits exactly one job. It does
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

### Independent rollout batch size

Set `ROLLOUT_PROMPT_BSZ=128` in `config.env` to dispatch **128 prompts × 8
responses = 1,024 agent episodes** per batch. The full-memory experiment profile
now uses this value. `TRAIN_PROMPT_BSZ=64`, `PPO_MINI_BATCH_SIZE=16`, and
`PARAMETER_SYNC_STEP=4` stay unchanged, so optimizer update sizes do not grow.

The corresponding Hydra setting is
`trainer.v1.separate_async.rollout_prompt_batch_size`. Null uses the training
batch size for backward compatibility. `setup.sh` and `submit.sh` apply the
small runtime extension to the pinned source archive, including existing
installations; unrelated source changes are preserved. `data.gen_batch_size`
remains a dataloader-fetch size, not this rollout-dispatch size.

This increases finite dispatch batches; it does not implement continuous
refill. With one warmup batch, startup can launch two batches (2,048 episodes
at this setting), before failed-group refills. Subsequent steps add 1,024
episodes while consuming 512; monitor ready-data backlog, policy staleness,
and sandbox CPU pressure. More generated data is not necessarily more data
used for training, and short runs can end with unused work.

### Continuous rollout refill (enabled in the experiment profile)

The full-memory experiment profile now sets:

```bash
CONTINUOUS_ROLLOUT=True
ROLLOUT_TARGET_PROMPTS=128
ROLLOUT_READY_PROMPTS=0
ROLLOUT_REFILL_PROMPTS=16
ROLLOUT_REFILL_INTERVAL=1.0
```

This maintains up to **128 pending/running prompt groups** (up to 1,024 agent
episodes at 8 responses per prompt), submitting at most 16 groups per second.
`ROLLOUT_READY_PROMPTS=0` disables admission backpressure from completed groups,
so generation can continue during training even when results accumulate.
The 128-group in-flight limit remains active. Completed results can accumulate
without a count bound; the existing sampler still applies policy staleness rules.
A positive value restores the completed-backlog cap: admissions pause at that
threshold and total unsampled groups are bounded by the in-flight target plus
the threshold. Failed terminal groups also count toward a positive threshold.

Continuous mode replaces the warmup/per-step dispatch policy; it ignores
`ROLLOUT_PROMPT_BSZ` and `NUM_WARMUP_BATCHES`. The independent larger finite
batch setting above applies when continuous mode is disabled. Optimizer batches
remain 64 prompts, split into four 16-prompt updates, and existing policy
staleness checks continue to apply. A prompt group refills only after all its
response sessions settle, not after each individual session finishes.

A dedicated producer reads an independent TransferQueue metadata snapshot.
It does not mutate the sampler's working sets. Checkpoint and validation pause
new submissions across dataloader/queue snapshots. Driver shutdown stops and
joins the producer, including early returns and exceptions; outstanding remote
episodes retain the existing Ray lifecycle. Producer errors surface at step
boundaries and replay polling. The installed training image's synchronous TQ
client schedules calls onto its dedicated event-loop thread.

`setup.sh` and `submit.sh` install `apply_continuous_rollout.py` and the bundled
runtime extension. The base trainer default remains disabled for other recipes.
Configuration lives at `trainer.v1.separate_async.continuous_refill.*`.
The producer writes `feeder-<pid>.jsonl` under `UNI_AGENT_TRACE_DIR` and reports
`training/refill/*` metrics (in-flight, terminal, failed and submitted groups).
These are prompt-group counts, not vLLM active request counts. The feature has
been syntax checked and statically reviewed; a fresh training run is still
needed to establish its runtime behavior and throughput.

Continuous mode also sets `data.dataloader_num_workers=0`: fetching runs in the
producer thread, so it must not fork DataLoader subprocesses at initial startup
or dataset rollover. The extension rejects nonzero worker counts in this mode.
This removes loader subprocess parallelism; model, optimizer and GPU topology
settings remain the same.


### GH200 episode dispatch and host memory

For the AgentENV saturation experiment, use
`AGENT_RUNNER_DISPATCH_MODE=inline_async` with the Clariden operational launcher
`examples/quickstart/training/submit_qwen35_27b_pro_v2_agentenv_clariden.sh`.
This uses the existing async runner path instead of starting a heavyweight Ray
worker for each episode. Harbor still runs its CLI in the native client venv;
sandbox commands execute remotely through AgentENV. The runner concurrency and
128-group active target remain independent settings.

`CPU_MEMORY_BIND=True` binds ordinary host allocations to permitted NUMA nodes
with CPUs, before Ray starts. GH200 also exposes GPU HBM as NUMA memory; host
allocations in HBM can reduce memory available to CUDA. The run-local
`bind_cpu_memory.sh` uses Linux MPOL_BIND and does not require numactl in the
image. It limits usable host memory to CPU-attached RAM and fails explicitly if
binding is unsupported. Explicit CUDA allocations are unaffected by this host
allocation policy. Collect per-node NUMA memory to check its effect.

The operational launcher accepts `MODEL_PATH`, `TOTAL_NODES`, `TIME_LIMIT`,
`TRAINING_STEPS`, batch settings, and parallelism overrides. For the 9B smoke:
one trainer node, one rollout node, TP4/PP1/CP1, 8 training prompts with 8
responses each, 2-prompt minibatches, one step, 4096 prompt tokens and 8192
response tokens. Debug permits 90 node-minutes per job, so two nodes use a
45-minute limit. This smaller run does not establish full-size memory safety.

An explicit `AGENTENV_ENDPOINT` survives loading the private credential file.
The launcher sets NO_PROXY only for that endpoint host, preserving other proxy
settings. AWS deployments must permit the actual compute egress address on the
AgentENV port; login and compute egress addresses may differ.
