# Reproduce training on CSCS Clariden

The [native AgentENV training package](native_agentenv/README.md) reproduces the
Qwen3.5-27B SWE-bench Pro V2 baseline on Clariden GH200 nodes. It connects
directly to the native AgentENV service on CloudLab using the SDK, without a
scheduler proxy or SSH tunnel.

## Quick start

On a Clariden login node, from this repository or a copy of `cscs/`:

```bash
cd cscs/native_agentenv
cp config.env.example config.env
# Edit config.env for your account; see the required inputs below.
bash setup.sh
# Create your private AgentENV credential file at AGENTENV_CREDENTIALS.
bash submit.sh
```

Another account can copy the entire `native_agentenv/` directory and run these
same commands from inside it. Its scripts locate their own directory and do
not depend on the surrounding checkout.

### Required inputs

- `SCRATCH`: the receiving account's shared scratch directory.
- `SLURM_ACCOUNT`: the receiving account's Slurm allocation.
- `TRAINING_IMAGE`: a readable GH200/aarch64
  `harbor-swe-megatron-lean.aarch64.sqsh` image.
- `MODEL_PATH`: readable Qwen3.5-27B weights.
- `AGENTENV_CREDENTIALS`: a private shell file exporting `AGENTENV_API_KEY`.
- `uv` on PATH and compute-node access to the configured AgentENV endpoint.

The large training image, model weights and credentials are supplied separately.
Sources, all 642 prepared task definitions, dependency pins and dataset
relocation scripts are included in the directory. Sandbox images are
`ghcr.io/scaleapi/swe-bench_pro-v2:<instance_id>` and are pulled by AgentENV.

## Baseline settings

| Setting | Value |
| --- | --- |
| Partition | `preemptable` |
| Allocation | 10 GH200 nodes: 8 trainer, 2 rollout |
| Trainer / rollout parallelism | TP4/PP2/CP1 / TP4 |
| Prompt batch / samples / mini-batch | 64 / 8 / 16 |
| Parameter sync interval | 4 steps |
| Sandbox concurrency | 64 |
| Prompt / response token limits | 8192 / 32768 |
| Training / checkpoint | 5 steps / step 5 |
| Validation | Disabled |
| Automatic requeue | Disabled |

Change account paths, partition, concurrency and step settings in `config.env`.
Each submission creates a fresh run under `WORK_ROOT/runs/`, freezes the launch
configuration and scripts, and saves logs, Harbor artifacts, native workload
traces and checkpoints there.

## Progress and trace export

```bash
bash status.sh "$SCRATCH/swe-pro-native/runs/<experiment>"
bash collect_traces.sh "$SCRATCH/swe-pro-native/runs/<experiment>" "$SCRATCH/replay-traces.tar.gz"
```

Run these from `native_agentenv/`; adjust the run path if you changed WORK_ROOT.
See the [package README](native_agentenv/README.md) for trace format, provenance
and reproduction limits. Setup and relocation of all 642 tasks succeeded in a
separate Clariden scratch directory. A second-account training run has not yet
been executed.
