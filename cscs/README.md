# Reproduce training on CSCS Clariden

The [native AgentENV training package](native_agentenv/README.md) provides
Qwen3.5 **4B, 9B and 27B** examples for **SWE-bench Verified, SWE-bench Pro V2,
and WebArena** on Clariden GH200 nodes. It connects
directly to the native AgentENV service on CloudLab using the SDK, without a
scheduler proxy or SSH tunnel.

For the latest instrumented 27B setup (16 trainer GPUs, 24 rollout GPUs,
uncapped runner concurrency), copy
`native_agentenv/config.fullmem_uncapped.env.example` to `native_agentenv/config.env`
before setup. Both profiles request exclusive nodes and the current maximum
permitted memory, 850,000 MiB/node (~830 GiB); Clariden rejects `--mem=0` because
its advertised node memory exceeds the configured per-job limit.

GPU/SM/DCGM, vLLM batch/KV metrics and incremental model captures are enabled by
the portable launcher. See [collection and replay](observability/README.md)
for the separate CloudLab host collector and download/replay commands.

## Model and task scripts

Run from the repository root, after the setup below. Each script submits one
job, with a distinct model/task experiment name and output directory.

| Model | SWE-bench Verified | SWE-bench Pro V2 | WebArena |
| --- | --- | --- | --- |
| 4B | [submit](qwen3.5-4b/submit_swebench_verified.sh) | [submit](qwen3.5-4b/submit_swebench_pro.sh) | [submit](qwen3.5-4b/submit_webarena.sh) |
| 9B | [submit](qwen3.5-9b/submit_swebench_verified.sh) | [submit](qwen3.5-9b/submit_swebench_pro.sh) | [submit](qwen3.5-9b/submit_webarena.sh) |
| 27B | [submit](qwen3.5-27b/submit_swebench_verified.sh) | [submit](qwen3.5-27b/submit_swebench_pro.sh) | [submit](qwen3.5-27b/submit_webarena.sh) |

```bash
bash cscs/qwen3.5-4b/submit_swebench_verified.sh
bash cscs/qwen3.5-9b/submit_swebench_pro.sh
bash cscs/qwen3.5-27b/submit_webarena.sh
```

The wrappers share `native_agentenv/config.env` by default. Use CONFIG_FILE to
select another account/config file. Model paths default to
`$SCRATCH/data/models/Qwen3.5-{4B,9B,27B}`; override MODEL_PATH_4B, MODEL_PATH_9B
or MODEL_PATH_27B in the config. Copy the entire `cscs/` directory to retain
these wrappers. The portable package also works alone via
`bash submit_preset.sh 9B swebench-pro`.

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
- `MODEL_PATH_4B`, `MODEL_PATH_9B`, `MODEL_PATH_27B`: weights for the chosen model.
  MODEL_PATH remains the fallback for the original 27B launcher.
- `AGENTENV_CREDENTIALS`: a private shell file exporting `AGENTENV_API_KEY`.
- `uv` on PATH and compute-node access to the configured AgentENV endpoint.

The example config points to the shared training image used by the original run:

```bash
TRAINING_IMAGE="/iopsstor/scratch/cscs/xyao/harbor-swe-megatron-lean.aarch64.sqsh"
```

It is readable by cluster accounts and has read-only permissions (`0444`). No
copy or image build is needed on Clariden. The containing scratch directory
permits traversal but not directory listing for other accounts. The image is
stored on scratch; override TRAINING_IMAGE if it is moved or removed.

Model weights and credentials are supplied separately.
Sources, all 642 prepared task definitions, dependency pins and dataset
relocation scripts are included in the directory. Sandbox images are
`ghcr.io/scaleapi/swe-bench_pro-v2:<instance_id>` and are pulled by AgentENV.

### Prepare the selected benchmark

From `cscs/native_agentenv/`, after `setup.sh`:

| Task | Preparation | Native sandbox image |
| --- | --- | --- |
| SWE-bench Pro | Included; setup relocates all 642 V2 rows | `ghcr.io/scaleapi/swe-bench_pro-v2:<instance_id>` |
| SWE-bench Verified | `bash prepare_verified.sh` downloads and prepares the full Verified dataset | `ghcr.io/swe-images/sweb.eval.x86_64.<instance>` |
| WebArena | Configure sites/image, export registry manifest, then `bash prepare_webarena.sh` | Your WebArena-capable browser image |

See [WebArena setup](native_agentenv/webarena/README.md). Its hosted websites,
browser image and any required evaluator credentials are external inputs.
Training scripts do not deploy them. Preprocessing prepares training rows;
it does not execute benchmark validation.

## Baseline settings

| Setting | Value |
| --- | --- |
| Partition | `preemptable` |
| Allocation | 10 GH200 nodes: 8 trainer, 2 rollout |
| Trainer / rollout parallelism | TP4/PP2/CP1 / TP4 |
| Prompt batch / samples / mini-batch | 64 / 8 / 16 |
| Parameter sync interval | 4 steps |
| Sandbox concurrency | 64 for SWE tasks; 1 for shared-site WebArena |
| Prompt / response token limits | 8192 / 32768 |
| Training / checkpoint | 5 steps / step 5 |
| Validation | Disabled |
| Automatic requeue | Disabled |

Change account paths, partition, concurrency and step settings in `config.env`.
All three sizes use this conservative topology; it has not been tuned for the
smaller models. The 4B/9B and new task combinations have not been trained as
part of adding these examples.
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
