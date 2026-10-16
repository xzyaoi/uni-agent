# SWE-bench Pro

Preprocess [ScaleAI/SWE-bench_Pro](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro)
directly from Hugging Face. The preprocessor supports all three public configurations:

| `--dataset-config` | Release | Tasks | Uni-Agent task |
| --- | --- | ---: | --- |
| `default` (CLI default) | V2 | 642 | `harbor` |
| `hard` | V2 HARD-51 | 51 | `harbor` |
| `v1` | Legacy V1 | 731 | `swe_bench_pro` |

## V2 and HARD-51

Install Harbor in each worker environment. Upstream requires Harbor 0.22 or later
and Modal 1.5.1 or later for the V2 network policy:

```bash
pip install 'harbor[modal]>=0.22' 'modal>=1.5.1'
modal setup
python -m uni_agent.tasks.swe_bench_pro.preprocess \
    --dataset-config default \
    --local-save-dir ~/data/swe_agent
```

This writes `swe_bench_pro_v2.parquet`. Use `--dataset-config hard` to write
`swe_bench_pro_v2_hard.parquet`, or add `--max-instances 1` for a small initial run.
The preprocessor pins both the Hugging Face revision and the official V2 task
archive. It selects tasks by Hugging Face instance ID and preserves the official
instructions, GHCR images, verifier assets, and reference solutions through Harbor.

Rows reference absolute paths to the extracted task assets in the datasets cache.
Keep that cache available for inference. For multiple Ray nodes, place the
official release's `v2/tasks` directory on shared storage and supply
`--tasks-dir /mnt/shared/SWE-bench_Pro-os/v2/tasks`; use the revision recorded in
`V2_TASKS_COMMIT` in the preprocessor. Every worker must see the same absolute paths.

Verify reference solutions using the existing Harbor oracle configuration:

```bash
python examples/inference/parallel_infer_api.py \
    --data-path ~/data/swe_agent/swe_bench_pro_v2.parquet \
    --task-config examples/quickstart/harbor/task_config_oracle.yaml \
    --base-url http://unused.invalid/v1 \
    --limit 1
```

Harbor's oracle ignores the model endpoint placeholder.

For model inference, use a Harbor agent configuration, such as
`examples/quickstart/harbor/task_config_glm52.yaml`, with
`examples/inference/parallel_infer_api.py`. See [Harbor Integration](harbor-integration.md)
for endpoint configuration and execution examples. V2 rows use task name `harbor`,
so select a YAML entry with that name.

These commands run ordinary Harbor trials. The upstream
[locked evaluation protocol](https://github.com/scaleapi/SWE-bench_Pro-os/blob/main/v2/README.md)
also requires offline agent execution, endpoint allowlisting, locked agent wrappers,
and a separate replay in a fresh sandbox. Use upstream tooling for that protocol;
ordinary trial scores alone do not establish leaderboard comparability.

## V2 training with Flash Sandbox

The Clariden Qwen3.5-27B training recipe uses Harbor V2 tasks with Flash Sandbox's
`libkrun` backend. Prepare a **fresh directory** on shared storage:

```bash
python -m uni_agent.tasks.swe_bench_pro.prepare_training \
    --dataset-config default \
    --local-save-dir /iopsstor/scratch/cscs/xyao/data/uni_agent/swe_bench_pro_v2
```

This writes `swe_bench_pro_v2_train.parquet` and copies the selected official task
directories into `tasks/<instance_id>` underneath the destination. Use
`--dataset-config hard` for `swe_bench_pro_v2_hard_train.parquet`,
`--max-instances 1` for a small preparation run, or `--tasks-dir` to supply the
official V2 task directory. Dataset rows reference the copies by absolute path;
every worker must see those paths. Existing task copies or an existing output
parquet cause preparation to fail rather than overwrite a previous run.

The copies preserve the official instructions, container image references,
verifier files, and reference solutions. The only task file change is
`[agent].network_mode` from `no-network` to `public`; each row records
`training_network_mode: public`. The original source tree remains untouched.
These training tasks allow network access because the Flash SDK cannot enforce
Harbor's offline agent policy. This training setup does not follow the official
leaderboard locked evaluation protocol.

Install pinned `harbor==0.24.0` and `flash-sandbox==0.4.0` in an isolated environment
available on every worker. The cluster recipe uses
`/iopsstor/scratch/cscs/xyao/venvs/uni-harbor-v2/bin/harbor`; provision that environment on each
worker before starting trials. The host ReAct agent runs through Uni-Agent's
model gateway while Harbor manages the Flash environment and official verifier.
Keep the task configuration's Harbor artifact log directory on shared storage
to retain trial configuration, verifier output, and other trial artifacts.

From the remote repository on a Clariden login node, launch the prepared training
run using:

```bash
bash examples/quickstart/training/submit_qwen35_27b_pro_v2_clariden.sh
```

The recipe runs five training steps with Qwen3.5-27B and performs no validation.
Its run directory records the launch configuration and logs. The wrapper uses
the parquet prepared at the shared path above.

## Legacy V1

```bash
python -m uni_agent.tasks.swe_bench_pro.preprocess \
    --dataset-config v1 \
    --local-save-dir ~/data/swe_agent

BASE_URL=http://localhost:8000/v1 MODEL=<served-model-name> \
python examples/inference/parallel_infer_api.py \
    --data-path ~/data/swe_agent/swe_bench_pro_v1.parquet \
    --task-config examples/quickstart/inference/task_config_swe_bench_pro.yaml \
    --limit 1
```

V1 uses Docker Hub images and the native `swe_bench_pro` evaluator, with per-instance
run scripts pinned to the legacy release. `--scripts-dir` optionally supplies a
local legacy `run_scripts` directory. The Python function `build_swe_bench_pro()`
keeps V1 as its default for compatibility; pass `dataset_config="default"` or
`dataset_config="hard"` explicitly to build V2 rows.
