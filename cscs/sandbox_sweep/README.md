# AgentENV group-concurrency sweep

Measures the wall time to finish **128 fixed groups × 8 episodes = 1,024 episode
attempts**, at active-group limits **8, 16, 32, 64**. Corresponding episode
ceilings are **64, 128, 256, 512**. A group retains its slot until all eight
sessions finish and its result is persisted. Failed episodes count as terminal
attempts and are reported separately; no replacement episodes are added.

The runner reuses the training gateway, ReAct agent, native AgentENV SDK,
Harbor reward verifier, and TransferQueue. It loads Qwen3.5-9B on two GH200 nodes
with eight GPUs and two TP4 replicas, 4K prompt capacity, 8K response capacity, temperature 1,
and top-p 1. No PPO trainer, optimizer, backward pass, or held-out dataset is
created. One framework actor runs episodes inline, with an additional group
semaphore. This isolates sandbox/rollout throughput from the trainer memory
failure.

## Submit on Clariden

```bash
bash cscs/sandbox_sweep/submit.sh
```

Set `SWEEP_INCLUDE_128=True` to append a conditional 128-active-group point
(1,024 simultaneous episode ceiling, still 1,024 total attempts). It runs only
if the 64-group point finishes with Slurm `COMPLETED`, zero failed groups, and
all 1,024 episodes both verified and usable. Otherwise `groups-128/skipped.json`
records the reason. `sweep-plan.json` drives queueing, reporting, and teardown;
the teardown watcher waits for the optional point or its recorded skip.

Defaults use the established CSCS model/data/image/private-key paths. Override
`SCRATCH`, `MODEL_PATH`, `DATA_PATH`, and `AGENTENV_SECRET_DIR` for another
account. The existing Clariden environment TOML and native Harbor wrapper must
also be available for that account. The AgentENV endpoint uses the EC2 DNS name
because the CSCS transparent web filter blocks the bare IP:
`http://ec2-3-229-113-27.compute-1.amazonaws.com:8000`.

Four debug-partition jobs run sequentially with `afterany` dependencies,
each requesting two exclusive nodes and 45 minutes. Debug permits only two
submitted jobs per user; `queue_points.py` submits later points as slots open. Each point separately
initializes vLLM, then starts the workload timer after engine/framework
readiness. Model startup is recorded separately. A timeout or failed point is
reported incomplete; subsequent points still run after cleanup. This is one
pass per concurrency, not a statistical estimate across repetitions. Later
points benefit from warmed server image caches; record that order effect.

`jobs.tsv` maps group limits to jobs and frozen per-point launch scripts.
`manifest.json` records deterministic task selection (seed 42) and hashes;
selection must match across points. Sampling is stochastic, so matching task
selection does not guarantee identical trajectories.

## Results and traces

- `progress.json` and `status.jsonl`: group completion, usable trajectories,
  episode errors and incomplete verifier counts.
- `result.json`: initialization time, rollout wall time, completion state and
  outcome counts. A terminal result is not automatically a successful episode.
- `telemetry/model-calls/group-status-*.jsonl`: observed group admission and
  completion, with active-group counts to check the cap.
- `telemetry/model-calls/`: raw model traces and available framework events.
- `native-traces/`: native sandbox lifecycle and command/file traces.
- `harbor/`: full agent/verifier artifacts and episode timing metadata.
- `telemetry/gpu/`: host/NUMA/GPU/DCGM/vLLM samples and final Ray logs.

AWS CPU/memory collectors and recorder traces are captured separately by the
study monitor. Keep their time windows aligned with `model_ready_at` and
`finished_at`. Compare groups/hour, usable episodes/hour, failures, sampled
peak VMs, CPU pressure, GPU SM/tensor activity and vLLM active requests. Failed
or incomplete points must not be treated as completed 128-group measurements.

Cleanup deletes only sandbox IDs successfully created and recorded by the
corresponding point. It does not enumerate and delete unrelated sandboxes.
An interrupted create whose response was never recorded may be left for the
server lifetime policy; record that limitation when interpreting the next point.


The primary comparison is observed **completed episodes per minute**, using
elapsed rollout time after model/framework readiness. Reports also show usable
trajectories and completed reward verifications per minute, plus error counts.
Running and interrupted points contribute observed rates, with their state and
measurement duration shown; unfinished episodes are excluded from the numerator.
A 128-group completion time is reported only for points that finish all groups.
