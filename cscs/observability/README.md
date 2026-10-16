# Training telemetry and replay captures

Each instrumented run stores:

- `telemetry/gpu/`: per-node JSONL host/GPU/cgroup readings every five seconds,
  DCGM profiling fields every second, and full vLLM Prometheus scrapes (running
  and waiting sequences, KV usage, token counters, iteration-token histograms,
  prefix-cache statistics, preemptions and latency histograms).
- `telemetry/model-calls/`: incremental full chat request/response bodies,
  timestamps, durations, usage, errors and native sandbox/trial correlation.
  Authentication headers are excluded and credential fields are redacted.
- `native-traces/`: lifecycle, commands, persistent shells, inputs/file bytes,
  outputs and errors. Model captures refer to these files by `sandbox_trace`.
- `harbor/` and `agent-logs/`: trial artifacts, verifier results and framework
  logs, stored separately for each run.
- `provenance/`: source snapshot, checksums, public settings, model metadata
  and the training image path/size/mtime. Model weights and credentials are not
  copied into these captures.

CloudLab monitoring is saved on that host under
`/sandbox/study/observability/<experiment>/`: host CPU/memory/I/O/pressure,
filesystem usage, service cgroup accounting, VM processes and service journal.
Its collector has a 24-hour limit; GPU collectors terminate with the job.
Protected service and kernel journals are read with `--sudo-journal` on CloudLab.
Start this separate collector on the sandbox host before submitting:

```bash
python3 collect.py --role sandbox --output /sandbox/study/observability/<experiment> \
  --disk /sandbox --duration 86400 --sudo-journal
```

Run it in a persistent terminal or redirect it with `nohup`; the service-journal
option requires noninteractive sudo access. This does not change service
admission limits or the training runner concurrency.
The two hosts use epoch timestamps so their records can be aligned. SSH
collection timestamps are not a substitute for synchronized system clocks.

## Interpretation

Compare useful token throughput and active batches over the **same time window**
as GPU counters and sandbox operations. Separate startup, rollout, verification,
training and weight-sync periods. GPU busy percentage alone does not establish
saturation. Record sandbox startup errors and timeouts as failed capacity
points rather than successful zero-reward episodes.

## Replay

`replay.py` consumes native and model JSONL captures. It creates fresh AgentENV
sandboxes and replays the recorded operations. Persistent-shell IDs and file
contents are retained; nested readiness/setup operations are omitted because
the provider performs them itself. Modes:

- `sandbox`: replay commands/files with recorded model think times if pacing
  is enabled; no model requests are submitted.
- `gpu`: replay recorded model requests against a supplied OpenAI-compatible
  endpoint; no sandboxes are started.
- `coupled`: replay both, keeping recorded sandbox actions after each model
  request. New model responses are measured but do not change recorded actions.

```bash
export PYTHONPATH=/path/to/runtime:/path/to/runtime/verl:/path/to/native-shell
export AGENTENV_ENDPOINT=http://your-agentenv-host:8000
# Supply AGENTENV_API_KEY privately, outside captured commands.
python replay.py --native-dir /path/to/native-traces \
  --model-dir /path/to/telemetry/model-calls --mode sandbox \
  --concurrency 16 --limit 64 --output /path/to/replay.jsonl
```

For `gpu` or `coupled`, also specify `--llm-base-url http://host:port/v1`.
`--pace recorded` preserves relative operation start times within each lane;
the default `asap` runs each lane as fast as its dependencies allow.
Replay arrival patterns across lanes are not preserved: concurrency is an
explicit experiment variable. Recreated sandbox IDs are new. Incomplete traces,
external network side effects, ephemeral files and random model outputs can
prevent exact reproduction. This is workload replay, not deterministic recovery
of the training optimizer or external websites. The default skips unfinished
sandbox lifecycles; `--include-incomplete` enables partial workloads explicitly.

## Download for analysis

From a machine with SSH access to both hosts and `rsync` installed:

```bash
bash cscs/observability/fetch.sh /remote/clariden/run \
  /sandbox/study/observability/<experiment> ./analysis/<job-id>
```

The output contains `run/`, `native-traces/` and `cloudlab/`. Use
`run/telemetry/model-calls/` as the replay model directory. This can download
an interim snapshot during training; collect again after completion for a
complete archive. Capture writes are incremental but an in-progress final
JSONL line may be partial. Raw Ray logs are copied by the node cleanup trap;
an abrupt node loss can prevent that final copy.

Raw captures may contain repository and tool content. Keep them with the run's
private storage; do not commit them. The scripts are separate from model
training validation and do not change the runner's concurrency setting.

## Resource correlation figures

`export_metrics.py` extracts metric-only JSONL snapshots without model prompts,
responses, tool contents, or file bytes. Run it beside the captures, then copy
its compressed output to the analysis machine. Use the **same epoch cutoff**
on both hosts for a bounded snapshot. Sources remain intact.

```bash
# GPU cluster (Python 3.6+):
python3 export_metrics.py --telemetry "$RUN/telemetry/gpu" \
  --models "$RUN/telemetry/model-calls" --native "$NATIVE_TRACE_DIR" \
  --cutoff "$CUTOFF" --output "$RUN/analysis-tools/cluster.jsonl.gz"
# AgentENV host:
python3 export_metrics.py --telemetry "$SANDBOX_METRICS_DIR" \
  --cutoff "$CUTOFF" --output "$SANDBOX_METRICS_DIR/cloudlab.jsonl.gz"
```

Check collector coverage before interpreting it. The image used by job 3666780
lacks `dcgmi` and produced no in-container vLLM scrapes. A supplementary collector
started **on the allocated hosts, outside the container**, captures both without
changing training. For an existing allocation, run from the cluster login host:

```bash
srun --jobid="$JOB_ID" --overlap --kill-on-bad-exit=0 \
  -N "$NNODES" -n "$NNODES" --ntasks-per-node=1 -c 1 \
  python3 "$RUN/collect.py" --role gpu \
  --output "$RUN/telemetry/host-supplement" --disk "$WORK_ROOT" --duration 21600
```

Keep this command in a persistent terminal or start it with `nohup` and redirect
its log. It uses the existing job and terminates when that allocation ends.
Export that directory separately with `export_metrics.py --telemetry ...`.
Its later coverage does not recover missing historical DCGM/vLLM samples.

On the analysis machine:

```bash
uv run --no-project --with matplotlib --with numpy python \
  cscs/observability/plot_resources.py cluster.jsonl.gz cloudlab.jsonl.gz \
  supplement.jsonl.gz \
  --trainer-nodes nid006127,nid006768,nid007637,nid007643 \
  --start 2026-10-15T07:55:13+00:00 \
  --title 'Job 3666780: 16 trainer / 24 rollout GPUs' \
  --output outputs/resource-study-3666780
```

The node names above belong to job 3666780. Determine actual trainer placement
for each run; Ray placement need not match Slurm node order. The plotting
packages are analysis tools, not training dependencies.

Outputs:

- `01_pipeline.{png,pdf}`: GPU activity, delivered model tokens, in-flight
  requests, VM count and operation latency on the same timeline.
- `02_sandbox.{png,pdf}`: CPU, load, memory/swap, PSI, disk/network rates and
  exception rates. Physical disk/NIC counters exclude virtual interfaces and
  device-mapper layers to avoid double counting.
- `03_vllm.{png,pdf}`: active/waiting requests, KV cache and engine token rates;
  zooms to the window in which scrapes actually exist.
- `04_associations.{png,pdf}`: exploratory resource/performance scatter plots.
- `metrics_1min.csv`, `summary.json`, `README.md`: numerical data, coverage and
  interpretation limits. Summary percentiles describe minute aggregates.

All figures use UTC. Successful and failed operation durations are kept
separate; unfinished operations remain right-censored. Model output tokens are
attributed to response completion, so this series is delivered throughput,
not instantaneous decode throughput. vLLM figures average endpoint samples;
they do not sum endpoint capacities. Native `exec` and `shell_run` can overlap.
Correlations mix workload phases and do not establish the limiting resource by
themselves. A controlled concurrency sweep is needed to identify a saturation
point. Missing metrics remain gaps rather than fabricated zeros.

Pass `--training-log training.out` to add `05_training_steps.{png,pdf}` and
`training_steps.json`: completed-step timing split into waiting for rollout,
log-probability computation, actor update, weight sync and remaining time.
`gen` here is time the trainer waits for a sampled batch; it is not the total
amount of concurrent generation work performed by rollout servers.

## Per-step rollout demand and episode stages

To identify whether batches drain during backward, include existing session logs
in the metric export. Their timestamps are local wall times: supply the actual
UTC offset for the capture period (for example, Zurich on 2026-10-15 is +2).
An offset that changes during a run requires splitting the capture window.

```bash
python3 export_metrics.py --telemetry "$RUN/telemetry/host-supplement" \
  --models "$RUN/telemetry/model-calls" --native "$NATIVE_TRACE_DIR" \
  --episode-logs "$FRAMEWORK_LOG_ROOT" --log-utc-offset-hours 2 \
  --output "$RUN/analysis-tools/demand.jsonl.gz"

uv run --no-project --with matplotlib --with numpy python \
  cscs/observability/plot_rollout_demand.py demand.jsonl.gz \
  --trainer-nodes nid006127,nid006768,nid007637,nid007643 \
  --output outputs/resource-study-3666780
```

`06_step_cohorts.{png,pdf}` groups admissions, active episodes, model-waiting
episodes and model completions by the trainer step at dispatch. This is the
originating step, not the model's eventual policy version. Existing captures
provide first **admitted episode** times, not exact driver dispatch timestamps.
Warmup and normal batches at the same step cannot be separated retrospectively.

`07_episode_stages.{png,pdf}` partitions admitted episode wall time into model
requests, sandbox startup, sandbox commands/files, and other/unmapped work.
Stage priority avoids counting nested operations twice. It correlates client
model waits with actual vLLM running/waiting counts. A request waiting on the
client/gateway/network is not necessarily executing in vLLM. Verifier commands
share the sandbox-operation category; the capture cannot reliably label every
command as setup, agent tool or verifier. Unmapped native traces stay unknown.
`episode_stage_metrics.csv`, `demand_summary.json`, and `DEMAND.md` retain the
numbers and coverage gaps.

Future framework runs with `UNI_AGENT_TRACE_DIR` also write compact
`pipeline-<pid>.jsonl` events: batch/episode begin, end and error, IDs, dispatch
step, counts, duration and exception class. Episode begin is after runner
semaphore admission; waiting for that semaphore is excluded. Episode end
includes scoring/artifact persistence but precedes TransferQueue writes.
The exporter preserves these events; the demand plot uses explicit batch
boundaries when present. This instrumentation does not change dispatch/refill
policy, concurrency, or trainer/rollout allocation.

### Feeding vLLM

The current `separate_async` loop submits one full batch at the beginning of
each training step and uses `num_warmup_batches` for initial prefetch. It is not
a continuously replenished episode producer. Increasing the existing warmup
setting can deepen prefetch, but also increases the initial sandbox burst and
policy staleness; the current server already exhibits substantial CPU pressure.

A sustained feeder would need a producer independent of the backward loop,
a target number of admitted in-flight prompt groups, a bounded ready-data
backlog, replacements as groups complete, and coordination with policy-version
staleness, failed-group refills, checkpointing, and final shutdown. It must not
launch unbounded new work merely to keep GPUs busy when the trainer cannot
consume it. That policy is implemented by the optional continuous refill extension described
in `cscs/native_agentenv/README.md`; the full-memory experiment profile enables it.

First shorten sandbox gaps (warm/cache images and distribute CPU-intensive
commands/verifiers across adequate sandbox hosts). Then compare finite prefetch
against a bounded continuous feeder at identical topology and policy-staleness
settings. Increase active request targets in measured plateaus and stop when
useful throughput flattens or errors/latency rise. If the trainer remains the
consumer bottleneck, rebalance GPUs toward training rather than accumulate
unused or stale trajectories.

Continuous refill records `feeder_status` pipeline events with prompt-group
counts and submission totals. The metric exporter preserves these records.
Existing per-step cohort figures remain useful: admission should now spread
through backward rather than appear only at the beginning of a training step.

### Automatic run monitoring

`watch_gpu_job.sh JOB_ID RUN_DIR` waits for the main Ray container step, then
starts host-side GPU/DCGM/vLLM collectors in the existing allocation. Waiting
until the main step starts avoids having the monitoring step occupy CPUs needed
by the main container step. It records through job termination, with a six-hour
maximum per collector invocation.

For a run with both SSH endpoints reachable, `monitor_run.py` periodically
exports metric-only snapshots from both hosts, discovers actual trainer node
placement, and regenerates the figures until the job exits. Run it from a
persistent terminal or with `nohup`:

```bash
uv run --no-project --with matplotlib --with numpy python \
  cscs/observability/monitor_run.py --job-id JOB_ID --run /remote/run \
  --cloudlab-dir /sandbox/study/observability/EXPERIMENT \
  --output outputs/resource-study-JOB_ID --interval 600
```

This operator helper targets the ten-node Clariden experiment and its existing
framework-log layout. Scheduler state is saved separately from training
completion: a job can report all five optimizer steps and still fail during
teardown. Snapshots and figures are refreshed every ten minutes while running,
and once more at termination. Missing startup/episode data is reported as a gap.
Local monitoring requires a live SSH agent; SSH failures are logged and retried.

`08_continuous_refill.{png,pdf}` aligns feeder in-flight/terminal group counts,
vLLM active/waiting requests, rollout SM activity, sandbox CPU pressure and
delivered output tokens. Interpret producer backpressure separately from sandbox
or GPU capacity: a full ready-data backlog deliberately stops replenishment.

### DCGM inside the training container

The container image lacks the DCGM client. New run preparation bundles the
site's ARM64 `dcgmi` and its matching `libdcgm.so` under `RUN/dcgm-client/` using
`bundle_dcgm.py`. The existing scratch mount exposes that prefix inside the
container. The node bootstrap sets `DCGMI_PATH` and adds its `bin/` to `PATH`;
the client's RPATH finds the bundled library without changing global library
search paths. It connects to the existing site host engine through the shared
network, rather than launching another daemon or rebuilding the image.

The collector records fields 1001–1005 every second: graphics activity, SM
active, SM occupancy, tensor activity and DRAM activity. These are device-wide
fields, not per-process attribution. Run provenance records client version,
architecture and binary/library SHA-256 hashes. Host-side monitoring remains
available as an independent fallback. `dcgm-container-proof.txt` in job
3668373's analysis directory records a successful query inside the same image
using an existing owned allocation; those readings are not measurements of the
pending training run.

### CPU attribution for the sandbox host

`collect_cpu.py --output DIR --interval 5 --duration 86400` is an optional,
read-only sidecar on the AgentENV host. It records interval CPU tick deltas for
Firecracker, registry clients, image preparation, VM storage, the server and
other processes. It executes no guest commands and records no command bodies.
The exporter includes `*-cpu-attribution.jsonl`; the standard plotter creates
`09_cpu_attribution.{png,pdf}` when samples exist.

All process percentages use total logical host CPU capacity. Guest execution is
already included in Firecracker CPU; do not add the two. VM guest time includes
boot and installation as well as agent commands/tests. This data does not yet
separate those guest stages. Processes that start/exit between samples or have
unreadable proc files may be missed; remaining host busy time is unattributed.
`ps %CPU` lifetime averages are not used as interval attribution.

Historical guest-versus-host CPU is also derived from the existing `/proc/stat`
captures, with guest ticks counted only once. Detailed process attribution starts
when the sidecar starts and cannot be backfilled from lifetime process averages.

`compare_cpu_gpu.py --study outputs/resource-study-JOB_ID` creates
`10_cpu_gpu_comparison.{png,pdf}`, `CPU_GPU.md` and `cpu_gpu_comparison.json`.
It uses identical complete one-minute bins for CPU busy, trainer/rollout GPU
busy, SM active and tensor activity, and records feeder backlog/backpressure
in the same window. These are distinct counters, not equal measures of compute
efficiency. The automatic monitor regenerates this comparison after each
snapshot when the script is available.


### AgentENV telemetry on AWS

`monitor_run.py` accepts `--sandbox ubuntu@3.229.113.27`,
`--sandbox-identity ~/.ssh/agentenv-xiayao.pem`, and `--sandbox-disk /nvme`.
Use `--nodes 2` for the smoke topology. `--cloudlab-dir` names the remote
telemetry directory on either platform; it need not be a CloudLab path.
`--manage-sandbox-collector` starts host/service and process CPU collectors
when the Slurm job begins. GPU snapshots now include per-NUMA node memory.

For run 3670472, server-side recorder traces remain under
`/nvme/aenv-recorder/` and are linked from its server telemetry directory.
Client model/native traces remain in the CSCS run and its recorded trace paths.
The recorder owns port 8000 and forwards to native AgentENV on localhost; port
9103 contains storage metrics and is not the AgentENV API.
