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
