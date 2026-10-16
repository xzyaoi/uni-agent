#!/usr/bin/env python3
"""Show per-step rollout cohorts and episode stages using metric-only exports."""
import argparse
import collections
import csv
import datetime as dt
import gzip
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

STAGES = ["model_request", "sandbox_start", "sandbox_operation", "other_or_unmapped"]
LABELS = ["Model request in flight (client)", "Sandbox start", "Sandbox command / file operation", "Other / unmapped / missing terminal"]


def overlap_bins(start, end):
    while start < end:
        minute = int(start // 60) * 60
        right = min(end, minute + 60)
        yield minute, (right - start) / 60
        start = right


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trainer-nodes", required=True)
    args = parser.parse_args()
    trainers = set(args.trainer_nodes.split(","))
    episodes = {}
    native_sessions = {}
    operations = []
    pipeline = []
    engine = collections.defaultdict(lambda: collections.defaultdict(list))
    cutoff = 0
    parse_errors = 0
    with gzip.open(args.input, "rt") as stream:
        for line in stream:
            try:
                r = json.loads(line)
            except ValueError:
                parse_errors += 1
                continue
            kind, t = r.get("kind"), r.get("timestamp", 0)
            if kind == "snapshot":
                cutoff = max(cutoff, t)
            elif kind == "episode":
                episodes[r["session_id"]] = r
            elif kind == "pipeline":
                pipeline.append(r)
            elif kind in ("model", "native"):
                if kind == "native" and r.get("operation") == "agent" and r.get("session_id"):
                    native_sessions[r["trace_file"]] = r["session_id"]
                if r.get("phase") in ("begin", "end", "error"):
                    operations.append((kind, r.get("trace_file"), r.get("operation_id"), r.get("operation"),
                                       r.get("phase"), t, r.get("session_id")))
            elif kind == "vllm" and r.get("node") not in trainers:
                identity = (r.get("node"), r.get("address"))
                minute = int(t // 60) * 60
                values = {}
                for text in r.get("prometheus", "").splitlines():
                    match = re.match(r"vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc)(?:\{.*\})?\s+([\d.eE+\-]+)$", text)
                    if match:
                        values[match[1]] = values.get(match[1], 0) + float(match[2])
                for key, value in values.items():
                    engine[(minute, key)][identity].append(value)

    if not cutoff:
        raise SystemExit("Input requires snapshot timestamp")
    # Prefer explicit episode boundaries when future pipeline captures exist.
    for r in sorted(pipeline, key=lambda record: record["timestamp"]):
        sid = r.get("session_id")
        if not sid:
            continue
        if r.get("event") == "episode_begin":
            episodes[sid] = {"session_id": sid, "timestamp": r["timestamp"], "global_steps": r["global_steps"], "batch_id": r.get("batch_id")}
        elif r.get("event") in ("episode_end", "episode_error") and sid in episodes:
            episodes[sid]["end_timestamp"] = r["timestamp"]
    if not episodes:
        raise SystemExit("No episode logs or explicit episode pipeline events captured")

    changes = collections.defaultdict(list)
    pending = {}
    mapped = collections.Counter()
    starts = collections.Counter()
    total_operations = collections.Counter()
    completed = collections.Counter()
    for kind, file, opid, op, phase, t, sid in operations:
        identity = (kind, file, opid)
        if phase == "begin":
            pending[identity] = (t, op, sid)
            continue
        begin = pending.pop(identity, None)
        if begin is None:
            continue
        left, op, sid0 = begin
        sid = sid or sid0 or (native_sessions.get(file) if kind == "native" else None)
        total_operations[kind] += 1
        if sid in episodes:
            mapped[kind] += 1
            stage = "model_request" if kind == "model" else ("sandbox_start" if op == "start" else "sandbox_operation")
            if kind == "native" and op not in ("start", "exec", "shell_run", "read_file", "write_file", "open_shell", "shell_close", "stop"):
                continue
            changes[sid].append((left, stage, 1))
            changes[sid].append((t, stage, -1))
            if kind == "model":
                completed[(int(t // 60) * 60, int(episodes[sid]["global_steps"]))] += 1
    for (kind, file, _), (left, op, sid) in pending.items():
        sid = sid or (native_sessions.get(file) if kind == "native" else None)
        if sid in episodes:
            stage = "model_request" if kind == "model" else ("sandbox_start" if op == "start" else "sandbox_operation")
            changes[sid].append((left, stage, 1))
            changes[sid].append((cutoff, stage, -1))
    del operations

    values = collections.defaultdict(collections.Counter)
    admissions = collections.Counter()
    cohort_starts = {}
    for sid, episode in episodes.items():
        left = episode["timestamp"]
        right = min(episode.get("end_timestamp", cutoff), cutoff)
        step = int(episode["global_steps"])
        if right <= left:
            continue
        cohort_starts[step] = min(cohort_starts.get(step, left), left)
        admissions[(int(left // 60) * 60, step)] += 1
        events = sorted(changes.get(sid, []))
        counts = collections.Counter()
        previous = left
        def account(begin, end):
            if end <= begin:
                return
            stage = next((name for name in STAGES[:3] if counts[name] > 0), STAGES[-1])
            for minute, weight in overlap_bins(begin, end):
                values[minute]["step_%d_episodes" % step] += weight
                values[minute]["stage_" + stage] += weight
                if stage == "model_request":
                    values[minute]["step_%d_model_episodes" % step] += weight
                if stage in ("sandbox_start", "sandbox_operation"):
                    values[minute]["step_%d_sandbox_episodes" % step] += weight
        for t, stage, delta in events:
            if t < left:
                counts[stage] += delta
                continue
            if t > right:
                break
            account(previous, t)
            counts[stage] += delta
            previous = t
        account(previous, right)
    minutes = list(range(min(values), int(cutoff // 60) * 60 + 1, 60))
    steps = sorted(cohort_starts)
    rows = []
    for minute in minutes:
        row = {"epoch": minute, "timestamp_utc": dt.datetime.fromtimestamp(minute, dt.timezone.utc).isoformat()}
        row.update(values[minute])
        for step in steps:
            for suffix in ("episodes", "model_episodes", "sandbox_episodes"):
                row.setdefault("step_%d_%s" % (step, suffix), 0)
            row["step_%d_admissions" % step] = admissions[(minute, step)]
            row["step_%d_model_completions" % step] = completed[(minute, step)]
        for key in STAGES:
            row.setdefault("stage_" + key, 0)
        for metric, label in [("num_requests_running", "vllm_running_total"), ("num_requests_waiting", "vllm_waiting_total")]:
            endpoint_values = engine.get((minute, metric))
            if endpoint_values:
                row[label] = sum(np.mean(v) for v in endpoint_values.values())
                row["vllm_endpoint_count"] = len(endpoint_values)
        rows.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with (args.output / "episode_stage_metrics.csv").open("w") as out:
        writer = csv.DictWriter(out, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    times = [dt.datetime.fromtimestamp(r["epoch"], dt.timezone.utc) for r in rows]
    def series(key):
        return np.array([r.get(key, np.nan) for r in rows])
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25})
    def finish(fig, axes, name, title):
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=dt.timezone.utc))
            ax.legend(loc="upper left", fontsize=8)
        axes[-1].set_xlabel("UTC on " + times[0].strftime("%Y-%m-%d"))
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, .94])
        for ext in ("png", "pdf"):
            fig.savefig(args.output / (name + "." + ext), dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    for step in steps:
        label = "Dispatched at step %d" % step
        axes[0].plot(times, series("step_%d_admissions" % step), label=label)
        axes[1].plot(times, series("step_%d_episodes" % step), label=label)
        axes[2].plot(times, series("step_%d_model_episodes" % step), label=label)
        axes[3].plot(times, series("step_%d_model_completions" % step), label=label)
    for ax, label in zip(axes, ["Episode admissions/min", "Open episodes in logs", "Model-waiting episodes", "Model completions/min"]):
        ax.set_ylabel(label)
    batch_begins = [record for record in pipeline if record.get("event") == "batch_begin"]
    boundaries = [(record.get("global_steps"), record["timestamp"]) for record in batch_begins] if batch_begins else list(cohort_starts.items())
    for step, t in boundaries:
        for ax in axes:
            ax.axvline(dt.datetime.fromtimestamp(t, dt.timezone.utc), color="grey", linestyle=":", alpha=.6)
    boundary_label = "captured batch begin" if batch_begins else "first observed admitted episode, not exact batch dispatch"
    finish(fig, axes, "06_step_cohorts", "Rollout cohorts by dispatch step\nDotted lines: " + boundary_label + "; counts are time-weighted minute means")

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    axes[0].stackplot(times, *[series("stage_" + key) for key in STAGES], labels=LABELS, alpha=.85)
    axes[0].set_ylabel("Open episodes in logs")
    axes[1].plot(times, series("stage_model_request"), label="Episodes with client request in flight")
    axes[1].plot(times, series("vllm_running_total"), label="vLLM running requests (sum of sampled endpoints)")
    axes[1].plot(times, series("vllm_waiting_total"), label="vLLM waiting requests (sum)")
    axes[1].set_ylabel("Count")
    total = sum(series("stage_" + key) for key in STAGES)
    for key, label in zip(STAGES, LABELS):
        axes[2].plot(times, np.divide(series("stage_" + key) * 100, total, out=np.full_like(total, np.nan), where=total > 0), label=label)
    axes[2].set_ylabel("Observed episode time (%)")
    finish(fig, axes, "07_episode_stages", "Where admitted rollout episodes spend time\nClient model time includes gateway/network/retries; sandbox operations include setup/tools/verification; missing mappings stay unknown")

    totals = {key: sum(r.get("stage_" + key, 0) for r in rows) for key in STAGES}
    total = sum(totals.values())
    summary = {
        "snapshot_end_utc": dt.datetime.fromtimestamp(cutoff, dt.timezone.utc).isoformat(),
        "episode_count": len(episodes), "episodes_without_terminal": sum("end_timestamp" not in episode for episode in episodes.values()), "episodes_by_dispatch_step": dict(collections.Counter(int(e["global_steps"]) for e in episodes.values())),
        "cohort_first_admission_utc": {str(k): dt.datetime.fromtimestamp(v, dt.timezone.utc).isoformat() for k, v in cohort_starts.items()},
        "stage_episode_minutes": totals, "stage_share_pct": {key: value / total * 100 for key, value in totals.items()},
        "mapped_completed_operations": dict(mapped), "completed_operations": dict(total_operations),
        "pipeline_event_count": len(pipeline), "captured_batch_begins": len(batch_begins), "parse_errors": parse_errors,
    }
    (args.output / "demand_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = ["# Rollout demand and episode stages", "", "Snapshot ends at " + summary["snapshot_end_utc"] + ".", "", "[Per-step rollout cohorts](06_step_cohorts.png) · [Episode stages](07_episode_stages.png)", "", "## Coverage", "", "- %d episodes; dispatch-step counts: `%s`." % (len(episodes), json.dumps(summary["episodes_by_dispatch_step"])), "- Matched completed operations / total: `%s` / `%s`." % (json.dumps(dict(mapped)), json.dumps(dict(total_operations))), "- Explicit pipeline events: %d. When absent, cohort admission times are reconstructed from session logs." % len(pipeline), "", "## Observed episode time", "", "| Stage | Episode minutes | Share |", "|---|---:|---:|"]
    for key, label in zip(STAGES, LABELS):
        text.append("| %s | %.1f | %.1f%% |" % (label, totals[key], summary["stage_share_pct"][key]))
    text += ["", "These are mutually exclusive wall-time stages of admitted episodes, not host CPU attribution. Model requests take precedence over startup, then command/file work. Native operations can nest and are counted once per episode. Unmapped traces, absent boundaries and uninstrumented work remain unknown. Episodes without a terminal log are treated as open until the cutoff, which can overstate activity if cancellation omitted a terminal record. Unmapped/other time must not be attributed to sandbox tools or CPU. Open episodes and operations are right-censored at the snapshot cutoff. Log timestamps have one-second precision and use the explicit export UTC offset. Final bins may be partial. Model-client waits include retries/gateway/network; they are not equivalent to engine-active batches. More prefetched work may increase sandbox pressure or policy staleness; this plot does not select a safe refill target.", ""]
    (args.output / "DEMAND.md").write_text("\n".join(text))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
