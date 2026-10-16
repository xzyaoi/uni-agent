#!/usr/bin/env python3
"""Plot aligned resource and workload metrics from export_metrics.py snapshots.

Run with: uv run --no-project --with matplotlib --with numpy python ...
"""
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


def read_records(paths):
    for path in paths:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as stream:
            for line in stream:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue


def numbers(text):
    result = {}
    for line in (text or "").splitlines():
        fields = line.replace(":", " ").split()
        if len(fields) > 1:
            try:
                result[fields[0]] = float(fields[1])
            except ValueError:
                pass
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trainer-nodes", required=True, help="Comma-separated observed trainer node names")
    parser.add_argument("--start", help="ISO timestamp with explicit timezone")
    parser.add_argument("--training-log", type=Path, help="Optional training.out for completed-step timing bars")
    parser.add_argument("--title", default="Training / AgentENV resource correlation")
    args = parser.parse_args()
    start = dt.datetime.fromisoformat(args.start).timestamp() if args.start else 0
    trainers = set(args.trainer_nodes.split(","))
    data = collections.defaultdict(lambda: collections.defaultdict(list))
    counts = collections.Counter()
    op_counts = collections.Counter()
    errors = collections.Counter()
    events = collections.defaultdict(list)
    pending = collections.defaultdict(dict)
    previous_cpu = {}
    previous_disk = {}
    previous_net = {}
    previous_counter = {}
    dcgm_errors = set()
    bounds = collections.defaultdict(list)
    end = start

    def put(t, key, value):
        if t >= start and value is not None and np.isfinite(value):
            data[int(t // 60) * 60][key].append(float(value))

    # Files are written chronologically within each node. Separate supplements
    # need independent counter baselines to avoid negative time differences.
    for source in args.inputs:
        previous_cpu.clear()
        previous_disk.clear()
        previous_net.clear()
        previous_counter.clear()
        for r in read_records([source]):
            kind = r.get("kind")
            t = r.get("timestamp", 0)
            if not isinstance(t, (int, float)) or t < start:
                continue
            end = max(end, t)
            counts[kind] += 1
            bounds[kind].append(t)
            node = r.get("node", "unknown")
            if kind == "host":
                mem = numbers(r.get("meminfo"))
                role = "sandbox" if r.get("role") == "sandbox" else ("trainer" if node in trainers else "rollout")
                if mem:
                    put(t, role + "_ram_available_gib", mem.get("MemAvailable", np.nan) / 1048576)
                    put(t, role + "_swap_used_gib", (mem.get("SwapTotal", 0) - mem.get("SwapFree", 0)) / 1048576)
                for line in r.get("gpus", {}).get("stdout", "").splitlines():
                    fields = [x.strip() for x in line.split(",")]
                    if len(fields) >= 6:
                        try:
                            put(t, role + "_gpu_busy_pct", float(fields[2]))
                            put(t, role + "_gpu_memory_gib", float(fields[4]) / 1024)
                            put(t, role + "_gpu_memory_pct", float(fields[4]) / float(fields[5]) * 100)
                        except ValueError:
                            pass
                if role != "sandbox":
                    continue
                put(t, "sandbox_vm_count", r.get("vm_count"))
                disk = r.get("disk", {}).get("bytes", [])
                if len(disk) == 3:
                    put(t, "sandbox_disk_free_gib", disk[2] / 2**30)
                load = (r.get("loadavg") or "").split()
                if load:
                    put(t, "sandbox_load1", float(load[0]))
                cpu = (r.get("cpu_stat") or "").split()
                if len(cpu) >= 9:
                    # guest time is already included in user/nice; do not add it again.
                    vals = np.array([float(x) for x in cpu[1:11]])
                    prev = previous_cpu.get(node)
                    previous_cpu[node] = (t, vals)
                    if prev and t > prev[0]:
                        delta = vals - prev[1]
                        total = delta[:8].sum()
                        if total > 0 and np.all(delta >= 0):
                            put(t, "sandbox_cpu_busy_pct", 100 * (total - delta[3] - delta[4]) / total)
                            put(t, "sandbox_cpu_iowait_pct", 100 * delta[4] / total)
                            put(t, "sandbox_cpu_kernel_pct", 100 * (delta[2] + delta[5] + delta[6]) / total)
                            if len(delta) >= 10:
                                put(t, "sandbox_cpu_guest_pct", 100 * delta[8:10].sum() / total)
                for resource, text in r.get("pressure", {}).items():
                    match = re.search(r"some avg10=([\d.]+)", text or "")
                    if match:
                        put(t, "sandbox_" + resource + "_psi_pct", float(match[1]))
                net = np.zeros(2)
                interfaces = []
                for line in (r.get("netdev") or "").splitlines():
                    if ":" not in line:
                        continue
                    name, counters = line.split(":", 1)
                    fields = counters.split()
                    if re.fullmatch(r"(?:en\w+|eth\d+)", name.strip()) and len(fields) >= 9:
                        net += [float(fields[0]), float(fields[8])]
                        interfaces.append(name.strip())
                if interfaces:
                    prev = previous_net.get(node)
                    previous_net[node] = (t, net)
                    if prev and t > prev[0] and np.all(net >= prev[1]):
                        delta = (net - prev[1]) / (t - prev[0]) / 2**20
                        put(t, "sandbox_network_rx_mib_s", delta[0])
                        put(t, "sandbox_network_tx_mib_s", delta[1])
                # Sum whole physical devices only, avoiding partitions and dm layers.
                io = np.zeros(3)
                devices = []
                for line in (r.get("diskstats") or "").splitlines():
                    f = line.split()
                    if len(f) >= 14 and re.fullmatch(r"(?:sd[a-z]+|nvme\d+n\d+|vd[a-z]+)", f[2]):
                        io += [float(f[5]), float(f[9]), float(f[12])]
                        devices.append(f[2])
                if devices:
                    prev = previous_disk.get(node)
                    previous_disk[node] = (t, io)
                    if prev and t > prev[0] and np.all(io >= prev[1]):
                        delta = (io - prev[1]) / (t - prev[0])
                        put(t, "sandbox_disk_read_mib_s", delta[0] * 512 / 2**20)
                        put(t, "sandbox_disk_write_mib_s", delta[1] * 512 / 2**20)
                        # Busy milliseconds summed across devices: may exceed 100%.
                        put(t, "sandbox_disk_busy_device_pct", delta[2] / 10)
            elif kind == "cpu_attribution":
                put(t, "attribution_host_busy_pct", r.get("host_busy_pct"))
                put(t, "attribution_unattributed_pct", r.get("unattributed_busy_pct"))
                for name, values in r.get("groups", {}).items():
                    put(t, "attribution_" + name + "_pct", values.get("host_capacity_pct"))
                    if name == "vm":
                        put(t, "attribution_vm_guest_pct", values.get("guest_capacity_pct"))
            elif kind == "pipeline" and r.get("event") == "feeder_status":
                for key in ["inflight_prompts", "ready_prompts", "failed_prompts", "target_prompts", "ready_threshold"]:
                    put(t, "refill_" + key, r.get(key))
                put(t, "refill_submitted_prompts", r.get("submitted_now"))
                if "ready_prompts" in r and "ready_threshold" in r:
                    put(t, "refill_backlog_pause_fraction", int(r.get("backlog_paused", r["ready_threshold"] > 0 and r["ready_prompts"] >= r["ready_threshold"])))
            elif kind == "dcgm":
                if r.get("error"):
                    dcgm_errors.add(r["error"])
                f = (r.get("line") or "").split()
                if len(f) >= 7 and f[0] == "GPU":
                    role = "trainer" if node in trainers else "rollout"
                    for key, val in zip(["graphics", "sm_active", "sm_occupancy", "tensor_active", "dram_active"], f[2:7]):
                        try:
                            put(t, role + "_" + key + "_pct", float(val) * 100)
                        except ValueError:
                            pass
            elif kind == "vllm":
                endpoint = node + ":" + r.get("address", "")
                metrics = collections.defaultdict(list)
                for line in r.get("prometheus", "").splitlines():
                    match = re.match(r"(vllm:[\w:]+)(?:\{.*\})?\s+([\d.eE+\-]+)(?:\s|$)", line)
                    if match:
                        metrics[match[1]].append(float(match[2]))
                role = "trainer" if node in trainers else "rollout"
                for metric, key in [("num_requests_running", "active_requests"), ("num_requests_waiting", "waiting_requests"), ("kv_cache_usage_perc", "kv_cache_pct"), ("gpu_cache_usage_perc", "kv_cache_pct")]:
                    values = metrics.get("vllm:" + metric)
                    if values:
                        put(t, role + "_vllm_" + key, np.mean(values) * 100 if key == "kv_cache_pct" else sum(values))
                for metric in ["generation_tokens_total", "prompt_tokens_total"]:
                    values = metrics.get("vllm:" + metric)
                    if values:
                        value = sum(values)
                        identity = (endpoint, metric)
                        prev = previous_counter.get(identity)
                        previous_counter[identity] = (t, value)
                        if prev and t > prev[0] and value >= prev[1]:
                            put(t, role + "_vllm_" + metric.replace("_total", "_s"), (value - prev[1]) / (t - prev[0]))
            elif kind in ("model", "native"):
                op = r.get("operation", "unknown")
                phase = r.get("phase")
                identity = (r.get("trace_file"), r.get("operation_id"))
                if phase == "begin":
                    events[kind].append((t, 1))
                    pending[kind][identity] = t
                elif phase in ("end", "error"):
                    events[kind].append((t, -1))
                    pending[kind].pop(identity, None)
                    op_counts[(kind, op, phase)] += 1
                    if phase == "error":
                        errors[(kind, op, r.get("error", "unknown"))] += 1
                    prefix = "model" if kind == "model" else "native_" + op
                    put(t, prefix + "_finished", 1)
                    put(t, prefix + "_errors", int(phase == "error"))
                    if phase == "end":
                        put(t, prefix + "_duration_s", r.get("duration"))
                    else:
                        put(t, prefix + "_error_duration_s", r.get("duration"))
                    if kind == "model" and phase == "end":
                        usage = r.get("usage") or {}
                        if "completion_tokens" in usage:
                            put(t, "model_completed_output_tokens", usage["completion_tokens"])
                        if "prompt_tokens" in usage:
                            put(t, "model_completed_input_tokens", usage["prompt_tokens"])

    if not data:
        raise SystemExit("No usable metric records in this time window")
    first = min(data)
    last = int(end // 60) * 60
    rows = {}
    # Include zero-count bins only where the corresponding trace coverage exists.
    for minute in range(first, last + 1, 60):
        row = {}
        for key, values in data[minute].items():
            if key.endswith(("_finished", "_errors", "_tokens")):
                row[key] = sum(values)
            else:
                row[key] = float(np.mean(values))
            if key.endswith("duration_s"):
                row[key + "_p50"] = float(np.median(values))
                row[key + "_p95"] = float(np.percentile(values, 95))
        for kind in ("model", "native"):
            if bounds[kind] and min(bounds[kind]) <= minute and minute + 60 <= max(bounds[kind]):
                prefixes = ["model"] if kind == "model" else ["native_" + op for k, op, _ in op_counts if k == "native"]
                for prefix in set(prefixes):
                    row.setdefault(prefix + "_finished", 0)
                    row.setdefault(prefix + "_errors", 0)
                if kind == "model":
                    row.setdefault("model_completed_output_tokens", 0)
        if "model_completed_output_tokens" in row:
            row["model_completed_output_tokens_s"] = row["model_completed_output_tokens"] / 60
        for prefix in ["model", "native_start", "native_exec", "native_shell_run"]:
            n = row.get(prefix + "_finished", 0)
            if n:
                row[prefix + "_error_pct"] = row.get(prefix + "_errors", 0) / n * 100
        rows[minute] = row
    for kind, changes in events.items():
        changes.sort()
        i = 0
        active = 0
        for minute in rows:
            # Snapshot at end of minute, including still pending operations.
            while i < len(changes) and changes[i][0] < minute + 60:
                active += changes[i][1]
                i += 1
            if bounds[kind] and min(bounds[kind]) <= minute and minute < max(bounds[kind]):
                rows[minute][kind + "_inflight"] = max(active, 0)

    args.output.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows.values() for key in row})
    with (args.output / "metrics_1min.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["timestamp_utc", "epoch"] + keys)
        writer.writeheader()
        for t, row in rows.items():
            writer.writerow({"timestamp_utc": dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat(), "epoch": t, **row})
    times = [dt.datetime.fromtimestamp(t, dt.timezone.utc) for t in rows]
    def series(key):
        return np.array([row.get(key, np.nan) for row in rows.values()])
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25})
    def panel(ax, items, ylabel):
        found = False
        for key, label in items:
            values = series(key)
            if np.isfinite(values).any():
                ax.plot(times, values, label=label, linewidth=1.4)
                found = True
        if found:
            ax.legend(loc="upper left", fontsize=8)
        else:
            ax.text(.5, .5, "Not captured in this window", transform=ax.transAxes, ha="center", color="darkred")
        ax.set_ylabel(ylabel)
    def finish(fig, axes, filename, subtitle):
        for ax in np.asarray(axes).flat:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=dt.timezone.utc))
        np.asarray(axes).flat[-1].set_xlabel("UTC on " + times[0].strftime("%Y-%m-%d"))
        fig.suptitle(args.title + "\n" + subtitle)
        fig.tight_layout(rect=[0, 0, 1, .95])
        for ext in ("png", "pdf"):
            fig.savefig(args.output / (filename + "." + ext), dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(5, 1, figsize=(13, 13), sharex=True)
    panel(axes[0], [(r + "_gpu_busy_pct", r.title()) for r in ("trainer", "rollout")], "GPU busy (%)")
    panel(axes[1], [(r + "_sm_active_pct", r.title() + " SM") for r in ("trainer", "rollout")] + [(r + "_tensor_active_pct", r.title() + " tensor") for r in ("trainer", "rollout")], "DCGM activity (%)")
    panel(axes[2], [("model_completed_output_tokens_s", "Completed model output tokens/s")], "Output tokens/s")
    panel(axes[3], [("model_inflight", "Model requests"), ("sandbox_vm_count", "Firecracker processes")], "Count")
    panel(axes[4], [("model_duration_s_p50", "Model p50"), ("model_duration_s_p95", "Model p95"), ("native_start_duration_s_p95", "Sandbox start p95"), ("native_shell_run_duration_s_p95", "Shell run p95")], "Duration (s)")
    finish(fig, axes, "01_pipeline", "One-minute bins; token counts attributed to response completion, not instantaneous decode throughput")

    fig, axes = plt.subplots(8, 1, figsize=(13, 19), sharex=True)
    panel(axes[0], [("sandbox_cpu_busy_pct", "CPU busy"), ("sandbox_cpu_guest_pct", "Guest VM execution"), ("sandbox_cpu_kernel_pct", "Host kernel / IRQ"), ("sandbox_cpu_iowait_pct", "I/O wait")], "CPU (%)")
    panel(axes[1], [("sandbox_load1", "Load average (1 min)"), ("sandbox_vm_count", "Firecracker processes")], "Count")
    panel(axes[2], [("sandbox_ram_available_gib", "Available RAM"), ("sandbox_swap_used_gib", "Swap used")], "GiB")
    panel(axes[3], [("sandbox_" + r + "_psi_pct", r.upper() + " some avg10") for r in ("cpu", "memory", "io")], "Stall time (%)")
    panel(axes[4], [("sandbox_disk_read_mib_s", "Read"), ("sandbox_disk_write_mib_s", "Write")], "Disk MiB/s")
    panel(axes[5], [("sandbox_network_rx_mib_s", "Physical NIC receive"), ("sandbox_network_tx_mib_s", "Physical NIC transmit")], "Network MiB/s")
    panel(axes[6], [("sandbox_disk_free_gib", "Sandbox filesystem free")], "Disk free (GiB)")
    panel(axes[7], [("native_start_error_pct", "Startup errors"), ("native_exec_error_pct", "Exec errors"), ("native_shell_run_error_pct", "Shell errors")], "Error (%)")
    finish(fig, axes, "02_sandbox", "Host resources; errors are exceptions, not command exit codes; nested exec and shell operations shown separately")

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    panel(axes[0], [("rollout_vllm_active_requests", "Running / endpoint"), ("rollout_vllm_waiting_requests", "Waiting / endpoint")], "Mean requests")
    panel(axes[1], [("rollout_vllm_kv_cache_pct", "KV cache / endpoint")], "KV cache (%)")
    panel(axes[2], [("rollout_vllm_generation_tokens_s", "Decode tokens/s / endpoint")], "Decode tokens/s")
    panel(axes[3], [("rollout_vllm_prompt_tokens_s", "Prompt tokens/s / endpoint")], "Prompt tokens/s")
    if bounds["vllm"]:
        if max(bounds["vllm"]) - min(bounds["vllm"]) < 600:
            axes[-1].xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
        axes[-1].set_xlim(dt.datetime.fromtimestamp(min(bounds["vllm"]), dt.timezone.utc), dt.datetime.fromtimestamp(max(bounds["vllm"]), dt.timezone.utc))
    finish(fig, axes, "03_vllm", "Mean of captured rollout endpoint samples; missing scrapes remain gaps")

    if any("refill_" in key for key in keys):
        fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
        panel(axes[0], [("refill_inflight_prompts", "Pending/running groups"), ("refill_target_prompts", "In-flight target"), ("refill_ready_prompts", "Terminal groups"), ("refill_ready_threshold", "Ready pause threshold")], "Prompt groups")
        panel(axes[1], [("rollout_vllm_active_requests", "Running / endpoint"), ("rollout_vllm_waiting_requests", "Waiting / endpoint")], "Mean requests")
        panel(axes[2], [("rollout_sm_active_pct", "Rollout SM active"), ("sandbox_cpu_busy_pct", "Sandbox CPU busy"), ("sandbox_cpu_psi_pct", "Sandbox CPU pressure")], "Activity / stall (%)")
        panel(axes[3], [("model_completed_output_tokens_s", "Delivered output tokens/s")], "Output tokens/s")
        finish(fig, axes, "08_continuous_refill", "Groups stay in flight across backward; terminal backlog can intentionally pause new admission")

    if counts["cpu_attribution"]:
        fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
        panel(axes[0], [("attribution_vm_pct", "Firecracker total"), ("attribution_vm_guest_pct", "Guest execution (subset)"), ("attribution_host_busy_pct", "Host total busy")], "Host CPU capacity (%)")
        panel(axes[1], [("attribution_registry_pct", "Registry / download"), ("attribution_image_prepare_pct", "Image preparation"), ("attribution_vm_storage_pct", "VM storage daemon"), ("attribution_agentenv_pct", "AgentENV server")], "Host CPU capacity (%)")
        panel(axes[2], [("attribution_other_processes_pct", "Other processes"), ("attribution_kernel_threads_pct", "Kernel threads"), ("attribution_unattributed_pct", "Unattributed busy time")], "Host CPU capacity (%)")
        axes[0].set_ylim(0, 100)
        axes[1].set_ylim(bottom=0)
        axes[2].set_ylim(bottom=0)
        if bounds["cpu_attribution"]:
            axes[-1].set_xlim(dt.datetime.fromtimestamp(min(bounds["cpu_attribution"]), dt.timezone.utc), dt.datetime.fromtimestamp(max(bounds["cpu_attribution"]), dt.timezone.utc))
            if max(bounds["cpu_attribution"]) - min(bounds["cpu_attribution"]) < 600:
                axes[-1].xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
        finish(fig, axes, "09_cpu_attribution", "Process tick deltas; guest execution is included in Firecracker total, not added to it; new capture coverage only")

    pairs = [
        ("sandbox_cpu_busy_pct", "native_start_duration_s_p95"),
        ("sandbox_memory_psi_pct", "native_start_duration_s_p95"),
        ("sandbox_io_psi_pct", "native_start_duration_s_p95"),
        ("sandbox_vm_count", "native_start_error_pct"),
        ("sandbox_vm_count", "model_completed_output_tokens_s"),
        ("rollout_gpu_busy_pct", "model_completed_output_tokens_s"),
    ]
    correlations = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (xkey, ykey) in zip(axes.flat, pairs):
        x, y = series(xkey), series(ykey)
        valid = np.isfinite(x) & np.isfinite(y)
        n = int(valid.sum())
        corr = None
        if n >= 10 and np.std(x[valid]) > 0 and np.std(y[valid]) > 0:
            corr = float(np.corrcoef(x[valid], y[valid])[0, 1])
        correlations.append({"x": xkey, "y": ykey, "n": n, "pearson_r": corr})
        if n:
            ax.scatter(x[valid], y[valid], c=np.arange(len(x))[valid], cmap="viridis", s=16, alpha=.7)
        ax.set_xlabel(xkey.replace("sandbox_", "").replace("_", " "))
        ax.set_ylabel(ykey.replace("_", " "))
        ax.set_title("n=" + str(n) + (", r=%.2f" % corr if corr is not None else "; insufficient variation/data"))
    fig.suptitle("Same-minute associations; color runs early (purple) to late (yellow)\nCorrelation does not establish causation; startup and workload changes confound these comparisons")
    fig.tight_layout(rect=[0, 0, 1, .91])
    for ext in ("png", "pdf"):
        fig.savefig(args.output / ("04_associations." + ext), dpi=160)
    plt.close(fig)

    training_steps = []
    if args.training_log:
        for line in args.training_log.open(errors="replace"):
            if re.search(r"\bstep:\d+", line):
                values = {key: float(value) for key, value in re.findall(r"([\w./-]+):(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", line)}
                if "step" in values:
                    training_steps.append(values)
        # Requeued logs can repeat step numbers: keep the last report per step.
        training_steps = list({int(row["step"]): row for row in training_steps}.values())
        training_steps.sort(key=lambda row: row["step"])
        if training_steps:
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(training_steps))
            bottom = np.zeros(len(x))
            for key, label in [("gen", "Wait for rollout batch"), ("old_log_prob", "Log probabilities"), ("update_actor", "Actor update"), ("update_weights", "Weight sync")]:
                values = np.array([row.get("timing_s/" + key, 0) / 60 for row in training_steps])
                ax.bar(x, values, bottom=bottom, label=label)
                bottom += values
            total = np.array([row.get("timing_s/step", np.nan) / 60 for row in training_steps])
            other = np.maximum(0, total - bottom)
            ax.bar(x, other, bottom=bottom, label="Other measured step time")
            for position, value in zip(x, total):
                ax.text(position, value + 1, "%.1f min" % value, ha="center")
            ax.set_xticks(x, ["Step " + str(int(row["step"])) for row in training_steps])
            ax.set_ylabel("Minutes")
            ax.set_title("Completed training step timing\nRollout collection continues concurrently; gen is trainer wait time")
            ax.legend(loc="upper right")
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(args.output / ("05_training_steps." + ext), dpi=160)
            plt.close(fig)
            (args.output / "training_steps.json").write_text(json.dumps(training_steps, indent=2) + "\n")

    summary = {
        "start_utc": times[0].isoformat(), "snapshot_end_utc": dt.datetime.fromtimestamp(end, dt.timezone.utc).isoformat(),
        "record_counts": dict(counts), "trainer_nodes": sorted(trainers),
        "completed_training_steps": len(training_steps),
        "inputs": [str(path) for path in args.inputs], "training_log": str(args.training_log) if args.training_log else None,
        "incomplete_at_snapshot": {kind: len(values) for kind, values in pending.items()},
        "operation_counts": {"/".join(k): v for k, v in op_counts.items()},
        "error_classes": {"/".join(k): v for k, v in errors.items()},
        "dcgm_capture_errors": sorted(dcgm_errors), "correlations": correlations,
        "metrics": {key: {"min": float(np.nanmin(series(key))), "median": float(np.nanmedian(series(key))), "p95": float(np.nanpercentile(series(key), 95)), "max": float(np.nanmax(series(key)))} for key in keys if np.isfinite(series(key)).any()},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Resource correlation snapshot", "", "Window: " + summary["start_utc"] + " to " + summary["snapshot_end_utc"], "", "## Coverage", "", "Record counts: `" + json.dumps(dict(counts)) + "`.", "", "Incomplete operations at cutoff (right-censored): `" + json.dumps(summary["incomplete_at_snapshot"]) + "`.", ""]
    if not counts["vllm"]:
        lines += ["**No vLLM scrapes captured.** Active batches, KV cache, and engine token rates cannot be inferred from GPU busy or model traces.", ""]
    if not any("sm_active" in key for key in keys):
        lines += ["**No usable historical DCGM samples.** " + "; ".join(sorted(dcgm_errors)), ""]
    lines += ["## Interpretation limits", "", "- One-minute bins mix startup, warmup, rollout, training, and cleanup phases; correlations are exploratory, not causal.", "- Model tokens are assigned to completion time. This is delivered throughput, not instantaneous engine throughput.", "- GPU busy may include communication. DCGM SM/tensor activity is needed to distinguish it from computation.", "- Memory capacity usage does not measure memory bandwidth saturation.", "- Native operation timings include client/network/service/guest work; exec and shell_run may overlap.", "- Failed operations and pending operations are not included in successful-operation latency percentiles.", "- Export snapshots use source wall clocks without measured cross-host clock skew. Final minutes can be partial.", "", "## Same-minute correlations", "", "| Resource | Outcome | Paired minutes | Pearson r |", "|---|---|---:|---:|"]
    for item in correlations:
        value = "n/a" if item["pearson_r"] is None else "%.3f" % item["pearson_r"]
        lines.append("| %s | %s | %s | %s |" % (item["x"], item["y"], item["n"], value))
    if any("refill_" in key for key in keys):
        lines += ["", "## Continuous refill signals", "", "[Feeder / GPU / sandbox timeline](08_continuous_refill.png).", "", "- Ready backlog at the threshold intentionally pauses admission; compare this with trainer consumption and GPU idle time.", "- In-flight groups at the target with small vLLM batches mean episodes are spending time outside engine execution; compare stage, CPU, startup and I/O figures.", "- A vLLM waiting queue plus high SM activity is a serving-capacity signal; low KV occupancy alone does not establish compute headroom.", "- These signals are descriptive and do not establish a single causal bottleneck without comparable workload windows."]
    if counts["cpu_attribution"]:
        lines += ["", "## CPU attribution", "", "[VM versus image-preparation CPU](09_cpu_attribution.png).", "", "Percentages use total host CPU capacity. Guest execution is a subset of Firecracker CPU and includes boot, installs, tools and verification. It does not identify guest processes. Short-lived processes that exit between samples, unreadable processes and IRQ accounting can leave unattributed CPU. Historical ps %CPU is a lifetime average and is not used for interval attribution. New attribution samples cannot reconstruct historical per-process CPU."]
    (args.output / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(args.output), "records": dict(counts), "figures": 4 + int(bool(training_steps)) + int(any("refill_" in key for key in keys)) + int(bool(counts["cpu_attribution"]))}))


if __name__ == "__main__":
    main()
