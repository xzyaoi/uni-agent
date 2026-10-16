#!/usr/bin/env python3
"""Export a bounded, body-free snapshot from existing collectors and traces."""
import argparse
import datetime as dt
import re
import gzip
import json
import time
from pathlib import Path


def records(path):
    with path.open(errors="replace") as stream:
        for line in stream:
            try:
                yield json.loads(line)
            except (ValueError, TypeError):
                continue  # A live writer may leave an incomplete final line.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--episode-logs", type=Path, help="Framework log root containing step_*/session-*/{framework,task}.log")
    parser.add_argument("--log-utc-offset-hours", type=float, help="Explicit UTC offset of session log wall times")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, default=time.time())
    args = parser.parse_args()
    if args.episode_logs is not None and args.log_utc_offset_hours is None:
        parser.error("--episode-logs requires --log-utc-offset-hours")
    episodes = []
    trials = {}
    if args.episode_logs is not None:
        timezone = dt.timezone(dt.timedelta(hours=args.log_utc_offset_hours))
        for path in sorted(args.episode_logs.glob("step_*/*/framework.log")):
            match = re.fullmatch(r"step_(\d+)", path.parent.parent.name)
            if not match:
                continue
            episode = {"kind": "episode", "global_steps": int(match[1]), "session_id": path.parent.name}
            for line in path.open(errors="replace"):
                try:
                    stamp = dt.datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone).timestamp()
                except ValueError:
                    continue
                if stamp > args.cutoff:
                    continue
                if " start: runner=" in line:
                    episode["timestamp"] = stamp
                if re.search(r"session .*: \d+ trajectory", line) or "failed (runner=" in line:
                    episode["end_timestamp"] = stamp
                    episode["status"] = "error" if "failed (runner=" in line else "end"
            task_path = path.with_name("task.log")
            if task_path.exists():
                for line in task_path.open(errors="replace"):
                    trial = re.search(r"\btrial=([a-f0-9]+)", line)
                    if trial:
                        episode["trial_id"] = trial[1]
                        trials[trial[1]] = episode
                        break
            if "timestamp" in episode:
                episodes.append(episode)
    counts = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt") as out:
        def emit(kind, record):
            stamp = record.get("timestamp")
            if not isinstance(stamp, (float, int)) or stamp > args.cutoff:
                return
            record["kind"] = kind
            out.write(json.dumps(record) + "\n")
            counts[kind] = counts.get(kind, 0) + 1

        for episode in episodes:
            emit("episode", episode)

        for path in sorted(args.telemetry.glob("*.jsonl")):
            if path.name.endswith("-cpu-attribution.jsonl"):
                kind = "cpu_attribution"
            elif any(path.name.endswith("-" + k + ".jsonl") for k in ("host", "dcgm", "vllm")):
                kind = path.stem.rsplit("-", 1)[-1]
            else:
                continue
            for record in records(path):
                if kind == "host":
                    processes = record.pop("processes", {}).get("stdout", "")
                    vm_rows = [line.split() for line in processes.splitlines() if "firecracker" in line]
                    if processes:
                        record["vm_count"] = len(vm_rows)
                        record["vm_rss_kib"] = sum(int(r[3]) for r in vm_rows if len(r) >= 4 and r[3].isdigit())
                    record.pop("vm_arguments", None)
                    record.pop("service_log_deltas", None)
                    record["cpu_stat"] = (record.get("cpu_stat") or "").split("\n", 1)[0]
                emit(kind, record)
        for kind, directory in (("model", args.models), ("native", args.native)):
            if directory is None:
                continue
            for path in sorted(directory.glob("*.jsonl")):
                for record in records(path):
                    slim = {key: record[key] for key in (
                        "timestamp", "phase", "operation", "operation_id", "duration", "sandbox_id",
                        "model_client_id", "native_trace_file", "trace_file", "pid", "sandbox_trace",
                        "global_steps", "event", "batch_id", "partition_id", "num_prompts", "num_sessions",
                        "session_id", "uid", "sample_index", "session_index", "error_class", "num_outputs",
                        "num_input_prompts", "num_success_sessions", "num_failed_sessions", "num_success_outputs",
                        "num_unfinished_episodes", "num_failed_uids", "submitted_now", "inflight_prompts",
                        "ready_prompts", "failed_prompts", "submitted_prompts", "target_prompts", "ready_threshold", "backlog_paused", "active_groups", "terminal_groups", "group_limit"
                    ) if key in record}
                    slim["trace_file"] = path.name
                    logs_dir = record.get("agent_logs_dir", record.get("logs_dir"))
                    if logs_dir:
                        trial_id = Path(logs_dir).parent.name
                        if trial_id in trials:
                            slim["trial_id"] = trial_id
                            slim["global_steps"] = trials[trial_id]["global_steps"]
                            slim["session_id"] = trials[trial_id]["session_id"]
                    # Preserve error class, not arbitrary exception text or content.
                    if "error" in record:
                        slim["error"] = str(record["error"]).split(":", 1)[0]
                    if kind == "model" and isinstance(record.get("response"), dict):
                        slim["usage"] = record["response"].get("usage")
                    emit("pipeline" if record.get("kind") == "pipeline" else kind, slim)
        out.write(json.dumps({"kind": "snapshot", "timestamp": args.cutoff, "counts": counts}) + "\n")
    print(json.dumps({"output": str(args.output), "cutoff": args.cutoff, "counts": counts}))


if __name__ == "__main__":
    main()
