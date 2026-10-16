#!/usr/bin/env python3
"""Periodically save metric-only snapshots and regenerate figures for a Slurm run."""
import argparse
import json
import datetime as dt
from zoneinfo import ZoneInfo
import os
import shlex
import re
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--nodes", type=int, default=10)
    parser.add_argument("--rollout-only", action="store_true")
    parser.add_argument("--run", required=True)
    parser.add_argument("--cloudlab-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cluster", default="clariden")
    parser.add_argument("--sandbox", default="xzyao@clnode286.clemson.cloudlab.us")
    parser.add_argument("--sandbox-identity", type=Path)
    parser.add_argument("--sandbox-disk", default="/sandbox")
    parser.add_argument("--manage-sandbox-collector", action="store_true")
    parser.add_argument("--release-cscs-job", help="Release this held Clariden job after this monitored job terminates")
    parser.add_argument("--release-job", help="Release this held job after the monitored job terminates")
    parser.add_argument("--release-cluster", default="clariden")
    parser.add_argument("--host-metrics", default="telemetry/host-supplement")
    parser.add_argument("--interval", type=int, default=600)
    parser.add_argument("--duration", type=int, default=30000)
    args = parser.parse_args()
    release_job = args.release_job or args.release_cscs_job
    if not args.job_id.isdigit() or args.interval <= 0 or args.duration <= 0:
        parser.error("job-id must be numeric; interval and duration must be positive")
    output = args.output.resolve()
    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    script_root = Path(__file__).resolve().parent
    ssh_base = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    def run(argv, timeout=600):
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        if result.returncode:
            raise RuntimeError("Command failed: " + result.stdout[-1600:])
        return result.stdout
    def ssh(host, argv, timeout=600):
        identity = ["-i", str(args.sandbox_identity.expanduser()), "-o", "IdentitiesOnly=yes"] if host == args.sandbox and args.sandbox_identity else []
        return run(ssh_base + identity + [host, shlex.join(argv)], timeout)
    def copy(host, remote, local):
        # These are operator-supplied absolute paths; modern scp uses SFTP.
        identity = ["-i", str(args.sandbox_identity.expanduser()), "-o", "IdentitiesOnly=yes"] if host == args.sandbox and args.sandbox_identity else []
        run(["scp", "-q"] + identity + [host + ":" + remote, str(local)])
    copy(args.cluster, args.run + "/launch.env", inputs / "launch.env")
    config = {}
    for line in (inputs / "launch.env").read_text().splitlines():
        key, _, value = line.removeprefix("export ").partition("=")
        fields = shlex.split(value)
        if fields:
            config[key] = fields[0]
    native = config["AGENTENV_TRACE_DIR"]
    framework_logs = config.get("AGENT_LOG_DIR") or ("/iopsstor/scratch/cscs/xyao/logs/" + config["PROJECT_NAME"] + "/" + config["EXP_NAME"])
    trainers = []
    sandbox_started = False
    last_snapshot = 0
    if (output / "summary.json").exists():
        try:
            last_snapshot = dt.datetime.fromisoformat(json.loads((output / "summary.json").read_text())["snapshot_end_utc"]).timestamp()
        except (ValueError, KeyError, OSError):
            pass
    started = time.time()
    while time.time() - started < args.duration:
        try:
            try:
                queue_line = ssh(args.cluster, ["squeue", "-h", "-j", args.job_id, "-o", "%T|%r"], 40).strip()
                state, _, reason = queue_line.partition("|")
            except RuntimeError as exc:
                if "Invalid job id" not in str(exc):
                    raise
                state = ""
                reason = ""
            finished = not state
            if finished:
                state = ssh(args.cluster, ["sacct", "-X", "-n", "-P", "-j", args.job_id, "--format=State"], 40).strip().strip("|")
            record = {"timestamp": time.time(), "scheduler_state": state, "scheduler_reason": reason}
            print(json.dumps(record), flush=True)
            status_lines = [("# Rollout sweep job " if args.rollout_only else "# Training job ") + args.job_id, "", "Scheduler state: **" + state + "** (" + reason + ").",
                            "", "Last checked: " + dt.datetime.now(dt.timezone.utc).isoformat(), "",
                            "Continuous refill: " + config.get("ROLLOUT_TARGET_PROMPTS", "128") + " target prompt groups; completed-backlog threshold: " + config.get("ROLLOUT_READY_PROMPTS", "64") + " (0 disables admission backpressure); submission chunks: " + config.get("ROLLOUT_REFILL_PROMPTS", "16") + ".",
                            "", "Topology: " + str((args.nodes - int(config["NNODES_ROLLOUT"])) * 4) + " trainer GPUs + " + str(int(config["NNODES_ROLLOUT"]) * 4) + " standalone rollout GPUs; " + config.get("TRAINING_STEPS", "5") + " steps, no validation; time limit " + config.get("TIME_LIMIT", "04:00:00") + ".",
                            "", "Run: `" + args.run + "`", "", "Raw model/native traces remain on the cluster; sandbox telemetry: `" + args.cloudlab_dir + "`.",
                            "", "Metric-only snapshots and figures refresh every " + str(args.interval) + " seconds while running and at termination."]
            if args.rollout_only:
                status_lines[6] = "Fixed workload: 128 groups x 8 episodes; active-group limit " + config.get("GROUP_CONCURRENCY", "unknown") + ". No trainer or optimizer."
            if (output / "summary.json").exists():
                summary = json.loads((output / "summary.json").read_text())
                status_lines += ["", "Latest metric snapshot: " + summary["snapshot_end_utc"],
                                 "", "Reported completed training steps in that snapshot: " + str(summary.get("completed_training_steps", 0)),
                                 "", "[Coverage and figures](README.md) · [One-minute metrics](metrics_1min.csv)"]
                if (output / "08_continuous_refill.png").exists():
                    status_lines += ["", "[Continuous refill / GPU / sandbox timeline](08_continuous_refill.png)"]
            else:
                status_lines += ["", "Performance figures await sufficient allocated-node telemetry; no bottleneck conclusion yet."]
            (output / "STATUS.md").write_text("\n".join(status_lines) + "\n")
            with (output / "monitor-history.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            if state.startswith("RUNNING") and args.manage_sandbox_collector and not sandbox_started:
                directory = shlex.quote(args.cloudlab_dir)
                code = "dir=" + directory + '; pidfile="$dir/collector.pid"; if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then echo "Sandbox collector already running"; else umask 077; nohup python3 "$dir/collect.py" --role sandbox --output "$dir" --disk ' + shlex.quote(args.sandbox_disk) + ' --duration 86400 --sudo-journal > "$dir/collector.log" 2>&1 < /dev/null & echo $! > "$pidfile"; fi'
                code += '; if [ -f "$dir/collect_cpu.py" ]; then cpu_pidfile="$dir/cpu-collector.pid"; if [ -f "$cpu_pidfile" ] && kill -0 "$(cat "$cpu_pidfile")" 2>/dev/null; then true; else nohup python3 "$dir/collect_cpu.py" --output "$dir" --interval 5 --duration 86400 > "$dir/cpu-collector.log" 2>&1 < /dev/null & echo $! > "$cpu_pidfile"; fi; fi'
                ssh(args.sandbox, ["bash", "-c", code])
                sandbox_started = True
            if state.startswith("RUNNING") and not trainers and not args.rollout_only:
                # Discover actual Ray trainer placement; it need not follow Slurm node order.
                code = 'if ps -eo args | grep -q "^ray::WorkerDict"; then hostname; fi; true'
                text = ssh(args.cluster, ["timeout", "30", "srun", "--jobid=" + args.job_id,
                           "--overlap", "--kill-on-bad-exit=0", "-N", str(args.nodes), "-n", str(args.nodes),
                           "--ntasks-per-node=1", "-c", "1", "bash", "-c", code], 45)
                trainers = sorted({line.strip().split(".")[0] for line in text.splitlines()
                                   if re.fullmatch(r"(?:nid\d+|jpbo-\d+-\d+)(?:\.[\w.-]+)?", line.strip())})
                expected_trainers = args.nodes - int(config["NNODES_ROLLOUT"])
                if len(trainers) != expected_trainers:
                    trainers = []
                if trainers:
                    (output / "trainer_nodes.json").write_text(json.dumps(trainers) + "\n")
            due = time.time() - last_snapshot >= args.interval
            if (state.startswith("RUNNING") and due) or finished:
                cutoff = str(time.time())
                ssh(args.cluster, ["python3", args.run + "/analysis-tools/export_metrics.py",
                    "--telemetry", args.run + "/" + args.host_metrics.strip("/"),
                    "--models", args.run + "/telemetry/model-calls", "--native", native,
                    "--episode-logs", framework_logs, "--log-utc-offset-hours", "2",
                    "--cutoff", cutoff, "--output", args.run + "/analysis-tools/demand.jsonl.gz"])
                ssh(args.sandbox, ["python3", args.cloudlab_dir + "/export_metrics.py",
                    "--telemetry", args.cloudlab_dir, "--cutoff", cutoff,
                    "--output", args.cloudlab_dir + "/cloudlab.jsonl.gz"])
                copy(args.cluster, args.run + "/analysis-tools/demand.jsonl.gz", inputs / "demand.jsonl.gz")
                copy(args.sandbox, args.cloudlab_dir + "/cloudlab.jsonl.gz", inputs / "cloudlab.jsonl.gz")
                copy(args.cluster, args.run + ("/rollout.out" if args.rollout_only else "/training.out"), inputs / "training.out")
                if trainers or args.rollout_only:
                    last_step_text = ssh(args.cluster, ["env", "SLURM_TIME_FORMAT=%Y-%m-%dT%H:%M:%S", "sacct", "-X", "-n", "-P", "-j", args.job_id, "--format=Start"], 40).strip().splitlines()[0].strip("|")
                    # Cluster accounting/log timestamps are Europe/Zurich; date-specific offset provided explicitly.
                    start_iso = dt.datetime.fromisoformat(last_step_text).replace(tzinfo=ZoneInfo("Europe/Zurich")).isoformat()
                    run([sys.executable, str(script_root / "plot_resources.py"), str(inputs / "demand.jsonl.gz"),
                         str(inputs / "cloudlab.jsonl.gz"), "--output", str(output), "--trainer-nodes", ",".join(trainers),
                         "--start", start_iso, "--training-log", str(inputs / "training.out"),
                         "--title", "Job " + args.job_id + ": bounded continuous refill"])
                    try:
                        run([sys.executable, str(script_root / "plot_rollout_demand.py"), str(inputs / "demand.jsonl.gz"),
                             "--output", str(output), "--trainer-nodes", ",".join(trainers)])
                    except RuntimeError as exc:
                        print(str(exc), flush=True)  # Episodes may not exist during initial model loading.
                    comparison_script = script_root / "compare_cpu_gpu.py"
                    if comparison_script.exists() and not args.rollout_only:
                        run([sys.executable, str(comparison_script), "--study", str(output)])
                if args.rollout_only:
                    for name in ("progress.json", "result.json", "manifest.json", "settings.json"):
                        try:
                            copy(args.cluster, args.run + "/" + name, output / name)
                        except RuntimeError:
                            pass  # Driver may still be initializing.
                last_snapshot = time.time()
                print("Snapshot saved: " + str(output), flush=True)
            if finished:
                (output / "scheduler-final.json").write_text(json.dumps(record, indent=2) + "\n")
                if release_job:
                    if not release_job.isdigit():
                        raise ValueError("release-job must be numeric")
                    try:
                        held = ssh(args.release_cluster, ["squeue", "-h", "-j", release_job, "-o", "%T|%r"], 40).strip()
                        if held.startswith("PENDING|") and "held" in held.lower():
                            ssh(args.release_cluster, ["scontrol", "release", release_job], 40)
                            print("Released " + args.release_cluster + " job " + release_job + " after monitored experiment termination", flush=True)
                    except Exception as exc:
                        print("Held-job release requires follow-up: " + str(exc), flush=True)
                return
        except Exception as exc:
            print("Monitor retry: " + str(exc), flush=True)
        time.sleep(30)
    print("Monitor duration limit reached; raw collectors retain their independent limits", flush=True)


if __name__ == "__main__":
    main()
