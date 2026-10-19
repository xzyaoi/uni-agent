#!/usr/bin/env python3
"""Persist vLLM Prometheus scrapes or local GPU samples without extra services."""

import argparse
import concurrent.futures
import csv
import io
import json
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

GPU_FIELDS = (
    "index",
    "uuid",
    "name",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "temperature.gpu",
)


def timestamp():
    return time.time()


class Records:
    """Append durable, immediately readable records; no compression footer needed."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("ab", buffering=0)

    def append(self, record):
        data = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        self.stream.write(data)
        os.fsync(self.stream.fileno())

    def close(self):
        self.stream.close()


def discover(path, max_bytes, max_targets, discovery_format="file-sd"):
    with Path(path).open("rb") as stream:
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("discovery file exceeds byte limit")
    if discovery_format == "prometheus-yaml":
        import yaml  # Already installed for verl's generated Prometheus configuration.

        try:
            config = yaml.safe_load(data)
        except yaml.YAMLError as exc:
            raise ValueError("invalid Prometheus YAML: " + str(exc)) from exc
        if not isinstance(config, dict) or not isinstance(config.get("scrape_configs"), list):
            raise ValueError("Prometheus config must contain scrape_configs")
        groups = []
        for scrape_config in config["scrape_configs"]:
            if isinstance(scrape_config, dict) and scrape_config.get("job_name") == "rollout":
                static_groups = scrape_config.get("static_configs", [])
                if not isinstance(static_groups, list):
                    raise ValueError("rollout static_configs must be a list")
                for group in static_groups:
                    if not isinstance(group, dict) or not isinstance(group.get("labels", {}), dict):
                        raise ValueError("invalid rollout target group")
                    groups.append({**group, "labels": {"job": "rollout", **group.get("labels", {})}})
        if not groups:
            raise ValueError("Prometheus rollout targets are not available yet")
    else:
        groups = json.loads(data)
    if not isinstance(groups, list):
        raise ValueError("file_sd root must be a list")
    endpoints = []
    seen = set()
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("targets"), list):
            raise ValueError("file_sd group must contain a targets list")
        labels = group.get("labels", {})
        if not isinstance(labels, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in labels.items()
        ):
            raise ValueError("discovery labels must map strings to strings")
        for target in group["targets"]:
            if not isinstance(target, str):
                raise ValueError("discovery targets must be strings")
            url = target if "://" in target else "http://" + target
            parsed = urllib.parse.urlsplit(url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError("target must be an HTTP endpoint without credentials")
            if parsed.query or parsed.fragment:
                raise ValueError("target must not contain query parameters or fragments")
            url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/metrics", "", ""))
            identity = (url, tuple(sorted(labels.items())))
            if identity not in seen:
                endpoints.append(
                    {
                        "source_url": url,
                        "address": parsed.netloc,
                        "labels": {"replica_index": str(len(endpoints)), **labels},
                        "replica_endpoint": parsed.netloc,
                    }
                )
                seen.add(identity)
            if len(endpoints) > max_targets:
                raise ValueError("discovery exceeds target limit")
    return endpoints


def scrape(endpoint, timeout, max_bytes):
    record = {"kind": "prometheus", "timestamp": timestamp(), **endpoint}
    started = time.monotonic()
    try:
        # Node-to-node metrics never need the submitting user's proxy or credentials.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request(endpoint["source_url"], headers={"Accept": "text/plain"})
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(max_bytes + 1)
            record["http_status"] = response.status
        if len(raw) > max_bytes:
            raise ValueError("metrics response exceeds byte limit")
        record.update(ok=True, prometheus=raw.decode("utf-8", errors="replace"))
    except (OSError, ValueError, urllib.error.URLError) as exc:
        record.update(ok=False, error=str(exc)[:500])
        if isinstance(exc, urllib.error.HTTPError):
            record["http_status"] = exc.code
            exc.close()
    record.update(finished_at=timestamp(), duration_seconds=round(time.monotonic() - started, 6))
    return record


def collect_prometheus(args, records, context):
    try:
        endpoints = discover(args.discovery_file, args.max_bytes, args.max_targets, args.discovery_format)
    except (OSError, ValueError) as exc:
        records.append({**context, "kind": "discovery", "timestamp": timestamp(), "ok": False, "error": str(exc)[:500]})
        return
    if not endpoints:
        records.append({**context, "kind": "discovery", "timestamp": timestamp(), "ok": True, "targets": 0})
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(endpoints))) as pool:
        futures = [pool.submit(scrape, endpoint, args.timeout_seconds, args.max_bytes) for endpoint in endpoints]
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            record["labels"] = {**record["labels"], "job_id": context["job_id"]}
            records.append({**context, **record})


def parse_gpu_rows(text):
    rows = []
    for values in csv.reader(io.StringIO(text), skipinitialspace=True):
        if not values:
            continue
        if len(values) != len(GPU_FIELDS):
            raise ValueError("unexpected nvidia-smi CSV field count")
        rows.append(dict(zip(GPU_FIELDS, (value.strip() for value in values), strict=True)))
    if not rows:
        raise ValueError("nvidia-smi returned no GPU rows")
    return rows


def collect_gpu(args, records, context, writer, csv_stream):
    sampled_at = timestamp()
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(GPU_FIELDS), "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds,
            check=True,
        )
        if len(result.stdout.encode()) > args.max_bytes:
            raise ValueError("nvidia-smi output exceeds byte limit")
        rows = parse_gpu_rows(result.stdout)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        records.append({**context, "kind": "gpu_error", "timestamp": sampled_at, "ok": False, "error": str(exc)[:500]})
        return
    for row in rows:
        writer.writerow({**context, "timestamp": sampled_at, **row})
    csv_stream.flush()
    os.fsync(csv_stream.fileno())


def positive(value):
    result = float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("prometheus", "gpu"))
    parser.add_argument("--discovery-file", type=Path)
    parser.add_argument("--discovery-format", choices=("file-sd", "prometheus-yaml"), default="file-sd")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--interval-seconds", type=positive, default=5)
    parser.add_argument("--timeout-seconds", type=positive, default=3)
    parser.add_argument("--max-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--max-targets", type=int, default=128)
    parser.add_argument("--hostname", default=socket.gethostname())
    parser.add_argument("--job-id", default=os.environ.get("SLURM_JOB_ID", "unknown"))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "prometheus" and args.discovery_file is None:
        parser.error("--discovery-file is required for prometheus mode")
    if args.max_bytes <= 0 or args.max_targets <= 0:
        parser.error("byte and target limits must be positive")
    if any("/" in part or part in ("", ".", "..") for part in (args.hostname, args.job_id)):
        parser.error("hostname and job ID must be nonempty file name components")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / (args.hostname + ("-vllm" if args.mode == "prometheus" else "-gpu"))
    context = {"job_id": args.job_id, "hostname": args.hostname, "node": args.hostname}
    stop = threading.Event()
    old_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, lambda _signum, _frame: stop.set())
    records = Records(str(stem) + ".jsonl")
    csv_stream = None
    try:
        records.append(
            {
                **context,
                "kind": "collector_started",
                "timestamp": timestamp(),
                "mode": args.mode,
                "interval_seconds": args.interval_seconds,
            }
        )
        if args.mode == "gpu":
            csv_path = Path(str(stem) + ".csv")
            needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
            csv_stream = csv_path.open("a", newline="")
            writer = csv.DictWriter(csv_stream, fieldnames=("job_id", "hostname", "node", "timestamp", *GPU_FIELDS))
            if needs_header:
                writer.writeheader()
                csv_stream.flush()
        while not stop.is_set():
            started = time.monotonic()
            if args.mode == "prometheus":
                collect_prometheus(args, records, context)
            else:
                collect_gpu(args, records, context, writer, csv_stream)
            if args.once:
                break
            stop.wait(max(0, args.interval_seconds - (time.monotonic() - started)))
        records.append({**context, "kind": "collector_stopped", "timestamp": timestamp()})
    finally:
        if csv_stream is not None:
            csv_stream.close()
        records.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
