"""Warm selected OCI images through AgentENV's native conversion and SDK readiness path.

Run on the AgentENV host with /sandbox/bench/venv220/bin/python. This only
executes the provider's ``true`` readiness command: no task setup or verifier.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time


def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def cache_metrics(root, disk_root):
    try:
        usage = shutil.disk_usage(disk_root)
    except PermissionError:
        usage = shutil.disk_usage(disk_root.parent)
    result = {"disk_root": str(disk_root), "cache_root": str(root), "disk_total_bytes": usage.total, "disk_free_bytes": usage.free}
    proc = subprocess.run(["sudo", "-n", "du", "-s", "--block-size=1", str(root)], capture_output=True, text=True)
    if proc.returncode == 0:
        result["cache_disk_bytes"] = int(proc.stdout.split()[0])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--inspect-tools", action="store_true", help="record uv/python versions once; never run task tests")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--provider-root", type=Path, default=Path("/sandbox/bench/flash-sandbox"))
    parser.add_argument("--disk-root", type=Path, default=Path("/nvme"))
    parser.add_argument("--cache-root", type=Path, default=Path("/sandbox/aenv/image-cache"))
    args = parser.parse_args()
    if args.concurrency < 1 or args.timeout <= 0:
        parser.error("concurrency and timeout must be positive")
    key = args.api_key_file.read_text().strip() if args.api_key_file else os.getenv("AGENTENV_API_KEY", "")
    if not key:
        parser.error("provide AGENTENV_API_KEY or --api-key-file (raw key, not an env file)")
    sys.path.insert(0, str(args.provider_root))
    from meta.baselines.sbench_providers.agentenv import create_provider

    manifest = json.loads(args.manifest.read_text())
    entries = manifest["images"] if isinstance(manifest, dict) else manifest
    images = {}
    for entry in entries:
        item = {"image": entry} if isinstance(entry, str) else dict(entry)
        original = item.get("original_ref") or item.get("image") or item.get("original")
        if not original:
            raise ValueError("each image entry requires original_ref or image")
        item["original_ref"] = original
        images.setdefault(original, item)
    if not images:
        raise ValueError("image manifest is empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "progress.jsonl").exists():
        raise ValueError("use a fresh output directory to preserve prior warmup evidence")
    lock = threading.Lock()
    records = []
    tools_claimed = threading.Event()
    start = time.monotonic()
    initial_cache = cache_metrics(args.cache_root, args.disk_root)
    (args.output_dir / "status.json").write_text(json.dumps({
        "state": "warming", "total_images": len(images), "finished_images": 0,
        "ready_images": 0, "failed_images": 0, "concurrency": args.concurrency,
        "started_at": timestamp(), "initial_cache": initial_cache}, indent=2) + "\n")

    def safe_error(exc):
        return str(exc).replace(key, "[REDACTED]")[:4000]

    def emit(record):
        with lock:
            with (args.output_dir / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")

    def warm(item):
        record = dict(item, started_at=timestamp(), status="running")
        t0 = time.monotonic()
        sid = None
        provider = create_provider({"base-url": args.endpoint, "api-key": key,
            "ttl": str(int(args.timeout + 600)), "create-timeout": str(args.timeout),
            "prepare-timeout": str(args.timeout), "stop-timeout": "120"})
        try:
            pinned = item.get("resolved_ref") or item.get("pinned_ref")
            if not pinned:
                proc = subprocess.run(["sudo", "-n", "-u", "aenv", "env",
                    "REGCTL_CONFIG=/sandbox/aenv/config/regctl.json",
                    "/sandbox/aenv/deps/regctl/v0.11.5/regctl", "manifest", "digest",
                    "--platform", "linux/amd64", item["original_ref"]],
                    capture_output=True, text=True, timeout=120)
                if proc.returncode:
                    raise RuntimeError("manifest resolution failed: " + proc.stderr)
                digest = proc.stdout.strip()
                if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
                    raise RuntimeError("manifest resolution returned an invalid digest")
                repository = item["original_ref"].split("@", 1)[0]
                if ":" in repository.rsplit("/", 1)[-1]:
                    repository = repository.rsplit(":", 1)[0]
                pinned = repository + "@" + digest
            if not re.search(r"@sha256:[0-9a-f]{64}$", pinned):
                raise ValueError("resolved image must contain a sha256 manifest digest")
            record["resolved_ref"] = pinned
            record["manifest_digest"] = pinned.rsplit("@", 1)[1]
            emit(dict(record, event="resolved"))
            create_start = time.monotonic()
            # These are the installed provider.prepare() operations, kept
            # explicit so the created ID is journaled before SDK readiness.
            sid = provider._create_sandbox(pinned, timeout=args.timeout)
            record["sandbox_id"] = sid
            record["create_seconds"] = time.monotonic() - create_start
            emit(dict(record, event="created"))
            ready_start = time.monotonic()
            remaining = max(0.001, args.timeout - (ready_start - create_start))
            provider._settle_envd(sid, timeout=remaining)
            record["readiness_seconds"] = time.monotonic() - ready_start
            record["status"] = "ready"
            inspect = False
            if args.inspect_tools:
                with lock:
                    if not tools_claimed.is_set():
                        tools_claimed.set()
                        inspect = True
            if inspect:
                try:
                    result = provider._t().attached(sid).commands.run(
                        "command -v uv || true; uv --version 2>/dev/null || true; python --version 2>&1 || true; python3 --version 2>&1 || true; command -v conda || true; ls -d /opt/conda /root/miniconda3 /opt/miniconda3 2>/dev/null || true",
                        timeout=30, request_timeout=45)
                    record["tools"] = {"stdout": result.stdout, "stderr": result.stderr,
                                       "exit_code": result.exit_code}
                except Exception as exc:
                    record["tools_error"] = safe_error(exc)
        except Exception as exc:
            record.update(status="failed", error=safe_error(exc))
        finally:
            if sid is not None:
                delete_start = time.monotonic()
                try:
                    provider._destroy_sandbox(sid, timeout=120)
                    record["cleanup"] = "deleted"
                except Exception as exc:
                    record.update(status="failed", cleanup="failed", cleanup_error=safe_error(exc))
                record["delete_seconds"] = time.monotonic() - delete_start
            else:
                record["cleanup"] = "no_known_sandbox_id"
            record.update(event="finished", finished_at=timestamp(), seconds=time.monotonic() - t0)
            emit(record)
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(warm, item) for item in images.values()]
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
            report = {"updated_at": timestamp(), "total_images": len(images),
                "finished_images": len(records), "ready_images": sum(r["status"] == "ready" for r in records),
                "failed_images": sum(r["status"] != "ready" for r in records),
                "concurrency": args.concurrency, "elapsed_seconds": time.monotonic() - start,
                "initial_cache": initial_cache, "results": sorted(records, key=lambda r: r["original_ref"])}
            temp = args.output_dir / "status.json.tmp"
            temp.write_text(json.dumps(report, indent=2) + "\n")
            temp.replace(args.output_dir / "status.json")
            print(f"finished={len(records)}/{len(images)} ready={report['ready_images']} failed={report['failed_images']}", flush=True)
    report["final_cache"] = cache_metrics(args.cache_root, args.disk_root)
    report["finished_at"] = timestamp()
    (args.output_dir / "status.json").write_text(json.dumps(report, indent=2) + "\n")
    pinned_manifest = {"images": [{k: r[k] for k in ("original_ref", "resolved_ref", "manifest_digest", "status") if k in r} for r in report["results"]]}
    (args.output_dir / "pinned_manifest.json").write_text(json.dumps(pinned_manifest, indent=2) + "\n")
    return 1 if report["failed_images"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
