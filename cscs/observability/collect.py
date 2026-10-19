#!/usr/bin/env python3
"""Continuous read-only host, GPU, DCGM and vLLM telemetry (Python 3.6+)."""

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import signal
import socket
import struct
import subprocess
import threading
import time
import urllib.request
from pathlib import Path


def vllm_address(host, port):
    if host in ("*", "0.0.0.0"):
        host = "127.0.0.1"
    elif host == "::":
        host = "::1"
    return ("[" + host + "]" if ":" in host else host) + ":" + str(port)


def proc_vllm_candidates(proc_root="/proc"):
    """Read only LISTEN sockets owned by Ray vLLM HTTP actors, never arbitrary ports."""
    root = Path(proc_root)
    inodes = set()
    info = {"method": "procfs", "peer_process_count": 0, "unreadable_proc_entries": 0}
    try:
        processes = list(root.iterdir())
    except OSError:
        info["unreadable_proc_entries"] += 1
        return [], info
    for process in processes:
        if not process.name.isdigit():
            continue
        tags = []
        for filename in ("comm", "cmdline"):
            try:
                tags.append((process / filename).read_bytes().split(b"\0", 1)[0].decode(errors="replace").strip())
            except OSError:
                pass
        if not any(tag.startswith("ray::vLLMHttp") for tag in tags):
            continue
        info["peer_process_count"] += 1
        try:
            descriptors = list((process / "fd").iterdir())
        except OSError:
            info["unreadable_proc_entries"] += 1
            continue
        for descriptor in descriptors:
            try:
                match = re.fullmatch(r"socket:\[(\d+)\]", os.readlink(str(descriptor)))
                if match:
                    inodes.add(match[1])
            except OSError:
                info["unreadable_proc_entries"] += 1
    candidates = set()
    if not inodes:
        return [], info
    for filename in ("tcp", "tcp6"):
        try:
            lines = (root / "net" / filename).read_text().splitlines()[1:]
        except OSError:
            info["unreadable_proc_entries"] += 1
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A" or fields[9] not in inodes:
                continue
            try:
                address, raw_port = fields[1].split(":")
                port = int(raw_port, 16)
                if not 0 < port <= 65535:
                    continue
                if filename == "tcp":
                    host = socket.inet_ntoa(struct.pack("=I", int(address, 16)))
                else:
                    if len(address) != 32:
                        continue
                    raw = b"".join(struct.pack("=I", int(address[i:i + 8], 16)) for i in range(0, 32, 8))
                    host = socket.inet_ntop(socket.AF_INET6, raw)
                candidates.add(vllm_address(host, port))
            except (ValueError, OSError, struct.error):
                continue
    return sorted(candidates), info


def vllm_candidates(ss_result, proc_root="/proc"):
    if ss_result.get("returncode") != 0:
        return proc_vllm_candidates(proc_root)
    candidates = set()
    for line in ss_result.get("stdout", "").splitlines():
        if "ray::vLLMHttp" not in line:
            continue
        fields = line.split()
        if len(fields) < 4:
            continue
        host, separator, port = fields[3].rpartition(":")
        if separator and port.isdigit() and 0 < int(port) <= 65535:
            candidates.add(vllm_address(host.strip("[]"), int(port)))
    return sorted(candidates), {"method": "ss"}


parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--role", choices=("gpu", "sandbox"), required=True)
parser.add_argument("--service", default="aenv")
parser.add_argument("--interval", type=float, default=5)
parser.add_argument("--duration", type=float, default=86400)
parser.add_argument("--disk", default="/tmp")
parser.add_argument("--sudo-journal", action="store_true", help="Read protected service/kernel journals with sudo -n")
args = parser.parse_args()
root = Path(args.output)
root.mkdir(parents=True, exist_ok=True)
node = socket.gethostname().split(".")[0]
stopping = threading.Event()
signal.signal(signal.SIGTERM, lambda *_: stopping.set())
signal.signal(signal.SIGINT, lambda *_: stopping.set())


def read(path):
    try:
        return Path(path).read_text()
    except OSError:
        return None


def command(argv, timeout=3):
    try:
        # Host login Python is 3.6; capture_output requires 3.7.
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)  # noqa: UP022
        return {
            "returncode": result.returncode,
            "stdout": result.stdout.decode(errors="replace"),
            "stderr": result.stderr.decode(errors="replace"),
        }
    except Exception as exc:
        return {"error": str(exc)}


def dump(stream, record):
    stream.write(json.dumps({"timestamp": time.time(), "node": node, **record}) + "\n")
    stream.flush()


children = []


def stream_command(argv, filename):
    target = (root / filename).open("a")
    try:
        process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except OSError as exc:
        dump(target, {"error": str(exc), "command": argv})
        target.close()
        return
    children.append(process)

    def consume():
        try:
            for line in iter(process.stdout.readline, b""):
                dump(target, {"line": line.decode(errors="replace").rstrip()})
        finally:
            target.close()

    threading.Thread(target=consume, daemon=True).start()


if args.role == "gpu":
    stream_command([os.environ.get("DCGMI_PATH", "dcgmi"), "dmon", "-e", "1001,1002,1003,1004,1005", "-d", "1000"], node + "-dcgm.jsonl")
else:
    prefix = ["sudo", "-n"] if args.sudo_journal else []
    stream_command(
        prefix + ["journalctl", "-u", args.service, "--since", "now", "--follow", "--no-pager", "-o", "json"],
        node + "-service-journal.jsonl",
    )
    stream_command(
        prefix + ["journalctl", "-k", "--since", "now", "--follow", "--no-pager", "-o", "json"],
        node + "-kernel-journal.jsonl",
    )

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def scrape(address):
    try:
        with opener.open("http://" + address + "/metrics", timeout=2) as response:
            text = response.read().decode()
        if "vllm:" in text:
            return {"address": address, "prometheus": text}
    except Exception:
        pass
    return None


endpoints = []
service_logs = {}
if args.role == "sandbox":
    properties = command(["systemctl", "show", args.service, "-p", "StandardOutput", "-p", "StandardError"])
    for line in properties.get("stdout", "").splitlines():
        value = line.partition("=")[2]
        if value.startswith(("append:", "file:")):
            path = value.partition(":")[2]
            try:
                stat = os.stat(path)
                service_logs[path] = (stat.st_ino, stat.st_size)
            except OSError:
                service_logs[path] = (None, 0)
started = time.monotonic()
last_discovery = -100
discovery_info = {"method": "not_started", "candidate_count": 0, "endpoint_count": 0}
with (root / (node + "-host.jsonl")).open("a") as host, (root / (node + "-vllm.jsonl")).open("a") as vllm:
    try:
        while not stopping.is_set() and time.monotonic() - started < args.duration:
            tick = time.monotonic()
            record = {
                "role": args.role,
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "meminfo": read("/proc/meminfo"),
                "cpu_stat": read("/proc/stat"),
                "numa_meminfo": {p.name: read(str(p / "meminfo")) for p in Path("/sys/devices/system/node").glob("node[0-9]*")},
                "numa_allowed": read("/proc/self/status"),
                "loadavg": read("/proc/loadavg"),
                "diskstats": read("/proc/diskstats"),
                "netdev": read("/proc/net/dev"),
                "pressure": {kind: read("/proc/pressure/" + kind) for kind in ("cpu", "memory", "io")},
            }
            try:
                record["disk"] = {"path": args.disk, "bytes": list(shutil.disk_usage(args.disk))}
            except OSError as exc:
                record["disk"] = {"error": str(exc)}
            cgroup = None
            if args.role == "sandbox":
                info = command(
                    [
                        "systemctl",
                        "show",
                        args.service,
                        "-p",
                        "ControlGroup",
                        "-p",
                        "MemoryCurrent",
                        "-p",
                        "MemoryPeak",
                        "-p",
                        "CPUUsageNSec",
                        "-p",
                        "NRestarts",
                        "-p",
                        "MainPID",
                        "-p",
                        "ActiveState",
                        "-p",
                        "TasksCurrent",
                    ]
                )
                record["service"] = info
                match = re.search(r"^ControlGroup=(.+)$", info.get("stdout", ""), re.M)
                cgroup = match.group(1) if match else None
                record["processes"] = command(["ps", "-eo", "pid,ppid,comm,rss,%cpu"])
                record["vm_arguments"] = {}
                for line in record["processes"].get("stdout", "").splitlines():
                    fields = line.split()
                    if len(fields) > 2 and fields[2] == "firecracker":
                        raw = read("/proc/" + fields[0] + "/cmdline")
                        if raw:
                            record["vm_arguments"][fields[0]] = raw.split("\x00")
                record["service_log_deltas"] = {}
                for path, (inode, offset) in list(service_logs.items()):
                    try:
                        stat = os.stat(path)
                        if stat.st_ino != inode or stat.st_size < offset:
                            offset = 0
                        with open(path, "rb") as log:
                            log.seek(offset)
                            chunk = log.read(1024 * 1024)
                            service_logs[path] = (stat.st_ino, log.tell())
                        if chunk:
                            record["service_log_deltas"][path] = chunk.decode(errors="replace")
                    except OSError as exc:
                        record["service_log_deltas"][path] = {"error": str(exc)}
            else:
                record["gpus"] = command(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,uuid,utilization.gpu,utilization.memory,"
                        "memory.used,memory.total,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ]
                )
                record["gpu_processes"] = command(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,process_name,used_gpu_memory",
                        "--format=csv,noheader,nounits",
                    ]
                )
                for line in (read("/proc/self/cgroup") or "").splitlines():
                    if line.startswith("0::"):
                        cgroup = line[3:]
                if tick - last_discovery >= 30:
                    candidates, discovery_info = vllm_candidates(command(["ss", "-ltnp"]))
                    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                        discovered = [item for item in pool.map(scrape, candidates) if item]
                    endpoints = [item["address"] for item in discovered]
                    discovery_info.update(candidate_count=len(candidates), endpoint_count=len(endpoints))
                    last_discovery = tick
                record["vllm_discovery"] = dict(discovery_info)
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                    for item in pool.map(scrape, endpoints):
                        if item:
                            dump(vllm, item)
            if cgroup:
                record["cgroup"] = {
                    "path": cgroup,
                    "files": {
                        name: read("/sys/fs/cgroup" + cgroup + "/" + name)
                        for name in (
                            "memory.current",
                            "memory.peak",
                            "memory.max",
                            "memory.events",
                            "memory.stat",
                            "memory.swap.current",
                            "cpu.stat",
                            "io.stat",
                            "memory.pressure",
                            "cpu.pressure",
                            "io.pressure",
                        )
                    },
                }
            dump(host, record)
            stopping.wait(max(0, args.interval - (time.monotonic() - tick)))
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
