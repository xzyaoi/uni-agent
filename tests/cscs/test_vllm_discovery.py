"""Exercise collector discovery without starting its command-line sampling loop."""

import ast
import io
import json
import os
import re
import runpy
import signal
import socket
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "cscs/observability/collect.py"
# collect.py intentionally starts collectors at module level. Compile its actual
# discovery helpers separately so tests never invoke GPU commands or network I/O.
tree = ast.parse(SOURCE.read_text())
helpers = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)
                          and node.name in {"vllm_address", "proc_vllm_candidates", "vllm_candidates"}],
                     type_ignores=[])
namespace = {"os": os, "re": re, "socket": socket, "struct": struct, "Path": Path}
exec(compile(helpers, str(SOURCE), "exec"), namespace)
discover = namespace["vllm_candidates"]


def process(root, pid, name, inodes, *, comm=None):
    directory = root / str(pid)
    (directory / "fd").mkdir(parents=True)
    (directory / "comm").write_text(comm if comm is not None else name[:15])
    (directory / "cmdline").write_bytes(name.encode() + b"\0")
    for index, inode in enumerate(inodes):
        (directory / "fd" / str(index)).symlink_to("socket:[" + str(inode) + "]")
    return directory


def tcp_row(ip, port, inode, state="0A"):
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    raw = socket.inet_pton(family, ip)
    address = "".join(f'{struct.unpack("=I", raw[i:i + 4])[0]:08X}' for i in range(0, len(raw), 4))
    return f"0: {address}:{port:04X} 00000000:0000 {state} 0:0 0:0 0 29484 0 {inode} 1\n"


def tables(root, tcp="", tcp6=""):
    (root / "net").mkdir(exist_ok=True)
    for name, body in (("tcp", tcp), ("tcp6", tcp6)):
        (root / "net" / name).write_text("sl local_address rem_address st queues timer retr uid timeout inode\n" + body)


def test_legacy_ss_discovery_remains_preferred(tmp_path):
    result = {"returncode": 0, "stdout": "\n".join([
        'LISTEN 0 128 172.28.46.140:43755 *:* users:(("ray::vLLMHttpSe",pid=123,fd=9))',
        'LISTEN 0 128 *:10300 *:* users:(("ray::vLLMHttpSe",pid=123,fd=10))',
        'LISTEN 0 128 [::]:8000 *:* users:(("ray::vLLMHttpSe",pid=123,fd=11))',
        'LISTEN 0 128 127.0.0.1:9999 *:* users:(("python",pid=456,fd=1))',
    ])}
    endpoints, info = discover(result, tmp_path / "missing-proc")
    assert endpoints == ["127.0.0.1:10300", "172.28.46.140:43755", "[::1]:8000"]
    assert info == {"method": "ss"}


@pytest.mark.parametrize("ss_result", [{"error": "ss unavailable"}, {"returncode": 127, "stdout": ""}])
def test_procfs_fallback_only_probes_actor_owned_listeners(tmp_path, ss_result):
    process(tmp_path, 100, "ray::vLLMHttpServer", [111, 222, 333])
    process(tmp_path, 200, "unrelated-server", [444])
    process(tmp_path, 300, "ray::IDLE", [555])
    tables(tmp_path, tcp=(tcp_row("172.28.46.140", 43755, 111)
                         + tcp_row("0.0.0.0", 8000, 222)
                         + tcp_row("127.0.0.1", 9090, 333, state="01")
                         + tcp_row("127.0.0.1", 9999, 444)
                         + tcp_row("127.0.0.1", 8888, 555)))
    endpoints, info = discover(ss_result, tmp_path)
    assert endpoints == ["127.0.0.1:8000", "172.28.46.140:43755"]
    assert info == {"method": "procfs", "peer_process_count": 1, "unreadable_proc_entries": 0}


def test_procfs_ipv6_wildcard_and_specific_addresses(tmp_path):
    process(tmp_path, 100, "ray::vLLMHttpServer", [111, 222, 333])
    tables(tmp_path, tcp6=(tcp_row("::", 8000, 111) + tcp_row("::1", 8001, 222)
                          + tcp_row("2001:db8::10", 8002, 333)))
    endpoints, _ = discover({"error": "ss unavailable"}, tmp_path)
    assert endpoints == ["[2001:db8::10]:8002", "[::1]:8000", "[::1]:8001"]


def test_truncated_comm_matches_when_cmdline_is_python(tmp_path):
    process(tmp_path, 100, "/usr/bin/python3", [111], comm="ray::vLLMHttpSe")
    tables(tmp_path, tcp=tcp_row("127.0.0.1", 8000, 111))
    endpoints, _ = discover({"error": "ss unavailable"}, tmp_path)
    assert endpoints == ["127.0.0.1:8000"]


def test_fd_permission_denial_is_reported_without_scanning_other_ports(tmp_path, monkeypatch):
    peer = process(tmp_path, 100, "ray::vLLMHttpServer", [111])
    process(tmp_path, 200, "unrelated-server", [444])
    tables(tmp_path, tcp=tcp_row("127.0.0.1", 8000, 111) + tcp_row("127.0.0.1", 9999, 444))
    original = Path.iterdir

    def iterdir(path):
        if path == peer / "fd":
            raise PermissionError("procfs peer descriptors denied")
        return original(path)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    endpoints, info = discover({"error": "ss unavailable"}, tmp_path)
    assert endpoints == []
    assert info["peer_process_count"] == 1
    assert info["unreadable_proc_entries"] == 1


def test_descriptor_race_and_malformed_tables_do_not_crash(tmp_path, monkeypatch):
    peer = process(tmp_path, 100, "ray::vLLMHttpServer", [111, 222])
    tables(tmp_path, tcp=(tcp_row("127.0.0.1", 8000, 111)
                         + "malformed\n" + tcp_row("127.0.0.1", 8001, 222)))
    original = os.readlink

    def readlink(path):
        if path == str(peer / "fd/1"):
            raise FileNotFoundError("peer descriptor exited")
        return original(path)

    monkeypatch.setattr(os, "readlink", readlink)
    endpoints, info = discover({"returncode": 1}, tmp_path)
    assert endpoints == ["127.0.0.1:8000"]
    assert info["unreadable_proc_entries"] == 1


def test_missing_procfs_is_a_safe_empty_discovery(tmp_path):
    endpoints, info = discover({"error": "ss unavailable"}, tmp_path / "absent")
    assert endpoints == []
    assert info["unreadable_proc_entries"] == 1


def test_collector_stays_python36_compatible():
    ast.parse(SOURCE.read_text(), feature_version=(3, 6))


def test_gpu_sample_records_discovery_counts_without_changing_raw_schema(tmp_path, monkeypatch):
    import urllib.request

    body = b"vllm:kv_cache_usage_perc 0.25\n"
    response = Mock()
    response.read.return_value = body
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    opener = Mock()
    opener.open.return_value = response
    monkeypatch.setattr(urllib.request, "build_opener", Mock(return_value=opener))

    def command(argv, **kwargs):
        stdout = (b'LISTEN 0 128 127.0.0.1:12345 *:* users:(("ray::vLLMHttpSe",pid=1,fd=2))\n'
                  if argv[0] == "ss" else b"")
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(subprocess, "run", command)
    monkeypatch.setattr(subprocess, "Popen", Mock(return_value=SimpleNamespace(
        stdout=io.BytesIO(b""), poll=lambda: 0)))
    monkeypatch.setattr(signal, "signal", Mock())
    monkeypatch.setattr(sys, "argv", [str(SOURCE), "--role", "gpu", "--output", str(tmp_path),
                                      "--duration", "0.05", "--interval", "0"])
    runpy.run_path(str(SOURCE), run_name="__main__")
    host_path = next(tmp_path.glob("*-host.jsonl"))
    host_rows = [json.loads(line) for line in host_path.read_text().splitlines()]
    assert host_rows and all(row["vllm_discovery"] == {
        "method": "ss", "candidate_count": 1, "endpoint_count": 1,
    } for row in host_rows)
    metric_path = next(tmp_path.glob("*-vllm.jsonl"))
    rows = [json.loads(line) for line in metric_path.read_text().splitlines()]
    assert rows and all(row["prometheus"] == body.decode() and row["address"] == "127.0.0.1:12345" for row in rows)
    assert all(set(row) == {"timestamp", "node", "address", "prometheus"} for row in rows)
