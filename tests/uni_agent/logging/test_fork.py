"""Fork must not inherit buffered logger locks owned by a vanished thread."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.cpu, pytest.mark.level0, pytest.mark.skipif(os.name != "posix", reason="POSIX fork")]


@pytest.mark.parametrize("mode", ["fork", "uvloop"])
def test_fork_during_buffered_flush_completes_and_preserves_log_ownership(tmp_path, mode):
    if mode == "uvloop" and importlib.util.find_spec("uvloop") is None:
        pytest.skip("uvloop is an optional runtime dependency")
    root = Path(__file__).resolve().parents[3]
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).with_name("_fork_probe.py")), mode, str(tmp_path)],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        try:
            stdout, stderr = process.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate(timeout=5)
            pytest.fail(f"fork probe deadlocked\nstdout: {stdout}\nstderr: {stderr}")
        assert process.returncode == 0, f"stdout: {stdout}\nstderr: {stderr}"
        records = [json.loads(line) for line in stdout.splitlines()]
        assert records == [{"forked_while_locked": False}, {"children": 128, "reaped": True}]
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=5)
