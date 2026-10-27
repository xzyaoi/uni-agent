"""Exercise Harbor CLI process ownership with real subprocesses."""

import asyncio
import os
import signal
import sys
from pathlib import Path

import pytest

from uni_agent.tasks.harbor import task as harbor_task

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def _running(pid: int) -> bool:
    try:
        # A terminated orphan may remain a zombie until the host's init reaps it.
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0] != "Z"
    except FileNotFoundError:
        return False


async def _wait_for(predicate, *, timeout=5):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.02)

    await asyncio.wait_for(poll(), timeout=timeout)


async def _finish_cancelled(task):
    done, _ = await asyncio.wait({task}, timeout=5)
    assert task in done, "Harbor cancellation hung while waiting for inherited pipes"
    with pytest.raises(asyncio.CancelledError):
        await task


async def _cleanup(task, *pid_paths):
    for path in pid_paths:
        if path.exists():
            try:
                os.kill(int(path.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
    if not task.done():
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_captures_exit_code_output_and_environment():
    result = await harbor_task.run_harbor_cli(
        [
            sys.executable,
            "-c",
            "import os,sys; print(os.environ['HARBOR_TEST_VALUE']); "
            "sys.stderr.buffer.write(b'error\\xff'); sys.exit(7)",
        ],
        env={"HARBOR_TEST_VALUE": "supplied"},
    )
    assert result.exit_code == 7
    assert result.stdout == "supplied\n"
    assert result.stderr == "error\ufffd"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc process-state assertions")
@pytest.mark.parametrize("parent_exits", [False, True], ids=["running-parent", "exited-parent"])
@pytest.mark.asyncio
async def test_cancellation_kills_descendant_holding_captured_pipes(tmp_path, monkeypatch, parent_exits):
    monkeypatch.setattr(harbor_task, "_PROCESS_TERMINATE_TIMEOUT_SECONDS", 0.2, raising=False)
    monkeypatch.setattr(harbor_task, "_PROCESS_KILL_TIMEOUT_SECONDS", 0.2, raising=False)
    parent_pid = tmp_path / "parent.pid"
    child_pid = tmp_path / "child.pid"
    child_code = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(child_pid)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    parent_code = (
        "import os,subprocess,sys,time; from pathlib import Path; "
        f"Path({str(parent_pid)!r}).write_text(str(os.getpid())); "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        + ("sys.exit(0)" if parent_exits else "time.sleep(60)")
    )
    task = asyncio.create_task(harbor_task.run_harbor_cli([sys.executable, "-c", parent_code]))
    try:
        await _wait_for(child_pid.exists)
        if parent_exits:
            await _wait_for(lambda: not _running(int(parent_pid.read_text())))
        task.cancel()
        await _finish_cancelled(task)
        await _wait_for(lambda: not _running(int(child_pid.read_text())))
        assert not _running(int(parent_pid.read_text()))
    finally:
        await _cleanup(task, parent_pid, child_pid)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc process-state assertions")
@pytest.mark.asyncio
async def test_cancellation_allows_graceful_shutdown(tmp_path):
    pid_path = tmp_path / "parent.pid"
    stopped_path = tmp_path / "terminated"
    code = (
        "import os,signal,time; from pathlib import Path\n"
        "def stop(signum, frame):\n"
        f"    Path({str(stopped_path)!r}).write_text('graceful')\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        f"Path({str(pid_path)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    task = asyncio.create_task(harbor_task.run_harbor_cli([sys.executable, "-c", code]))
    try:
        await _wait_for(pid_path.exists)
        task.cancel()
        await _finish_cancelled(task)
        assert stopped_path.read_text() == "graceful"
        assert not _running(int(pid_path.read_text()))
    finally:
        await _cleanup(task, pid_path)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc process-state assertions")
@pytest.mark.asyncio
async def test_repeated_cancellation_does_not_abandon_cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr(harbor_task, "_PROCESS_TERMINATE_TIMEOUT_SECONDS", 0.3)
    pid_path = tmp_path / "parent.pid"
    term_path = tmp_path / "term"
    code = (
        "import os,signal,time; from pathlib import Path\n"
        "def stop(signum, frame):\n"
        f"    Path({str(term_path)!r}).write_text('received')\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        f"Path({str(pid_path)!r}).write_text(str(os.getpid()))\n"
        "while True: time.sleep(60)\n"
    )
    task = asyncio.create_task(harbor_task.run_harbor_cli([sys.executable, "-c", code]))
    try:
        await _wait_for(pid_path.exists)
        task.cancel()
        await _wait_for(term_path.exists)
        task.cancel()
        await _finish_cancelled(task)
        assert not _running(int(pid_path.read_text()))
    finally:
        await _cleanup(task, pid_path)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc process-state assertions")
@pytest.mark.asyncio
async def test_cancellation_during_spawn_still_owns_child(tmp_path, monkeypatch):
    create_subprocess_exec = asyncio.create_subprocess_exec
    spawned = asyncio.Event()
    release = asyncio.Event()
    pid_path = tmp_path / "parent.pid"

    async def delayed_spawn(*args, **kwargs):
        process = await create_subprocess_exec(*args, **kwargs)
        spawned.set()
        await release.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)
    code = (
        "import os,time; from pathlib import Path; "
        f"Path({str(pid_path)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    task = asyncio.create_task(harbor_task.run_harbor_cli([sys.executable, "-c", code]))
    try:
        await asyncio.wait_for(spawned.wait(), timeout=5)
        await _wait_for(pid_path.exists)
        task.cancel()
        release.set()
        await _finish_cancelled(task)
        assert not _running(int(pid_path.read_text()))
    finally:
        release.set()
        await _cleanup(task, pid_path)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc process-state assertions")
@pytest.mark.asyncio
async def test_pipe_drain_is_bounded_when_a_process_escapes_the_owned_group(tmp_path, monkeypatch):
    monkeypatch.setattr(harbor_task, "_PROCESS_TERMINATE_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(harbor_task, "_PROCESS_KILL_TIMEOUT_SECONDS", 0.2)
    parent_pid = tmp_path / "parent.pid"
    child_pid = tmp_path / "child.pid"
    child_code = (
        "import os,time; from pathlib import Path; "
        f"Path({str(child_pid)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    code = (
        "import os,subprocess,sys,time; from pathlib import Path; "
        f"Path({str(parent_pid)!r}).write_text(str(os.getpid())); "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}], start_new_session=True); time.sleep(60)"
    )
    task = asyncio.create_task(harbor_task.run_harbor_cli([sys.executable, "-c", code]))
    try:
        await _wait_for(child_pid.exists)
        task.cancel()
        await _finish_cancelled(task)
        assert not _running(int(parent_pid.read_text()))
        # Cleanup closes our pipes without signalling another process group.
        assert _running(int(child_pid.read_text()))
    finally:
        await _cleanup(task, parent_pid, child_pid)


def _task_with_local_trials(tmp_path, monkeypatch):
    task_path = tmp_path / "task"
    task_path.mkdir()
    (task_path / "task.toml").write_text("")
    config = harbor_task.HarborTaskConfig(
        agent={"name": "oracle"}, metadata={"instance_id": "test", "task_path": str(task_path)}
    )
    monkeypatch.setattr(harbor_task, "_TEMP_TRIALS_DIR", tmp_path / "trials")
    monkeypatch.setattr(harbor_task.execution_logging, "get_current_log_context", lambda: None)
    return harbor_task.HarborTask(config)


@pytest.mark.asyncio
async def test_cancelled_trial_retains_artifacts_without_waiting_on_filesystem_cleanup(tmp_path, monkeypatch, caplog):
    trial = _task_with_local_trials(tmp_path, monkeypatch)
    started = asyncio.Event()
    trial_paths = []

    async def run_trial(config, trials_dir, *, trial_name, output_dir):
        trial_dir = trials_dir / trial_name
        trial_dir.mkdir(parents=True)
        (trial_dir / "progress").write_text("diagnostic data")
        trial_paths.append(trial_dir)
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(trial, "_run_trial", run_trial)
    task = asyncio.create_task(trial.run())
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    await _finish_cancelled(task)
    assert (trial_paths[0] / "progress").read_text() == "diagnostic data"
    assert str(trial_paths[0]) in caplog.text


@pytest.mark.parametrize("fails", [False, True], ids=["completed", "failed"])
@pytest.mark.asyncio
async def test_non_cancelled_trial_still_cleans_temporary_directory(tmp_path, monkeypatch, fails):
    trial = _task_with_local_trials(tmp_path, monkeypatch)
    trial_paths = []

    async def run_trial(config, trials_dir, *, trial_name, output_dir):
        trial_dir = trials_dir / trial_name
        trial_dir.mkdir(parents=True)
        trial_paths.append(trial_dir)
        if fails:
            raise RuntimeError("trial failed")
        return harbor_task.TaskResult(reward=1.0)

    monkeypatch.setattr(trial, "_run_trial", run_trial)
    if fails:
        with pytest.raises(RuntimeError, match="trial failed"):
            await trial.run()
    else:
        assert (await trial.run()).reward == 1.0
    assert not trial_paths[0].exists()
