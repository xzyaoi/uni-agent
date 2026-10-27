"""Exercise startup variables through real shells, without a sandbox service."""

import asyncio
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from uni_agent.sandbox.base import ExecResult
from uni_agent.sandbox.exec_context import bind_exec_metadata, current_exec_metadata
from uni_agent.sandbox.flash_sandbox import FlashSandboxSandbox, _wrap_command


@pytest.fixture
def guest(tmp_path):
    tools = tmp_path / "tool directory"
    tools.mkdir()
    go = tools / "go"
    go.write_text('#!/bin/sh\nprintf "go-found:%s\\n" "$TASK_VALUE"\n')
    go.chmod(0o755)
    profile = tmp_path / "profile"
    profile.write_text("export PATH=/missing-from-login-profile\n")
    return tools, {**os.environ, "BASH_ENV": str(profile)}


class LocalExecClient:
    def __init__(self, environment):
        self.environment = environment

    async def exec_command(self, sandbox_id, command, **kwargs):
        result = subprocess.run(command, env=self.environment, capture_output=True, text=True, timeout=5)
        return ExecResult(result.returncode, result.stdout, result.stderr)


def attach(sandbox, client):
    sandbox._client = client
    sandbox._sandbox_id = "local-test"
    return sandbox


def test_explicit_path_survives_login_profile(guest):
    tools, environment = guest
    result = subprocess.run(
        _wrap_command(["go"], None, {"PATH": str(tools), "TASK_VALUE": "from-task"}),
        env=environment,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "go-found:from-task\n"


def test_wrapper_preserves_literal_values_and_working_directory(tmp_path, guest):
    _, environment = guest
    cwd = tmp_path / "working 'directory"
    cwd.mkdir()
    value = "literal $HOME $(echo expanded) `echo expanded` ' \"\nnext line"
    command = [
        sys.executable,
        "-c",
        "import os,json; print(json.dumps([os.getcwd(),os.environ['VALUE'],os.environ['TAR_OPTIONS']]))",
    ]
    result = subprocess.run(
        _wrap_command(command, str(cwd), {"VALUE": value}),
        env=environment,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [str(cwd), value, "--no-same-owner"]


@pytest.mark.asyncio
async def test_one_shot_overrides_do_not_mutate_startup_environment(guest):
    tools, environment = guest
    startup = {"PATH": str(tools), "TASK_VALUE": "startup"}
    sandbox = attach(FlashSandboxSandbox(env_vars=startup), LocalExecClient(environment))
    startup["TASK_VALUE"] = "caller-mutation"
    overridden = await sandbox.exec_shell("go", env={"TASK_VALUE": "per-command"})
    default = await sandbox.exec_shell("go")
    assert overridden.stdout == "go-found:per-command\n"
    assert default.stdout == "go-found:startup\n"
    assert overridden.exit_code == default.exit_code == 0


@pytest.mark.asyncio
async def test_one_shot_explicitly_avoids_sdks_persistent_shell_default():
    async def exec_command(sandbox_id, command, *, timeout_ms, persistent=True):
        assert persistent is False
        return ExecResult(0, "isolated", "")

    sandbox = attach(FlashSandboxSandbox(), types.SimpleNamespace(exec_command=exec_command))
    assert (await sandbox.exec_shell("true")).stdout == "isolated"


@pytest.mark.asyncio
async def test_bootstrap_runs_executable_commands(tmp_path, monkeypatch):
    import uni_agent.sandbox.flash_sandbox as provider

    shim = tmp_path / "apt-get"
    git_log = tmp_path / "git-invocations"
    git = tmp_path / "git"
    git.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {shlex.quote(str(git_log))}\nexit 0\n")
    git.chmod(0o755)

    async def write_file(sandbox_id, path, data):
        Path(path).write_bytes(data)

    client = LocalExecClient(dict(os.environ))
    client.write_file = write_file
    sandbox = attach(FlashSandboxSandbox(env_vars={"PATH": f"{tmp_path}:/usr/bin:/bin"}), client)
    monkeypatch.setattr(provider, "_ROOTLESS_APT_GET_PATH", str(shim))
    await sandbox._bootstrap()
    assert os.access(shim, os.X_OK)
    assert "config --system --get-all safe.directory" in git_log.read_text()


@pytest.mark.asyncio
async def test_creation_receives_independent_startup_mapping(monkeypatch):
    seen = []

    async def create(**kwargs):
        seen.append(dict(kwargs["env"]))
        kwargs["env"]["TASK_VALUE"] = "sdk-mutation"
        return types.SimpleNamespace(id="local-test")

    client = types.SimpleNamespace(start_sandbox=create, stop_sandbox=AsyncMock(), close=AsyncMock())
    monkeypatch.setitem(sys.modules, "flash_sandbox", types.SimpleNamespace(AsyncHTTPClient=lambda *a, **k: client))
    monkeypatch.setenv("OTELA_API", "test-only")
    sandbox = FlashSandboxSandbox(env_vars={"TASK_VALUE": "startup"}, keep_alive=False)
    monkeypatch.setattr(sandbox, "_bootstrap", AsyncMock())
    try:
        await sandbox.start()
        assert seen == [{"TASK_VALUE": "startup"}]
        assert sandbox.env_vars == {"TASK_VALUE": "startup"}
    finally:
        await sandbox.stop()


class LocalShellClient:
    """Model SDK string wrapping and server argv quoting in a real persistent Bash."""

    async def start(self):
        self.process = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "--noprofile",
            "--norc",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

    async def shell_exec(self, sandbox_id, command, **kwargs):
        argv = ["sh", "-c", command] if isinstance(command, str) else command
        command = shlex.join(argv)
        marker = "done_" + uuid4().hex
        self.process.stdin.write((command + f"\nresult=$?; printf '\\n{marker}:%s\\n' \"$result\"\n").encode())
        await self.process.stdin.drain()
        lines = []
        while True:
            line = await asyncio.wait_for(self.process.stdout.readline(), 5)
            assert line, "Local shell exited unexpectedly"
            if line.startswith((marker + ":").encode()):
                return ExecResult(int(line.split(b":")[1]), b"".join(lines).decode().rstrip("\n"), "")
            lines.append(line)

    async def close(self):
        self.process.terminate()
        await self.process.wait()


@pytest.mark.asyncio
async def test_persistent_shell_initializes_once_and_preserves_agent_changes(tmp_path, guest):
    tools, _ = guest
    cwd = tmp_path / "working directory"
    cwd.mkdir()
    client = LocalShellClient()
    await client.start()
    sandbox = attach(FlashSandboxSandbox(env_vars={"PATH": str(tools), "TASK_VALUE": "startup"}), client)
    try:
        shell = await sandbox.open_shell(cwd=str(cwd), env={"TASK_VALUE": "tool-override"})
        result = await shell.run('go; printf "%s\\n" "$PWD"')
        assert result.stdout == f"go-found:tool-override\n{cwd}"
        changed = "agent 'change' $HOME; literal"
        await shell.run(f"export TASK_VALUE={shlex.quote(changed)}; cd -- {shlex.quote(str(tmp_path))}")
        result = await shell.run('go; printf "%s\\n" "$PWD"')
        assert result.stdout == f"go-found:{changed}\n{tmp_path}"
        assert sandbox.env_vars["TASK_VALUE"] == "startup"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_persistent_shell_setup_errors_are_not_silently_ignored(tmp_path):
    client = LocalShellClient()
    await client.start()
    sandbox = attach(FlashSandboxSandbox(), client)
    try:
        with pytest.raises(RuntimeError, match="initialize Flash Sandbox"):
            await sandbox.open_shell(cwd=str(tmp_path / "missing"))
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_shell_setup_does_not_report_the_callers_terminal_outcome():
    calls = []

    async def shell_exec(sandbox_id, command, *, timeout_ms, metadata=None):
        calls.append(metadata)
        return ExecResult(0, "", "")

    sandbox = attach(
        FlashSandboxSandbox(env_vars={"TASK_VALUE": "startup"}),
        types.SimpleNamespace(shell_exec=shell_exec),
    )
    async with bind_exec_metadata(rlvr={"terminal": True}):
        shell = await sandbox.open_shell()
        assert current_exec_metadata() == {"rlvr": {"terminal": True}}
        await shell.run("true")
    assert calls == [None, {"rlvr": {"terminal": True}}]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("setup transport failed"), asyncio.CancelledError()])
async def test_shell_setup_restores_metadata_on_failure(failure):
    async def shell_exec(sandbox_id, command, *, timeout_ms, metadata=None):
        assert metadata is None
        raise failure

    sandbox = attach(
        FlashSandboxSandbox(env_vars={"TASK_VALUE": "startup"}),
        types.SimpleNamespace(shell_exec=shell_exec),
    )
    async with bind_exec_metadata(rlvr={"terminal": True}):
        with pytest.raises(type(failure)):
            await sandbox.open_shell()
        assert current_exec_metadata() == {"rlvr": {"terminal": True}}


@pytest.mark.asyncio
async def test_real_harbor_task_trial_and_scoped_environment_precedence(tmp_path, guest):
    pytest.importorskip("harbor", reason="Harbor is an optional dependency")
    from harbor.models.task.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

    from uni_agent.tasks.harbor.flash_environment import FlashSandboxEnvironment

    tools, environment = guest
    adapter = FlashSandboxEnvironment(
        environment_dir=tmp_path,
        environment_name="startup-env-test",
        session_id="startup-env-test",
        trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
        task_env_config=EnvironmentConfig(
            docker_image="task-image",
            env={"PATH": str(tools), "TASK_VALUE": "task"},
        ),
        persistent_env={"TASK_VALUE": "trial"},
    )
    attach(adapter.sandbox, LocalExecClient(environment))
    assert adapter.sandbox.env_vars["TASK_VALUE"] == "trial"
    assert (await adapter.exec("go")).stdout == "go-found:trial\n"
    assert (await adapter.exec("go", env={"TASK_VALUE": "per-command"})).stdout == "go-found:per-command\n"
    with adapter.scoped_exec_env({"TASK_VALUE": "scoped"}):
        result = await adapter.exec("go", env={"TASK_VALUE": "per-command"})
        assert result.stdout == "go-found:scoped\n"
    assert (await adapter.exec("go")).stdout == "go-found:trial\n"


@pytest.mark.asyncio
async def test_harbor_startup_environment_reaches_actual_verifier_command(monkeypatch, guest):
    """Stub only optional Harbor's boundary; execute the real adapter/provider/Bash."""
    tools, environment = guest
    startup = {"PATH": str(tools), "TASK_VALUE": "harbor-task"}

    class BaseEnvironment:
        def __init__(self, **kwargs):
            self.task_env_config = types.SimpleNamespace(docker_image="task-image", workdir=None)
            self._effective_cpus = kwargs["override_cpus"]
            self._effective_memory_mb = kwargs["override_memory_mb"]
            self._startup_env = Mock(return_value=startup)

        def _merge_env(self, env):
            return env

        def _resolve_user(self, user):
            return user

    base = types.ModuleType("harbor.environments.base")
    base.BaseEnvironment = BaseEnvironment
    base.ExecResult = types.SimpleNamespace
    monkeypatch.setitem(sys.modules, "harbor", types.ModuleType("harbor"))
    monkeypatch.setitem(sys.modules, "harbor.environments", types.ModuleType("harbor.environments"))
    monkeypatch.setitem(sys.modules, "harbor.environments.base", base)
    path = Path(__file__).resolve().parents[3] / "uni_agent/tasks/harbor/flash_environment.py"
    spec = importlib.util.spec_from_file_location("tested_flash_harbor_environment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    adapter = module.FlashSandboxEnvironment()
    adapter._startup_env.assert_called_once_with()
    attach(adapter.sandbox, LocalExecClient(environment))
    result = await adapter.exec("go", env={"TASK_VALUE": "verifier-override"})
    assert result.return_code == 0, result.stderr
    assert result.stdout == "go-found:verifier-override\n"
    assert adapter.sandbox.env_vars == startup
