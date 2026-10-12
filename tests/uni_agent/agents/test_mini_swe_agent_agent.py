"""Tests for mini-swe-agent's white-box host-side glue.

mini-swe-agent's own loop runs on the cluster (in this process), pointing its
model at the gateway and driving the sandbox purely over HTTP as the
environment. These tests pin that wiring -- config -> model + env -> agent.run
-> AgentResult -- without a real mini-swe-agent install or network, by
monkeypatching minisweagent's ``DefaultAgent`` + ``get_model`` with fakes and
driving the real :class:`RemoteSandboxEnvironment` against an in-memory fake
sandbox (so the cross-thread ``exec_shell`` bridge is exercised too).
"""

from __future__ import annotations

import asyncio

import pytest

from uni_agent.agents.base import AgentResult, ModelConfig
from uni_agent.agents.mini_swe_agent.agent import (
    _DEFAULT_ENV,
    _DEFAULT_INSTANCE_TEMPLATE,
    MiniSweAgentAgent,
    MiniSweAgentConfig,
    RemoteSandboxEnvironment,
)
from uni_agent.sandbox.base import ExecResult


class _FakeSandbox:
    """Records ``exec_shell`` calls and answers them (drives the env bridge)."""

    def __init__(self, *, stdout: str = "", exit_code: int = 0):
        self._stdout = stdout
        self._exit_code = exit_code
        self.exec_shell_calls: list[dict] = []

    async def exec_shell(self, command, *, timeout=None, workdir=None, env=None):
        self.exec_shell_calls.append({"command": command, "timeout": timeout, "workdir": workdir, "env": env})
        return ExecResult(exit_code=self._exit_code, stdout=self._stdout, stderr="")


class _FakeModel:
    def __init__(self, config: dict):
        self.config = config


class _FakeAgent:
    """Stands in for mini-swe-agent's DefaultAgent: records config + the run call."""

    last: _FakeAgent | None = None
    result: dict = {"exit_status": "Submitted", "submission": "diff --git a b\n"}
    raise_on_run: BaseException | None = None

    def __init__(self, model: _FakeModel, env: RemoteSandboxEnvironment, **kwargs):
        self.model = model
        self.env = env
        self.kwargs = kwargs
        self.n_calls = 0
        self.cost = 0.0
        _FakeAgent.last = self

    def run(self, task: str) -> dict:
        self.task = task
        self.n_calls += 1
        if self.raise_on_run is not None:
            raise self.raise_on_run
        return dict(self.result)


@pytest.fixture(autouse=True)
def _patched_mswe(monkeypatch):
    """Swap minisweagent's DefaultAgent + get_model for fakes; quiet the banner."""
    monkeypatch.setenv("MSWEA_SILENT_STARTUP", "1")
    import minisweagent.agents.default as default_mod
    import minisweagent.models as models_mod

    monkeypatch.setattr(models_mod, "get_model", lambda *, config: _FakeModel(config))
    monkeypatch.setattr(default_mod, "DefaultAgent", _FakeAgent)
    _FakeAgent.last = None
    _FakeAgent.result = {"exit_status": "Submitted", "submission": "diff --git a b\n"}
    _FakeAgent.raise_on_run = None
    yield


def _agent(**config_kwargs) -> MiniSweAgentAgent:
    model = ModelConfig(base_url="http://gateway:8000/v1", model_name="policy")
    return MiniSweAgentAgent(MiniSweAgentConfig(model=model, **config_kwargs))


# --------------------------- validation (no import yet) ---------------------------


def test_missing_base_url_raises():
    agent = MiniSweAgentAgent(MiniSweAgentConfig())
    with pytest.raises(ValueError, match="base_url"):
        asyncio.run(agent.run(sandbox=_FakeSandbox(), messages=[{"role": "user", "content": "fix the bug"}]))


def test_missing_user_message_raises():
    agent = _agent()
    with pytest.raises(ValueError, match="requires a 'user' message"):
        asyncio.run(agent.run(sandbox=_FakeSandbox(), messages=[{"role": "system", "content": "sys"}]))


def test_too_many_user_messages_raises():
    agent = _agent()
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "user", "content": "u2"}]
    with pytest.raises(ValueError, match="at most one 'user' message"):
        asyncio.run(agent.run(sandbox=_FakeSandbox(), messages=messages))


# --------------------------- RemoteSandboxEnvironment bridge ---------------------------


def test_env_exec_forwards_command_workdir_env_timeout():
    sandbox = _FakeSandbox(stdout="hello")

    async def main() -> None:
        loop = asyncio.get_running_loop()
        env = RemoteSandboxEnvironment(
            sandbox=sandbox, loop=loop, cwd="/testbed", timeout=42, extra_env={"PAGER": "cat"}
        )
        # Called from a worker thread, exactly as the agent loop would.
        out = await asyncio.to_thread(env.execute, {"command": "echo hi"})
        assert out == {"output": "hello", "returncode": 0, "exception_info": ""}
        call = sandbox.exec_shell_calls[0]
        assert call["command"] == "echo hi"
        assert call["workdir"] == "/testbed"
        assert call["env"] == {"PAGER": "cat"}
        assert call["timeout"] == 42

    asyncio.run(main())


def test_env_exec_raises_submitted_on_complete_signal():
    from minisweagent.exceptions import Submitted

    sandbox = _FakeSandbox(stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndiff --git a b\n")

    async def main() -> None:
        loop = asyncio.get_running_loop()
        env = RemoteSandboxEnvironment(sandbox=sandbox, loop=loop)
        with pytest.raises(Submitted) as exc_info:
            await asyncio.to_thread(env.execute, {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff"})
        assert exc_info.value.messages[0]["extra"]["exit_status"] == "Submitted"
        assert exc_info.value.messages[0]["extra"]["submission"] == "diff --git a b\n"

    asyncio.run(main())


def test_env_exec_does_not_signal_without_first_line():
    sandbox = _FakeSandbox(stdout="echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndiff --git a b\n")

    async def main() -> None:
        loop = asyncio.get_running_loop()
        env = RemoteSandboxEnvironment(sandbox=sandbox, loop=loop)
        out = await asyncio.to_thread(env.execute, {"command": "echo hi"})
        assert out["exception_info"] == ""  # signal is line[1], not line[0]

    asyncio.run(main())


# --------------------------- run() wiring ---------------------------


def test_run_builds_model_and_runs_agent():
    sandbox = _FakeSandbox()
    agent = _agent(step_limit=25)
    messages = [{"role": "system", "content": "be careful"}, {"role": "user", "content": "fix the off-by-one bug"}]

    result = asyncio.run(agent.run(sandbox=sandbox, messages=messages))

    fake = _FakeAgent.last
    assert fake is not None

    # model: litellm 'openai/<served>' routed at the gateway; cost stays 0.
    assert fake.model.config["model_name"] == "openai/policy"
    assert fake.model.config["cost_tracking"] == "ignore_errors"
    assert fake.model.config["model_kwargs"]["api_base"] == "http://gateway:8000/v1"
    assert fake.model.config["model_kwargs"]["api_key"] == "EMPTY"
    assert fake.model.config["model_kwargs"]["drop_params"] is True

    # agent: system message flows in as system_template; instance defaults.
    assert fake.kwargs["system_template"] == "be careful"
    assert fake.kwargs["instance_template"] == _DEFAULT_INSTANCE_TEMPLATE
    assert fake.kwargs["step_limit"] == 25
    assert fake.kwargs["cost_limit"] == 3.0

    # env: real bridge bound to the sandbox at /testbed with quiet pagers.
    assert isinstance(fake.env, RemoteSandboxEnvironment)
    assert fake.env.config.cwd == "/testbed"
    assert fake.env.config.env == _DEFAULT_ENV

    # the loop ran with the user message as the task.
    assert fake.task == "fix the off-by-one bug"

    # result mapped onto AgentResult; Submitted => finished.
    assert isinstance(result, AgentResult)
    assert result.output["exit_status"] == "Submitted"
    assert result.output["submission"] == "diff --git a b\n"
    assert result.finished is True
    assert result.info["exit_status"] == "Submitted"


def test_default_model_name_used_when_unset():
    sandbox = _FakeSandbox()
    config = MiniSweAgentConfig(model=ModelConfig(base_url="http://gateway:8000/v1"))
    agent = MiniSweAgentAgent(config)

    asyncio.run(agent.run(sandbox=sandbox, messages=[{"role": "user", "content": "task"}]))

    assert _FakeAgent.last.model.config["model_name"] == "openai/default"


def test_custom_step_and_cost_limits_flow_through():
    sandbox = _FakeSandbox()
    agent = _agent(step_limit=99, cost_limit=5.0, wall_time_limit_seconds=1800)

    asyncio.run(agent.run(sandbox=sandbox, messages=[{"role": "user", "content": "task"}]))

    fake = _FakeAgent.last
    assert fake.kwargs["step_limit"] == 99
    assert fake.kwargs["cost_limit"] == 5.0
    assert fake.kwargs["wall_time_limit_seconds"] == 1800


def test_run_returns_unfinished_when_not_submitted():
    _FakeAgent.result = {"exit_status": "LimitsExceeded", "submission": ""}
    sandbox = _FakeSandbox()
    agent = _agent()

    result = asyncio.run(agent.run(sandbox=sandbox, messages=[{"role": "user", "content": "task"}]))

    assert result.finished is False
    assert result.output["exit_status"] == "LimitsExceeded"
    assert result.info["exit_status"] == "LimitsExceeded"


def test_run_swallows_loop_failure_and_reports():
    _FakeAgent.raise_on_run = RuntimeError("litellm blew up")
    sandbox = _FakeSandbox()
    agent = _agent()

    result = asyncio.run(agent.run(sandbox=sandbox, messages=[{"role": "user", "content": "task"}]))

    assert result.finished is False
    assert result.output["exit_status"] == "RuntimeError"
    assert "RuntimeError: litellm blew up" in result.info["error"]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
