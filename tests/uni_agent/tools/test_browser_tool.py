"""Unit tests for the ``browser`` tool: JSON bridge parsing + error paths.

No sandbox / network -- the backend is a fake returning canned ExecResults,
mirroring the style of ``test_toolbox_call_errors.py``.
"""

from __future__ import annotations

import asyncio
import json

from uni_agent.tools import Toolbox
from uni_agent.tools.browser import BrowserTool, BrowserToolConfig


class FakeBackend:
    """Records exec calls; returns queued ExecResults in order."""

    def __init__(self, results):
        from uni_agent.sandbox import ExecResult

        self._results = list(results)
        self.calls: list[tuple] = []

        def make(res):
            if isinstance(res, ExecResult):
                return res
            return ExecResult(
                exit_code=res.get("exit_code", 0), stdout=res.get("stdout", ""), stderr=res.get("stderr", "")
            )

        self._results = [make(r) for r in self._results]

    async def exec(self, argv, *, timeout=None, workdir=None, env=None):
        self.calls.append((tuple(argv), timeout))
        if not self._results:
            raise AssertionError("no canned result left")
        return self._results.pop(0)


def _ok_payload(**extra) -> str:
    return json.dumps({"ok": True, "obs_text": "[1] button 'OK'", "reward": 0.0, "done": False, "n_steps": 1, **extra})


def _tool(backend, **kwargs) -> Toolbox:
    return Toolbox([BrowserTool(backend, **kwargs)])


def test_ok_step_returns_observation():
    backend = FakeBackend([{"stdout": _ok_payload()}])
    result = asyncio.run(_tool(backend).call("browser", {"action": "click(bid='1')"}))
    assert result.status == "ok"
    assert "[1] button 'OK'" in result.text
    argv = backend.calls[0][0]
    assert argv[:3] == ("python", "/opt/bgym/bridge.py", "step")
    assert "--action" in argv and "click(bid='1')" in argv


def test_invalid_action_is_tool_error_with_env_reason():
    backend = FakeBackend(
        [{"stdout": json.dumps({"ok": False, "error": "Invalid action (ValueError): unknown bid '99'"})}]
    )
    result = asyncio.run(_tool(backend).call("browser", {"action": "click(bid='99')"}))
    assert result.status == "error"
    assert "unknown bid '99'" in result.text


def test_bridge_crash_is_tool_error():
    backend = FakeBackend([{"exit_code": 1, "stderr": "URLError: connection refused"}])
    result = asyncio.run(_tool(backend).call("browser", {"action": "noop()"}))
    assert result.status == "error"
    assert "connection refused" in result.text


def test_garbage_json_is_tool_error():
    backend = FakeBackend([{"stdout": "<html>gateway noise</html>"}])
    result = asyncio.run(_tool(backend).call("browser", {"action": "noop()"}))
    assert result.status == "error"
    assert "invalid JSON" in result.text


def test_done_appends_marker_with_reward():
    backend = FakeBackend([{"stdout": _ok_payload(done=True, reward=1.0, success=True)}])
    result = asyncio.run(_tool(backend).call("browser", {"action": "stop(answer='42')"}))
    assert "episode has finished" in result.text
    assert "reward=1.0" in result.text


def test_observation_clipped_at_obs_max_chars():
    long_obs = "x" * 5000
    backend = FakeBackend([{"stdout": _ok_payload(obs_text=long_obs)}])
    tool = BrowserTool(backend, obs_max_chars=100)
    result = asyncio.run(tool.run({"action": "noop()"}))
    assert len(result.text) < 500
    assert "clipped at 100 chars" in result.text


def test_timeout_forwarded_as_default_command_timeout():
    backend = FakeBackend([{"stdout": _ok_payload()}, {"stdout": _ok_payload()}])
    tool = BrowserTool(backend, command_timeout=7.0)
    asyncio.run(tool.run({"action": "noop()"}))
    assert backend.calls[0][1] == 7.0
    asyncio.run(tool.run({"action": "noop()"}, timeout=42.0))
    assert backend.calls[1][1] == 42.0


def test_missing_action_argument_is_format_error():
    backend = FakeBackend([{"stdout": _ok_payload()}])
    result = asyncio.run(_tool(backend).call("browser", {}))  # Toolbox.call reports, never raises
    assert result.status == "format_error"
    assert "action" in result.text


def test_config_model_defaults():
    config = BrowserToolConfig()
    assert config.bridge_port == 8027
    assert config.command_timeout == 120.0
