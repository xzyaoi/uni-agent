"""Unit tests for the BrowserGym task family: config glue, prompts, reward parsing."""

from __future__ import annotations

import asyncio
import json

from uni_agent.tasks.browsergym.prompts import build_messages
from uni_agent.tasks.browsergym.reward import compute_reward, parse_result_payload
from uni_agent.tasks.browsergym.task import BrowserGymTaskConfig, build_browsergym_sandbox_config


def _base_config(provider: str = "flash_sandbox", **sandbox_overrides):
    return BrowserGymTaskConfig(
        sandbox={"provider": provider, **sandbox_overrides},
        metadata={
            "task_id": "browsergym.miniwob.miniwob_click-test",
            "benchmark": "browsergym.miniwob",
            "seed": 3,
            "timeout_s": 300.0,
        },
    )


class FakeSandbox:
    """Serves one exec result then fails the result-file read (or vice versa)."""

    def __init__(self, exec_result=None, file_bytes=None):
        self.exec_result = exec_result
        self.file_bytes = file_bytes
        self.exec_calls = 0

    async def exec(self, argv, **kwargs):
        self.exec_calls += 1
        if self.exec_result is not None:
            return self.exec_result
        raise RuntimeError("daemon dead")

    async def read_file(self, path):
        if self.file_bytes is None:
            raise RuntimeError("no result file")
        return self.file_bytes


def test_flash_sandbox_defaults_image_resources_and_ttl(monkeypatch):
    monkeypatch.delenv("BROWSERGYM_IMAGE", raising=False)
    from uni_agent.tasks.browsergym import task as task_mod

    config = build_browsergym_sandbox_config(_base_config(), _base_config().metadata)
    assert config.sandbox_kwargs["cpu_cores"] == 2.0
    assert config.sandbox_kwargs["memory_mb"] == 4096
    assert config.sandbox_kwargs["keep_alive"] is True
    # runtime timeout must cover daemon boot + episode + margin
    assert config.runtime_timeout >= 300.0 + 240.0 + 300.0
    # untouched default image is replaced by the browsergym default
    assert config.image == task_mod.DEFAULT_IMAGE


def test_env_var_overrides_default_image(monkeypatch):
    monkeypatch.setenv("BROWSERGYM_IMAGE", "ghcr.io/me/browsergym:0.2.0")
    config = build_browsergym_sandbox_config(_base_config(), _base_config().metadata)
    assert config.image == "ghcr.io/me/browsergym:0.2.0"


def test_explicit_image_is_kept():
    config = build_browsergym_sandbox_config(_base_config(image="ghcr.io/me/browsergym:0.1.0"), _base_config().metadata)
    assert config.image == "ghcr.io/me/browsergym:0.1.0"


def test_unknown_provider_rejected():
    try:
        build_browsergym_sandbox_config(_base_config(provider="local"), _base_config().metadata)
    except ValueError as exc:
        assert "flash_sandbox" in str(exc)
    else:
        raise AssertionError("local provider should be rejected")


def test_prompt_contains_goal_obs_and_actions():
    messages = build_messages(goal="Click the button labelled OK.", obs="[1] button 'OK'", url="http://x/y")
    assert messages[0]["role"] == "system"
    user = messages[1]["content"]
    assert "Click the button labelled OK." in user
    assert "[1] button 'OK'" in user
    assert "click(bid=" in user
    assert "send_msg_to_user" in user  # answers go through the chat, not stop()


def test_parse_result_payload_clips_and_marks_success():
    parsed = parse_result_payload({"reward": 1.0, "done": True, "truncated": False, "n_steps": 4})
    assert parsed["reward"] == 1.0 and parsed["success"] is True
    parsed = parse_result_payload({"reward": 2.5, "done": True, "truncated": False})  # clipped
    assert parsed["reward"] == 1.0
    parsed = parse_result_payload({"reward": 1.0, "done": True, "truncated": True})  # truncation != success
    assert parsed["success"] is False


def test_compute_reward_prefers_live_daemon_then_file():
    from uni_agent.sandbox import ExecResult

    payload = json.dumps({"ok": True, "reward": 1.0, "done": True, "n_steps": 2})
    sandbox = FakeSandbox(exec_result=ExecResult(exit_code=0, stdout=payload, stderr=""))
    result = asyncio.run(
        compute_reward(
            sandbox,
            python_bin="python",
            bridge_path="/opt/bgym/bridge.py",
            bridge_port=8027,
            result_file="/logs/bgym/result.json",
        )
    )
    assert result["reward"] == 1.0 and result["success"] is True and sandbox.exec_calls == 1

    sandbox = FakeSandbox(file_bytes=payload.encode())
    result = asyncio.run(
        compute_reward(
            sandbox,
            python_bin="python",
            bridge_path="/opt/bgym/bridge.py",
            bridge_port=8027,
            result_file="/logs/bgym/result.json",
        )
    )
    assert result["reward"] == 1.0

    sandbox = FakeSandbox()
    result = asyncio.run(
        compute_reward(
            sandbox,
            python_bin="python",
            bridge_path="/opt/bgym/bridge.py",
            bridge_port=8027,
            result_file="/logs/bgym/result.json",
        )
    )
    assert result["reward"] == 0.0 and result["error"]


def test_task_registered_and_buildable_from_flat_mapping():
    from uni_agent.tasks import get_task
    from uni_agent.tasks.registry import TASK_MODULES

    assert "browsergym" in TASK_MODULES
    task = get_task(
        {"name": "browsergym", "sandbox": {"provider": "flash_sandbox"}, "metadata": {"task_id": "x", "benchmark": "y"}}
    )
    assert task.name == "browsergym"
