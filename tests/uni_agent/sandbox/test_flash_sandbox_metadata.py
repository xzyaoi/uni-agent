"""Metadata forwarding for the flash-sandbox provider.

The provider reads :func:`current_exec_metadata` (threaded from the training
loop via :mod:`uni_agent.sandbox.exec_context`) and forwards it onto the SDK's
``start_sandbox`` / ``exec_command`` / ``shell_exec`` when (a) the ambient bag
is non-empty and (b) the installed SDK method advertises a ``metadata``
keyword. A stale SDK (e.g. one whose ``shell_exec`` predates the metadata
kwarg) must be a silent no-op, never a ``TypeError`` on the agent's primary
command path.

These run with no real SDK / network: a fake async client records the kwargs
and a ``sys.modules``-injected stub stands in for the ``flash_sandbox`` package
on the ``start()`` path.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from uni_agent.sandbox.exec_context import (
    _EXEC_METADATA,
    bind_exec_metadata,
    current_exec_metadata,
    reset_exec_metadata,
    set_exec_metadata,
)
from uni_agent.sandbox.flash_sandbox import FlashSandboxSandbox


@pytest.fixture(autouse=True)
def _isolate_exec_metadata():
    """Start and end every test with an empty ambient bag.

    The provider reads the ambient contextvar, so a leaked non-empty bag from
    a prior test would spuriously mark every later client call with metadata.
    Pin the var to empty on both sides of each test for order-independent
    results (``reset(set(None))`` restores the *previous* value, which is the
    very leak this guard prevents).
    """
    token = set_exec_metadata(None)
    try:
        yield
    finally:
        reset_exec_metadata(token)
        if _EXEC_METADATA.get() not in (None, {}):
            set_exec_metadata(None)


class _ExecResult:
    def __init__(self, exit_code: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class _NoMetaClient:
    """SDK shape whose ``shell_exec`` predates the metadata kwarg (today's SDK)."""

    def __init__(self) -> None:
        self.start_calls: list[dict] = []
        self.exec_calls: list[dict] = []
        self.shell_calls: list[dict] = []
        self.write_calls: list[dict] = []

    async def start_sandbox(self, *, metadata=None, **kwargs: Any) -> Any:
        self.start_calls.append({"metadata": metadata, **kwargs})
        return types.SimpleNamespace(id="sbx-1")

    async def exec_command(self, sandbox_id, command, *, timeout_ms=None, metadata=None) -> _ExecResult:
        self.exec_calls.append({
            "sandbox_id": sandbox_id, "command": command,
            "timeout_ms": timeout_ms, "metadata": metadata,
        })
        return _ExecResult(0, "ok", "")

    async def shell_exec(self, sandbox_id, command, *, timeout_ms=None) -> _ExecResult:
        # Faithful to today's SDK: no ``metadata`` and no ``**kwargs``. The
        # provider must NOT forward metadata here (would raise TypeError).
        self.shell_calls.append({"sandbox_id": sandbox_id, "command": command, "timeout_ms": timeout_ms})
        return _ExecResult(0, "ok", "")

    async def write_file(self, sandbox_id, path, data, **kw: Any) -> None:
        self.write_calls.append({"sandbox_id": sandbox_id, "path": path})


class _WithMetaClient(_NoMetaClient):
    """SDK shape whose ``shell_exec`` now advertises a ``metadata`` kwarg."""

    async def shell_exec(self, sandbox_id, command, *, timeout_ms=None, metadata=None) -> _ExecResult:  # noqa: D401
        self.shell_calls.append({
            "sandbox_id": sandbox_id, "command": command,
            "timeout_ms": timeout_ms, "metadata": metadata,
        })
        return _ExecResult(0, "ok", "")


def _attach(sb: FlashSandboxSandbox, client: _NoMetaClient) -> FlashSandboxSandbox:
    sb._client = client
    sb._sandbox_id = "sbx-1"
    return sb


# --------------------------- _exec (one-shot /exec) ---------------------------


@pytest.mark.asyncio
async def test_exec_forwards_ambient_metadata():
    sb = _attach(FlashSandboxSandbox(image="x"), _NoMetaClient())
    async with bind_exec_metadata(
        group_id="g1",
        step=12,
        staleness=3,
        rlvr={"group": "g1", "rollout": "r1"},
        llm={"prompt_tokens": 10},
    ):
        await sb._exec(["echo", "hi"], timeout=5)
    call = sb._client.exec_calls[-1]  # type: ignore[attr-defined]
    assert call["metadata"] == {
        "group_id": "g1",
        "step": 12,
        "staleness": 3,
        "rlvr": {"group": "g1", "rollout": "r1"},
        "llm": {"prompt_tokens": 10},
    }
    assert call["timeout_ms"] == 5000


@pytest.mark.asyncio
async def test_exec_omits_metadata_when_ambient_bag_empty():
    sb = _attach(FlashSandboxSandbox(image="x"), _NoMetaClient())
    await sb._exec(["echo", "hi"], timeout=3)
    assert sb._client.exec_calls[-1]["metadata"] is None  # type: ignore[attr-defined]


# --------------------------- _shell_exec (persistent shell) ---------------------------


@pytest.mark.asyncio
async def test_shell_exec_skips_metadata_on_a_stale_sdk():
    """No metadata kwarg on the SDK -> forwarded kwargs omit it, no TypeError."""
    sb = _attach(FlashSandboxSandbox(image="x"), _NoMetaClient())
    async with bind_exec_metadata(rlvr={"group": "g1"}):
        res = await sb._shell_exec("echo hi", timeout_ms=1000)
    assert res.exit_code == 0
    call = sb._client.shell_calls[-1]  # type: ignore[attr-defined]
    assert "metadata" not in call
    assert call["timeout_ms"] == 1000


@pytest.mark.asyncio
async def test_shell_exec_forwards_metadata_when_sdk_advertises_it():
    sb = _attach(FlashSandboxSandbox(image="x"), _WithMetaClient())
    async with bind_exec_metadata(rlvr={"group": "g1", "terminal": True}):
        await sb._shell_exec("pytest", timeout_ms=60000)
    call = sb._client.shell_calls[-1]  # type: ignore[attr-defined]
    assert call["metadata"] == {"rlvr": {"group": "g1", "terminal": True}}


@pytest.mark.asyncio
async def test_shell_exec_omits_metadata_when_ambient_bag_empty_even_if_sdk_supports_it():
    sb = _attach(FlashSandboxSandbox(image="x"), _WithMetaClient())
    await sb._shell_exec("ls", timeout_ms=500)
    assert sb._client.shell_calls[-1]["metadata"] is None  # type: ignore[attr-defined]


# --------------------------- start() (create) ---------------------------


def _install_fake_flash_sandbox(monkeypatch, client: _NoMetaClient) -> None:
    """Inject a stub ``flash_sandbox`` package so ``start()`` can import it."""
    module = types.ModuleType("flash_sandbox")
    module.AsyncHTTPClient = lambda *args, **kwargs: client  # type: ignore[attr-defined]
    module.SandboxNotFoundError = type("SandboxNotFoundError", (Exception,), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "flash_sandbox", module)


@pytest.mark.asyncio
async def test_start_forwards_session_metadata_on_create(monkeypatch):
    client = _NoMetaClient()
    _install_fake_flash_sandbox(monkeypatch, client)
    sb = FlashSandboxSandbox(image="python:3.12", keep_alive=False)
    try:
        async with bind_exec_metadata(rlvr={"group": "g1", "rollout": "r1", "q": 8, "deadline_ms": 1800000}):
            await sb.start()
        assert client.start_calls, "start_sandbox must be called"
        assert client.start_calls[0]["metadata"] == {
            "rlvr": {"group": "g1", "rollout": "r1", "q": 8, "deadline_ms": 1800000}
        }
        assert sb._sandbox_id == "sbx-1"
    finally:
        await sb.stop()


@pytest.mark.asyncio
async def test_start_omits_metadata_when_ambient_bag_empty(monkeypatch):
    client = _NoMetaClient()
    _install_fake_flash_sandbox(monkeypatch, client)
    sb = FlashSandboxSandbox(image="python:3.12", keep_alive=False)
    try:
        await sb.start()
        assert client.start_calls[0]["metadata"] is None
    finally:
        await sb.stop()


# --------------------------- _accepts_kwarg caching ---------------------------


@pytest.mark.asyncio
async def test_accepts_kwarg_distinguishes_method_variants():
    from uni_agent.sandbox.flash_sandbox import _accepts_kwarg

    no_meta = _NoMetaClient()
    with_meta = _WithMetaClient()
    assert _accepts_kwarg(no_meta.exec_command, "metadata") is True
    assert _accepts_kwarg(no_meta.shell_exec, "metadata") is False
    assert _accepts_kwarg(with_meta.shell_exec, "metadata") is True
    # Cached on second read (same answer, no re-introspection cost).
    assert _accepts_kwarg(no_meta.shell_exec, "metadata") is False


def test_set_exec_metadata_none_exposes_empty_bag_and_cleans_up():
    # Defensive: a Ray worker that binds nothing (set_exec_metadata(None)) must
    # expose an empty bag and reset cleanly -- no leakage to a later test.
    token = set_exec_metadata(None)
    assert current_exec_metadata() == {}
    reset_exec_metadata(token)
    assert current_exec_metadata() == {}
