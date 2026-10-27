"""A missing exec response must not launch overlapping copies of a command."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from uni_agent.sandbox import flash_sandbox as provider


class SandboxHTTPError(Exception):
    def __init__(self, status_code, *, code=None, origin="", outcome="uncertain"):
        super().__init__("response unavailable")
        self.status_code = status_code
        self.code = code
        self.origin = origin
        self.outcome = outcome


@pytest.fixture
def sandbox(monkeypatch):
    local_asyncio = SimpleNamespace(**vars(asyncio))
    local_asyncio.sleep = AsyncMock()
    monkeypatch.setattr(provider, "asyncio", local_asyncio)
    value = provider.FlashSandboxSandbox(keep_alive=False)
    value._sandbox_id = "verifier"
    value._http_retries = 2
    return value


async def command(sandbox, mode):
    if mode == "exec":
        return await sandbox._exec(["bash", "/tests/test.sh"], timeout=3000)
    return await sandbox._shell_exec("bash /tests/test.sh", timeout_ms=3000000)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["exec", "shell"])
async def test_gateway_timeout_cannot_launch_three_overlapping_verifiers(sandbox, mode):
    running = []

    async def dispatch(*args, **kwargs):
        running.append("verifier still running")
        if len(running) < 3:
            raise SandboxHTTPError(502)
        return SimpleNamespace(exit_code=0, stdout="late result", stderr="")

    sandbox._client = SimpleNamespace(exec_command=dispatch, shell_exec=dispatch)
    with pytest.raises(SandboxHTTPError):
        await command(sandbox, mode)
    assert len(running) == 1
    provider.asyncio.sleep.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["exec", "shell"])
@pytest.mark.parametrize("error", [
    SandboxHTTPError(400), SandboxHTTPError(502), SandboxHTTPError(503), SandboxHTTPError(504),
    SandboxHTTPError(503, code="execution_uncertain", origin="worker"),
    SandboxHTTPError(503, code="admission_cancelled", origin="worker", outcome="uncertain"),
    SandboxHTTPError(503, code="admission_cancelled", origin="gateway", outcome="not_dispatched"),
    OSError("response lost"), ConnectionResetError("reset after dispatch"), TimeoutError("response timed out"),
])
async def test_ambiguous_command_failure_is_not_retried(sandbox, mode, error):
    dispatch = AsyncMock(side_effect=error)
    sandbox._client = SimpleNamespace(exec_command=dispatch, shell_exec=dispatch)
    with pytest.raises(type(error)) as caught:
        await command(sandbox, mode)
    assert caught.value is error
    assert dispatch.await_count == 1
    provider.asyncio.sleep.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [408, 503])
@pytest.mark.parametrize("origin", ["coordinator", "worker"])
async def test_authoritative_admission_refusal_can_retry_before_dispatch(sandbox, status, origin):
    refusal = SandboxHTTPError(status, code="admission_cancelled", origin=origin, outcome="not_dispatched")
    result = SimpleNamespace(exit_code=0, stdout="one command", stderr="")
    dispatch = AsyncMock(side_effect=[refusal, result])
    sandbox._client = SimpleNamespace(exec_command=dispatch)
    value = await command(sandbox, "exec")
    assert value.stdout == "one command"
    assert dispatch.await_count == 2
    provider.asyncio.sleep.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [SandboxHTTPError(502), OSError("temporary connection failure")])
async def test_idempotent_operation_keeps_transient_retries(sandbox, error):
    operation = AsyncMock(side_effect=[error, b"file contents"])
    assert await sandbox._with_retry("read_file", operation, idempotent=True) == b"file contents"
    assert operation.await_count == 2


@pytest.mark.asyncio
async def test_cancellation_is_never_retried(sandbox):
    operation = AsyncMock(side_effect=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await sandbox._with_retry("exec", operation, idempotent=False)
    assert operation.await_count == 1
