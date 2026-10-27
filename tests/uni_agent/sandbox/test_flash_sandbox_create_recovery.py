"""Stable create ownership across the framework's outer startup retry loop."""
import asyncio
import copy
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from uni_agent.sandbox import base
from uni_agent.sandbox import flash_sandbox as provider
from uni_agent.sandbox.exec_context import bind_exec_metadata, current_exec_metadata


@pytest.fixture
def harness(monkeypatch):
    state = SimpleNamespace(calls=[], clients=[], creates={}, deletes=[], lost_reply=True,
                            cancel=False, after_dispatch=None)

    class Client:
        def __init__(self, *args, **kwargs):
            self.closed = False
            self.timeout = kwargs['timeout']
            state.clients.append(self)

        async def start_sandbox(self, *, sandbox_id, operation_id, infrastructure_deadline_unix_ms,
                                request_timeout, **kwargs):
            payload = dict(sandbox_id=sandbox_id, operation_id=operation_id,
                           infrastructure_deadline_unix_ms=infrastructure_deadline_unix_ms,
                           request_timeout=request_timeout, **kwargs)
            state.calls.append(copy.deepcopy(payload))
            state.creates.setdefault(operation_id, sandbox_id)
            if state.after_dispatch:
                state.after_dispatch(payload)
            if state.cancel:
                raise asyncio.CancelledError()
            if state.lost_reply:
                state.lost_reply = False
                raise TimeoutError('reply lost after creation')
            return SimpleNamespace(id=state.creates[operation_id])

        async def stop_sandbox(self, sid, **kwargs):
            state.deletes.append(sid)

        async def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, 'flash_sandbox', SimpleNamespace(AsyncHTTPClient=Client))
    monkeypatch.setenv('OTELA_API', 'fixture')
    # Exercise the real retry/cleanup implementation without multi-second backoff.
    local_asyncio = SimpleNamespace(**vars(asyncio))
    local_asyncio.sleep = AsyncMock()
    monkeypatch.setattr(base, 'asyncio', local_asyncio)
    sandbox = provider.FlashSandboxSandbox(fenced_creates=True, keep_alive=False,
                                           start_timeout=60, env_vars={'TASK': 'original'})
    monkeypatch.setattr(sandbox, '_bootstrap', AsyncMock())
    return sandbox, state


@pytest.mark.asyncio
async def test_lost_reply_outer_retry_reuses_one_create_and_frozen_payload(harness):
    sandbox, state = harness

    def mutate(payload):
        if len(state.calls) == 1:
            payload['env']['TASK'] = 'sdk-mutated'
            current_exec_metadata()['rlvr']['group'] = 'later'
            sandbox.env_vars['TASK'] = 'later'

    state.after_dispatch = mutate
    async with bind_exec_metadata(rlvr={'group': 'original'}):
        await sandbox.__aenter__()
    assert len(state.clients) == 2 and state.clients[0].closed
    assert len(state.calls) == 2 and len(state.creates) == 1
    assert state.calls[0] == state.calls[1]
    assert state.calls[1]['metadata'] == {'rlvr': {'group': 'original'}}
    assert state.calls[1]['env'] == {'TASK': 'original'}
    assert state.calls[0]['request_timeout'] >= 65
    assert state.deletes == []
    assert sandbox.create_recovery['state'] == 'acknowledged'
    await sandbox.stop()
    assert state.deletes == [state.calls[0]['sandbox_id']]
    assert sandbox.create_recovery['state'] == 'closed'
    with pytest.raises(RuntimeError, match='cleanup'):
        await sandbox.start()
    assert len(state.calls) == 2


@pytest.mark.asyncio
async def test_expired_pending_start_never_renews_deadline_or_allocates(harness, monkeypatch):
    sandbox, state = harness
    clock = [1000.]
    monkeypatch.setattr(provider, 'time', SimpleNamespace(time=lambda: clock[0]))
    state.after_dispatch = lambda _: clock.__setitem__(0, 1061.)
    with pytest.raises(provider.FlashCreateRecoveryError, match='deadline expired') as failure:
        await sandbox.__aenter__()
    assert failure.value.operation_id == sandbox.create_recovery['operation_id']
    assert len(state.calls) == len(state.creates) == 1
    assert state.deletes == []
    assert sandbox.create_recovery['state'] == 'pending'
    assert sandbox.create_recovery['infrastructure_deadline_unix_ms'] == 1060000
    assert all(client.closed for client in state.clients)


@pytest.mark.asyncio
async def test_cancelled_start_retains_pending_ownership_without_delete(harness):
    sandbox, state = harness
    state.cancel = True
    with pytest.raises(asyncio.CancelledError):
        await sandbox.__aenter__()
    assert len(state.calls) == 1
    assert state.deletes == []  # even a 404 could not cancel queued creation
    assert sandbox.create_recovery['sandbox_id'] == state.calls[0]['sandbox_id']
    assert sandbox.create_recovery['operation_id'] == state.calls[0]['operation_id']
    assert sandbox.create_recovery['state'] == 'pending'
    assert state.clients[0].closed


@pytest.mark.asyncio
async def test_failed_bootstrap_does_not_retry_deleted_create_fence(harness, monkeypatch):
    sandbox, state = harness
    state.lost_reply = False
    monkeypatch.setattr(sandbox, '_bootstrap', AsyncMock(side_effect=RuntimeError('bootstrap failed')))
    with pytest.raises(RuntimeError, match='cleanup'):
        await sandbox.__aenter__()
    assert len(state.calls) == 1 and len(state.creates) == 1
    assert state.deletes == [state.calls[0]['sandbox_id']]
    assert sandbox.create_recovery['state'] == 'closed'


@pytest.mark.asyncio
async def test_fencing_rejects_sdk_without_operation_deadline_support(harness):
    sandbox, state = harness

    async def old_start(*, image):
        raise AssertionError('old SDK must never dispatch')

    client = sys.modules['flash_sandbox'].AsyncHTTPClient('fixture', timeout=100)
    client.start_sandbox = old_start
    sandbox._client = client
    with pytest.raises(RuntimeError, match='SDK'):
        await sandbox.start()
    assert state.calls == [] and sandbox.create_recovery is None
    await sandbox.stop()


@pytest.mark.asyncio
async def test_benchmark_client_marker_makes_fencing_mandatory(harness):
    sandbox, state = harness
    sandbox._fenced_creates = False
    sys.modules['flash_sandbox'].AsyncHTTPClient.requires_fenced_create = True
    state.lost_reply = False
    await sandbox.start()
    assert state.calls[0]['sandbox_id'] and state.calls[0]['operation_id']
    assert sandbox.create_recovery['state'] == 'acknowledged'
    await sandbox.stop()


@pytest.mark.asyncio
async def test_failed_delete_can_retry_exact_owned_id_without_create(harness, monkeypatch):
    sandbox, state = harness
    state.lost_reply = False
    await sandbox.start()
    sid = sandbox.create_recovery['sandbox_id']
    client = sandbox._client
    monkeypatch.setattr(client, 'stop_sandbox', AsyncMock(side_effect=OSError('delete response lost')))
    await sandbox.stop()
    assert sandbox.create_recovery['state'] == 'cleanup_pending'
    assert sandbox._sandbox_id == sid and sandbox._client is None
    await sandbox.stop()
    assert sandbox.create_recovery['state'] == 'closed'
    assert state.deletes == [sid]
    assert len(state.calls) == 1
