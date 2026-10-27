"""Existing Flash adapter with explicit Firecracker shape and owned-ID tracing."""
import hashlib
import inspect
import json
import os
import time
from pathlib import Path

import flash_sandbox

from uni_agent.tasks.harbor.agentenv_environment import AgentEnvEnvironment
from uni_agent.tasks.harbor.flash_environment import FlashSandboxEnvironment

from .client import SynchronousDeleteClient


def record_owned(sandbox_id: str, endpoint: str, event: str = 'sandbox_created', *,
                 operation_id: str | None = None, infrastructure_deadline_unix_ms: int | None = None) -> None:
    trace = Path(os.environ['FLASH_SWEEP_TRACE_DIR'])
    trace.mkdir(parents=True, exist_ok=True)
    row = {'event': event, 'timestamp': time.time(),
           'run_id': os.environ['FLASH_SWEEP_RUN_ID'], 'sandbox_id': sandbox_id,
           'backend': 'firecracker', 'vcpu_count': 1, 'cpu_cores': 1, 'memory_mb': 4096,
           'endpoint_sha256': hashlib.sha256(endpoint.rstrip('/').encode()).hexdigest()}
    if operation_id is not None:
        row['operation_id'] = operation_id
        row['infrastructure_deadline_unix_ms'] = infrastructure_deadline_unix_ms
    descriptor = os.open(trace / f'owned-{os.getpid()}.jsonl', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, 'a') as stream:
        stream.write(json.dumps(row) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


class OwnedShapeClient(SynchronousDeleteClient):
    requires_fenced_create = True

    def __init__(self, endpoint, *args, **kwargs):
        self._sweep_endpoint = endpoint
        super().__init__(endpoint, *args, **kwargs)

    async def start_sandbox(self, **kwargs):
        if (kwargs.get('type'), kwargs.get('cpu_cores'), kwargs.get('memory_mb')) != ('firecracker', 1, 4096):
            raise ValueError('Matched Flash comparison requires Firecracker, 1 vCPU and 4096 MiB')
        kwargs['vcpu_count'] = 1  # Guest topology is distinct from the cpu_cores quota.
        # The provider owns this identity across client recreation and outer
        # retries. Never replace it with a new wrapper-generated sandbox ID.
        fields = ('sandbox_id', 'operation_id', 'infrastructure_deadline_unix_ms')
        claimed_id, operation_id = kwargs.get('sandbox_id'), kwargs.get('operation_id')
        deadline = kwargs.get('infrastructure_deadline_unix_ms')
        if (not isinstance(claimed_id, str) or not claimed_id
                or not isinstance(operation_id, str) or not operation_id
                or type(deadline) is not int or deadline <= 0):
            raise ValueError('Matched Flash creation requires a stable sandbox ID, operation ID and absolute deadline')
        parameters = inspect.signature(super().start_sandbox).parameters
        if not all(name in parameters for name in fields):
            raise RuntimeError('Matched Flash creation requires an SDK with explicit operation/deadline support')
        ownership = dict(operation_id=operation_id, infrastructure_deadline_unix_ms=deadline)
        record_owned(claimed_id, self._sweep_endpoint, event='sandbox_claimed', **ownership)
        result = await super().start_sandbox(**kwargs)
        returned_id = str(getattr(result, 'id', result))
        if returned_id != claimed_id:
            # A mismatched response cannot authorize deletion of an unrelated
            # sandbox. Keep the original claim for reconciliation only.
            record_owned(claimed_id, self._sweep_endpoint, event='sandbox_create_identity_mismatch', **ownership)
            raise RuntimeError('Flash create returned an ID different from the claimed ID')
        record_owned(claimed_id, self._sweep_endpoint, **ownership)
        return result


class OwnedFlashSandboxEnvironment(FlashSandboxEnvironment):
    async def start(self, force_build: bool = False) -> None:
        if self._started:
            return
        # The provider imports AsyncHTTPClient inside start and has no factory
        # hook. Harbor runs each trial in its own subprocess; scope this SDK
        # substitution to startup and restore it even on cancellation/failure.
        original = flash_sandbox.AsyncHTTPClient
        flash_sandbox.AsyncHTTPClient = OwnedShapeClient
        try:
            await super().start(force_build=force_build)
        finally:
            flash_sandbox.AsyncHTTPClient = original
        try:
            # Pure pinned-tool bootstrap, with no AgentENV client/API access.
            await AgentEnvEnvironment._prepare_native_tools(self)
        except BaseException:
            await self.stop(delete=True)
            raise
