"""Existing Flash adapter with explicit Firecracker shape and owned-ID tracing."""
import hashlib
import json
import os
import time
from pathlib import Path
from uuid import uuid4

import flash_sandbox

from uni_agent.tasks.harbor.agentenv_environment import AgentEnvEnvironment
from uni_agent.tasks.harbor.flash_environment import FlashSandboxEnvironment

from .client import SynchronousDeleteClient


def record_owned(sandbox_id: str, endpoint: str, event: str = 'sandbox_created') -> None:
    trace = Path(os.environ['FLASH_SWEEP_TRACE_DIR'])
    trace.mkdir(parents=True, exist_ok=True)
    row = {'event': event, 'timestamp': time.time(),
           'run_id': os.environ['FLASH_SWEEP_RUN_ID'], 'sandbox_id': sandbox_id,
           'backend': 'firecracker', 'vcpu_count': 1, 'cpu_cores': 1, 'memory_mb': 4096,
           'endpoint_sha256': hashlib.sha256(endpoint.rstrip('/').encode()).hexdigest()}
    descriptor = os.open(trace / f'owned-{os.getpid()}.jsonl', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, 'a') as stream:
        stream.write(json.dumps(row) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


class OwnedShapeClient(SynchronousDeleteClient):
    def __init__(self, endpoint, *args, **kwargs):
        self._sweep_endpoint = endpoint
        super().__init__(endpoint, *args, **kwargs)

    async def start_sandbox(self, **kwargs):
        if (kwargs.get('type'), kwargs.get('cpu_cores'), kwargs.get('memory_mb')) != ('firecracker', 1, 4096):
            raise ValueError('Matched Flash comparison requires Firecracker, 1 vCPU and 4096 MiB')
        kwargs['vcpu_count'] = 1  # Guest topology is distinct from the cpu_cores quota.
        # Claim and fsync a client-generated ID before the HTTP request. A lost
        # response or cancellation can still leave a created VM on the server.
        claimed_id = str(uuid4())
        kwargs['sandbox_id'] = claimed_id
        record_owned(claimed_id, self._sweep_endpoint, event='sandbox_claimed')
        result = await super().start_sandbox(**kwargs)
        returned_id = str(getattr(result, 'id', result))
        if returned_id != claimed_id:
            # The create response authorizes this additional owned ID; retain
            # both IDs in the trace even if immediate deletion fails.
            record_owned(returned_id, self._sweep_endpoint)
            await self.stop_sandbox(returned_id, cleanup=True, timeout=120)
            raise RuntimeError('Flash create returned an ID different from the claimed ID')
        record_owned(claimed_id, self._sweep_endpoint)
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
