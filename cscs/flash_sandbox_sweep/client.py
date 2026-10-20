"""Sweep-scoped SDK deletion that waits for node cleanup and route removal."""
from flash_sandbox import AsyncHTTPClient, SandboxHTTPError, SandboxNotFoundError


class SynchronousDeleteClient(AsyncHTTPClient):
    async def stop_sandbox(self, sandbox_id, cleanup: bool = True, *, timeout: float | None = None) -> None:
        if isinstance(sandbox_id, str):
            sid = sandbox_id
        else:
            sid = getattr(sandbox_id, 'id', None)
            if sid is None:
                raise TypeError('Expected a sandbox ID or an object with an id')
            sid = str(sid)
        # Reuse the SDK's authenticated transport and timeout/error handling.
        # The deployed SDK has no async-stop option; its default DELETE returns
        # before the node is gone and leaves the coordinator's route behind.
        kwargs = {'params': {'cleanup': 'true' if cleanup else 'false', 'async': 'false'}}
        if timeout is not None:
            kwargs['timeout'] = timeout
        transport = getattr(self, '_transport', self)
        try:
            await transport._delete(f'sandboxes/{sid}', **kwargs)
        except SandboxHTTPError as error:
            # The coordinator's absent-route 404 is generic in this SDK;
            # owned cleanup must recognize it just like a worker's missing ID.
            if error.status_code != 404 or isinstance(error, SandboxNotFoundError):
                raise
            missing = SandboxNotFoundError(error.status_code, error.detail)
            # Preserve transport metadata across both supported SDK signatures.
            missing.__dict__.update(error.__dict__)
            raise missing from error
