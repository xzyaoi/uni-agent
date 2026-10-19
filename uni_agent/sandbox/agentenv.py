"""Native AgentENV lifecycle and E2B data plane, without a scheduler.

Persistent shells reuse the experiment's proven E2B shell implementation in
``meta.baselines.clients.persistent_shell``. Make the flash-sandbox source root
available on PYTHONPATH when using ``stateful_shell``; one-shot commands and
file operations only require the ``e2b`` SDK and ``httpx``.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import quote
from uuid import uuid4

import httpx

from .agentenv_trace import NativeTrace, traced
from .base import ExecResult, Sandbox
from .registry import register_sandbox

if TYPE_CHECKING:
    from .base import SandboxConfig


class _NativeShell:
    def __init__(self, shell: Any, trace: NativeTrace):
        self._shell = shell
        self._trace = trace
        self._shell_id = uuid4().hex
        self._lock = asyncio.Lock()

    @traced("shell_run")
    async def run(self, command: str, *, timeout: float = 120.0) -> ExecResult:
        # The reference shell merges stderr into its captured output.
        async with self._lock:
            if self._shell._timeout_cleanup_confirmed:
                # Cleanup killed the old process tree. Reopen only for the next
                # command, never replay the command whose outcome was uncertain.
                self._shell = type(self._shell)(self._shell._sbx)
                await asyncio.to_thread(self._shell.start, timeout=30.0)
            code, output = await asyncio.to_thread(self._shell.run, command, timeout_s=timeout)
        return ExecResult(exit_code=code, stdout=output, stderr="")

    @traced("shell_close")
    async def close(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._shell.close)


@register_sandbox("agentenv")
class AgentEnvSandbox(Sandbox):
    """Create cold sandboxes directly on AgentENV and attach the native SDK."""

    supports_shell: ClassVar[bool] = True

    def __init__(
        self,
        *,
        image: str,
        runtime_timeout: float = 3600.0,
        endpoint: str | None = None,
        api_key_env: str = "AGENTENV_API_KEY",
        start_timeout: float = 600.0,
        request_timeout: float = 30.0,
        exec_timeout: float = 180.0,
        secure: bool | None = None,
        env_vars: dict[str, str] | None = None,
        guest_env_file: str | None = None,
    ):
        self.image = image
        self.runtime_timeout = runtime_timeout
        self.endpoint = (endpoint or os.getenv("AGENTENV_ENDPOINT", "http://127.0.0.1:8000")).rstrip("/")
        self.api_key_env = api_key_env
        self.start_timeout = start_timeout
        self.request_timeout = request_timeout
        self.exec_timeout = exec_timeout
        self.secure = secure if secure is not None else os.getenv("AGENTENV_SECURE", "false").lower() == "true"
        self.env_vars = dict(env_vars or {})
        if guest_env_file:
            with open(guest_env_file, encoding="utf-8") as stream:
                guest_env = json.load(stream)
            if not isinstance(guest_env, dict) or any(
                not isinstance(key, str) or not isinstance(value, str) for key, value in guest_env.items()
            ):
                raise ValueError("guest_env_file must contain a string-to-string JSON mapping")
            self.env_vars.update(guest_env)
        self._id: str | None = None
        self._create_attempted = False
        self._sdk: Any = None
        self._shells: list[_NativeShell] = []
        self._trace = NativeTrace(image)
        self._trace.emit(
            operation="configure", env_vars=self.env_vars, runtime_timeout=runtime_timeout, exec_timeout=exec_timeout
        )

    @classmethod
    def from_config(cls, config: SandboxConfig) -> AgentEnvSandbox:
        kwargs = dict(config.sandbox_kwargs)
        image = kwargs.pop("image", config.image)
        if not image:
            raise ValueError("agentenv requires a container image")
        return cls(image=image, runtime_timeout=config.runtime_timeout, **kwargs)

    async def __aenter__(self, retry: int = 1) -> AgentEnvSandbox:
        # A lost create response can hide a live sandbox ID. Never automatically
        # repeat a create; AgentENV's lifetime bounds such ambiguous orphans.
        await super().__aenter__(retry=1)
        return self

    def _headers(self) -> dict[str, str]:
        key = os.getenv(self.api_key_env, "")
        return {"X-API-Key": key, "Authorization": f"Bearer {key}"} if key else {}

    @traced("start")
    async def start(self) -> None:
        if self._create_attempted:
            raise RuntimeError("AgentENV sandbox creation was already attempted; construct a new provider")
        try:
            from e2b import Sandbox as E2BSandbox
        except ImportError as exc:
            raise ImportError("agentenv requires the optional e2b SDK") from exc

        self._create_attempted = True
        async with httpx.AsyncClient(timeout=self.start_timeout) as client:
            response = await client.post(
                f"{self.endpoint}/sandboxes-cold",
                headers=self._headers(),
                json={
                    "image": self.image,
                    "timeout": int(self.runtime_timeout),
                    "autoPause": False,
                    **({"secure": True} if self.secure else {}),
                    **({"envVars": self.env_vars} if self.env_vars else {}),
                },
            )
            response.raise_for_status()
            body = response.json()
        if isinstance(body, list) and body:
            body = body[0]
        sid = (body.get("sandboxID") or body.get("sandboxId")) if isinstance(body, dict) else None
        if not isinstance(sid, str) or not sid:
            raise RuntimeError("AgentENV create returned no sandbox ID")
        self._id = sid
        self._sdk = await asyncio.to_thread(
            E2BSandbox.connect,
            sid,
            api_url=self.endpoint,
            sandbox_url=self.endpoint,
            api_key=os.getenv(self.api_key_env, ""),
            headers={"E2b-Sandbox-Id": sid, "E2b-Sandbox-Port": "49983"},
            request_timeout=self.request_timeout,
        )
        # Cold create can return before envd accepts process RPCs. Readiness
        # probes are safe to repeat; user commands and creates are never retried.
        deadline = asyncio.get_running_loop().time() + 30
        while True:
            try:
                await self._exec(["true"], timeout=5)
                break
            except Exception:
                if asyncio.get_running_loop().time() >= deadline:
                    raise
                await asyncio.sleep(1)

    @traced("stop")
    async def stop(self) -> None:
        if self._id is None:
            return
        sid = self._id
        # Deletion kills guest processes, including native shell sessions.
        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            response = await client.delete(f"{self.endpoint}/sandboxes/{quote(sid, safe='')}", headers=self._headers())
            if response.status_code != 404:
                response.raise_for_status()
        self._shells.clear()
        self._sdk = None
        self._id = None

    def _attached(self) -> Any:
        if self._sdk is None:
            raise RuntimeError("AgentENV sandbox is not started")
        return self._sdk

    @traced("exec")
    async def _exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        duration = self.exec_timeout if timeout is None else timeout
        env = {**self.env_vars, **(env or {})}
        # E2B launches a login shell, whose profile can overwrite PATH from
        # envs. Apply explicit caller variables to the actual command afterward.
        command = (
            shlex.join(["env", "--", *[f"{key}={value}" for key, value in env.items()], *argv])
            if env
            else shlex.join(argv)
        )
        try:
            result = await asyncio.to_thread(
                self._attached().commands.run,
                command,
                timeout=duration,
                request_timeout=max(self.request_timeout, duration + 10),
                cwd=workdir,
                envs=env,
                user="root",
            )
        except Exception as exc:
            if type(exc).__name__ != "CommandExitException":
                raise
            result = exc
        return ExecResult(exit_code=result.exit_code, stdout=result.stdout or "", stderr=result.stderr or "")

    def _is_timeout_error(self, exc: BaseException) -> bool:
        return (
            super()._is_timeout_error(exc)
            or isinstance(exc, httpx.TimeoutException)
            or type(exc).__name__ == "TimeoutException"  # E2B command deadline.
        )

    async def is_alive(self) -> bool:
        try:
            return (await self._exec(["true"], timeout=5)).exit_code == 0
        except Exception:
            return False

    @traced("read_file")
    async def read_file(self, path: str) -> bytes:
        content = await asyncio.to_thread(
            self._attached().files.read, path, format="bytes", user="root", request_timeout=self.request_timeout
        )
        return bytes(content)

    @traced("write_file")
    async def write_file(self, path: str, content: bytes | str) -> None:
        await asyncio.to_thread(
            self._attached().files.write, path, content, user="root", request_timeout=self.request_timeout
        )

    @traced("open_shell")
    async def open_shell(self, *, cwd: str | None = None, env: dict[str, str] | None = None) -> _NativeShell:
        try:
            from meta.baselines.clients.persistent_shell import _AgentShell
        except ImportError as exc:
            raise ImportError("AgentENV native shells require the flash-sandbox source root on PYTHONPATH") from exc
        raw_shell = _AgentShell(self._attached())
        await asyncio.to_thread(raw_shell.start, timeout=self.request_timeout)
        shell = _NativeShell(raw_shell, self._trace)
        self._shells.append(shell)
        setup = [f"cd -- {shlex.quote(cwd)}"] if cwd else []
        for key, value in {**self.env_vars, **(env or {})}.items():
            # Quote the complete assignment; invalid variable names fail in bash.
            setup.append(f"export {shlex.quote(f'{key}={value}')}")
        if setup:
            result = await shell.run(" && ".join(setup), timeout=self.request_timeout)
            if result.exit_code != 0:
                await shell.close()
                raise RuntimeError(f"AgentENV shell setup failed: {result.stdout}")
        return shell
