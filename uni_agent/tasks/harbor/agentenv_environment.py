"""Harbor environment using native AgentENV, without scheduler overhead.

Tasks require prebuilt images and public networking. The CloudLab baseline
server has a fixed resource shape; overrides must match the actual server.
"""

from __future__ import annotations

import hashlib
import os
import shlex
from pathlib import Path
from typing import Any

from harbor.environments.base import BaseEnvironment, ExecResult

from uni_agent.sandbox.agentenv import AgentEnvSandbox


class AgentEnvEnvironment(BaseEnvironment):
    """Adapt Uni-Agent's SDK provider to Harbor's environment contract."""

    def __init__(
        self,
        *args: Any,
        runtime_timeout: float = 14400,
        api_key_env: str = "AGENTENV_API_KEY",
        endpoint: str | None = None,
        fixed_cpu_cores: int | None = None,
        fixed_memory_mb: int = 4096,
        start_timeout: float = 600.0,
        exec_timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        if fixed_cpu_cores is None:
            fixed_cpu_cores = int(os.environ.get("AGENTENV_FIXED_CPU_CORES", "4"))
        kwargs.setdefault("override_cpus", fixed_cpu_cores)
        kwargs.setdefault("override_memory_mb", fixed_memory_mb)
        super().__init__(*args, **kwargs)
        if self._effective_cpus != fixed_cpu_cores or self._effective_memory_mb != fixed_memory_mb:
            raise ValueError(
                "AgentENV cannot change the server's fixed resource shape: "
                f"requested {self._effective_cpus} CPUs/{self._effective_memory_mb} MiB, "
                f"server configured for {fixed_cpu_cores} CPUs/{fixed_memory_mb} MiB"
            )
        self._sandbox = AgentEnvSandbox(
            image=self.task_env_config.docker_image,
            runtime_timeout=runtime_timeout,
            api_key_env=api_key_env,
            endpoint=endpoint,
            start_timeout=start_timeout,
            # Harbor enforces its task/phase deadline outside environment.exec.
            # Without an explicit command deadline, do not impose a shorter
            # SDK default; the sandbox lifetime remains the RPC ceiling.
            exec_timeout=runtime_timeout if exec_timeout is None else exec_timeout,
            env_vars=self._startup_env(),
        )
        self._started = False

    @staticmethod
    def type() -> str:
        return "agentenv"

    @property
    def sandbox(self) -> AgentEnvSandbox:
        """Expose the SDK provider to the host-side Uni-Agent Harbor agent."""
        return self._sandbox

    def _validate_definition(self) -> None:
        if not self.task_env_config.docker_image:
            raise ValueError("AgentENV requires environment.docker_image; building task images is unsupported")
        if (self.environment_dir / "docker-compose.yaml").exists() or (
            self.environment_dir / "docker-compose.yml"
        ).exists():
            raise ValueError("AgentENV does not support Docker Compose task environments")

    async def start(self, force_build: bool = False) -> None:
        if force_build:
            raise ValueError("AgentENV uses prebuilt images and cannot force an image build")
        if self._started:
            return
        await self._sandbox.__aenter__()
        self._started = True
        try:
            result = await self.ensure_dirs(["/logs/agent", "/logs/verifier", "/logs/artifacts"])
            if result is not None and result.return_code != 0:
                raise RuntimeError(f"Unable to create Harbor log directories: {result.stderr or result.stdout}")
            await self._upload_environment_dir_after_start()
            await self._prepare_native_tools()
        except BaseException:
            await self.stop(delete=True)
            raise

    async def _prepare_native_tools(self) -> None:
        """Supply the upstream Verified Dockerfile's pinned uv tool on native images."""
        archive_name = os.environ.get("AGENTENV_UV_ARCHIVE")
        if not archive_name:
            return
        archive = Path(archive_name)
        expected = os.environ.get("AGENTENV_UV_ARCHIVE_SHA256")
        if not expected or hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
            raise ValueError("Native uv archive checksum is missing or incorrect")
        guest_archive = "/tmp/uni-agent-uv.tar.gz"
        await self._sandbox.upload(archive, guest_archive)
        command = (
            "mkdir -p /root/.local/bin && tar -xzf " + shlex.quote(guest_archive)
            + " --strip-components=1 -C /root/.local/bin"
            + " uv-x86_64-unknown-linux-gnu/uv uv-x86_64-unknown-linux-gnu/uvx"
            + " && rm -f " + shlex.quote(guest_archive)
            + " && /root/.local/bin/uv --version"
        )
        result = await self._sandbox.exec_shell(command, timeout=120)
        if result.exit_code != 0 or not result.stdout.strip().startswith("uv 0.7.13"):
            raise RuntimeError("Unable to prepare native Verified uv tool: " + result.stderr)

    async def stop(self, delete: bool = True) -> None:
        # Native AgentENV sandboxes are ephemeral; stop always deletes the instance.
        await self._sandbox.__aexit__(None, None, None)
        self._started = False

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        if not Path(source_path).is_file():
            raise ValueError(f"Upload source is not a file: {source_path}")
        await self._sandbox.upload(source_path, target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        if not Path(source_dir).is_dir():
            raise ValueError(f"Upload source is not a directory: {source_dir}")
        # The shared provider packs directories with tar and preserves modes.
        await self._sandbox.upload(source_dir, target_dir)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        if not await self.is_file(source_path, user="root"):
            raise FileNotFoundError(f"Download source is not a file: {source_path}")
        await self._sandbox.download(source_path, target_path)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        if not await self.is_dir(source_dir, user="root"):
            raise FileNotFoundError(f"Download source is not a directory: {source_dir}")
        # Harbor's transfer helper applies its safe extraction filter.
        await self.download_dir_with_exclusions(source_dir=source_dir, target_dir=target_dir, exclude=[])

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        effective_user = self._resolve_user(user)
        if effective_user not in (None, "root", "0", 0):
            raise ValueError(f"AgentENV cannot execute as user {effective_user!r}; only root is supported")
        result = await self._sandbox.exec_shell(
            command,
            workdir=cwd or self.task_env_config.workdir,
            env=self._merge_env(env),
            timeout=timeout_sec,
        )
        if result.exit_code == -1:
            raise TimeoutError(result.stderr or "AgentENV command timed out")
        return ExecResult(stdout=result.stdout, stderr=result.stderr, return_code=result.exit_code)
