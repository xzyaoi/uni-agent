"""Run-local Flash environment supplying the official Verified verifier's UV."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from uni_agent.tasks.harbor.flash_environment import FlashSandboxEnvironment

UV_VERSION = "0.7.13"
UV_SHA256 = "909278eb197c5ed0e9b5f16317d1255270d1f9ea4196e7179ce934d48c4c2545"


class VerifiedFlashEnvironment(FlashSandboxEnvironment):
    """Install a locally verified, pinned UV archive before agent execution."""

    async def start(self, force_build: bool = False) -> None:
        if self._started and not force_build:
            return
        archive_name = os.environ.get("FLASH_SANDBOX_UV_ARCHIVE")
        expected = os.environ.get("FLASH_SANDBOX_UV_ARCHIVE_SHA256")
        if not archive_name or expected != UV_SHA256:
            raise ValueError("Verified Flash requires its pinned UV archive and SHA256")
        archive = Path(archive_name)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
            raise ValueError("Verified Flash UV archive checksum is incorrect")
        await super().start(force_build=force_build)
        try:
            guest_archive = "/tmp/uni-agent-uv.tar.gz"
            await self.sandbox.upload(archive, guest_archive)
            result = await self.sandbox.exec_shell(
                "mkdir -p /root/.local/bin"
                " && tar -xzf /tmp/uni-agent-uv.tar.gz --strip-components=1 -C /root/.local/bin"
                " uv-x86_64-unknown-linux-gnu/uv uv-x86_64-unknown-linux-gnu/uvx"
                " && rm -f /tmp/uni-agent-uv.tar.gz"
                " && /root/.local/bin/uv --version",
                timeout=120,
            )
            if result.exit_code != 0 or result.stdout.strip().split()[:2] != ["uv", UV_VERSION]:
                raise RuntimeError("Unable to prepare Verified UV tool: " + (result.stderr or result.stdout))
        except BaseException as error:
            try:
                await self.stop(delete=True)
            except BaseException as cleanup_error:
                error.add_note(f"Verified Flash cleanup also failed: {cleanup_error}")
            raise
