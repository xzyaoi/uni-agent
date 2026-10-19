"""Bootstrap lifecycle checks requiring no Harbor installation or remote VM."""

import hashlib
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


class ParentEnvironment:
    async def start(self, force_build=False):
        if force_build:
            raise ValueError("force_build unsupported")
        self.parent_starts += 1
        self._started = True

    async def stop(self, delete=True):
        self.stops.append(delete)
        self._started = False


stub = types.ModuleType("uni_agent.tasks.harbor.flash_environment")
stub.FlashSandboxEnvironment = ParentEnvironment
spec = importlib.util.spec_from_file_location(
    "verified_flash_environment_test_target", Path(__file__).with_name("verified_flash_environment.py")
)
module = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {stub.__name__: stub}):
    spec.loader.exec_module(module)


class BootstrapTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.archive = Path(self.tmp.name) / "uv.tar.gz"
        self.archive.write_bytes(b"test archive bytes")
        self.digest = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.env = module.VerifiedFlashEnvironment()
        self.env._started = False
        self.env.parent_starts = 0
        self.env.stops = []
        self.env.sandbox = types.SimpleNamespace(
            upload=AsyncMock(),
            exec_shell=AsyncMock(return_value=types.SimpleNamespace(exit_code=0, stdout="uv 0.7.13\n", stderr="")),
        )
        self.addCleanup(patch.stopall)
        patch.object(module, "UV_SHA256", self.digest).start()
        patch.dict(os.environ, {
            "FLASH_SANDBOX_UV_ARCHIVE": str(self.archive),
            "FLASH_SANDBOX_UV_ARCHIVE_SHA256": self.digest,
        }).start()

    async def test_bootstrap_once_and_repeated_start_is_idempotent(self):
        await self.env.start()
        await self.env.start()
        self.assertEqual(self.env.parent_starts, 1)
        self.env.sandbox.upload.assert_awaited_once_with(self.archive, "/tmp/uni-agent-uv.tar.gz")
        self.env.sandbox.exec_shell.assert_awaited_once()
        self.assertTrue(self.env._started)

    async def test_missing_configuration_does_not_start_or_execute(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "pinned UV"):
            await self.env.start()
        self.assertEqual(self.env.parent_starts, 0)
        self.env.sandbox.exec_shell.assert_not_awaited()

    async def test_corrupt_archive_does_not_start_or_execute(self):
        self.archive.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            await self.env.start()
        self.assertEqual(self.env.parent_starts, 0)
        self.env.sandbox.upload.assert_not_awaited()
        self.env.sandbox.exec_shell.assert_not_awaited()

    async def test_noncanonical_expected_hash_is_rejected(self):
        with patch.dict(os.environ, {"FLASH_SANDBOX_UV_ARCHIVE_SHA256": "arbitrary"}):
            with self.assertRaisesRegex(ValueError, "pinned UV"):
                await self.env.start()
        self.assertEqual(self.env.parent_starts, 0)

    async def test_failed_install_deletes_started_sandbox(self):
        self.env.sandbox.exec_shell.return_value.exit_code = 1
        with self.assertRaisesRegex(RuntimeError, "prepare Verified UV"):
            await self.env.start()
        self.assertEqual(self.env.stops, [True])
        self.assertFalse(self.env._started)

    async def test_wrong_uv_version_deletes_started_sandbox(self):
        self.env.sandbox.exec_shell.return_value.stdout = "uv 0.7.130\n"
        with self.assertRaisesRegex(RuntimeError, "prepare Verified UV"):
            await self.env.start()
        self.assertEqual(self.env.stops, [True])

    async def test_upload_failure_deletes_started_sandbox(self):
        self.env.sandbox.upload.side_effect = OSError("upload failed")
        with self.assertRaisesRegex(OSError, "upload failed"):
            await self.env.start()
        self.assertEqual(self.env.stops, [True])
        self.env.sandbox.exec_shell.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
