#!/usr/bin/env python3
"""Apply the native Harbor deadline fix to a deployed checkout without replacing unrelated edits."""
import argparse
import ast
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('repo', type=Path)
parser.add_argument('--backup-dir', type=Path, required=True)
args = parser.parse_args()
args.backup_dir.mkdir(parents=True, exist_ok=True)
changes = {}
p = args.repo / 'uni_agent/tasks/harbor/agentenv_environment.py'
s = p.read_text()
s = s.replace('exec_timeout: float = 180.0,', 'exec_timeout: float | None = None,')
s = s.replace('            exec_timeout=exec_timeout,', '            exec_timeout=runtime_timeout if exec_timeout is None else exec_timeout,')
old = '        return ExecResult(stdout=result.stdout, stderr=result.stderr, return_code=result.exit_code)'
new = '        if result.exit_code == -1:\n            raise TimeoutError(result.stderr or "AgentENV command timed out")\n' + old
if new not in s:
    if s.count(old) != 1:
        raise RuntimeError('Unexpected native Harbor exec result path')
    s = s.replace(old, new)
if "import hashlib\n" not in s:
    s = s.replace("from pathlib import Path", "import hashlib\nimport os\nimport shlex\nfrom pathlib import Path", 1)
s = s.replace("fixed_cpu_cores: int = 4,", "fixed_cpu_cores: int | None = None,")
if 'fixed_cpu_cores = int(os.environ.get("AGENTENV_FIXED_CPU_CORES"' not in s:
    s = s.replace('        kwargs.setdefault("override_cpus", fixed_cpu_cores)',
                  '        if fixed_cpu_cores is None:\n            fixed_cpu_cores = int(os.environ.get("AGENTENV_FIXED_CPU_CORES", "4"))\n        kwargs.setdefault("override_cpus", fixed_cpu_cores)', 1)
if "    async def _prepare_native_tools" not in s:
    s = s.replace("            await self._upload_environment_dir_after_start()", "            await self._upload_environment_dir_after_start()\n            await self._prepare_native_tools()", 1)
    s = s.replace("    async def stop(self, delete: bool = True) -> None:", '    async def _prepare_native_tools(self) -> None:\n        """Supply the upstream Verified Dockerfile\'s pinned uv tool on native images."""\n        archive_name = os.environ.get("AGENTENV_UV_ARCHIVE")\n        if not archive_name:\n            return\n        archive = Path(archive_name)\n        expected = os.environ.get("AGENTENV_UV_ARCHIVE_SHA256")\n        if not expected or hashlib.sha256(archive.read_bytes()).hexdigest() != expected:\n            raise ValueError("Native uv archive checksum is missing or incorrect")\n        guest_archive = "/tmp/uni-agent-uv.tar.gz"\n        await self._sandbox.upload(archive, guest_archive)\n        command = (\n            "mkdir -p /root/.local/bin && tar -xzf " + shlex.quote(guest_archive)\n            + " --strip-components=1 -C /root/.local/bin"\n            + " uv-x86_64-unknown-linux-gnu/uv uv-x86_64-unknown-linux-gnu/uvx"\n            + " && rm -f " + shlex.quote(guest_archive)\n            + " && /root/.local/bin/uv --version"\n        )\n        result = await self._sandbox.exec_shell(command, timeout=120)\n        if result.exit_code != 0 or not result.stdout.strip().startswith("uv 0.7.13"):\n            raise RuntimeError("Unable to prepare native Verified uv tool: " + result.stderr)\n\n' + "    async def stop(self, delete: bool = True) -> None:", 1)
changes[p] = s
p = args.repo / 'uni_agent/sandbox/agentenv.py'
s = p.read_text()
if '    def _is_timeout_error(self, exc: BaseException)' not in s:
    marker = '    async def is_alive(self) -> bool:'
    if s.count(marker) != 1:
        raise RuntimeError('Unexpected native provider timeout path')
    s = s.replace(marker, '''    def _is_timeout_error(self, exc: BaseException) -> bool:
        return (
            super()._is_timeout_error(exc)
            or isinstance(exc, httpx.TimeoutException)
            or type(exc).__name__ == "TimeoutException"
        )

''' + marker)
changes[p] = s
for source in changes.values():
    ast.parse(source)
for path, source in changes.items():
    backup = args.backup_dir / path.name
    if not backup.exists():
        shutil.copyfile(path, backup)
    path.write_text(source)
    print('Updated', path)
