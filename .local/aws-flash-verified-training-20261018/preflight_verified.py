"""Exercise one real Verified image and run-local environment before training."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
import uuid
from pathlib import Path

import pyarrow.parquet as pq
from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths
from uni_agent.tasks.harbor.verified_flash_environment import VerifiedFlashEnvironment


async def preflight(args: argparse.Namespace) -> dict:
    row = pq.read_table(args.dataset).slice(0, 1).to_pylist()[0]
    task_path = Path(row["extra_info"]["tools_kwargs"]["task"]["metadata"]["task_path"])
    task = Task(task_path)
    test_script = (task.paths.tests_dir / "test.sh").read_text()
    # Upstream generates parser.py at verifier execution; it is not a separate
    # task file. Validate the embedded parser without running the full tests.
    parser = re.search(r"cat > parser.py <<EOF\n(.*?)\nEOF", test_script, re.DOTALL)
    if parser is None or "uv run parser.py" not in test_script:
        raise ValueError("Unexpected official Verified verifier/parser contract")
    compile(parser.group(1), "official_verified_parser.py", "exec")
    session_id = "verified-preflight-" + uuid.uuid4().hex[:12]
    trial_dir = Path(args.artifacts_dir) / session_id
    trial_dir.mkdir(parents=True)
    environment = VerifiedFlashEnvironment(
        environment_dir=task.paths.environment_dir,
        environment_name=task.short_name,
        session_id=session_id,
        trial_paths=TrialPaths(trial_dir),
        task_env_config=task.config.environment,
        override_cpus=args.cpus,
        override_memory_mb=args.memory_mb,
        runtime_timeout=1200,
        start_timeout=900,
        exec_timeout=120,
    )
    receipt = {
        "task_path": str(task_path),
        "image": task.config.environment.docker_image,
        "endpoint": os.environ.get("FLASH_SANDBOX_ENDPOINT"),
        "backend": os.environ.get("FLASH_SANDBOX_TYPE"),
        "cpus": args.cpus,
        "memory_mb": args.memory_mb,
        "passed": False,
    }
    started = time.monotonic()
    try:
        await environment.start()
        receipt["start_seconds"] = round(time.monotonic() - started, 3)
        receipt["sandbox_id"] = environment.sandbox._sandbox_id
        await environment.sandbox.write_file("/tmp/verified-preflight", b"verified-ready\n")
        assert await environment.sandbox.read_file("/tmp/verified-preflight") == b"verified-ready\n"
        shell = await environment.sandbox.open_shell(cwd="/testbed")
        try:
            result = await shell.run("export VERIFIED_PREFLIGHT=ready")
            assert result.exit_code == 0, result
            result = await shell.run('printf "%s" "$VERIFIED_PREFLIGHT"')
            assert result.exit_code == 0 and result.stdout == "ready", result
        finally:
            await shell.close()
        await environment.upload_dir(task.paths.tests_dir, "/tests")
        result = await environment.exec(
            "set -e; test -d /testbed/.git; test -r /tests/test.sh; test -r /tests/config.json;"
            " bash -n /tests/test.sh; command -v uv; uv --version;"
            " test -r /opt/miniconda3/bin/activate;"
            " /opt/miniconda3/bin/python --version; printf 'VERIFIED_PREFLIGHT_OK\\n'",
            timeout_sec=120,
        )
        assert result.return_code == 0 and "VERIFIED_PREFLIGHT_OK" in result.stdout, result
        receipt["stdout"] = result.stdout
        receipt["official_verifier_shell_syntax"] = "passed"
        receipt["official_embedded_parser_syntax"] = "passed"
        # Check repeat startup does not create another VM or lose the current one.
        await environment.start()
        assert environment.sandbox._sandbox_id == receipt["sandbox_id"]
        receipt["passed"] = True
    finally:
        await environment.stop(delete=True)
        receipt["cleanup_requested"] = True
        receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
        if args.output:
            Path(args.output).write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=os.environ.get("TRAIN_FILE"), required=not os.environ.get("TRAIN_FILE"))
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--memory-mb", type=int, default=8192)
    parser.add_argument("--output")
    print(json.dumps(asyncio.run(preflight(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
