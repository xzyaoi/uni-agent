"""Run local Harbor tasks through the Harbor CLI."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from uni_agent import logging as execution_logging
from uni_agent.agents import AgentConfig

from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task
from .reward import task_result_from_harbor_trial

logger = logging.getLogger(__name__)
_TEMP_TRIALS_DIR = Path("/tmp/harbor-trials")
_PROCESS_TERMINATE_TIMEOUT_SECONDS = 5.0
_PROCESS_KILL_TIMEOUT_SECONDS = 5.0
_MODEL_BASE_URL_ENV_VARS = (
    "LLM_BASE_URL",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "HOSTED_VLLM_BASE_URL",
    "HOSTED_VLLM_API_BASE",
    "VLLM_API_BASE",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_BASE",
)
_MODEL_API_KEY_ENV_VARS = (
    "LLM_API_KEY",
    "OPENAI_API_KEY",
    "HOSTED_VLLM_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
)


class HarborAgentConfig(AgentConfig):
    """Harbor agent name and model endpoint."""

    name: str = Field(default="", description="Harbor built-in agent name or custom agent import path.")
    kwargs: dict[str, Any] = Field(default_factory=dict, description="Keyword arguments passed to the Harbor agent.")
    timeout_sec: float | None = Field(default=None, gt=0, description="Harbor agent execution timeout in seconds.")


class HarborTaskConfig(TaskConfig):
    """Configuration for one Harbor CLI trial, including gateway-backed training."""

    name: str = "harbor"
    sandbox: None = Field(default=None, exclude=True)
    harbor_env: str = Field(default="docker", description="Harbor environment backend.")
    harbor_executable: str = Field(default="harbor", min_length=1, description="Harbor executable on every worker.")
    artifacts_dir: str | None = Field(
        default=None, description="Persistent parent directory for Harbor trial artifacts."
    )
    agent: HarborAgentConfig = Field(default_factory=HarborAgentConfig)
    timeout_multiplier: float = Field(default=1.0, gt=0)
    override_cpus: int | None = Field(default=None, gt=0)
    override_memory_mb: int | None = Field(default=None, gt=0)

    @field_validator("agent", mode="before")
    @classmethod
    def _resolve_agent(cls, value: Any) -> Any:
        return value

    @field_validator("sandbox", mode="before")
    @classmethod
    def _reject_sandbox(cls, value: Any) -> None:
        if value is not None:
            raise ValueError("HarborTask does not use a Uni-Agent sandbox; configure harbor_env instead")
        return None

    @model_validator(mode="after")
    def _validate_local_trial(self) -> HarborTaskConfig:
        instance_id = self.metadata.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise ValueError("HarborTask metadata requires a non-empty instance_id")

        raw_task_path = self.metadata.get("task_path")
        if not isinstance(raw_task_path, str) or not raw_task_path:
            raise ValueError("HarborTask metadata requires an absolute task_path")
        task_path = Path(raw_task_path).expanduser()
        if not task_path.is_absolute():
            raise ValueError(f"Harbor task_path must be absolute, got {raw_task_path!r}")
        if not task_path.is_dir() or not (task_path / "task.toml").is_file():
            raise ValueError(f"Harbor task_path is not a task directory: {task_path}")

        if not self.harbor_env.strip():
            raise ValueError("HarborTask harbor_env must be non-empty")
        if not self.agent.name.strip():
            raise ValueError("Harbor agent.name must be non-empty")
        return self


def build_harbor_trial_command(
    config: HarborTaskConfig,
    *,
    trial_name: str,
    trials_dir: Path,
) -> list[str]:
    """Build the argv for one collision-safe Harbor trial."""
    task_path = str(config.metadata["task_path"])
    command = [
        config.harbor_executable,
        "trial",
        "start",
        "--path",
        task_path,
        "--trial-name",
        trial_name,
        "--trials-dir",
        str(trials_dir),
        "--agent",
        config.agent.name,
        "--env",
        config.harbor_env,
        "--timeout-multiplier",
        f"{config.timeout_multiplier:g}",
    ]

    if config.agent.name != "oracle" and config.agent.model.model_name:
        command.extend(["--model", config.agent.model.model_name])
    if config.agent.timeout_sec is not None:
        command.extend(["--agent-timeout", f"{config.agent.timeout_sec:g}"])
    if config.override_cpus is not None:
        command.extend(["--override-cpus", str(config.override_cpus)])
    if config.override_memory_mb is not None:
        command.extend(["--override-memory-mb", str(config.override_memory_mb)])
    for key, value in sorted(config.agent.kwargs.items()):
        try:
            encoded_value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Harbor agent kwarg {key!r} is not JSON serializable") from exc
        command.extend(["--agent-kwarg", f"{key}={encoded_value}"])
    return command


def build_harbor_process_env(config: HarborTaskConfig) -> dict[str, str] | None:
    """Expose the runtime model endpoint under compatible Harbor aliases."""
    if config.agent.name == "oracle":
        return None

    process_env: dict[str, str] = {}
    if config.agent.model.base_url:
        process_env.update(dict.fromkeys(_MODEL_BASE_URL_ENV_VARS, config.agent.model.base_url))
    if config.agent.model.api_key:
        process_env.update(dict.fromkeys(_MODEL_API_KEY_ENV_VARS, config.agent.model.api_key))
    return process_env or None


@dataclass(frozen=True)
class HarborCLIResult:
    exit_code: int
    stdout: str
    stderr: str


async def _terminate_harbor_process(process: asyncio.subprocess.Process, communication: asyncio.Task) -> None:
    """Stop the owned process group and bound draining inherited output pipes."""

    def signal_process(*, force: bool) -> None:
        try:
            if os.name == "posix":
                # Signal descendants even if the CLI itself has already exited.
                os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
            elif process.returncode is None:
                if force:
                    process.kill()
                else:
                    process.terminate()
        except ProcessLookupError:
            pass

    signal_process(force=False)
    await asyncio.wait({communication}, timeout=_PROCESS_TERMINATE_TIMEOUT_SECONDS)
    # A descendant can outlive the CLI and close its output pipes, so always
    # clean up the remaining group even when communicate has already finished.
    signal_process(force=True)
    done, _ = await asyncio.wait({communication}, timeout=_PROCESS_KILL_TIMEOUT_SECONDS)
    if not done:
        # A descendant may have escaped the group while retaining a pipe. There
        # is no public asyncio Process API to close its owned read transports.
        process._transport.close()
        communication.cancel()
    await asyncio.gather(communication, return_exceptions=True)


async def run_harbor_cli(command: list[str], *, env: dict[str, str] | None = None) -> HarborCLIResult:
    """Run Harbor directly on the host and capture its terminal output."""
    spawning = asyncio.create_task(
        asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **env} if env else None,
            start_new_session=os.name == "posix",
        )
    )
    communication = None
    try:
        # Keep process ownership and pipe draining intact when an outer episode
        # deadline cancels this coroutine, including while spawn is in flight.
        process = await asyncio.shield(spawning)
        communication = asyncio.create_task(process.communicate())
        stdout, stderr = await asyncio.shield(communication)
    except BaseException:

        async def cleanup() -> None:
            try:
                process = await spawning
            except Exception:
                return  # Spawn failed without yielding a process to own.
            await _terminate_harbor_process(
                process, communication if communication is not None else asyncio.create_task(process.communicate())
            )

        cleanup_task = asyncio.create_task(cleanup())
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                # A second cancellation must not abandon the owned processes.
                continue
        cleanup_task.result()
        raise
    return HarborCLIResult(
        exit_code=process.returncode or 0,
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
    )


@register_task("harbor")
class HarborTask(Task):
    name = "harbor"
    config_model = HarborTaskConfig

    async def run(self) -> TaskResult:
        config: HarborTaskConfig = self.config  # type: ignore[assignment]
        context_getter = getattr(execution_logging, "get_current_log_context", None)
        log_context = context_getter() if context_getter is not None else None
        output_dir: Path | None = None
        if log_context is not None and log_context.log_path:
            rollout_dir = Path(log_context.log_path).expanduser().resolve().parent
            output_dir = rollout_dir / "harbor"
            await asyncio.to_thread(rollout_dir.mkdir, parents=True, exist_ok=True)
            if output_dir.exists():
                raise RuntimeError(f"Harbor output directory already exists: {output_dir}")

        trial_name = str(uuid4().hex)
        if config.artifacts_dir:
            output_dir = Path(config.artifacts_dir).expanduser().resolve() / trial_name
        trial_dir = _TEMP_TRIALS_DIR / trial_name
        cancelled = False
        try:
            return await self._run_trial(
                config,
                _TEMP_TRIALS_DIR,
                trial_name=trial_name,
                output_dir=output_dir,
            )
        except asyncio.CancelledError:
            cancelled = True
            # Cancellation must not enter another potentially blocked shared
            # filesystem operation. Keep the partial trial for diagnosis.
            logger.warning("Harbor trial cancelled; retaining temporary artifacts at %s", trial_dir)
            raise
        finally:
            if not cancelled:
                await asyncio.to_thread(shutil.rmtree, trial_dir, ignore_errors=True)

    async def _run_trial(
        self,
        config: HarborTaskConfig,
        trials_dir: Path,
        *,
        trial_name: str,
        output_dir: Path | None,
    ) -> TaskResult:
        instance_id = str(config.metadata["instance_id"])
        trial_dir = trials_dir / trial_name
        command = build_harbor_trial_command(config, trial_name=trial_name, trials_dir=trials_dir)
        process_env = build_harbor_process_env(config)

        try:
            await asyncio.to_thread(trials_dir.mkdir, parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(f"failed to create Harbor trials directory {trials_dir}: {exc}") from exc

        logger.info(
            "starting Harbor trial (instance_id=%s, agent=%s, model=%s, environment=%s, trial=%s, trials_dir=%s)",
            instance_id,
            config.agent.name,
            None if config.agent.name == "oracle" else config.agent.model.model_name,
            config.harbor_env,
            trial_name,
            trials_dir,
        )
        started = time.perf_counter()
        try:
            response = await run_harbor_cli(command, env=process_env)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Harbor CLI executable was not found; install Harbor 0.16.0 or later and ensure `harbor` is on PATH"
            ) from exc
        elapsed = time.perf_counter() - started

        if output_dir is not None and trial_dir.is_dir():
            try:
                await asyncio.to_thread(shutil.move, str(trial_dir), str(output_dir))
            except OSError as exc:
                raise RuntimeError(f"failed to move Harbor trial output to {output_dir}: {exc}") from exc
            trial_dir = output_dir

        result_path = trial_dir / "result.json"
        try:
            result_bytes = await asyncio.to_thread(result_path.read_bytes)
        except FileNotFoundError as exc:
            detail = (response.stderr or response.stdout).strip()[-2000:]
            message = f"Harbor trial did not write {result_path}"
            if detail:
                message += f": {detail}"
            raise RuntimeError(message) from exc

        try:
            payload = json.loads(result_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Harbor wrote an invalid trial result at {result_path}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Harbor trial result at {result_path} is not a JSON object")
        if output_dir is not None:
            payload["trial_uri"] = trial_dir.resolve().as_uri()
            await asyncio.to_thread(
                result_path.write_text,
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        result = task_result_from_harbor_trial(
            payload,
            trial_dir=trial_dir,
            cli_exit_code=response.exit_code,
            stdout=response.stdout,
            stderr=response.stderr,
            elapsed=elapsed,
        )
        info = result.extra_info or {}
        if not info.get("eval_completed"):
            logger.warning(
                "Harbor trial incomplete for %s: %s",
                instance_id,
                (info.get("eval_report") or {}).get("error"),
            )
        logger.info(
            "Harbor trial done: instance_id=%s reward=%.3f resolved=%s elapsed=%.1fs",
            instance_id,
            float(result.reward),
            info.get("resolved"),
            elapsed,
        )
        return result
