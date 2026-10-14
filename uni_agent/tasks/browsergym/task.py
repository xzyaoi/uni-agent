"""BrowserGym task lifecycle: sandbox + env daemon + agent + env-sourced reward.

Per-episode flow (mirrors ``terminal_bench``, with an env where Terminal-Bench
has a filesystem):

1. build the sandbox config (provider-dispatched; flash-sandbox is the target),
2. boot the bridge daemon inside the sandbox (headless Chromium + BrowserEnv),
3. fetch the episode opening (goal + first AXTree) from the daemon,
4. run the configured agent (its ``browser`` tool drives the daemon),
5. score from the daemon's own episode result -- never the agent's claim.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
from typing import Any

from pydantic import Field

from uni_agent.sandbox import SandboxConfig, build_sandbox

from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, build_messages
from .reward import compute_reward

logger = logging.getLogger(__name__)

#: Image used when neither the task-config YAML nor the sample row carries one.
#: The SandboxConfig default ("python:3.12") has no browsergym/chromium, so the
#: task replaces it; set ``BROWSERGYM_IMAGE`` to point at your GHCR image.
DEFAULT_IMAGE = "ghcr.io/uni-agent/browsergym:latest"
#: Digest-pinned production image. Providers cache mutable tags PER NODE and
#: serve stale bytes after a re-push (observed: 0.1.4 watchdog missing under
#: :latest on live jobs while GHCR's latest/0.1.4 digests matched). Code-level
#: pin because the env path (sbatch -> wrapper -> ray runtime-env yaml) proved
#: fragile. BROWSERGYM_FORCE_IMAGE (when it actually arrives) still overrides.
_PINNED_IMAGE = "ghcr.io/xzyaoi/browsergym@sha256:2b2a437db3f08f36062a9d8d54c3af5055a5d745e351a54e05c33c63737b7ffa"
_SANDBOX_DEFAULT_IMAGE = "python:3.12"  # SandboxConfig.image default sentinel

LOG_DIR = "/logs/bgym"
RESULT_FILE = f"{LOG_DIR}/result.json"


class BrowserGymTaskConfig(TaskConfig):
    """BrowserGym episode knobs; the sample lives in :attr:`metadata`."""

    name: str = "browsergym"
    bridge_path: str = Field(default="/opt/bgym/bridge.py", description="Bridge script inside the sandbox image.")
    bridge_port: int = Field(default=8027, description="Sandbox-local port of the bridge daemon.")
    python_bin: str = Field(default="python", description="Python binary inside the sandbox image.")
    daemon_timeout: float = Field(
        default=240.0,
        description="Budget (s) for booting the daemon incl. the first Chromium start.",
    )
    boot_attempts: int = Field(
        default=3,
        description=(
            "Daemon boot + warmup attempts. Each attempt relaunches the daemon with the "
            "same seed; ~15% of libkrun daemons die on the first page interaction "
            "(native chromium crash), which the warmup converts to a boot-time retry."
        ),
    )


def build_browsergym_sandbox_config(config: TaskConfig, metadata: dict[str, Any]) -> SandboxConfig:
    """Apply BrowserGym defaults (image, resources, timeouts) per provider."""
    sandbox_config = config.sandbox.model_copy(deep=True)
    timeout_s = float(metadata.get("timeout_s", 300.0))
    daemon_timeout = float(getattr(config, "daemon_timeout", 240.0))
    # Always leave enough lease for daemon boot + the episode + teardown margin;
    # a shorter user-set TTL than that can only truncate the episode.
    sandbox_config.runtime_timeout = max(sandbox_config.runtime_timeout, timeout_s + daemon_timeout + 300.0)

    if sandbox_config.image == _SANDBOX_DEFAULT_IMAGE:
        sandbox_config.image = os.environ.get("BROWSERGYM_IMAGE", DEFAULT_IMAGE)
        logger.info("browsergym: sandbox image defaulted to %s", sandbox_config.image)
    # Digest pinning (see _PINNED_IMAGE): env override first, code pin as the
    # reliable floor. Overrides parquet rows too.
    forced = os.environ.get("BROWSERGYM_FORCE_IMAGE", "").strip() or _PINNED_IMAGE
    if forced:
        if sandbox_config.image != forced:
            logger.info("browsergym: image forced %s -> %s", sandbox_config.image, forced)
        sandbox_config.image = forced

    kwargs = dict(sandbox_config.sandbox_kwargs)
    if sandbox_config.provider == "flash_sandbox":
        # Browsers want a bit more than a shell task: 2 cores / 4 GiB is a
        # comfortable default for headless Chromium on MiniWoB-sized pages.
        kwargs.setdefault("cpu_cores", 2.0)
        kwargs.setdefault("memory_mb", 4096)
        kwargs.setdefault("keep_alive", True)
    elif sandbox_config.provider == "docker":
        run_args = list(kwargs.get("run_args", []))
        run_args.extend(["--platform", "linux/amd64", "--shm-size", "1g"])
        kwargs["run_args"] = run_args
    elif sandbox_config.provider == "agentenv":
        kwargs.setdefault("start_timeout", 600.0)
    elif sandbox_config.provider == "modal":
        kwargs.setdefault("cpu", 2.0)
        kwargs.setdefault("memory", 4096)
    else:
        raise ValueError(
            f"BrowserGym supports sandbox providers 'docker', 'modal', 'agentenv' and 'flash_sandbox', "
            f"got {sandbox_config.provider!r}"
        )
    sandbox_config.sandbox_kwargs = kwargs
    return sandbox_config


async def _start_bridge_daemon(sandbox, config: BrowserGymTaskConfig, metadata: dict[str, Any]) -> None:
    """Boot ``bridge.py serve`` detached inside the sandbox and wait until ready."""
    env_kwargs = json.loads(metadata.get("env_kwargs_json") or "{}")
    # setsid is load-bearing: without it the daemon shares the flash-envd
    # session shell's process group (nohup only ignores SIGHUP), so any
    # session-side group sweep -- session teardown, the 0.2s-post-shutdown
    # sweep observed in production, or session-init failures -- kills the
    # daemon+Chromium tree with it (mid-episode "bridge call failed").
    # setsid gives the daemon its own session+process group.
    # NOTE: statement-per-line matters here -- `mkdir && nohup ... & echo` would
    # background the whole `mkdir && nohup` chain (shell precedence), racing the
    # pid-file redirect against mkdir. Sequential statements guarantee the log
    # dir exists before the daemon spawns and before the pid file is written.
    serve_cmd = (
        f"mkdir -p {LOG_DIR}\n"
        f"setsid nohup {shlex.quote(config.python_bin)} {shlex.quote(config.bridge_path)} serve"
        f" --task {shlex.quote(str(metadata['task_id']))}"
        f" --port {config.bridge_port}"
        f" --obs-max-chars {int(metadata.get('obs_max_chars', 30000))}"
        f" --result-file {shlex.quote(RESULT_FILE)}"
        + (f" --seed {int(metadata['seed'])}" if metadata.get("seed") is not None else "")
        + (f" --slow-mo {float(env_kwargs['slow_mo']):g}" if env_kwargs.get("slow_mo") else "")
        + (f" --viewport {shlex.quote(str(env_kwargs['viewport']))}" if env_kwargs.get("viewport") else "")
        + f" > {LOG_DIR}/stdout.log 2>&1 &\n"
        f"echo $! > {LOG_DIR}/daemon.pid || true\n"
        f"echo bridge-launching"
    )
    res = await sandbox.exec_shell(serve_cmd, timeout=30.0)
    if res.exit_code != 0:
        raise RuntimeError(f"failed to launch browsergym bridge daemon: {res.stderr.strip()[-2000:]}")

    wait = await sandbox.exec(
        [
            config.python_bin,
            config.bridge_path,
            "wait",
            "--port",
            str(config.bridge_port),
            "--timeout",
            str(config.daemon_timeout),
        ],
        timeout=config.daemon_timeout + 60.0,
    )
    if wait.exit_code != 0:
        logs = await sandbox.exec_shell(f"tail -n 80 {LOG_DIR}/stdout.log 2>/dev/null", timeout=15.0)
        raise RuntimeError(
            "browsergym bridge daemon not ready within "
            f"{config.daemon_timeout:g}s\nbridge log tail:\n{logs.stdout[-4000:]}"
        )


async def _bridge_call(
    sandbox, config: BrowserGymTaskConfig, command: str, *, extra: list[str] | None = None, timeout: float = 60.0
) -> dict:
    res = await sandbox.exec(
        [config.python_bin, config.bridge_path, command, *(extra or []), "--port", str(config.bridge_port)],
        timeout=timeout,
    )
    if res.exit_code != 0:
        raise RuntimeError(f"bridge {command!r} failed: {(res.stderr or res.stdout).strip()[-2000:]}")
    return json.loads(res.stdout)


@register_task("browsergym")
class BrowserGymTask(Task):
    name = "browsergym"
    config_model = BrowserGymTaskConfig

    async def run(self) -> TaskResult:
        cfg: BrowserGymTaskConfig = self.config  # type: ignore[assignment]
        metadata = cfg.metadata
        for key in ("task_id", "benchmark"):
            if not metadata.get(key):
                raise ValueError(f"browsergym metadata requires {key!r}")
        timeout_s = float(metadata.get("timeout_s", 300.0))
        sandbox_config = build_browsergym_sandbox_config(cfg, metadata)
        logger.info(
            "starting BrowserGym task (task_id=%s benchmark=%s seed=%s) timeout=%gs\nsandbox config: %s",
            metadata["task_id"],
            metadata["benchmark"],
            metadata.get("seed"),
            timeout_s,
            json.dumps(sandbox_config.model_dump(mode="json"), indent=2),
        )

        async with build_sandbox(sandbox_config) as sandbox:
            info = await self._boot_env_with_warmup(sandbox, cfg, metadata)
            messages = build_messages(
                goal=str(info.get("goal") or ""),
                obs=str(info.get("obs_text") or ""),
                url=str(info.get("url") or ""),
            )
            logger.info("BrowserGym episode prompt:\n%s", messages[1]["content"])

            agent = self.build_agent()
            try:
                agent_result = await asyncio.wait_for(
                    agent.run(sandbox=sandbox, messages=messages),
                    timeout=timeout_s,
                )
                agent_info: dict[str, Any] = {"timed_out": False, "error": None, **agent_result.info}
                finished = agent_result.finished
            except TimeoutError:
                logger.warning("BrowserGym agent timed out after %.0fs", timeout_s)
                agent_info = {"timed_out": True, "error": f"agent exceeded {timeout_s:g}s"}
                finished = False
            except Exception as exc:  # still score the env's own state
                logger.exception("BrowserGym agent failed; continuing to reward")
                agent_info = {"timed_out": False, "error": f"{type(exc).__name__}: {exc}"}
                finished = False

            result = await compute_reward(
                sandbox,
                python_bin=cfg.python_bin,
                bridge_path=cfg.bridge_path,
                bridge_port=cfg.bridge_port,
                result_file=RESULT_FILE,
            )
            result["agent"] = agent_info
            # Best-effort daemon teardown (the sandbox dies right after anyway).
            try:
                await _bridge_call(sandbox, cfg, "shutdown", timeout=15.0)
            except Exception:
                logger.debug("bridge shutdown skipped", exc_info=True)

        score = float(result["reward"])
        logger.info(
            "BrowserGym task done: task_id=%s reward=%.3f success=%s steps=%s",
            metadata["task_id"],
            score,
            result.get("success"),
            result.get("n_steps"),
        )
        return TaskResult(
            reward=score,
            accuracy=score,
            finished=finished,
            extra_info=result,
        )

    async def _boot_env_with_warmup(
        self, sandbox, cfg: BrowserGymTaskConfig, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Boot the daemon, warm it with a throwaway step, retry on early death.

        Measured on libkrun (job 3607020): ~15% of daemons die on the FIRST
        real page interaction (chromium fork/native crash under the microVM:
        the same restriction class that forced ``--no-zygote``). After that
        first step they live for minutes. A ``noop()`` warmup moves the death
        window BEFORE the agent sees the episode: a daemon that dies here is
        relaunched with the same seed (env.reset is deterministic), so the
        agent always starts against a warm, surviving daemon. The warmup step
        is a literal no-op for the env (it may tick MiniWoB timers once and
        bump ``n_steps`` by one).
        """
        attempts = int(getattr(cfg, "boot_attempts", 3))
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            if attempt > 1:
                # clear a possibly-hung previous daemon before rebinding the port
                await sandbox.exec_shell(f"kill $(cat {LOG_DIR}/daemon.pid) 2>/dev/null; sleep 1; true", timeout=15.0)
            try:
                await _start_bridge_daemon(sandbox, cfg, metadata)
                warm = await _bridge_call(sandbox, cfg, "step", extra=["--action", "noop(wait_ms=100)"], timeout=90.0)
                if not warm.get("ok"):
                    # Daemon alive; the env merely rejected the warmup action
                    # (some tasks reject noop). Fine -- the process survived.
                    logger.debug("browsergym warmup noop rejected (daemon alive): %s", warm.get("error"))
                return await _bridge_call(sandbox, cfg, "info", timeout=60.0)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "browsergym daemon died during warmup (attempt %d/%d): %s",
                    attempt,
                    attempts,
                    str(exc)[-300:],
                )
        raise RuntimeError(f"browsergym daemon failed to survive warmup after {attempts} attempts") from last_error


__all__ = [
    "BrowserGymTask",
    "BrowserGymTaskConfig",
    "SYSTEM_PROMPT",
    "USER_PROMPT_TEMPLATE",
    "build_browsergym_sandbox_config",
]
