"""mini-swe-agent: white-box solver (the loop runs on the cluster).

mini-swe-agent (https://github.com/SWE-agent/mini-swe-agent) drives a simple
``query -> bash action -> observe`` loop. We run that loop **in the framework's
Ray task on the cluster**, pointing its model at the gateway at the node's
private IP (reachable from the same node, unlike a sandbox in the cloud) and
using the cloud sandbox purely as the command-execution backend over HTTP.
This is the only mini-swe-agent shape that works behind the CSCS firewall: the
cloud sandbox makes no inbound connection to the (private) gateway.

mini-swe-agent's loop is **synchronous** (``LitellmModel.query`` via
``litellm.completion`` + ``Environment.execute``), so it runs in a worker
thread (``asyncio.to_thread``). Each ``env.execute(action)`` bridges back to the
main event loop's async :meth:`Sandbox.exec_shell` via
:func:`asyncio.run_coroutine_threadsafe`, so the agent's blocking model calls
and command execs never stall the loop.

Submission follows mini-swe-agent's own convention: when a command prints
``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` on its first line with exit code 0,
the environment raises ``Submitted`` and :meth:`DefaultAgent.run` returns
``{"exit_status": "Submitted", "submission": <rest of output>}``. The task's
:func:`compute_reward` then scores the resulting ``/testbed`` (it resets only
the test files, so the agent's source edits persist).

The agent is launched through a :class:`TaskConfig`; its ``sandbox`` selects the
provider (``flash_sandbox`` for Clariden) and its ``agent.model`` carries the
gateway endpoint (filled in by the runner from the live policy server).
"""

from __future__ import annotations

import asyncio
import copy
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from pydantic import Field

from ..base import Agent, AgentConfig, AgentResult
from ..registry import register_agent
from ...sandbox.exec_context import bind_exec_metadata, current_exec_metadata

if TYPE_CHECKING:
    from uni_agent.sandbox import Sandbox

logger = logging.getLogger(__name__)

#: Built-in system message when neither the config nor `messages` supplies one.
_DEFAULT_SYSTEM_TEMPLATE = (
    "You are a helpful assistant that can interact with a computer shell to solve programming tasks."
)

#: Built-in instance message: wraps the `user` message as ``{{ task }}`` and
#: teaches mini-swe-agent's submission protocol. Override via the config to use
#: upstream's ``swebench.yaml`` prompt or a custom one.
_DEFAULT_INSTANCE_TEMPLATE = """\
<pr_description>
{{ task }}
</pr_description>

You are operating in a remote shell. The working directory is `/testbed`, a copy
of the repository at the base commit. Modify **non-test source files** to fix
the issue described above, verify your fix with a relevant test command, then
submit your changes.

Command rules:

- Every command runs in its own subshell, so directory and environment changes
  are not persistent. Prefix with `cd /testbed && ...` when you need a cwd.
- Use non-interactive flags (`-y`, `-f`); avoid editors that wait for input.

When you are done, submit by running EXACTLY this single command (it must exit 0
or it will not submit):

    echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git -c core.fileMode=false diff

Do not run any further command after submitting.
"""

#: Env vars mirroring mini-swe-agent's SWE-bench config (quiet pagers / progress
#: bars so the observation stays short and parseable).
_DEFAULT_ENV: dict[str, str] = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}


class MiniSweAgentConfig(AgentConfig):
    """White-box launch params for mini-swe-agent (model on :attr:`AgentConfig.model`)."""

    name: str = "mini_swe_agent"

    step_limit: int = Field(
        default=250, description="Max tool-calling turns per episode (0 disables)."
    )
    cost_limit: float = Field(
        default=3.0,
        description="Stop after exceeding this USD cost (0 disables). With cost_tracking="
        "'ignore_errors' (set below) the cost stays 0, so this is effectively unbounded.",
    )
    wall_time_limit_seconds: int = Field(
        default=0, description="Wall-clock cap in seconds (0 disables)."
    )
    max_consecutive_format_errors: int = Field(
        default=3, description="Exit after this many format errors in a row (0 disables)."
    )

    command_timeout: int = Field(
        default=60, description="Per-command timeout (s) forwarded to the sandbox, plus a "
        "30s margin on the cross-thread bridge."
    )
    cwd: str = Field(default="/testbed", description="Working directory for commands in the sandbox.")
    env: dict[str, str] = Field(
        default_factory=lambda: dict(_DEFAULT_ENV),
        description="Extra env vars set on every command in the sandbox.",
    )

    system_template: str | None = Field(
        default=None,
        description="Jinja system message template. Defaults to the 'system' message from "
        "`messages`, then a built-in SWE prompt.",
    )
    instance_template: str | None = Field(
        default=None,
        description="Jinja instance message template rendered with the 'user' message as "
        "`{{ task }}`. Defaults to a built-in SWE prompt with the "
        "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT submission mechanism.",
    )

    model_class: str = Field(
        default="",
        description="mini-swe-agent model class (dotted path or registry key) passed to "
        "minisweagent.get_model; empty infers LitellmModel (OpenAI tool-calling). Set to "
        "'litellm_textbased' to skip tool-calling and parse ```mswea_bash_command``` blocks "
        "out of the completion instead -- handy if the gateway's tool-call format is finicky.",
    )
    extra_model_kwargs: dict[str, Any] = Field(
        default_factory=lambda: {"drop_params": True, "parallel_tool_calls": True},
        description="Extra model_kwargs passed to litellm.completion. api_base/api_key are "
        "always injected from the agent's model; `drop_params` lets litellm drop anything the "
        "endpoint rejects.",
    )


class RemoteSandboxEnvironment:
    """mini-swe-agent ``Environment`` that runs commands in a remote sandbox.

    Mirrors :class:`minisweagent.environments.local.LocalEnvironment`'s output
    shape (``{output, returncode, exception_info}``) and its ``Submitted``
    finish signal, but instead of a local ``subprocess`` it forwards each
    command to the framework's async :meth:`Sandbox.exec_shell` over HTTP. The
    agent loop calls :meth:`execute` synchronously from a worker thread, so the
    bridge uses :func:`asyncio.run_coroutine_threadsafe` to land the exec on the
    main event loop (free while the worker blocks on ``model.query``). Each
    bridged exec re-binds the session-level RLVR metadata ``base`` (captured by
    :meth:`MiniSweAgentAgent.run` via :func:`current_exec_metadata`) so the
    flash-sandbox provider observes the same ``rlvr`` group/rollout tags on
    execs as on the create -- the ``run_coroutine_threadsafe`` task is a fresh
    context in which the ambient ``exec_metadata`` ContextVar does not reliably
    survive the worker-thread -> main-loop boundary.
    """

    def __init__(
        self,
        *,
        sandbox: Sandbox,
        loop: asyncio.AbstractEventLoop,
        cwd: str = "/testbed",
        timeout: int = 60,
        extra_env: dict[str, str] | None = None,
        base: dict[str, Any] | None = None,
    ) -> None:
        self.config = SimpleNamespace(cwd=cwd, env=dict(extra_env or {}), timeout=timeout)
        self._sandbox = sandbox
        self._loop = loop
        self._base = base or None

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Run ``action['command']`` in the sandbox and return its captured result.

        Raises mini-swe-agent's :class:`Submitted` when the command's first line
        is ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` and it exited 0, so
        :class:`DefaultAgent` ends the episode with that submission.
        """
        command = action.get("command", "")
        workdir = cwd or self.config.cwd
        cmd_timeout = timeout if timeout is not None else self.config.timeout

        try:
            from minisweagent.exceptions import Submitted
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError("mini_swe_agent requires the 'mini-swe-agent' package") from exc

        async def _bridged() -> Any:
            # Re-bind the session-level RLVR base in this new task's context.
            # ``run_coroutine_threadsafe`` schedules a fresh Task on the main
            # loop; the ambient ``exec_metadata`` ContextVar does not reliably
            # cross the worker-thread -> main-loop boundary, so without an
            # explicit re-bind execs could silently lose their rlvr
            # group/rollout tags. The create carries them via ``_run_session``'s
            # ``bind_exec_metadata``; execs must too for the scheduler's
            # per-group progress accounting (``k_g``) to match the create's
            # ``rlvr.group`` / ``rlvr.rollout``.
            if self._base:
                async with bind_exec_metadata(**self._base):
                    return await self._sandbox.exec_shell(
                        command,
                        workdir=workdir,
                        env=self.config.env or None,
                        timeout=cmd_timeout,
                    )
            return await self._sandbox.exec_shell(
                command,
                workdir=workdir,
                env=self.config.env or None,
                timeout=cmd_timeout,
            )

        try:
            future = asyncio.run_coroutine_threadsafe(_bridged(), self._loop)
            # Wait a little longer than the server-side cap so a command that
            # completes just under it still returns rather than tripping the
            # cross-thread future.
            result = future.result(timeout=cmd_timeout + 30)
        except (TimeoutError, asyncio.TimeoutError) as exc:
            return {
                "output": "",
                "returncode": -1,
                "exception_info": f"command timed out after {cmd_timeout}s: {exc}",
            }

        output: dict[str, Any] = {
            "output": result.stdout or "",
            "returncode": result.exit_code,
            "exception_info": "",
        }
        lines = output["output"].lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )
        return output

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        """Expose the env's own knobs to Jinja templates (mirrors LocalEnvironment)."""
        from minisweagent.utils.serialize import recursive_merge

        return recursive_merge(
            {
                "cwd": self.config.cwd,
                "env": self.config.env,
                "timeout": self.config.timeout,
            },
            kwargs,
        )

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": {
                        "cwd": self.config.cwd,
                        "env": self.config.env,
                        "timeout": self.config.timeout,
                    },
                    "environment_type": f"{__name__}.RemoteSandboxEnvironment",
                }
            }
        }


@register_agent("mini_swe_agent")
class MiniSweAgentAgent(Agent):
    """White-box mini-swe-agent solver: loop on the cluster, sandbox as the tool backend."""

    config_model = MiniSweAgentConfig

    async def run(
        self,
        *,
        sandbox: Sandbox,
        messages: list[dict[str, Any]],
    ) -> AgentResult:
        cfg: MiniSweAgentConfig = self.config  # type: ignore[assignment]
        if not cfg.model.base_url:
            raise ValueError("mini_swe_agent: config.model.base_url is not set (the gateway endpoint the policy calls)")

        system_msg = next((m for m in messages if isinstance(m, dict) and m.get("role") == "system"), None)
        user_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
        if not user_msgs:
            raise ValueError("mini_swe_agent: messages requires a 'user' message (the task)")
        if len(user_msgs) > 1:
            raise ValueError("mini_swe_agent: messages may contain at most one 'user' message")
        user_msg = user_msgs[0]

        system_template = cfg.system_template or (system_msg["content"] if system_msg else _DEFAULT_SYSTEM_TEMPLATE)
        instance_template = cfg.instance_template or _DEFAULT_INSTANCE_TEMPLATE
        task = user_msg.get("content", "")

        try:
            from minisweagent.agents.default import DefaultAgent
            from minisweagent.models import get_model
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "mini_swe_agent requires the 'mini-swe-agent' package: pip install mini-swe-agent"
            ) from exc

        loop = asyncio.get_running_loop()
        # Capture the session-level RLVR base (group/rollout/q/deadline) from
        # the ambient context established by ``_run_session``'s
        # ``bind_exec_metadata``. The mini-swe-agent loop runs in a worker
        # thread (``asyncio.to_thread``) and bridges each exec back to the main
        # loop via ``run_coroutine_threadsafe``; re-binding the captured base
        # on the bridge ensures execs carry the same rlvr tags as the create,
        # regardless of how the ContextVar propagates across the
        # thread/loop boundary. Deep-copy so the capture is fully independent
        # of the ambient bag (cheap once per session; the per-request body is
        # deep-copied again by ``_forward_metadata``).
        base = copy.deepcopy(current_exec_metadata()) or None
        env = RemoteSandboxEnvironment(
            sandbox=sandbox,
            loop=loop,
            cwd=cfg.cwd,
            timeout=cfg.command_timeout,
            extra_env=cfg.env,
            base=base,
        )

        # Point litellm at the gateway: 'openai/<served>' routes through the
        # openai provider to api_base, and drop_params lets the call succeed
        # even if the gateway rejects some OpenAI fields. cost_tracking=
        # 'ignore_errors' keeps cost 0 when the served model isn't in litellm's
        # registry, so cost_limit never prematurely ends the loop.
        model_config: dict[str, Any] = {
            "model_name": f"openai/{cfg.model.model_name or 'default'}",
            "cost_tracking": "ignore_errors",
            "model_kwargs": {
                "api_base": cfg.model.base_url,
                "api_key": cfg.model.api_key,
                **cfg.extra_model_kwargs,
            },
        }
        if cfg.model_class:
            model_config["model_class"] = cfg.model_class
        model = get_model(config=model_config)

        agent = DefaultAgent(
            model,
            env,
            system_template=system_template,
            instance_template=instance_template,
            step_limit=cfg.step_limit,
            cost_limit=cfg.cost_limit,
            wall_time_limit_seconds=cfg.wall_time_limit_seconds,
        )

        logger.info(
            "mini_swe_agent start: model=%s base_url=%s step_limit=%s cwd=%s",
            model_config["model_name"],
            cfg.model.base_url,
            cfg.step_limit,
            cfg.cwd,
        )
        try:
            # The loop is synchronous; offload it so its blocking model query
            # and (bridged) exec calls never stall the main event loop.
            result = await asyncio.to_thread(agent.run, task)
        except Exception as exc:
            logger.exception("mini_swe_agent loop failed")
            return AgentResult(
                output={"exit_status": type(exc).__name__, "submission": ""},
                info={"error": f"{type(exc).__name__}: {exc}"},
                finished=False,
            )

        exit_status = result.get("exit_status", "") if isinstance(result, dict) else ""
        logger.info("mini_swe_agent done: exit_status=%s n_calls=%s", exit_status, getattr(agent, "n_calls", 0))
        return AgentResult(
            output=result if isinstance(result, dict) else {"raw": result},
            info={
                "exit_status": exit_status,
                "n_calls": getattr(agent, "n_calls", 0),
                "cost": getattr(agent, "cost", 0.0),
            },
            finished=(exit_status == "Submitted"),
        )
