"""``browser``: one BrowserGym action per call against the in-sandbox env daemon.

The task layer (``uni_agent/tasks/browsergym``) boots the bridge daemon
(``bridge.py serve``) inside the sandbox before the agent starts; this tool is
the model-facing unit: it takes a single BrowserGym action string, forwards it
over one sandbox ``exec`` (the bridge's stateless CLI), and returns the next
flattened AXTree observation. The env / Chromium never run on this side -- the
agent loop keeps living in the framework process, exactly like the shell tool.

Invalid actions (bad syntax, unknown ``bid``) come back from the env as
``ok: False`` and surface as a :class:`ToolError` observation, so the policy
can self-correct on the next turn instead of killing the rollout.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from uni_agent.sandbox import SandboxBackend

from .base import Tool, ToolError, ToolResult, register_tool

logger = logging.getLogger(__name__)

DESCRIPTION = """
Execute one action in the web browser and return the updated page observation
(flattened accessibility tree). Elements are referenced by their bid shown in
square brackets in the observation, e.g. click(bid='12').
""".strip()


class BrowserToolConfig(BaseModel):
    """Construction kwargs for the ``browser`` tool (a tools entry's kwargs)."""

    bridge_path: str = Field(default="/opt/bgym/bridge.py", description="Path of the bridge script inside the sandbox.")
    bridge_port: int = Field(default=8027, description="Sandbox-local port the bridge daemon listens on.")
    python_bin: str = Field(default="python", description="Python binary inside the sandbox image.")
    command_timeout: float = Field(default=120.0, description="Per-action exec timeout (s).")
    obs_max_chars: int = Field(
        default=30000,
        description="Hard cap on the observation text returned to the model (0 = no cap).",
    )


class BrowserActionArguments(BaseModel):
    action: str = Field(
        description=(
            "One BrowserGym action string, e.g. click(bid='12'), fill(bid='3', value='hello'), "
            "press(key='Enter'), scroll(direction='down'), or stop(answer='...') to finish."
        )
    )


@register_tool("browser")
class BrowserTool(Tool):
    """Drive the in-sandbox BrowserGym env, one action per call."""

    name = "browser"
    description = DESCRIPTION
    args_model = BrowserActionArguments
    config_model = BrowserToolConfig

    def __init__(self, sandbox: SandboxBackend, **kwargs: Any):
        super().__init__(sandbox, **kwargs)
        self.config: BrowserToolConfig

    async def run(self, args: dict[str, Any], *, timeout: float | None = None) -> ToolResult:
        cfg = self.config
        action = str(args["action"]).strip()
        argv = [
            cfg.python_bin,
            cfg.bridge_path,
            "step",
            "--action",
            action,
            "--port",
            str(cfg.bridge_port),
        ]
        budget = timeout if timeout is not None else cfg.command_timeout
        result = await self.sandbox.exec(argv, timeout=budget)
        if result.exit_code != 0:
            stderr = (result.stderr or "").strip()[-1000:]
            autopsy = await self._autopsy_daemon()
            raise ToolError(
                f"bridge call failed (exit {result.exit_code})"
                + (f": {stderr}" if stderr else "; is the bridge daemon running?")
                + (f"\n[daemon autopsy]\n{autopsy}" if autopsy else "")
            )
        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            raise ToolError(f"bridge returned invalid JSON: {(result.stdout or '')[:500]}") from None

        if not payload.get("ok"):
            # The env rejected the action (syntax, unknown bid, hung page).
            # Show the env's own reason so the policy can self-correct.
            raise ToolError(str(payload.get("error") or "the environment rejected the action"))

        text = self._render(payload)
        return ToolResult(text=text)

    async def _autopsy_daemon(self) -> str:
        """Capture why the daemon is unreachable, while the sandbox is still alive.

        Runs when a bridge CLI call fails (daemon dead / unreachable): the
        sandbox itself responds to exec, so we can read the daemon's own log,
        the process table, and the memory picture. Embedded in the ToolError,
        which lands in rollout session logs -> self-documenting failures.
        Best-effort: any exception returns "" (the failure itself matters more).
        """
        try:
            res = await self.sandbox.exec_shell(
                "echo === daemon.log tail ===; tail -n 25 /logs/bgym/stdout.log 2>/dev/null; "
                "echo === processes ===; ps aux 2>/dev/null | grep -E 'bridge|chrome|python' | grep -v grep | head -8; "
                "echo === memory ===; head -3 /proc/meminfo 2>/dev/null; cat /sys/fs/cgroup/memory.max 2>/dev/null; "
                "cat /sys/fs/cgroup/memory.current 2>/dev/null; "
                "echo === pid ===; cat /logs/bgym/daemon.pid 2>/dev/null; "
                "kill -0 $(cat /logs/bgym/daemon.pid 2>/dev/null) 2>&1 && echo DAEMON-ALIVE || echo DAEMON-DEAD",
                timeout=15.0,
            )
            return (res.stdout or "")[-2500:]
        except Exception:
            return ""

    def _render(self, payload: dict) -> str:
        cfg = self.config
        text = str(payload.get("obs_text") or "")
        limit = cfg.obs_max_chars
        if limit > 0 and len(text) > limit:
            text = f"{text[:limit]}\n<NOTE>observation clipped at {limit} chars</NOTE>"
        if payload.get("done"):
            text += (
                "\n<NOTE>The episode has finished "
                f"(reward={payload.get('reward', 0)}, success={bool(payload.get('success'))}). "
                "No further browser actions are accepted.</NOTE>"
            )
        return text
