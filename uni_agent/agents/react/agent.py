"""ReAct: the white-box agent driven by our own framework loop (reason + tool-call act)."""

from __future__ import annotations

import logging
import os
import shlex
from typing import TYPE_CHECKING, Any

from pydantic import Field

from uni_agent.sandbox.exec_context import bind_exec_metadata
from uni_agent.tools import Toolbox

from ..base import Agent, AgentConfig, AgentResult, RequestSamplingConfig
from ..registry import register_agent
from .model import OpenAICompatibleChatModel

if TYPE_CHECKING:
    from uni_agent.sandbox import Sandbox

logger = logging.getLogger(__name__)

#: Tool names that end the episode when the policy calls them.
_FINISH_TOOLS = {"submit", "finish"}

#: Max chars of LLM ``content`` / tool ``arguments`` retained in exec metadata
#: (the rest is truncated) so the sandbox exec trace + structured event log
#: stays manageable. Override with the env var of the same name.
_LLM_PREVIEW_CHARS = int(os.getenv("FLASH_SANDBOX_LLM_PREVIEW_CHARS", "2048"))


def _truncate(value: Any, limit: int = _LLM_PREVIEW_CHARS) -> Any:
    """Truncate a string ``value`` to ``limit`` chars (appending ``…``).

    Non-strings pass through unchanged so the metadata stays JSON-safe and the
    helper is usable for both LLM ``content`` and tool ``arguments``.
    """
    if not isinstance(value, str) or limit <= 0 or len(value) <= limit:
        return value
    return value[:limit] + "…"


class ReActConfig(AgentConfig):
    """White-box launch params: host-side tools + step / timeout budgets."""

    name: str = "react"
    sampling_params_override: RequestSamplingConfig = Field(default_factory=RequestSamplingConfig)
    tools: list[dict] = Field(
        default_factory=lambda: [
            {"name": "str_replace_editor"},
            {"name": "stateful_shell", "command_timeout": 120},
            {"name": "submit"},
        ],
        description="Host-side tools exposed to the policy (each a {name, ...kwargs} entry). "
        "Include a finish tool (submit/finish) so the policy can end the episode explicitly.",
    )
    max_steps: int = Field(default=50, description="Max tool-calling turns per episode.")
    action_timeout: float | None = Field(
        default=None,
        description="Per-call timeout (s) forwarded to each tool call, overriding the tool's own "
        "default (e.g. the shell's command_timeout); None defers to the tool's own timeout.",
    )
    timeout_budget: int = Field(
        default=3,
        description="Tool-call timeouts tolerated per episode before it stops with a timeout-limit reason.",
    )


@register_agent("react")
class ReActAgent(Agent):
    """White-box solver: framework loop + host-side tools over an OpenAI endpoint."""

    config_model = ReActConfig

    async def run(
        self,
        *,
        sandbox: Sandbox,
        messages: list[dict[str, Any]],
        workdir: str | None = None,
    ) -> AgentResult:
        cfg: ReActConfig = self.config  # type: ignore[assignment]
        if cfg.model.base_url is None:
            raise ValueError("react: config.model.base_url is not set (the endpoint the policy calls)")

        toolbox = Toolbox.from_specs(cfg.tools, sandbox=sandbox)
        model = OpenAICompatibleChatModel(
            base_url=cfg.model.base_url,
            api_key=cfg.model.api_key,
            model_name=cfg.model.model_name,
            sampling_params=cfg.sampling_params_override.sampling_params(),
            tools_schemas=toolbox.schemas(),
        )

        transcript: list[dict[str, Any]] = list(messages)
        for message in messages:
            logger.info(f"{str(message.get('role', '')).upper()} PROMPT:\n{message.get('content', '')}")
        trajectory_info: dict[str, Any] = {
            "steps": 0,
            "num_tool_calls": 0,
            "timeouts": 0,
            "errors": 0,
            "total_tokens": 0,
        }
        termination_reason = "unknown"
        try:
            async with toolbox.entered(retry=3, timeout=60):
                if workdir is not None and "shell" in toolbox.names():
                    await toolbox.call("shell", {"command": f"cd -- {shlex.quote(workdir)}"})
                for step_idx in range(1, cfg.max_steps + 1):
                    trajectory_info["steps"] = step_idx
                    stop_reason = await self.step(cfg, model, toolbox, transcript, trajectory_info)
                    if stop_reason != "completed":
                        termination_reason = stop_reason
                        break
                else:  # loop ran the full step budget without an early stop
                    termination_reason = "max_steps"
                    logger.warning(f"Reached max steps ({cfg.max_steps}) without finishing.")
        except Exception as exc:  # keep the partial transcript; the task buckets the failure
            logger.exception("react loop failed at step %s", trajectory_info["steps"])
            termination_reason = "unknown_error"
            trajectory_info["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            await model.aclose()  # release the reused HTTP session

        logger.info(
            f"Episode done: termination_reason={termination_reason} steps={trajectory_info['steps']} "
            f"tool_calls={trajectory_info['num_tool_calls']} timeouts={trajectory_info['timeouts']} "
            f"errors={trajectory_info['errors']} total_tokens={trajectory_info['total_tokens']}"
        )
        return AgentResult(
            transcript=transcript,
            info=trajectory_info,
            finished=termination_reason == "finished",
        )

    async def step(
        self,
        cfg: ReActConfig,
        model: OpenAICompatibleChatModel,
        toolbox: Toolbox,
        transcript: list[dict[str, Any]],
        info: dict[str, Any],
    ) -> str:
        """Run one turn: query the policy, record its message, run its tool calls."""
        logger.info(f"{'=' * 25} STEP {info['steps']} {'=' * 25}")

        # step 1: query the model
        content, tool_calls, gen_info = await model.query(
            transcript,
            sampling_params=cfg.sampling_params_override.sampling_params(),
        )
        info["total_tokens"] = gen_info["prompt_tokens"] + gen_info["completion_tokens"]
        finish_reason = gen_info.get("finish_reason")
        logger.info(
            f"Prompt Tokens: {gen_info['prompt_tokens']}, Completion Tokens: {gen_info['completion_tokens']} "
            f"(total {info['total_tokens']})"
        )
        logger.info(f"💭 THOUGHT:\n{content}")

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        transcript.append(assistant_msg)

        if finish_reason == "length":
            logger.info("Exit: response truncated at a token cap.")
            return "token_limit"

        if not tool_calls:
            logger.info("💬 FINISHED: policy replied with plain text (no tool call).")
            return "finished"

        # Capture this turn's LLM input/output (token counts + truncated content
        # and tool calls) so a downstream consumer of the sandbox exec trace +
        # structured event log can reconstruct the full execution pattern: which
        # LLM turn, in which group/trajectory, produced which sandbox command.
        _llm_meta = {
            "prompt_tokens": int(gen_info.get("prompt_tokens") or 0),
            "completion_tokens": int(gen_info.get("completion_tokens") or 0),
            "total_tokens": int(info["total_tokens"]),
            "finish_reason": gen_info.get("finish_reason"),
            "content": _truncate(content),
            "tool_calls": [
                {
                    "name": tc.get("function", {}).get("name"),
                    "arguments": _truncate(tc.get("function", {}).get("arguments")),
                }
                for tc in tool_calls
            ],
            "step": int(info.get("steps") or 0),
        }

        # step 2: dispatch the tool calls
        saw_finish = False
        for idx, tool_call in enumerate(tool_calls):
            fn = tool_call.get("function", {})
            name = fn.get("name", "")
            logger.info(f"🎬 ACTION ({name}):\n{fn.get('arguments')}")
            # Tag this exec with the LLM input/output that produced it. The
            # session-level rlvr group/rollout base is already bound by the
            # framework (or re-bound in a Ray worker), so this overlay adds the
            # per-call ``llm`` bag the provider forwards onto the exec trace +
            # structured event log.
            async with bind_exec_metadata(
                llm={**_llm_meta, "tool_name": name, "tool_arguments": _truncate(fn.get("arguments"))},
            ):
                tool_result = await toolbox.call(name, fn.get("arguments"), timeout=cfg.action_timeout)
            observation = tool_result.to_observation()
            info["num_tool_calls"] += 1
            if tool_result.status == "timeout":  # a tool hit its own timeout (e.g. shell command_timeout)
                info["timeouts"] += 1
                logger.warning(
                    f"⏳ TIMEOUT ({name}): {info['timeouts']}/{cfg.timeout_budget} budget used\n{observation}"
                )
            elif tool_result.status != "ok":
                info["errors"] += 1
                logger.warning(f"❌ TOOL {tool_result.status.upper()} ({name}):\n{observation}")
            else:
                logger.info(f"👀 OBSERVATION ({name}):\n{observation}")

            transcript.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "name": name,
                    "content": observation,
                }
            )

            if info["timeouts"] > cfg.timeout_budget:
                logger.warning("Exit: timeout budget exhausted; skipping remaining tool calls this turn.")
                for tool_call in tool_calls[idx + 1 :]:
                    transcript.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.get("id"),
                            "name": tool_call.get("function", {}).get("name", ""),
                            "content": "Skipped: timeout budget exhausted.",
                        }
                    )
                return "timeout_limit"
            if name in _FINISH_TOOLS and tool_result.status == "ok":
                saw_finish = True
        if saw_finish:
            logger.info("💬 FINISHED: policy called a finish tool.")
        return "finished" if saw_finish else "completed"
