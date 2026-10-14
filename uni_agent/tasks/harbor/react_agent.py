"""Harbor agent that keeps Uni-Agent's ReAct model requests on the host."""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import os
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from uni_agent.agents.base import ModelConfig
from uni_agent.agents.react.agent import ReActConfig
from uni_agent.agents.registry import build_agent
from uni_agent.observability import capture_context


class UniAgentReAct(BaseAgent):
    """Use the same tools and session gateway as native Uni-Agent rollouts.

    The Flash Sandbox environment owns provisioning and teardown. Passing its
    existing sandbox to ReAct also preserves stateful shell sessions.
    """

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        max_steps: int = 200,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens_per_turn: int | None = None,
        max_total_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.max_steps = int(max_steps)
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens_per_turn = max_tokens_per_turn
        self.max_total_tokens = max_total_tokens

    @staticmethod
    def name() -> str:
        return "uni-agent-react"

    def version(self) -> str:
        return "1"

    async def setup(self, environment: BaseEnvironment) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def _agent_config(self) -> ReActConfig:
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not base_url:
            raise ValueError("UniAgentReAct requires OPENAI_BASE_URL pointing to the session gateway")
        if not self.model_name:
            raise ValueError("UniAgentReAct requires model_name")
        model: dict[str, Any] = {
            "base_url": base_url,
            "api_key": os.environ.get("OPENAI_API_KEY", "EMPTY"),
            "model_name": self.model_name,
        }
        sampling: dict[str, Any] = {"temperature": self.temperature, "top_p": self.top_p}
        # Older training installations keep sampling controls on ModelConfig.
        # Inspect the declared schema rather than retrying failed construction.
        config: dict[str, Any] = {"max_steps": self.max_steps}
        if "sampling_params_override" in ReActConfig.model_fields:
            if self.max_tokens_per_turn is not None:
                sampling["max_tokens_per_turn"] = self.max_tokens_per_turn
            config["sampling_params_override"] = sampling
        else:
            model.update({key: value for key, value in sampling.items() if key in ModelConfig.model_fields})
            if self.max_tokens_per_turn is not None and "max_tokens_per_turn" in ModelConfig.model_fields:
                model["max_tokens_per_turn"] = self.max_tokens_per_turn
        if self.max_total_tokens is not None:
            if "max_total_tokens" in ModelConfig.model_fields:
                model["max_total_tokens"] = self.max_total_tokens
            elif "max_total_tokens" in ReActConfig.model_fields:
                config["max_total_tokens"] = self.max_total_tokens
            else:
                raise ValueError("This ReAct installation does not support max_total_tokens")
        return ReActConfig(model=ModelConfig(**model), **config)

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        sandbox = environment.sandbox
        workdir = None
        for candidate in ("/app", "/testbed"):
            result = await sandbox.exec_shell(f"test -d {candidate}")
            if result.exit_code == 0:
                workdir = candidate
                break
        if workdir is None:
            raise RuntimeError("No repository directory found in the Harbor sandbox (/app or /testbed)")
        agent = build_agent(self._agent_config())
        context.metadata = {**(context.metadata or {}), "agent": self.name(), "workdir": workdir}
        trace = getattr(sandbox, "_trace", None)
        if trace is not None and trace.path is not None:
            context.metadata["sandbox_trace"] = str(trace.path)
            trace.emit(operation="agent", logs_dir=str(self.logs_dir), workdir=workdir)
        run_kwargs: dict[str, Any] = {
            "sandbox": sandbox,
            "messages": [{"role": "user", "content": f"Repository directory: {workdir}\n\n{instruction}"}],
        }
        if "workdir" in inspect.signature(agent.run).parameters:
            run_kwargs["workdir"] = workdir
        with capture_context(
            sandbox_id=getattr(sandbox, "_id", None),
            sandbox_trace=str(trace.path) if trace is not None and trace.path is not None else None,
            image=getattr(sandbox, "image", None),
            agent_logs_dir=str(self.logs_dir),
            workdir=workdir,
        ):
            result = await agent.run(**run_kwargs)
        context.metadata.update({"finished": result.finished, "agent_info": result.info})
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(
            (self.logs_dir / "uni-agent-result.json").write_text,
            json.dumps(dataclasses.asdict(result), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
