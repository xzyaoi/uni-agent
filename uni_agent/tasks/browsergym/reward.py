"""Reward scoring: read the final episode result from the in-sandbox bridge.

The daemon holds the ground truth (episode reward, done/truncated, step count)
and mirrors it to a result file. Never trust the agent's self-report: the score
always comes from the env via the bridge (HTTP first, file fallback).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from uni_agent.sandbox import Sandbox

logger = logging.getLogger(__name__)


def parse_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize a bridge result payload into the task's reward dict."""
    reward = float(payload.get("reward") or 0.0)
    score = min(1.0, max(0.0, reward))
    done = bool(payload.get("done"))
    truncated = bool(payload.get("truncated"))
    if "success" in payload:
        success = bool(payload.get("success"))
    else:  # derive for payloads written without the daemon's own verdict
        success = done and not truncated and reward > 0
    return {
        "reward": score,
        "raw_reward": reward,
        "success": success,
        "done": done,
        "truncated": truncated,
        "n_steps": int(payload.get("n_steps") or 0),
        "user_message": str(payload.get("user_message") or ""),
        "url": str(payload.get("url") or ""),
    }


async def compute_reward(
    sandbox: Sandbox,
    *,
    python_bin: str,
    bridge_path: str,
    bridge_port: int,
    result_file: str,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Fetch the final result: live daemon first, then the on-disk mirror.

    Both paths failing (dead daemon + no file) scores 0 with the error recorded,
    mirroring how Terminal-Bench still scores the filesystem after agent failure.
    """
    payload: dict[str, Any] | None = None
    error: str | None = None
    try:
        result = await sandbox.exec(
            [python_bin, bridge_path, "result", "--port", str(bridge_port)],
            timeout=timeout,
        )
        if result.exit_code == 0 and result.stdout.strip():
            payload = json.loads(result.stdout)
    except Exception as exc:
        error = f"bridge result call failed: {type(exc).__name__}: {exc}"

    if payload is None:
        try:
            data = await sandbox.read_file(result_file)
            payload = json.loads(data.decode("utf-8"))
            logger.info("bridge daemon unreachable; scored from %s", result_file)
        except Exception as exc:
            error = f"{error or ''} result file read failed: {type(exc).__name__}: {exc}".strip()

    if payload is None:
        return {
            "reward": 0.0,
            "success": False,
            "done": False,
            "truncated": False,
            "n_steps": 0,
            "error": (error or "no result available")[-2000:],
        }

    parsed = parse_result_payload(payload)
    if error:
        parsed["error"] = error[-2000:]
    return parsed
