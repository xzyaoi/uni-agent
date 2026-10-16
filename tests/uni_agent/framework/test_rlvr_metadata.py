"""Session-level RLVR metadata construction (``_rlvr_base_metadata``).

The framework maps each serialized prompt's rollout identifiers onto the
coordinator's RLVR value signal (``docs/design/rlvr-group-completion-scheduler.md``):
``group`` = ``uid``, ``rollout`` = ``session_id``, ``q`` = ``num_sessions``,
plus an optional ``deadline_ms`` derived from the task's per-episode
``sandbox.runtime_timeout``. These pin that mapping; the ambient-bag plumbing
that carries it to the flash-sandbox provider is covered by
``tests/uni_agent/sandbox/test_exec_context.py`` +
``tests/uni_agent/sandbox/test_flash_sandbox_metadata.py`` (which run with no
heavy deps). This module imports the framework and therefore runs where the
package (torch / verl) is installed, like ``test_generate_sequences_on_cpu``.
"""

from __future__ import annotations

import asyncio

import pytest

from uni_agent.framework.framework import (
    AgentRunnerTimeoutError,
    _await_runner_with_timeout,
    _rlvr_base_metadata,
    _sample_runtime_timeout,
)

# --------------------------- _sample_runtime_timeout ---------------------------


def test_runtime_timeout_extracted_from_task_sandbox():
    sample = {"tools_kwargs": {"task": {"sandbox": {"runtime_timeout": 30}}}}
    assert _sample_runtime_timeout(sample) == 30.0


def test_runtime_timeout_absent_when_task_sandbox_missing():
    assert _sample_runtime_timeout({"tools_kwargs": {"task": {}}}) is None
    assert _sample_runtime_timeout({"tools_kwargs": {}}) is None
    assert _sample_runtime_timeout({}) is None


def test_runtime_timeout_tolerates_junk():
    assert _sample_runtime_timeout({"tools_kwargs": {"task": {"sandbox": {"runtime_timeout": "later"}}}}) is None
    assert _sample_runtime_timeout({"tools_kwargs": {"task": {"sandbox": {"runtime_timeout": None}}}}) is None


def test_runtime_timeout_coerces_numeric_strings():
    # Best-effort: a numeric string is a valid float, so it is accepted.
    assert _sample_runtime_timeout({"tools_kwargs": {"task": {"sandbox": {"runtime_timeout": "45"}}}}) == 45.0


# --------------------------- _rlvr_base_metadata ---------------------------


def test_base_metadata_full_when_deadline_present():
    sample = {"uid": "uid-42", "tools_kwargs": {"task": {"sandbox": {"runtime_timeout": 60}}}}
    base = _rlvr_base_metadata(sample, "session-7", num_sessions=8, global_steps=12)
    assert base == {
        "group_id": "uid-42",
        "step": 12,
        "staleness": 0,
        "rlvr": {
            "group": "uid-42",
            "rollout": "session-7",
            "q": 8,
            "deadline_ms": 60_000,
        }
    }


def test_base_metadata_omits_deadline_when_timeout_missing():
    sample = {"uid": "uid-42"}
    base = _rlvr_base_metadata(sample, "session-7", num_sessions=8)
    assert base == {
        "group_id": "uid-42",
        "rlvr": {"group": "uid-42", "rollout": "session-7", "q": 8},
    }


def test_base_metadata_omits_deadline_when_timeout_nonpositive():
    sample = {"uid": "uid-42", "tools_kwargs": {"task": {"sandbox": {"runtime_timeout": 0}}}}
    assert "deadline_ms" not in _rlvr_base_metadata(sample, "s", num_sessions=1)["rlvr"]


def test_base_metadata_coerces_uid_to_str_and_q_to_int():
    sample = {"uid": 12345}
    base = _rlvr_base_metadata(sample, "rollout-0", num_sessions=4)
    assert base["rlvr"]["group"] == "12345"
    assert base["rlvr"]["q"] == 4
    assert isinstance(base["rlvr"]["group"], str)
    assert isinstance(base["rlvr"]["q"], int)
    assert base["group_id"] == "12345"


def test_base_metadata_empty_group_when_uid_missing():
    base = _rlvr_base_metadata({}, "rollout-0", num_sessions=1)
    assert base["rlvr"]["group"] == ""
    assert base["group_id"] == ""


def test_base_metadata_computes_staleness_from_policy_version():
    sample = {"uid": "uid-42", "max_global_steps": 8}
    base = _rlvr_base_metadata(sample, "session-7", num_sessions=8, global_steps=12)
    assert base["step"] == 12
    assert base["staleness"] == 3


def test_base_metadata_invalid_policy_version_defaults_staleness_to_zero():
    sample = {"uid": "uid-42", "max_global_steps": "unknown"}
    base = _rlvr_base_metadata(sample, "session-7", num_sessions=8, global_steps=12)
    assert base["staleness"] == 0


def test_base_metadata_does_not_set_terminal():
    # ``terminal`` is flipped per-exec by the task's eval (terminal_exec), never
    # by the session-level base -- otherwise every infra exec would advance k_g.
    assert "terminal" not in _rlvr_base_metadata({"uid": "g"}, "r", num_sessions=2)["rlvr"]


def test_runner_wall_clock_timeout_cancels_slow_session():
    async def slow_runner():
        await asyncio.sleep(60)

    with pytest.raises(AgentRunnerTimeoutError, match="runtime_timeout=0.01s"):
        asyncio.run(_await_runner_with_timeout(slow_runner(), 0.01))


def test_runner_wall_clock_timeout_allows_fast_session():
    assert asyncio.run(_await_runner_with_timeout(asyncio.sleep(0, result="ok"), 1)) == "ok"
