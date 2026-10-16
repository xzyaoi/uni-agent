"""Unit tests for :mod:`uni_agent.sandbox.exec_context`.

The ambient metadata bag is the seam that threads group/trajectory/LLM-IO from
the training loop down to the flash-sandbox provider without changing the
shared :meth:`Sandbox.exec` signature. These pin the contract with no
non-stdlib dependencies, so they run anywhere.
"""

from __future__ import annotations

import asyncio

import pytest

from uni_agent.sandbox.exec_context import (
    _EXEC_METADATA,
    bind_exec_metadata,
    current_exec_metadata,
    reset_exec_metadata,
    set_exec_metadata,
    terminal_exec,
)


@pytest.fixture(autouse=True)
def _isolate_exec_metadata():
    """Start and end every test with an empty ambient bag.

    Direct ``set_exec_metadata`` calls (without a matching ``reset``) would
    otherwise leak across tests, and ``reset(set(None))`` restores the
    *previous* value rather than the default -- so pin the var to empty on
    both sides of each test for order-independent results.
    """
    token = set_exec_metadata(None)
    try:
        yield
    finally:
        reset_exec_metadata(token)
        # Belt-and-suspenders: ensure we never leave a non-empty bag for the
        # next test even if a body reset to a stale token.
        if _EXEC_METADATA.get() not in (None, {}):
            set_exec_metadata(None)


def test_current_metadata_empty_when_unbound():
    assert current_exec_metadata() == {}


def test_set_then_reset_round_trips_to_none():
    token = set_exec_metadata({"rlvr": {"group": "g1"}})
    assert current_exec_metadata() == {"rlvr": {"group": "g1"}}
    reset_exec_metadata(token)
    assert current_exec_metadata() == {}
    assert _EXEC_METADATA.get() is None


def test_nested_binds_deep_merge_rlvr_and_llm():
    """The framework binds the session-level rlvr base; the agent overlays llm
    and a per-call rlvr field; the deep merge composes them without clobbering."""
    async def main() -> dict:
        async with bind_exec_metadata(rlvr={"group": "g1", "rollout": "r1", "q": 8, "deadline_ms": 60000}):
            async with bind_exec_metadata(
                rlvr={"terminal": True},
                llm={"prompt_tokens": 10, "completion_tokens": 2, "tool_name": "shell_exec"},
            ):
                return current_exec_metadata()

    merged = asyncio.run(main())
    assert merged["rlvr"] == {
        "group": "g1",
        "rollout": "r1",
        "q": 8,
        "deadline_ms": 60000,
        "terminal": True,
    }
    assert merged["llm"]["tool_name"] == "shell_exec"


def test_inner_bind_does_not_leak_into_outer():
    async def main() -> tuple[dict, dict]:
        async with bind_exec_metadata(rlvr={"group": "g1"}):
            async with bind_exec_metadata(llm={"prompt_tokens": 10}):
                inner = current_exec_metadata()
            return inner, current_exec_metadata()

    inner, outer = asyncio.run(main())
    assert inner == {"rlvr": {"group": "g1"}, "llm": {"prompt_tokens": 10}}
    assert outer == {"rlvr": {"group": "g1"}}  # llm did not leak; rlvr preserved


def test_concurrent_sessions_do_not_crosstalk():
    """Each asyncio.Task captures its own context, so two sessions bound in
    parallel never see each other's metadata."""
    async def session(group: str) -> str:
        async with bind_exec_metadata(rlvr={"group": group}):
            await asyncio.sleep(0)  # yield so the other task can run
            return current_exec_metadata()["rlvr"]["group"]

    async def main() -> tuple[str, str]:
        a, b = await asyncio.gather(session("g1"), session("g2"))
        return a, b

    assert asyncio.run(main()) == ("g1", "g2")


def test_empty_bind_is_a_no_op_when_unbound():
    async def main() -> None:
        async with bind_exec_metadata():
            assert current_exec_metadata() == {}

    asyncio.run(main())


def test_terminal_exec_sets_only_the_terminal_flag():
    async def main() -> dict:
        async with bind_exec_metadata(rlvr={"group": "g1"}):
            async with terminal_exec():
                return current_exec_metadata()["rlvr"]

    assert asyncio.run(main()) == {"group": "g1", "terminal": True}


def test_scalar_overlay_replaces_not_merges():
    async def main() -> dict:
        async with bind_exec_metadata(step=1):
            async with bind_exec_metadata(step=2):
                return current_exec_metadata()

    assert asyncio.run(main()) == {"step": 2}


def test_reset_exec_metadata_is_idempotent_safe_after_bind():
    """``set_exec_metadata(None)`` exposes an empty bag (a Ray worker that
    binds nothing must observe no metadata), and resetting the token it
    returned leaves the var empty -- no leakage to a later test."""
    token = set_exec_metadata(None)
    assert current_exec_metadata() == {}
    reset_exec_metadata(token)
    assert current_exec_metadata() == {}


if __name__ == "__main__":
    sys = __import__("sys")
    sys.exit(pytest.main([__file__, "-v"]))
