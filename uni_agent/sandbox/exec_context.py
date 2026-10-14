"""Ambient per-exec metadata threaded from the training loop to sandbox providers.

The flash-sandbox coordinator consumes an opaque ``metadata`` bag on create
(``POST /sandboxes``) and one-shot exec (``POST /sandboxes/{id}/exec``) to drive
its RLVR-aware scheduler (see ``docs/design/rlvr-group-completion-scheduler.md``
and ``docs/design/scheduling.md`` in the flash-sandbox repo). The training loop
already knows, for each rollout session, the identifiers that signal value:

* ``group``   -- the rollout-group id ``g`` (one prompt owns ``rollout.n``
                 sessions; the learner consumes a group once it has
                 ``q_g = |L_g|`` infra-valid terminal outcomes).
* ``rollout`` -- the rollout id ``l`` within ``g`` (one agent episode/session).
* ``q``       -- ``q_g``: infra-valid outcomes the group needs before the
                 learner can consume it.
* ``deadline_ms`` -- latest useful completion time, milliseconds-from-now.
* ``terminal``    -- set only on the command producing the rollout's terminal
                     application outcome (e.g. the eval), so the scheduler's
                     progress counter ``k_g`` advances once per rollout.

The outer metadata bag additionally records ``group_id``, training ``step``,
and dispatch-time ``staleness`` on every exec. ``group_id`` intentionally
duplicates ``rlvr.group``: the latter is the scheduler contract, while the
former gives trace consumers the exact training-facing field name without
requiring knowledge of the RLVR schema.

The LLM input/output that produced a given command is opaque to the backend
(forwarded onto the exec trace + structured event log under ``metadata["llm"]``)
so a downstream consumer can reconstruct the full execution pattern: which LLM
turn, in which group/trajectory, produced which sandbox command.

A :class:`~contextvars.ContextVar` carries the bag so the shared
:meth:`Sandbox.exec` signature stays unchanged -- only providers that read
:meth:`current_exec_metadata` (today: ``flash_sandbox``) are affected; the base
class, the :class:`SandboxBackend` protocol, every tool, and every other
provider keep working untouched. Nested binds compose (the framework sets the
session-level ``rlvr`` base; the agent overlays per-step ``llm``; the eval
flips ``rlvr["terminal"]``), and concurrent sessions (separate async tasks /
Ray workers, each its own context) never cross-talk.
"""

from __future__ import annotations

import contextvars
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

#: Ambient per-exec metadata bag. ``None`` until the training loop binds a
#: session-level base; ``None`` is equivalent to an empty bag for callers.
_EXEC_METADATA: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "uni_agent_flash_sandbox_exec_metadata", default=None
)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge ``overlay`` onto a shallow copy of ``base``.

    Dict-valued fields (``rlvr``, ``llm``) are merged recursively so a nested
    bind (e.g. ``rlvr={"terminal": True}``) augments rather than replaces the
    outer session-level ``rlvr``; scalars replace.
    """
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def current_exec_metadata() -> dict[str, Any]:
    """Return a shallow copy of the ambient metadata, or ``{}`` when unbound.

    Callers (sandbox providers) read this at each create/exec call and forward
    only the non-empty bag; an empty bag means "no metadata" and is omitted so
    the wire shape matches pre-instrumentation behaviour exactly.
    """
    meta = _EXEC_METADATA.get()
    return dict(meta) if meta else {}


def set_exec_metadata(metadata: dict[str, Any] | None) -> contextvars.Token[dict[str, Any] | None]:
    """Set the ambient bag, returning the token for :func:`reset_exec_metadata`.

    Used by Ray rollout workers (a sync ``@ray.remote`` function that runs the
    agent via :func:`asyncio.run`): the parent serializes the session-level base
    into the task kwargs, and the worker sets it before running so the provider
    (which runs inside the same ``asyncio.run`` context) observes it.
    """
    return _EXEC_METADATA.set(dict(metadata) if metadata else None)


def reset_exec_metadata(token: contextvars.Token[dict[str, Any] | None]) -> None:
    """Restore the ambient bag to the value captured by ``token``."""
    _EXEC_METADATA.reset(token)


@asynccontextmanager
async def bind_exec_metadata(**fields: Any):
    """Overlay ``fields`` onto the ambient metadata for the duration of the body.

    Dict-valued fields (``rlvr``, ``llm``) are merged with the current value;
    scalars replace. The prior value is restored on exit (even on exception), so
    nested binds compose and concurrent sessions never cross-talk. An empty
    ``fields`` (with nothing already bound) is a cheap no-op.

    Example::

        # framework: session-level base (uid -> group, session_id -> rollout)
        async with bind_exec_metadata(rlvr={"group": uid, "rollout": sid, "q": n}):
            await runner(...)

        # agent: per-step LLM input/output that produced this command
        async with bind_exec_metadata(llm={"prompt_tokens": 10, "tool_name": name, ...}):
            await toolbox.call(name, args)

        # eval: the rollout's terminal application outcome
        async with bind_exec_metadata(rlvr={"terminal": True}):
            await compute_reward(sample, sandbox)
    """
    current = _EXEC_METADATA.get()
    merged = _deep_merge(current, fields) if current else dict(fields) if fields else None
    token = _EXEC_METADATA.set(merged)
    try:
        yield
    finally:
        _EXEC_METADATA.reset(token)


def terminal_exec() -> AbstractAsyncContextManager:
    """Bind ``rlvr["terminal"] = True`` for the rollout's terminal command.

    Per the RLVR contract, the training loop tags the *one* command that
    produces a rollout's terminal application outcome (typically the eval) so
    the scheduler's per-group progress counter advances exactly once.
    """
    return bind_exec_metadata(rlvr={"terminal": True})
