"""Opt-in incremental model-call capture, correlated with native sandbox traces."""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)
_context: ContextVar[dict | None] = ContextVar("capture_context", default=None)


@contextmanager
def capture_context(**metadata):
    token = _context.set(metadata)
    try:
        yield
    finally:
        _context.reset(token)


def _redact(value):
    if isinstance(value, dict):
        return {
            key: "[redacted]"
            if str(key).upper() in {"AUTHORIZATION", "X-API-KEY", "API_KEY"}
            or str(key).upper().endswith(("_API_KEY", "_TOKEN", "_PASSWORD", "_SECRET"))
            else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


def _append(path, record):
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(_redact(record), ensure_ascii=False) + "\n")
    except Exception:
        logger.warning("Unable to append model capture %s", path, exc_info=True)


async def traced_chat_completion(model, body):
    """Capture full request/response bodies without HTTP authentication headers.

    One operation covers the existing client's retry loop. Its original retry
    behavior is retained; failed HTTP attempts remain in the worker logs.
    """
    directory = os.environ.get("UNI_AGENT_TRACE_DIR")
    if not directory:
        return await model._post_chat_completion(body)
    if not hasattr(model, "_capture_id"):
        model._capture_id = uuid4().hex
    path = Path(directory) / f"llm-{model._capture_id}.jsonl"
    identity = {
        "schema_version": 1,
        "operation": "chat_completion",
        "operation_id": uuid4().hex,
        "model_client_id": model._capture_id,
        "pid": os.getpid(),
        **(_context.get() or {}),
    }
    _append(path, {**identity, "phase": "begin", "timestamp": time.time(), "request": body})
    started = time.monotonic()
    try:
        response = await model._post_chat_completion(body)
    except BaseException as exc:
        _append(
            path,
            {
                **identity,
                "phase": "error",
                "timestamp": time.time(),
                "duration": time.monotonic() - started,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    _append(
        path,
        {
            **identity,
            "phase": "end",
            "timestamp": time.time(),
            "duration": time.monotonic() - started,
            "response": response,
        },
    )
    return response
