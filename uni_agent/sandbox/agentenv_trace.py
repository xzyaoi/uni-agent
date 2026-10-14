"""Optional incremental workload recording for direct native AgentENV clients."""

from __future__ import annotations

import base64
import dataclasses
import functools
import inspect
import json
import logging
import os
import time
from pathlib import Path
from uuid import uuid4

from uni_agent.observability import _redact

logger = logging.getLogger(__name__)


def _json_value(value):
    if isinstance(value, bytes):
        return {"encoding": "base64", "data": base64.b64encode(value).decode("ascii")}
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported trace value: {type(value).__name__}")


class NativeTrace:
    def __init__(self, image: str):
        directory = os.environ.get("AGENTENV_TRACE_DIR")
        self.path = Path(directory) / f"{uuid4().hex}.jsonl" if directory else None
        self.image = image
        self.sandbox_id = None

    def emit(self, **event):
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "schema_version": 1,
                "timestamp": time.time(),
                "image": self.image,
                "sandbox_id": self.sandbox_id,
                **event,
            }
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(_redact(record), default=_json_value, ensure_ascii=False) + "\n")
        except Exception:
            logger.warning("Unable to append native sandbox trace %s", self.path, exc_info=True)


def traced(operation):
    """Record invocation before awaiting it, including interrupted operations."""

    def decorate(function):
        signature = inspect.signature(function)

        @functools.wraps(function)
        async def wrapped(self, *args, **kwargs):
            trace = self._trace
            if trace.path is None:
                return await function(self, *args, **kwargs)
            arguments = dict(signature.bind(self, *args, **kwargs).arguments)
            arguments.pop("self")
            identity = {"operation": operation, "operation_id": uuid4().hex}
            if hasattr(self, "_shell_id"):
                identity["shell_id"] = self._shell_id
            trace.emit(**identity, phase="begin", arguments=arguments)
            started = time.monotonic()
            try:
                result = await function(self, *args, **kwargs)
            except BaseException as exc:
                trace.sandbox_id = getattr(self, "_id", trace.sandbox_id)
                trace.emit(
                    **identity, phase="error", duration=time.monotonic() - started, error=f"{type(exc).__name__}: {exc}"
                )
                raise
            if operation == "start":
                trace.sandbox_id = self._id
            output = {"shell_id": result._shell_id} if operation == "open_shell" else result
            trace.emit(**identity, phase="end", duration=time.monotonic() - started, result=output)
            return result

        return wrapped

    return decorate
