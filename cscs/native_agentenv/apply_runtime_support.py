"""Apply small, idempotent native WebArena extensions to the pinned source snapshot."""

import os
from pathlib import Path

runtime = Path(os.environ["WORK_ROOT"]) / "runtime"


def replace(path, old, new):
    source = path.read_text()
    if new in source:
        return
    if source.count(old) != 1:
        raise RuntimeError(f"Unexpected source snapshot: cannot apply native extension to {path}")
    path.write_text(source.replace(old, new, 1))


replace(
    runtime / "uni_agent/tasks/browsergym/task.py",
    '    elif sandbox_config.provider == "modal":',
    '    elif sandbox_config.provider == "agentenv":\n'
    '        kwargs.setdefault("start_timeout", 600.0)\n'
    '    elif sandbox_config.provider == "modal":',
)
provider = runtime / "uni_agent/sandbox/agentenv.py"
replace(
    provider,
    "        env_vars: dict[str, str] | None = None,\n",
    "        env_vars: dict[str, str] | None = None,\n        guest_env_file: str | None = None,\n",
)
replace(
    provider,
    "        self.env_vars = dict(env_vars or {})\n",
    "        self.env_vars = dict(env_vars or {})\n"
    "        if guest_env_file:\n"
    "            import json\n"
    '            with open(guest_env_file, encoding="utf-8") as stream:\n'
    "                guest_env = json.load(stream)\n"
    "            if not isinstance(guest_env, dict):\n"
    '                raise ValueError("guest_env_file must contain a JSON object")\n'
    "            if any(not isinstance(k, str) or not isinstance(v, str) for k, v in guest_env.items()):\n"
    '                raise ValueError("guest_env_file must contain a string-to-string JSON mapping")\n'
    "            self.env_vars.update(guest_env)\n",
)
trace = runtime / "uni_agent/sandbox/agentenv_trace.py"
replace(
    trace,
    "def _json_value(value):\n",
    "def _redact(value):\n"
    "    if isinstance(value, dict):\n"
    '        sensitive = ("API_KEY", "TOKEN", "PASSWORD", "SECRET", "AUTHORIZATION")\n'
    "        return {\n"
    '            k: "[redacted]" if any(word in str(k).upper() for word in sensitive) else _redact(v)\n'
    "            for k, v in value.items()\n"
    "        }\n"
    "    if isinstance(value, (list, tuple)):\n"
    "        return [_redact(item) for item in value]\n"
    "    return value\n\n\ndef _json_value(value):\n",
)
replace(
    trace,
    "json.dumps(record, default=_json_value, ensure_ascii=False)",
    "json.dumps(_redact(record), default=_json_value, ensure_ascii=False)",
)
print("Native BrowserGym and private guest environment support prepared")
