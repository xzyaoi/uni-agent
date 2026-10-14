"""Configure native guest websites at batch start; keep evaluator secrets private."""

import json
import os
from pathlib import Path

run = Path(os.environ["RUN_DIR"])
names = ("WA_SHOPPING", "WA_SHOPPING_ADMIN", "WA_REDDIT", "WA_GITLAB", "WA_MAP", "WA_WIKIPEDIA", "WA_HOMEPAGE")
missing = [name for name in names if not os.environ.get(name)]
if missing:
    raise ValueError("Missing WebArena URLs: " + ", ".join(missing))
guest_env = {name: os.environ[name] for name in names}
guest_env["PLAYWRIGHT_BROWSERS_PATH"] = "/ms-playwright"
if os.environ.get("WA_FULL_RESET"):
    guest_env["WA_FULL_RESET"] = os.environ["WA_FULL_RESET"]
if os.environ.get("WEBARENA_EVAL_API_KEY"):
    guest_env["OPENAI_API_KEY"] = os.environ["WEBARENA_EVAL_API_KEY"]
if os.environ.get("WEBARENA_EVAL_BASE_URL"):
    guest_env["OPENAI_BASE_URL"] = os.environ["WEBARENA_EVAL_BASE_URL"]
# Avoid including guest evaluator credentials in task configuration/log dumps.
env_file = run / "webarena-guest-env.json"
descriptor = os.open(env_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
with os.fdopen(descriptor, "w") as stream:
    json.dump(guest_env, stream)
config_path = run / "task_config.yaml"
config = json.loads(config_path.read_text())
config[0]["sandbox"]["sandbox_kwargs"]["guest_env_file"] = str(env_file)
config_path.write_text(json.dumps(config, indent=2) + "\n")
