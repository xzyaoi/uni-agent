import importlib
import json
import os
from pathlib import Path
import socket
import urllib.request

source = Path(os.environ["REPO_DIR"])
for name in ("flash_sandbox", "uni_agent.sandbox.flash_sandbox", "verl"):
    module = importlib.import_module(name)
    location = Path(module.__file__).resolve()
    if not location.is_relative_to(source):
        raise RuntimeError(f"Unexpected {name} import: {location}")
    print(f"[preflight] {socket.gethostname()} {name}={location}", flush=True)
from flash_sandbox import AsyncHTTPClient
from uni_agent.tasks.harbor.flash_environment import FlashSandboxEnvironment
request = urllib.request.Request(
    os.environ["FLASH_SANDBOX_ENDPOINT"].rstrip("/") + "/status",
    headers={"Authorization": "Bearer " + os.environ["OTELA_API_KEY"]},
)
with urllib.request.urlopen(request, timeout=20) as response:
    if response.status != 200:
        raise RuntimeError(f"Scheduler status HTTP {response.status}")
    status = json.load(response)
healthy = [node for node in status.get("nodes", [])
           if node.get("status") == "healthy" and "firecracker" in node.get("backend_types", [])]
if not healthy:
    raise RuntimeError("Scheduler has no healthy firecracker node")
print(f"[preflight] {socket.gethostname()} scheduler reachable", flush=True)
