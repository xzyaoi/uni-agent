"""Install the capture hook into the deployed ReAct client without replacing its implementation."""

import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("runtime", type=Path)
args = parser.parse_args()
path = args.runtime / "uni_agent/agents/react/model.py"
source = path.read_text()
marker = "data = await self._post_chat_completion(body)"
replacement = "data = await traced_chat_completion(self, body)"
if replacement not in source:
    if source.count(marker) != 1:
        raise RuntimeError("Unexpected model client: capture hook requires exactly one chat request")
    source = source.replace(marker, replacement)
    if source.count("import aiohttp\n") != 1:
        raise RuntimeError("Unexpected model client imports")
    source = source.replace(
        "import aiohttp\n", "import aiohttp\n\nfrom uni_agent.observability import traced_chat_completion\n", 1
    )
    path.write_text(source)
print("Incremental model capture installed")
