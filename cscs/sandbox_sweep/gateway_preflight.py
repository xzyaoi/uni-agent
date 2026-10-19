"""Check direct, authenticated gateway access before starting compute services."""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import urllib.error
import urllib.request


def check_gateway(endpoint: str, api_key: str, expected_nodes: int, *, timeout: float = 10) -> None:
    """Use no proxy and report only sanitized failure classes, never credentials."""
    failure = None
    if not endpoint or not api_key:
        failure = "MissingEndpointOrAPIKey"
    else:
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            request = urllib.request.Request(endpoint.rstrip("/") + "/nodes", headers={"X-API-Key": api_key})
            with opener.open(request, timeout=timeout) as response:
                nodes = json.load(response)
            if not isinstance(nodes, list):
                failure = "InvalidNodeList"
            elif len(nodes) != expected_nodes:
                failure = f"NodeCountMismatch expected={expected_nodes} observed={len(nodes)}"
            elif any(not isinstance(node, dict) or node.get("status") != "ready" for node in nodes):
                failure = "NodesNotReady"
        except urllib.error.HTTPError as exc:
            failure = f"HTTP{exc.code}"
            exc.close()
        except urllib.error.URLError as exc:
            failure = type(exc.reason).__name__
        except Exception as exc:
            failure = type(exc).__name__
    if failure:
        raise RuntimeError(f"AgentENV gateway preflight failed on {socket.gethostname()}: {failure}") from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-nodes", type=int, required=True)
    args = parser.parse_args()
    if args.expected_nodes < 1:
        parser.error("--expected-nodes must be positive")
    try:
        check_gateway(os.getenv("AGENTENV_ENDPOINT", ""), os.getenv("AGENTENV_API_KEY", ""), args.expected_nodes)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"AgentENV gateway preflight passed on {socket.gethostname()}: {args.expected_nodes} ready nodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
