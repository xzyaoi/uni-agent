#!/usr/bin/env python3
"""Probe every flash-sandbox backend type against api.opentela.ai via the
real SDK (flash-sandbox==0.3.0, same as the enroot image), with the OTELA
key from ~/.zshrc and a real swebench image. Prints HTTP status / exception
for each, so we can find any backend that has healthy nodes right now.
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
import traceback
from pathlib import Path


def load_key() -> str:
    k = os.environ.get("OTELA_API_KEY", "").strip()
    if not k:
        z = Path.home() / ".zshrc"
        if z.exists():
            for line in z.read_text(errors="ignore").splitlines():
                m = re.match(r'\s*export\s+OTELA_API_KEY=(\S+)', line)
                if m:
                    k = m.group(1).strip().strip('"').strip("'")
                    break
    return k


async def probe(endpoint: str, api_key: str, backend: str, image: str) -> None:
    label = "EMPTY(=cluster pick)" if backend == "" else backend
    print(f"\n--- backend={label!s} ---")
    from flash_sandbox import AsyncHTTPClient
    client = AsyncHTTPClient(endpoint, api_key=api_key, timeout=30.0)
    kw = {"image": image, "type": backend,
          "command": ["tail", "-f", "/dev/null"], "ttl_seconds": 300,
          "request_timeout": 30.0}
    try:
        sb = await client.start_sandbox(**kw)
        sid = getattr(sb, "id", sb)
        print(f"  ✅ STARTED id={sid!r}")
        try:
            await client.stop_sandbox(sid, cleanup=True, timeout=60.0)
            print("  stopped.")
        except Exception as e:
            print(f"  stop failed (ok): {e!r}")
    except Exception as e:
        # The SDK wraps status+detail; pull both for a clean one-liner.
        s = getattr(e, "status", None)
        d = getattr(e, "detail", None) or str(e)
        print(f"  ❌ status={s} detail={d!r}")


async def main() -> None:
    key = load_key()
    if not key:
        print("FATAL: no OTELA_API_KEY", file=sys.stderr); sys.exit(2)
    print(f"OTELA_API_KEY len={len(key)}")
    import flash_sandbox
    print(f"flash_sandbox at {os.path.dirname(flash_sandbox.__file__)}")
    ep = "https://api.opentela.ai/v1/service/flash-sandbox"
    img = "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907"
    print(f"endpoint={ep}\nimage={img}")
    for b in ("", "docker", "libkrun", "firecracker"):
        await probe(ep, key, b, img)


if __name__ == "__main__":
    asyncio.run(main())
