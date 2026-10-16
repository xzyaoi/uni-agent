#!/usr/bin/env python3
"""Faithful flash-sandbox SDK test (same v0.3.0 the enroot image uses).

Calls AsyncHTTPClient.start_sandbox() against api.opentela.ai with the OTELA
key from ~/.zshrc, for both the no-slash (307 -> parse error) and with-slash
endpoints. Prints the exact HTTP status / exception the job would see.
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
import traceback
from pathlib import Path


def load_key() -> str:
    # Try env first, then ~/.zshrc (matches run_train_qwen3p5_bristen.sh extraction)
    k = os.environ.get("OTELA_API_KEY", "").strip()
    if not k:
        z = Path.home() / ".zshrc"
        if z.exists():
            for line in z.read_text(errors="ignore").splitlines():
                m = re.match(r'\s*export\s+OTELA_API_KEY=(\S+)', line)
                if m:
                    k = m.group(1).strip().strip('"').strip("'")
                    break
    if not k:
        print("FATAL: no OTELA_API_KEY in env or ~/.zshrc", file=sys.stderr)
        sys.exit(2)
    return k


async def try_start(endpoint: str, api_key: str, label: str) -> None:
    print(f"\n===== {label} =====")
    print(f"endpoint: {endpoint}")
    try:
        from flash_sandbox import AsyncHTTPClient
    except Exception as e:
        print(f"  import failed: {e!r}")
        return

    client = AsyncHTTPClient(endpoint, api_key=api_key, timeout=30.0)
    image = "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907"
    print(f"  start_sandbox(type=docker, image={image}) ...")
    try:
        sb = await client.start_sandbox(type="docker", image=image)
        print(f"  ✅ STARTED: id={getattr(sb, 'id', sb)!r}")
        # best-effort cleanup
        try:
            await client.stop_sandbox(getattr(sb, "id", sb), cleanup=True, timeout=60.0)
            print("  stopped.")
        except Exception as e:
            print(f"  stop failed (ok): {e!r}")
    except Exception as e:
        print(f"  ❌ RAISED: {type(e).__name__}: {e!r}")
        # extract HTTP status if the SDK nests it
        s = repr(e)
        for tok in ("HTTP ", "status=", "status_code=", "SandboxHTTPError"):
            if tok in s:
                print(f"    ({tok} present in repr)")
        print("  --- traceback (last 8 frames) ---")
        tb = traceback.format_exc().splitlines()
        print("\n".join(tb[-8:] if len(tb) > 8 else tb))


async def main() -> None:
    key = load_key()
    print(f"OTELA_API_KEY loaded: len={len(key)}")
    # Confirm SDK version
    try:
        import flash_sandbox
        print(f"flash_sandbox version: {getattr(flash_sandbox, '__version__', '?')} "
              f"at {os.path.dirname(flash_sandbox.__file__)}")
    except Exception as e:
        print(f"flash_sandbox import failed before test: {e!r}")

    endpoints = [
        ("https://api.opentela.ai/v1/service/flash-sandbox",  "NO slash  (the old broken default)"),
        ("https://api.opentela.ai/v1/service/flash-sandbox/", "WITH slash (the fix in flash_sandbox.py:56)"),
    ]
    for ep, label in endpoints:
        await try_start(ep, key, label)


if __name__ == "__main__":
    asyncio.run(main())
