#!/usr/bin/env python3
"""End-to-end libkrun smoke: create a sandbox with the real SWE image (GHCR
remap via FLASH_SANDBOX_IMAGE_REGISTRY), wait ready, run uname + ls /testbed,
then stop. Validates the full path the training uses with libkrun.
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path


def load_key() -> str:
    k = os.environ.get("OTELA_API_KEY", "").strip()
    if not k:
        z = Path.home() / ".zshrc"
        if z.exists():
            for line in z.read_text(errors="ignore").splitlines():
                m = re.match(r'\s*export\s+OTELA_API_KEY=(\S+)', line)
                if m:
                    k = m.group(1).strip().strip('"').strip("'"); break
    return k


async def main() -> None:
    key = load_key()
    assert key, "no OTELA_API_KEY"
    os.environ["FLASH_SANDBOX_IMAGE_REGISTRY"] = os.environ.get(
        "FLASH_SANDBOX_IMAGE_REGISTRY", "ghcr.io/swe-images")
    os.environ["FLASH_SANDBOX_TYPE"] = "libkrun"
    from flash_sandbox import AsyncHTTPClient
    # Mirror our wrapper's registry remap exactly.
    image = "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907"
    reg = os.environ["FLASH_SANDBOX_IMAGE_REGISTRY"]
    first = image.split("/", 1)[0]
    if reg and "/" in image and not ("." in first or ":" in first or first == "localhost"):
        image = f"{reg.rstrip('/')}/{image.split('/', 1)[1]}"
    print(f"registry={reg}  resolved image={image}")
    ep = "https://api.opentela.ai/v1/service/flash-sandbox"
    cli = AsyncHTTPClient(ep, api_key=key, timeout=300.0)
    print("start_sandbox(type=libkrun) ...")
    try:
        sb = await cli.start_sandbox(image=image, type="libkrun",
                                     command=["tail", "-f", "/dev/null"],
                                     ttl_seconds=300, request_timeout=240.0)
        sid = getattr(sb, "id", sb)
        print(f"✅ STARTED id={sid!r}")
        for cmd in (["uname", "-a"], ["ls", "-la", "/testbed"],
                    ["ls", "/"], ["sh", "-c", "which git python3 pytest 2>&1"]):
            try:
                r = await cli.exec_sandbox(sid, command=cmd, timeout=60.0)
                out = getattr(r, "stdout", r) or ""
                print(f"$ {' '.join(cmd)} -> rc={getattr(r,'returncode','?')}\n{str(out)[:400]}")
            except Exception as e:
                print(f"$ {' '.join(cmd)} -> ERR {e!r}")
    finally:
        try: await cli.stop_sandbox(sid, cleanup=True, timeout=120.0); print("stopped.")
        except Exception as e: print(f"stop err: {e!r}")


if __name__ == "__main__":
    asyncio.run(main())
