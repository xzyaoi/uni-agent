#!/usr/bin/env python3
"""Final libkrun validation: create via start_sandbox(type=libkrun), then
shell_exec (the data plane our FlashSandboxSandbox wrapper uses) to confirm
the SWE image booted with /testbed + git + python3. Then stop.
"""
from __future__ import annotations
import asyncio, os, re, sys
from pathlib import Path

def load_key() -> str:
    k = os.environ.get("OTELA_API_KEY", "").strip()
    if not k:
        z = Path.home() / ".zshrc"
        if z.exists():
            for line in z.read_text(errors="ignore").splitlines():
                m = re.match(r'\s*export\s+OTELA_API_KEY=(\S+)', line)
                if m: k = m.group(1).strip().strip('"').strip("'"); break
    return k

async def main() -> None:
    key = load_key(); assert key, "no OTELA_API_KEY"
    reg = os.environ.get("FLASH_SANDBOX_IMAGE_REGISTRY", "ghcr.io/swe-images")
    image = "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907"
    first = image.split("/", 1)[0]
    if reg and "/" in image and not ("." in first or ":" in first or first == "localhost"):
        image = f"{reg.rstrip('/')}/{image.split('/', 1)[1]}"
    print(f"registry={reg} image={image}")
    from flash_sandbox import AsyncHTTPClient
    cli = AsyncHTTPClient("https://api.opentela.ai/v1/service/flash-sandbox",
                          api_key=key, timeout=300.0)
    sid = None
    try:
        sb = await cli.start_sandbox(image=image, type="libkrun",
                                     command=["tail", "-f", "/dev/null"],
                                     ttl_seconds=300, request_timeout=240.0)
        sid = str(getattr(sb, "id", sb))
        print(f"✅ STARTED id={sid}")
        cmd = "uname -a; echo ---; ls -la /testbed 2>&1 | head; echo ---; which git python3 pytest 2>&1; echo EXIT=$?"
        r = await cli.shell_exec(sid, cmd, timeout_ms=60000)
        out = getattr(r, "stdout", None) or getattr(r, "output", None) or str(r)
        print(f"shell_exec ->\n{str(out)[:800]}")
    finally:
        if sid:
            try: await cli.stop_sandbox(sid, cleanup=True, timeout=120.0); print("stopped.")
            except Exception as e: print(f"stop err: {e!r}")

if __name__ == "__main__":
    asyncio.run(main())
