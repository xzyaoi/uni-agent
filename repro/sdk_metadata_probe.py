#!/usr/bin/env python3
"""Decisive metadata probe: replicate the TRAINING request (metadata bag +
same SDK + same image) against BOTH no-slash and with-slash endpoints, plus
a no-metadata control. Tells us which of (a) trailing-slash-307-on-POST,
(b) metadata-is-the-culprit, (c) service-recovered is the real cause.
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

# Representative rollout-group metadata bag (matches exec_context shape:
# group_id, step, staleness, rlvr{group,terminal}, llm{model}).
META = {
    "group_id": "probe-group-001",
    "step": 0,
    "staleness": 0,
    "rlvr": {"group": "probe-group-001", "terminal": False},
    "llm": {"model": "qwen35-4b"},
}

async def probe(ep: str, api_key: str, image: str, *, with_meta: bool) -> None:
    tag = f"{'META' if with_meta else 'NO-META'}  ep={'NOslash' if ep.endswith('sandbox') else 'WITHslash'}"
    print(f"\n===== {tag} =====")
    from flash_sandbox import AsyncHTTPClient
    cli = AsyncHTTPClient(ep, api_key=api_key, timeout=60.0)
    kw = {"image": image, "type": "libkrun",
          "command": ["tail", "-f", "/dev/null"],
          "ttl_seconds": 300, "request_timeout": 240.0}
    if with_meta:
        kw["metadata"] = META
    try:
        sb = await cli.start_sandbox(**kw)
        sid = getattr(sb, "id", sb)
        print(f"  ✅ STARTED id={sid!r}")
        try: await cli.stop_sandbox(sid, cleanup=True, timeout=60.0); print("  stopped.")
        except Exception as e: print(f"  stop err (ok): {e!r}")
    except Exception as e:
        s = getattr(e, "status", None); d = getattr(e, "detail", None) or str(e)
        print(f"  ❌ status={s!r} detail={d!r}")

async def main() -> None:
    key = load_key(); assert key, "no OTELA_API_KEY"
    reg = os.environ.get("FLASH_SANDBOX_IMAGE_REGISTRY", "ghcr.io/swe-images")
    image = "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907"
    first = image.split("/", 1)[0]
    if reg and "/" in image and not ("." in first or ":" in first or first == "localhost"):
        image = f"{reg.rstrip('/')}/{image.split('/', 1)[1]}"
    print(f"registry={reg} image={image} keylen={len(key)}")
    # 1. Control: no metadata, both slash variants (reproduces my earlier probe)
    await probe("https://api.opentela.ai/v1/service/flash-sandbox",  key, image, with_meta=False)
    await probe("https://api.opentela.ai/v1/service/flash-sandbox/", key, image, with_meta=False)
    # 2. THE TEST: WITH metadata, both slash variants (replicates training)
    await probe("https://api.opentela.ai/v1/service/flash-sandbox",  key, image, with_meta=True)
    await probe("https://api.opentela.ai/v1/service/flash-sandbox/", key, image, with_meta=True)

if __name__ == "__main__":
    asyncio.run(main())
