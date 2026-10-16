#!/usr/bin/env python3
"""One-shot libkrun go/no-go for api.opentela.ai, mirroring exactly what
run_train_qwen3p5_bristen.sh drives:
  FLASH_SANDBOX_ENDPOINT = https://api.opentela.ai/v1/service/flash-sandbox/
  FLASH_SANDBOX_TYPE     = libkrun
  OTELA_API_KEY          = from ~/.zshrc
Starts one libkrun sandbox (tail -f /dev/null, ttl 300s), prints the HTTP
status / detail, then stops it. Exit 0 if a sandbox STARTED, non-zero if not.
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
                if m:
                    k = m.group(1).strip().strip('"').strip("'"); break
    return k

async def main() -> int:
    key = load_key()
    if not key:
        print("FATAL: no OTELA_API_KEY in env or ~/.zshrc", file=sys.stderr); return 2
    endpoint = os.environ.get("FLASH_SANDBOX_ENDPOINT", "https://api.opentela.ai/v1/service/flash-sandbox/")
    backend = os.environ.get("FLASH_SANDBOX_TYPE", "libkrun")
    image = os.environ.get("PROBE_IMAGE", "sweb.eval.x86_64.docker")
    print(f"[probe] flash-sandbox SDK vs api.opentela.ai")
    print(f"[probe] endpoint={endpoint}")
    print(f"[probe] backend(type)={backend!r}  image={image!r}")
    print(f"[probe] OTELA_API_KEY len={len(key)} starts_sk={key[:3]=='sk-'}")
    from flash_sandbox import AsyncHTTPClient
    print(f"[probe] flash_sandbox import OK from {os.path.dirname(__import__('flash_sandbox').__file__)}")
    client = AsyncHTTPClient(endpoint, api_key=key, timeout=45.0)
    kw = {"image": image, "type": backend,
          "command": ["tail", "-f", "/dev/null"], "ttl_seconds": 300,
          "request_timeout": 45.0}
    try:
        sb = await client.start_sandbox(**kw)
        sid = getattr(sb, "id", sb)
        print(f"[probe] ✅ STARTED id={sid!r}  -> libkrun is HEALTHY")
        try:
            await client.stop_sandbox(sid, cleanup=True, timeout=60.0)
            print("[probe] stopped.")
        except Exception as e:
            print(f"[probe] stop failed (non-fatal): {e!r}")
        return 0
    except Exception as e:
        s = getattr(e, "status", None)
        d = getattr(e, "detail", None) or str(e)
        print(f"[probe] ❌ FAILED  status={s!r}  detail={d!r}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
