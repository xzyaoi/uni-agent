#!/usr/bin/env python3
"""Minimal repro for the vLLM 0.24.0 `sample_tokens` RPC timeout
(EngineDeadError) that killed training job 3209975 on clariden (GH200 /
sm_90) after ~2h21m of rollout batches. The same setup reportedly works on
bristen (A100 / sm_80).

Root-cause hypothesis: vLLM's V1 AsyncLLM multiproc executor (shm_broadcast
IPC) deadlocks on GH200 with the stateful Qwen3.5-4B (GatedDeltaNet) model
under verl's "free_cache_engine" cycle that runs between every rollout batch:

    engine.sleep(level=1) -> engine.wake_up(tags=["kv_cache"]) ->
    engine.reset_prefix_cache(reset_connector=True)

This script drives AsyncLLM (the EXACT engine verl constructs, same args)
in a standalone loop -- generate CONCURRENCY requests (max_tokens) per
batch, then sleep/wake/reset -- WITHOUT verl, Ray, or the agent framework,
so a deadlock here localizes the bug to vLLM core on GH200.

Faithful to the failing rollout args:
  --distributed_executor_backend mp --tensor_parallel_size 2
  --enable_sleep_mode --enable_prefix_caching --enable_chunked_prefill
  --compilation_config {"cudagraph_mode": "FULL_AND_PIECEWISE"}
  --gpu_memory_utilization 0.7 --dtype bfloat16 --load_format dummy

Exit codes: 0 = ran to completion, NOT reproduced.
            1 = REPRODUCED (EngineDeadError / TimeoutError / sample_tokens).
            2 = other failure (API mismatch, OOM, etc.).
            3 = sleep API missing (cannot test hypothesis).

Tune via env: MODEL, TP, CONCURRENCY, MAX_TOKENS, MAX_MODEL_LEN,
  MAX_NUM_BATCHED_TOKENS, MAX_NUM_SEQS, GPU_MEMORY_UTILIZATION,
  CYCLES (0 = run until DURATION_S), DURATION_S.
"""
import asyncio
import inspect
import os
import sys
import time
import traceback

MODEL = os.environ.get("MODEL", "/iopsstor/scratch/cscs/xyao/data/models/Qwen3.5-4B")
TP = int(os.environ.get("TP", "2"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "8"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "16384"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "24576"))
MAX_NUM_BATCHED_TOKENS = int(os.environ.get("MAX_NUM_BATCHED_TOKENS", "24576"))
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "256"))
GPU_MEM_UTIL = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.7"))
CYCLES = int(os.environ.get("CYCLES", "0"))            # 0 = run until DURATION_S
DURATION_S = int(os.environ.get("DURATION_S", "1500"))  # 25 min default

PROMPT = ("Write a long Python program that prints the Fibonacci sequence "
          "using dynamic programming. Include comments and edge cases. Then "
          "extend it to compute the nth Fibonacci number modulo a prime, and "
          "print a table of the first 100 values. " * 6)


def _is_repro(e: BaseException) -> bool:
    name = type(e).__name__
    msg = str(e).lower()
    return ("EngineDeadError" in name or "TimeoutError" in name
            or "sample_tokens" in str(e)
            or "timed out" in msg
            or "enginecore encountered" in msg)


async def main() -> int:
    import torch
    import vllm
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.async_llm import AsyncLLM

    print("=" * 70, flush=True)
    print(f"[repro] vllm={vllm.__version__} torch={torch.__version__} "
          f"cuda={torch.version.cuda}", flush=True)
    try:
        print(f"[repro] GPU0={torch.cuda.get_device_name(0)} "
              f"cap={torch.cuda.get_device_capability(0)}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[repro] (could not read GPU name: {e})", flush=True)
    print(f"[repro] MODEL={MODEL}", flush=True)
    print(f"[repro] TP={TP} CONCURRENCY={CONCURRENCY} MAX_TOKENS={MAX_TOKENS} "
          f"MAX_MODEL_LEN={MAX_MODEL_LEN} MAX_NUM_BATCHED_TOKENS="
          f"{MAX_NUM_BATCHED_TOKENS}", flush=True)
    print(f"[repro] enable_sleep_mode=True enable_prefix_caching=True "
          f"cudagraph_mode=FULL_AND_PIECEWISE gpu_mem_util={GPU_MEM_UTIL}",
          flush=True)
    print(f"[repro] CYCLES={'until DURATION_S='+str(DURATION_S) if not CYCLES else CYCLES}",
          flush=True)
    print("=" * 70, flush=True)

    # ---- construct AsyncLLM exactly like verl vllm_async_server.run_server ----
    args = [
        "--model", MODEL,
        "--dtype", "bfloat16",
        "--load_format", "dummy",
        "--distributed_executor_backend", "mp",
        "--tensor_parallel_size", str(TP),
        "--max_model_len", str(MAX_MODEL_LEN),
        "--max_num_seqs", str(MAX_NUM_SEQS),
        "--max_num_batched_tokens", str(MAX_NUM_BATCHED_TOKENS),
        "--enable_chunked_prefill",
        "--enable_prefix_caching",
        "--enable_sleep_mode",
        "--gpu_memory_utilization", str(GPU_MEM_UTIL),
        "--compilation_config", '{"cudagraph_mode": "FULL_AND_PIECEWISE"}',
        "--disable_log_stats",
        "--trust_remote_code",
    ]
    engine_args = AsyncEngineArgs.from_cli_args(args)
    usage_context = UsageContext.OPENAI_API_SERVER
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)

    fn_args = set(inspect.signature(AsyncLLM.from_vllm_config).parameters.keys())
    kwargs = {}
    if "enable_log_requests" in fn_args:
        kwargs["enable_log_requests"] = engine_args.enable_log_requests
    if "disable_log_stats" in fn_args:
        kwargs["disable_log_stats"] = engine_args.disable_log_stats
    engine = AsyncLLM.from_vllm_config(
        vllm_config=vllm_config, usage_context=usage_context, **kwargs)
    print("[repro] AsyncLLM constructed", flush=True)

    # sanity: the sleep/wake/reset cycle is the WHOLE point. Fail loudly if the
    # engine lacks these (so we never silently run the wrong experiment).
    for m in ("sleep", "wake_up", "reset_prefix_cache", "generate"):
        if not hasattr(engine, m):
            print(f"[repro] engine is MISSING required method {m!r} -- "
                  f"cannot test the sleep/wake hypothesis in vllm "
                  f"{vllm.__version__}.", flush=True)
            return 3
    try:
        await engine.reset_mm_cache()
    except Exception as e:  # noqa: BLE001
        print(f"[repro] reset_mm_cache non-fatal: {type(e).__name__}: {e}",
              flush=True)

    sp = SamplingParams(temperature=1.0, top_p=1.0, top_k=-1,
                        max_tokens=MAX_TOKENS, ignore_eos=True)

    async def one_gen(i: int, cycle: int):
        rid = f"r{i}_c{cycle}"
        n = 0
        async for out in engine.generate(prompt=PROMPT, sampling_params=sp,
                                         request_id=rid):
            n = len(out.outputs[0].token_ids) if out.outputs else 0
        return n

    async def watchdog(stop_at: float):
        while time.time() < stop_at:
            await asyncio.sleep(15)
            print(f"[repro][hb] alive, elapsed {time.time()-start:.0f}s, "
                  f"cycle {cycle}", flush=True)

    cycle = 0
    start = time.time()
    total_tokens = 0
    stop_at = start + DURATION_S + 60
    wd = asyncio.create_task(watchdog(stop_at))
    try:
        while True:
            if CYCLES and cycle >= CYCLES:
                break
            if not CYCLES and (time.time() - start) > DURATION_S:
                break

            t0 = time.time()
            results = await asyncio.gather(
                *[one_gen(i, cycle) for i in range(CONCURRENCY)],
                return_exceptions=True)
            dt = time.time() - t0
            excs = [r for r in results if isinstance(r, Exception)]
            tok = sum(r for r in results if isinstance(r, int))
            total_tokens += tok
            print(f"[repro] cycle {cycle}: generate {dt:.1f}s, {tok} tok "
                  f"({tok/max(dt,1e-9):.0f} tok/s), {len(excs)} excs, "
                  f"elapsed {time.time()-start:.0f}s", flush=True)
            if excs:
                print(f"[repro] first exc in cycle {cycle}:", flush=True)
                traceback.print_exception(type(excs[0]), excs[0],
                                          excs[0].__traceback__)
                raise excs[0]

            # ---- the free_cache_engine cycle (COLOCATED), like verl ----
            t1 = time.time()
            await engine.sleep(level=1)
            await engine.wake_up()          # default tags include kv_cache
            await engine.reset_prefix_cache(reset_connector=True)
            print(f"[repro] cycle {cycle}: sleep/wake/reset took "
                  f"{time.time()-t1:.1f}s", flush=True)
            cycle += 1

        print("=" * 70, flush=True)
        print(f"[repro] DONE: {cycle} cycles, {total_tokens} tokens in "
              f"{time.time()-start:.0f}s. NO EngineDeadError / sample_tokens "
              f"timeout reproduced.", flush=True)
        print("=" * 70, flush=True)
        return 0
    except Exception as e:  # noqa: BLE001
        repro = _is_repro(e)
        print("=" * 70, flush=True)
        print(f"[repro] {'REPRODUCED' if repro else 'FAILED'}: "
              f"{type(e).__name__}: {str(e)[:500]}", flush=True)
        traceback.print_exc()
        print("=" * 70, flush=True)
        return 1 if repro else 2
    finally:
        wd.cancel()
        try:
            if hasattr(engine, "shutdown"):
                await engine.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
