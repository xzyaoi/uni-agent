"""No-LLM BrowserGym smoke against flash-sandbox: replay the agent's actions.

What this validates (everything except the model call):

1. the task plumbing boots on the real backend: flash sandbox from the image,
   bridge daemon + headless Chromium inside it;
2. the exact tool path a react episode uses: ``BrowserTool.run`` -> one
   sandbox ``exec`` per action -> bridge daemon -> env step -> AXTree obs;
3. malformed-action feedback, done-episode semantics, the reward readback, and
   daemon teardown;
4. CPU + memory of the sandbox over time (a stdlib sampler inside the sandbox
   polls /proc: system-wide CPU %, per-group CPU/RSS for chrome vs python,
   MemAvailable) -- the numbers you need to size ``cpu_cores``/``memory_mb``
   for rollouts.

Usage::

    export OTELA_API=...                       # flash-sandbox API key (default env)
    export BROWSERGYM_IMAGE=ghcr.io/<owner>/browsergym:0.1.2
    python examples/browsergym/smoke_flash_actions.py \
        [--task browsergym/miniwob.click-test] [--seed 0] [--interval 1.0]

Artifacts: prints a timeline + summary; writes raw samples to
``bgym_resources_<task>.jsonl`` next to this script.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time
from pathlib import Path

from uni_agent.sandbox import build_sandbox
from uni_agent.tasks.browsergym.reward import compute_reward
from uni_agent.tasks.browsergym.task import (
    BrowserGymTaskConfig,
    _bridge_call,
    _start_bridge_daemon,
    build_browsergym_sandbox_config,
)
from uni_agent.tools.browser import BrowserTool

logger = logging.getLogger("bgym-smoke")

LOG_DIR = "/logs/bgym"

#: Stdlib-only resource sampler, written into the sandbox as /tmp/res_sampler.py.
SAMPLER = r"""
import json, os, time

HZ = os.sysconf("SC_CLK_TCK")
PAGE = os.sysconf("SC_PAGE_SIZE")
OUT = os.environ.get("SAMPLER_OUT", "/logs/bgym/resources.jsonl")
INTERVAL = float(os.environ.get("SAMPLER_INTERVAL", "1.0"))

def read_procs():
    rows = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            stat = open(f"/proc/{pid}/stat").read()
            comm = stat[stat.index("(") + 1 : stat.rindex(")")]
            rest = stat[stat.rindex(")") + 2 :].split()
            utime, stime, rss = int(rest[11]), int(rest[12]), int(rest[21])
            rows.append((pid, comm, utime + stime, rss * PAGE))
        except (OSError, ValueError, IndexError):
            continue
    return rows

def read_sys_cpu():
    parts = open("/proc/stat").readline().split()[1:]
    vals = [int(x) for x in parts]
    return sum(vals), vals[3] + (vals[4] if len(vals) > 4 else 0)  # total, idle

def read_meminfo():
    info = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        info[k] = int(v.strip().split()[0]) * 1024
    return info.get("MemTotal", 0), info.get("MemAvailable", 0)

def group(comm):
    if "chrom" in comm:
        return "chrome"
    if comm.startswith("python"):
        return "python"
    return "other"

prev_groups = {"chrome": 0.0, "python": 0.0, "other": 0.0}  # summed jiffies per group
for _pid, _c, _j, _r in read_procs():
    prev_groups[group(_c)] += _j
prev_total, prev_idle = read_sys_cpu()
fh = open(OUT, "w", buffering=1)
while True:
    time.sleep(INTERVAL)
    now = time.time()
    procs = read_procs()
    total, idle = read_sys_cpu()
    mem_total, mem_avail = read_meminfo()
    dt = INTERVAL
    cpu = {"chrome": 0.0, "python": 0.0, "other": 0.0}
    rss = {"chrome": 0, "python": 0, "other": 0}
    groups = {"chrome": 0.0, "python": 0.0, "other": 0.0}
    for pid, comm, jiffies, rssb in procs:
        g = group(comm)
        rss[g] += rssb
        groups[g] += jiffies
    # Group-TOTAL deltas (not per-pid): tolerant of pid churn between samples
    # and of the backend's odd per-pid jiffy accounting; clamp negatives.
    for g in cpu:
        cpu[g] = max(0.0, (groups[g] - prev_groups[g]) / HZ / dt * 100.0)
    prev_groups = groups
    sys_cpu = (1.0 - (idle - prev_idle) / max(1, total - prev_total)) * 100.0
    fh.write(json.dumps({
        "ts": now,
        "sys_cpu": round(sys_cpu, 1),
        "chrome_cpu": round(cpu["chrome"], 1),
        "python_cpu": round(cpu["python"], 1),
        "other_cpu": round(cpu["other"], 1),
        "chrome_rss_mb": round(rss["chrome"] / 1e6, 1),
        "python_rss_mb": round(rss["python"] / 1e6, 1),
        "other_rss_mb": round(rss["other"] / 1e6, 1),
        "mem_avail_mb": round(mem_avail / 1e6, 1),
        "mem_total_mb": round(mem_total / 1e6, 1),
        "n_proc": len(procs),
    }) + "\n")
    prev_total, prev_idle = total, idle
"""


def sparkline(values: list[float], width: int = 40, ceiling: float | None = None) -> str:
    """Tiny ASCII bar row so trends are visible without plotting libs."""
    if not values:
        return ""
    top = ceiling if ceiling is not None else max(values) or 1.0
    blocks = " .:-=+*#%@"
    step = max(1, len(values) // width)
    picked = [max(values[i : i + step]) for i in range(0, len(values), step)][:width]
    return "".join(blocks[min(len(blocks) - 1, int(v / top * (len(blocks) - 1)))] for v in picked)


async def main() -> int:
    parser = argparse.ArgumentParser(description="No-LLM BrowserGym action smoke on flash-sandbox.")
    parser.add_argument("--task", default="browsergym/miniwob.click-test")
    parser.add_argument("--benchmark", default="browsergym.miniwob")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--interval", type=float, default=1.0, help="Sampler period (s).")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    task_config = BrowserGymTaskConfig(
        sandbox={"provider": "flash_sandbox"},
        metadata={
            "task_id": args.task,
            "benchmark": args.benchmark,
            "seed": args.seed,
            "timeout_s": args.timeout,
            "obs_max_chars": 30000,
            "env_kwargs_json": json.dumps({"headless": True}),
        },
    )
    sandbox_config = build_browsergym_sandbox_config(task_config, task_config.metadata)

    t_start = time.monotonic()
    async with build_sandbox(sandbox_config) as sandbox:
        t_sandbox_up = time.monotonic() - t_start

        # resource sampler first, so it sees the Chromium cold start
        await sandbox.write_file("/tmp/res_sampler.py", SAMPLER)
        res = await sandbox.exec_shell(
            f"mkdir -p {LOG_DIR}\n"
            f"SAMPLER_OUT={LOG_DIR}/resources.jsonl SAMPLER_INTERVAL={args.interval:g} "
            f"nohup python /tmp/res_sampler.py > {LOG_DIR}/sampler.log 2>&1 &\n"
            f"echo $! > /tmp/res_sampler.pid\n"
            f"echo sampler-started",
            timeout=30.0,
        )
        if res.exit_code != 0:
            raise RuntimeError(f"sampler failed to start: {res.stderr[-500:]}")

        await _start_bridge_daemon(sandbox, task_config, task_config.metadata)
        t_daemon_up = time.monotonic() - t_start
        logger.info("sandbox up in %.1fs, daemon+env ready at %.1fs", t_sandbox_up, t_daemon_up)

        info = await _bridge_call(sandbox, task_config, "info", timeout=60.0)
        print(f"\ngoal: {info['goal']}")
        print(f"obs:\n{info['obs_text']}\n")
        early = await _bridge_call(sandbox, task_config, "result", timeout=30.0)
        print(f"state right after info: done={early['done']} reward={early['reward']} n_steps={early['n_steps']}")

        # the model-facing tool, exactly as the react Toolbox binds it
        tool = BrowserTool(sandbox, command_timeout=90.0)
        match = re.search(r"\[([\d\s]+)\] button", str(info.get("obs_text", "")))
        bid = re.sub(r"\s+", "", match.group(1)) if match else "13"

        script = [
            ("noop", "noop(wait_ms=100)"),
            ("malformed (model self-correct path)", "click(bid="),  # invalid syntax -> ToolError
            ("scroll", "scroll(delta_x=0, delta_y=50)"),
            ("click target", f"click(bid='{bid}')"),
            ("post-done step", "noop(wait_ms=100)"),  # episode already finished
        ]
        latencies: list[float] = []
        for label, action in script:
            t0 = time.monotonic()
            try:
                result = await tool.run({"action": action}, timeout=90.0)
                head = (result.text or "").replace("\n", " | ")[:110]
                print(f"[{time.monotonic() - t_start:7.1f}s] {label:35s} -> ok  : {head}")
            except Exception as exc:  # ToolError: the feedback the model would see
                print(f"[{time.monotonic() - t_start:7.1f}s] {label:35s} -> ERR : {str(exc)[:110]}")
            latencies.append((time.monotonic() - t0) * 1000.0)
            await asyncio.sleep(2 * args.interval)  # let the sampler see the action

        reward = await compute_reward(
            sandbox,
            python_bin=task_config.python_bin,
            bridge_path=task_config.bridge_path,
            bridge_port=task_config.bridge_port,
            result_file=f"{LOG_DIR}/result.json",
        )
        await sandbox.exec_shell(
            f"kill $(cat /tmp/res_sampler.pid) 2>/dev/null || true; echo sampler-stopped", timeout=15.0
        )
        await _bridge_call(sandbox, task_config, "shutdown", timeout=15.0)
        await asyncio.sleep(1.0)
        raw = await sandbox.read_file(f"{LOG_DIR}/resources.jsonl")
        daemon_log = await sandbox.exec_shell(f"tail -n 60 {LOG_DIR}/stdout.log", timeout=15.0)
        print("\n===== daemon log (key lines) =====")
        print(daemon_log.stdout.strip())

    samples = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    t_total = time.monotonic() - t_start

    print("\n===== episode verdict =====")
    print(
        f"reward={reward['reward']} success={reward['success']} env_steps={reward['n_steps']} "
        f"done={reward['done']} truncated={reward['truncated']}"
    )
    print(f"sandbox up: {t_sandbox_up:.1f}s | daemon ready: {t_daemon_up:.1f}s | total: {t_total:.1f}s")
    print("action exec latency (ms): " + ", ".join(f"{x:.0f}" for x in latencies))

    if samples:
        t0 = samples[0]["ts"]
        print(f"\n===== resources over {len(samples)} samples ({args.interval:g}s) =====")
        print(f"{'t':>6} {'sys%':>6} {'chr%':>6} {'py%':>6} {'chrRSS':>8} {'pyRSS':>7} {'avail':>8} {'nproc':>5}")
        for s in samples:
            print(
                f"{s['ts'] - t0:6.1f} {s['sys_cpu']:6.1f} {s['chrome_cpu']:6.1f} {s['python_cpu']:6.1f} "
                f"{s['chrome_rss_mb']:7.1f}M {s['python_rss_mb']:6.1f}M {s['mem_avail_mb']:7.1f}M {s['n_proc']:5d}"
            )
        print("\nsys CPU%   " + sparkline([s["sys_cpu"] for s in samples], ceiling=100))
        print("chrome RSS " + sparkline([s["chrome_rss_mb"] for s in samples]))
        peaks = {
            "sys_cpu_pct": max(s["sys_cpu"] for s in samples),
            "chrome_cpu_pct": max(s["chrome_cpu"] for s in samples),
            "chrome_rss_mb": max(s["chrome_rss_mb"] for s in samples),
            "python_rss_mb": max(s["python_rss_mb"] for s in samples),
            "mem_avail_low_mb": min(s["mem_avail_mb"] for s in samples),
            "mem_total_mb": samples[-1]["mem_total_mb"],
        }
        print("\npeaks: " + json.dumps(peaks))
        out = Path(__file__).parent / f"bgym_resources_{args.task.rsplit('.', 1)[-1]}.jsonl"
        out.write_text("\n".join(json.dumps(s) for s in samples) + "\n", encoding="utf-8")
        print(f"raw samples -> {out}")

    ok = bool(reward.get("success"))
    print(f"\nSMOKE {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
