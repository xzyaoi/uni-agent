# BrowserGym × Uni-Agent × flash-sandbox

Train RL agents on [BrowserGym](https://github.com/servicenow/browsergym) web
tasks with flash-sandbox as the execution backend.

## Architecture

```
┌─ framework process (Ray node / local) ─┐        ┌─ flash sandbox (1 / episode) ────────────┐
│ react agent ──► browser tool ──► exec  │  HTTP  │ bridge daemon: BrowserEnv + headless     │
│      ▲                                 │ ─────► │ Chromium (+ bundled MiniWoB sites)       │
│ gateway / model                        │        │ result file: /logs/bgym/result.json      │
└────────────────────────────────────────┘        └───────────────────────────────────────────┘
```

- The **bridge daemon** (`bgym_bridge/bridge.py`) holds the stateful
  `BrowserEnv` (playwright's sync API is single-thread, so it serializes env
  ops on one worker thread) and serves `step`/`info`/`result`/`shutdown` on a
  sandbox-local port. The **client CLI** subcommands are stdlib-only, one
  sandbox `exec` per call.
- The **`browser` tool** (`uni_agent/tools/browser.py`) is the model-facing
  unit: one BrowserGym action string in, flattened AXTree out. Invalid actions
  return the env's own error so the policy self-corrects.
- The **task family** (`uni_agent/tasks/browsergym/`) boots the daemon, builds
  the prompt from the env's goal + first observation, runs the agent, and
  scores from the env's episode result — never the agent's claim.

## 1. Build & push the image

```bash
cd examples/browsergym
GHCR_OWNER=<your-github-user-or-org> TAG=0.1.0 ./build_and_push_ghcr.sh
# auth: docker login ghcr.io (PAT with write:packages), or export GHCR_TOKEN=...
```

The image (python:3.11-slim + browsergym-core/miniwob + playwright chromium +
the bridge) must be **public** on GHCR unless your flash-sandbox orchestrator
pulls GHCR with credentials. After the first build, pin the browsergym/playwright
versions in `Containerfile.browsergym` (`docker run <img> pip freeze | grep -E 'browsergym|playwright'`).

## 2. Smoke run (no trainer)

```bash
export OTELA_API_KEY=...                       # flash-sandbox API key
export BROWSERGYM_IMAGE=ghcr.io/<owner>/browsergym:0.1.0
python smoke_flash.py \
    --task browsergym.miniwob.miniwob_click-test \
    --base-url http://<your-model-endpoint>/v1 --model-name <model>
```

Watch for `bridge ready ...` in the logs (daemon boot ≈ first Chromium start,
budget 240 s), then browser-tool steps and finally the env's own
`reward`/`success` verdict.

## 3. Preprocess a benchmark into rows

```bash
pip install browsergym-core browsergym-miniwob     # listing only; episodes run in the image
python -m uni_agent.tasks.browsergym.preprocess \
    --benchmark browsergym.miniwob \
    --image ghcr.io/<owner>/browsergym:0.1.0 \
    --output browsergym_miniwob.parquet
```

## 4. Wire into the existing pipelines

- **Inference / trajectory generation**: point `run_infer.sh`-style entry points
  at the parquet + `task_config_browsergym_flash.yaml` (same shape as the
  SWE/Terminal-Bench flash configs).
- **RL**: reuse `examples/quickstart/training/train_qwen3_moe.sh` unchanged —
  the trainer is task-agnostic and consumes `TaskResult.reward` + gateway
  trajectories. Rows deep-merge onto the YAML defaults (image from the row,
  provider settings from the YAML).

## Choosing a benchmark

| Benchmark | Notes for flash-sandbox |
| --- | --- |
| MiniWoB++ (`browsergym.miniwob`) | start here — self-hosted sites, ~10–30-step episodes, binary rewards |
| WebArena | host the site stack **once** on a shared box; route base URLs via row `env_kwargs_json`; watch site-side concurrency at rollout scale |
| WorkArena | needs a ServiceNow instance (external, rate-limited) — avoid at scale |

## Gotchas

- **AXTree token budget**: MiniWoB pages are small; WebArena-class pages need a
  tighter `obs_max_chars` (and/or browsergym obs-builder filtering) plus a
  larger `max_total_tokens`.
- **Rootless Chromium** is handled by the bridge (launch patched to
  `chromium_sandbox=False`, `--disable-dev-shm-usage`); opt back in with
  `BROWSERGYM_CHROMIUM_SANDBOX=1` if your runtime allows.
- **Episode end**: teach the policy to finish with `stop(answer=...)`; calling
  `submit` also ends the rollout, and the task still queries the env's result.
- **Daemon death** mid-episode scores 0 with the error recorded in
  `extra_info` (bridge log: `/logs/bgym/stdout.log` inside the sandbox).
