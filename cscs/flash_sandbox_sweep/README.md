# Flash Sandbox matched sweep

Run from the Clariden repository checkout after the existing Flash Firecracker
cluster and its node are healthy. This directory does not deploy hosts, alter
AgentENV, mutate benchmark tasks, or stop AWS instances.

Each benchmark runs 128 fixed task groups × 8 responses (1024 episodes), seed
42, through the shared `cscs/sandbox_sweep/run_point.py` and group semaphore.
The four group limits 8/16/32/64 correspond to episode concurrency
64/128/256/512. Each point requests two debug nodes, four GH200 GPUs per node,
two TP4 replicas, and 45 minutes. Model: Qwen3.5-9B; prompt/response limits:
4096/8192; sampling: temperature=1, top_p=1; max_steps=200.

The two checked-in cohort pins match the exact parquet paths/hashes, ordered
128-task selections, task TOML hashes, and model config hash from
`outputs/pro_singlehost_c64_20261019` and
`outputs/verified_singlehost_c256_20261019`. Preparation reads shared inputs
without rewriting them, additionally freezes every selected task file's hash
and mode, and rechecks the complete proof before model initialization. Verified
also checks the original instruction/verifier/config hashes and pinned uv
archive. The AgentENV Verified finalizer is never called.

Stage the SDK and its runtime dependencies in an isolated `--target` directory,
without changing the existing AgentENV venv or container. Set `FLASH_SDK_PATH`
to that directory, containing `flash_sandbox/`, `httpx2/`, and dependencies.
Preparation fingerprints its Python sources and metadata; compute preflight
verifies those fingerprints and imports from this directory. The node script
re-adds this path after the container resets `PYTHONPATH`. `HARBOR_VENV` defaults
to `${SCRATCH}/venvs/uni-agentenv-v2`; its existing Harbor executable is used
through the point's `harbor-flash` wrapper.

Provide a user-owned mode-0600 `client.env` containing `OTELA_API_KEY`; it may
also contain `FLASH_SANDBOX_ENDPOINT`. An exported endpoint takes precedence.
Only the private file path is written to `launch.env`, never its API key.
The endpoint is public and the existing adapter logs it.

```bash
export FLASH_SDK_PATH=/iopsstor/scratch/cscs/xyao/flash-sdk/python-target
export FLASH_SECRET_FILE=/iopsstor/scratch/cscs/xyao/secrets/flash-sandbox/client.env
export FLASH_SANDBOX_ENDPOINT=http://YOUR_EXISTING_RUNTIME_DNS:8080
export FLASH_SANDBOX_TYPE=firecracker
BENCHMARK=swebench-pro-v2 \
  SWEEP_DIR="$PWD/runs/flash_pro_matched" \
  bash cscs/flash_sandbox_sweep/submit.sh
# After all four Pro jobs finish and owned sandbox cleanup succeeds:
BENCHMARK=swebench-verified \
  SWEEP_DIR="$PWD/runs/flash_verified_matched" \
  bash cscs/flash_sandbox_sweep/submit.sh
```

Set `SWEEP_PREPARE_ONLY=True` to build artifacts and validate inputs without
submitting. `SWEEP_PYTHON` defaults to the dataset-capable
`/iopsstor/scratch/cscs/xyao/venvs/uni-agentenv-v2/bin/python`.
The shared sequential queue controller submits subsequent points after each
previous job terminates and refuses to repeat a setup failure. A benchmark's
full sweep is prepared by default; `SWEEP_LEVELS` can select a bounded subset.

Authenticated preflight uses the public SDK's cluster status and verifies an
exactly one available node advertising Firecracker and zero sandboxes. The owned Harbor environment scopes
an SDK client subclass to its isolated trial process startup, restoring the
original class even on failure. It sends `vcpu_count=1`, `cpu_cores=1`,
`memory_mb=4096` explicitly and fsyncs a client-generated sandbox ID before dispatching create, then requires
the returned ID to match. Cleanup can recover a VM after a lost create response
or failed bootstrap. Cleanup stops only IDs whose trace matches the exact run
and endpoint; it never performs a global deletion or deletes inventory entries
it did not create. Normal adapter stop also deletes each completed trial. Both paths use the
same sweep-scoped SDK client and explicitly send `cleanup=true&async=false`
so DELETE waits for node teardown and coordinator route removal. A requested
pause retains `cleanup=false` while still waiting synchronously.

Verified calls the same pinned uv bootstrap helper as the baseline using only
`AGENTENV_UV_ARCHIVE` and its hash. These are tool-asset settings, with no
AgentENV credentials or endpoint. The existing Flash provider additionally
runs its rootless apt and git-safe-directory bootstrap for Firecracker as
well as other backends; this runner retains that adapter behavior.

GPU/DCGM, Ray/vLLM logs and model-call telemetry use the existing collector.
Sandbox host telemetry is managed separately. Offline checks:

```bash
python -m pytest cscs/flash_sandbox_sweep/tests -q
ruff check cscs/flash_sandbox_sweep
bash -n cscs/flash_sandbox_sweep/{submit,batch,node}.sh cscs/flash_sandbox_sweep/harbor-flash
```

Live SDK connectivity, guest topology, image readiness, the staged SDK imports
in the compute container, and a real model rollout require deployment smoke
checks; offline tests do not establish those properties.
