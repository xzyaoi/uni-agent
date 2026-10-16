# Native AgentENV baseline

The `agentenv` sandbox provider calls native AgentENV directly. Creation and
deletion use its control-plane API; commands and files use the E2B SDK. No
scheduler, transparent proxy, or scheduler boundary endpoint is involved.

Install `e2b==2.20.0` in each client environment. Persistent shells reuse
`meta.baselines.clients.persistent_shell` from the Flash Sandbox experiment
repository, so put that repository (or its deployed client source bundle) on
`PYTHONPATH`. This preserves the same native persistent-shell implementation
used by the reference baseline. One-shot commands and file operations do not
require the experiment source bundle.

```bash
export AGENTENV_ENDPOINT=http://127.0.0.1:8000
export AGENTENV_API_KEY=<native-agentenv-key>
export PYTHONPATH=/path/to/flash-sandbox:/path/to/uni-agent
```

For the Clariden training recipe, the native API is called directly at
`http://clnode286.clemson.cloudlab.us:8000`, with its API key loaded from a
private environment file. No SSH tunnel or scheduler is used. Use the hostname:
the compute network permits it but categorizes the bare IP as unrated and blocks
it. Model requests still use the training gateway; the AgentENV endpoint is only
the sandbox transport.

Native sandbox configuration:

```yaml
sandbox:
  provider: agentenv
  runtime_timeout: 14400
  sandbox_kwargs:
    api_key_env: AGENTENV_API_KEY
```

Explicit variables can be set with `sandbox_kwargs.env_vars` or per command
with `exec(..., env=...)`. The provider applies these after the SDK's login
shell initialization so `PATH` overrides take effect. The current AgentENV
runtime does not automatically reproduce every OCI image environment variable;
for example, the Flipt image's Go binary is `/usr/local/go/bin/go` but that
directory is absent from its default SDK `PATH`. Preserve the image's required
environment explicitly before using its verifier. Harbor task
`[environment].env` values are forwarded to the native provider.

For SWE-bench Pro V2, use
`task_config_harbor_swe_bench_pro_v2_native.yaml` with the prepared Harbor training
dataset. Harbor runs the native AgentENV environment and the host ReAct agent.
Public-network training copies remain required because this adapter does not
enforce Harbor's offline network policy.

The CloudLab baseline at `clnode286.clemson.cloudlab.us` currently runs native
AgentENV v0.2.3 with an existing header-timeout patch, **4 vCPUs and 4096 MiB per
VM**. The deployment manifest records the exact binary and configuration hashes.
AgentENV machine resources are server-side settings; the Harbor environment
rejects resource overrides that disagree with its declared fixed shape. Match
that shape in comparison runs.

The native training recipe is
`examples/quickstart/training/submit_qwen35_27b_pro_v2_agentenv_clariden.sh`.
It runs Qwen3.5-27B for five steps, saves at step 5, disables validation, and
limits the framework to 32 concurrent sandbox sessions. Prepare all 642 tasks
with their exact OCI environment and working directory before submission:

```bash
python -m uni_agent.tasks.swe_bench_pro.prepare_agentenv \
  --task-root /path/to/official/v2/tasks \
  --local-save-dir /path/to/fresh/native-dataset \
  --remote-task-root /iopsstor/scratch/cscs/xyao/data/uni_agent/swe_bench_pro_v2_agentenv/tasks
```

Preparation uses `skopeo inspect --config` and caches each image configuration.
It preserves the official verifier files and fails if any image cannot be
inspected. Only task environment, workdir, and the explicit online training
network policy are adapted.

Deployment files on CloudLab:

```text
/sandbox/study/cloudlab-agentenv-native-20261013/
  client.env       # private endpoint/key file; mode 0600
  manifest.json    # runtime provenance and resource shape
  src/            # Uni-Agent provider and reference native-shell client
  venv/           # pinned native SDK environment
```

Native creation and command execution are not automatically replayed after an
ambiguous failure. Persistent-shell output merges stdout and stderr; one-shot
commands retain their separate streams.

## Recording workloads for replay

Set `AGENTENV_TRACE_DIR` on the Harbor process to enable one incremental JSONL
file per sandbox. The native training wrapper enables this automatically under
`logs/Uni-Agent-Qwen3.5-27B-SWE-bench-Pro-V2-AgentENV/native-traces/<job-id>`.
Each operation has begin/end/error events, epoch timestamps, duration, image,
sandbox ID, and a unique operation ID. Shell operations also have a shell ID.
Records preserve commands, cwd, environment, timeouts, exit status and output;
file transfers preserve bytes as base64. API credentials are not recorded.
Harbor agent metadata links the trace file to the trial's retained transcript.

These are native client workload traces, not the scheduler's SQLite recording
format. The existing `bench_replay.py` harness needs an importer for this JSONL
schema before it can consume them. File transfers and shell IDs should be
preserved by that importer. Collection alone does not guarantee deterministic
replay of network requests or model responses. Tracing applies to newly started
Harbor processes; operations before activation cannot be recovered.
