# Native AgentENV runtime fleet

Preparation for the SWE-bench Pro concurrency sweep. The new run uses two
`m5d.metal` runtimes, providing 192 vCPUs in total. The first runtime also
hosts the Go gateway and scheduler; it is included in the runtime count.
The original four-node preparation remains the default (`--count 4`); use a
new state directory and explicit `--count 2 --name agentenv-pro-2x` for this run.

The topology follows the [official static multi-node guide](https://kvcache-ai.github.io/AgentENV/latest/deployment/static-multi-node.html).
Runtime HTTP 8000 and scheduler gRPC 9090 accept only traffic from the deployment
security group. Gateway 8080 and SSH 22 accept the scoped CIDRs copied from the
previous service group plus explicit `--allow-cidr` values. The previous public
`0.0.0.0/0` API rule is deliberately not inherited. The gateway requires the
shared API key. HTTP transport itself is not encrypted; TLS termination is a
separate deployment step if required by the network path.

## Prerequisites and plan

Use AWS CLI credentials for the existing account, Ubuntu SSH access via
`~/.ssh/agentenv-xiayao.pem`, and Python 3.10+. The existing instance profile
`agentenv-host` must retain access to
`s3://swe-bench-pro-templates/templates/`.

```bash
python3 cscs/agentenv_multinode/deploy.py plan \
  --count 2 --name agentenv-pro-2x --state outputs/agentenv-pro-2x/deployment
```

This queries AWS and writes private `plan.json` and `state.json`; it does not
create AWS resources. The Standard On-Demand quota must accommodate both
96-vCPU hosts plus existing usage. The original request for an increase to 576 vCPUs was filed
under request `ee5d9b5e0769486aabc287746157c840gXs68IdB`,
support case `179214207300203`. Quota status must be checked again at launch.

## Provision and inspect

```bash
python3 cscs/agentenv_multinode/deploy.py provision --apply \
  --count 2 --name agentenv-pro-2x --state outputs/agentenv-pro-2x/deployment
# After EC2 boot and SSH readiness:
python3 cscs/agentenv_multinode/deploy.py inspect \
  --count 2 --name agentenv-pro-2x --state outputs/agentenv-pro-2x/deployment \
  --source /actual/AgentENV/checkout
```

Provisioning checks quota before creating resources. One EC2 request uses
`MinCount=MaxCount=2` for this run, an immediately persisted client token, and experiment tags.
Repeat with the same state directory and arguments to recover interrupted
allocation. A partial existing fleet causes an explicit stop. Hosts use AMI
`ami-04643dd3d64c53242`, 350-GiB gp3 roots, the existing instance profile and
key pair, and the previous setup script's shared S3 user-data values.
Use `--subnet` to select a different default-VPC subnet before first launch;
otherwise the first sorted default subnet is selected. Capacity errors do not
silently allocate a second fleet in another availability zone.

Inspection records the runtime binary SHA256, source checkout commit, kernel,
KVM presence, service path, and health response without dumping environment
secrets. Source locations on the shared AMI must be established by inspection;
the default `/sandbox/AgentENV` is only a candidate.

## Configure the verified runtime build

The control plane is pinned to AgentENV stable v0.2.3 commit
`6cccaa7842bd5be2051111d4f74d9e37aa721244`. Go 1.25 is needed to build its
services. Install the pinned clean source checkout and Go toolchain on the
control host before configuring. Do not use the unpinned upstream installer.
The runtime binary is never upgraded by this script.

After establishing binary provenance, heartbeat capability, and shared S3
storage semantics, create a reviewed, non-secret `compatibility.json`:

```json
{
  "commit": "6cccaa7842bd5be2051111d4f74d9e37aa721244",
  "runtime_sha256": "<64 hex characters from verified installed server>",
  "heartbeat_supported": true,
  "shared_storage_verified": true,
  "runtime_api_addr": "127.0.0.1:8001",
  "recorder_active": true
}
```

These fields record an investigation, not automatic detection. Inspection reports
only the nonsensitive `API_ADDR` value and recorder service state. The previous
AMI topology is recorder 8000 forwarding to native 8001; preserve the recorder
and its trace collection. Static scheduler node endpoints remain private port
8000 so SDK and lifecycle traffic pass through the recorder. Set the reviewed
`runtime_api_addr` to the exact existing value (`127.0.0.1:8001` or
`0.0.0.0:8001`) and `recorder_active` to true. For a verified direct-runtime AMI,
use `0.0.0.0:8000` and false. Every host must match this topology before any
configuration changes. A matching source checkout alone does not establish the
installed binary's provenance.
If the AMI runtime differs, stop and prepare a reviewed, pinned runtime upgrade
separately; do not assert compatibility to bypass inspection.

```bash
python3 cscs/agentenv_multinode/deploy.py configure --apply \
  --count 2 --name agentenv-pro-2x --state outputs/agentenv-pro-2x/deployment \
  --source /actual/AgentENV/checkout --compatibility compatibility.json
```

All selected hosts must first pass the recorded runtime hash, pinned source commit,
KVM, and health checks. Then the script preserves existing runtime installation
paths, backs up `/etc/default/aenv`, installs unique runtime node IDs, and
configures shared secrets and heartbeats over private addresses. It builds the
Go services from the exact clean pinned source tree, configures round-robin
static discovery, and installs systemd services. Gateway request timeout is
**3600 seconds**, accommodating long sandbox command RPCs. API key and sandbox
seed are generated once, stored only in private `secrets.json`, and sent via
SSH stdin; they are absent from public state and command arguments.

## Required readiness gate before GPU allocation

The script deliberately leaves `gpu_launch_allowed=false` and phase
`configured_needs_heartbeat_auth_and_sdk_readiness`. Configuration is not a
readiness claim. The experiment launcher must establish and save evidence for:

1. Every runtime health hop reachable from the gateway.
2. Authenticated gateway `/nodes` showing every exact node ID with fresh,
   usable reports; unregistered static nodes must not be accepted.
3. Unauthenticated protected lifecycle requests rejected.
4. Native adapter creation, E2B SDK command execution, file transfer, and delete
   through the **new gateway URL**, including each runtime and secure envd token
   propagation. No benchmark verifier or held-out evaluation is needed.
5. Image warmup on every runtime for the exact selected Pro cohort, followed by
   successful temporary sandbox cleanup.
6. Per-node telemetry and gateway/scheduler logs collected before submission.

The readiness orchestration and sweep submission are intentionally outside this
preparation script. Never reuse the old `3.229.113.27:8000` endpoint.

## Recovery and teardown

`status` refreshes quota/plan information without AWS mutation. Keep the state
directory across retries; concurrent writers on the same directory are locked.
SSH/build failures leave persisted IDs for recovery and never auto-terminate a
partially inspected fleet. Remote reconfiguration reuses shared secrets and the
original backup. To revert configuration, stop the two Go services, restore
`/etc/agentenv/aenv.pre-multinode` to `/etc/default/aenv`, and restart `aenv`.

After archiving experiment artifacts and verifying checksums, terminate only
the IDs recorded in `state.json`, then delete this experiment's security
group once detached. Shared IAM profile, key pair, S3 bucket, AMI, and unrelated
instances are not deployment-owned resources. Automated teardown should check
experiment tags against the recorded IDs before mutation.

## Prepared cohort and monitoring

`outputs/pro_multinode_4x_20261016/EXPERIMENT.json` records the original four-node
preparation's quota blocker, exact workload, and readiness requirements at that
time. It is historical provenance; keep the two-node run's deployment and
sweep artifacts in new directories. The selected 128 tasks, order, and task TOML
hashes match the previous Pro sweep exactly; `preparation/cohort.json` contains
all 128 image references. Record immutable digests and warm each on every node
before starting the workload timer. The shared S3 templates do not establish
that every node's OCI conversion cache is warm.

`freeze_cohort.py` can reproduce that cohort manifest on Clariden with its
Python 3.12 environment and installed PyArrow:

```bash
python cscs/agentenv_multinode/freeze_cohort.py \
  --data /path/to/swe_bench_pro_v2_agentenv_train.parquet \
  --baseline-manifest /path/to/old/groups-8/manifest.json \
  --output /path/to/cohort.json
```

Once all deployment and image readiness requirements pass, use
`cscs/sandbox_sweep/submit.sh` with `BENCHMARK=swebench-pro-v2`, the **new**
`AGENTENV_ENDPOINT`, an isolated `AGENTENV_SECRET_DIR`,
`TASK_CONFIG_SOURCE=cscs/agentenv_multinode/task_config_pro.yaml`,
`AGENTENV_FIXED_CPU_CORES=1`, and `AGENTENV_SECURE=True`.
`SWEEP_INCLUDE_128=True` enables the existing clean-64-completion gate.
Endpoint settings survive sourcing the credentials file and populate `NO_PROXY`.
The deployed adapter must include the opt-in `AGENTENV_SECURE` change.

Start the fleet monitor after the sweep root exists:

```bash
python cscs/agentenv_multinode/monitor_cluster.py \
  --cluster-manifest outputs/agentenv-pro-2x/deployment/cluster.json \
  --remote-sweep-root /iopsstor/scratch/cscs/xyao/uni-agent/runs/NEW_SWEEP \
  --output outputs/NEW_SWEEP \
  --ssh-hostname clariden-ln001.cscs.ch
```

It saves per-node CPU/memory/VM metrics, control-plane metrics/journals, and GPU,
model, and native SDK metrics. Cluster CPU is the mean of equal-capacity 96-vCPU
nodes; VM and RAM totals require every selected node to report. The manifest's
`runtime_count` determines expected coverage and the CPU capacity label; legacy
manifests without it use the number of `instances`. Minute bins need at
least one sample per node; edge bins can be partial. GPU statistics retain the
existing sample-mean semantics without an all-eight-GPU coverage requirement.
`cluster_metrics_1min.csv` and resource figures record these definitions.

After every planned point terminates or is explicitly skipped, the monitor
stops its own collectors and each API recorder, archives raw host/journal/recorder
files (including rotated compressed traces), downloads and checksums every node's
archives, and writes `READY_TO_ARCHIVE.json`. It does not terminate EC2 instances.
Raw model/native/Harbor replay artifacts remain on persistent CSCS storage.

The previous Pro sweep had an accidental 180-second verifier command deadline.
The new sweep uses the corrected Harbor phase deadline, so a comparison against
old Pro results cannot isolate sandbox capacity alone. No held-out model
evaluation is required; per-episode reward verification remains enabled.

The non-secret `cluster.json` exposes `instances` with `instance_id`, `node_id`,
`public_ip`, `public_dns`, `private_ip`; plus `runtime_count`, `gateway_endpoint`, `region`,
`version`, `source_commit`, `secret_file` (path only), `readiness_state`, and
`gpu_launch_allowed`. Consumers must retain the readiness gate.
