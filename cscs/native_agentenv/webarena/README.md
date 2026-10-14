# WebArena prerequisites for the native presets

WebArena episodes use the host-side ReAct agent and `browser` tool. A native
AgentENV guest runs BrowserGym, the bridge daemon and Chromium. The websites
themselves run on a separate WebArena deployment, reachable from CloudLab.
The training scripts do not deploy those websites or publish an image.

Use the [official site deployment instructions](https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md)
and [BrowserGym WebArena configuration guide](https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/webarena/README.md).
BrowserGym uses `WA_`-prefixed site variables. Some task evaluators use an
external model judge; configure its key separately from the policy gateway.

## Prepare the browser image and dataset

On an amd64 Docker host:

```bash
bash build_image.sh ghcr.io/your-account/webarena:0.14.3
docker push ghcr.io/your-account/webarena:0.14.3
bash export_task_ids.sh ghcr.io/your-account/webarena:0.14.3 /tmp/webarena-task-ids.json
```

Transfer that JSON manifest to `WORK_ROOT/data/webarena-task-ids.json` on
Clariden. The manifest enumerates the pinned image's actual registry instead
of assuming a fixed task count. Prefer setting WEBARENA_IMAGE to the published
digest, `ghcr.io/your-account/webarena@sha256:...`.

In `native_agentenv/config.env`, configure:

```bash
WEBARENA_IMAGE=ghcr.io/your-account/webarena:0.14.3
WEBARENA_ENV_FILE="${WORK_ROOT}/secrets/webarena.env"
# Optional non-default manifest location:
# WEBARENA_TASK_IDS=/absolute/path/webarena-task-ids.json
```

Copy `sites.env.example` to WEBARENA_ENV_FILE, fill in your website URLs and
evaluator credentials, and make the directory/file private (0700/0600).
From `native_agentenv/`, prepare the dataset:

```bash
bash setup.sh
bash prepare_webarena.sh
```

Then use one of the three `submit_webarena.sh` model launchers under `cscs/`.
They require WebArena rows, so a MiniWoB parquet cannot be used accidentally.

## Isolation, evaluation and traces

These presets default to **one concurrent episode** because a shared WebArena
deployment contains mutable state. Reset the websites between episodes/runs as
required by your deployment. Set WA_FULL_RESET only when that deployment
supports the upstream reset service; these scripts do not implement a website
reset controller. Do not raise WEBARENA_CONCURRENCY without independent site
instances and a suitable assignment/reset policy. Serialization alone does
not restore website state between episodes.

The batch launcher reads the private environment file and writes a private
guest-environment JSON file. Task configs contain its path; evaluator keys
are redacted from the native environment trace. This guest file is excluded
from the trace-export archive. Replaying external judge calls requires
supplying credentials again. Traces do not snapshot the external websites.

All model presets retain five training steps and disable separate validation.
WebArena's terminal reward evaluator still runs during training. Browser image
builds, hosted websites and training runs have not been exercised by adding
these example scripts.
