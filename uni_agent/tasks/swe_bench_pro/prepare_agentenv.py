"""Prepare V2 Harbor training tasks with explicit OCI environment for AgentENV."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import Dataset

from .preprocess import build_swe_bench_pro


def _image_config(image: str, cache: Path) -> dict:
    """Read the amd64 OCI configuration without downloading image layers."""
    cached = cache / f"{hashlib.sha256(image.encode()).hexdigest()}.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    for attempt in range(3):
        try:
            result = subprocess.run(
                ["skopeo", "inspect", "--override-arch", "amd64", "--config", f"docker://{image}"],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            config = json.loads(result.stdout)["config"]
            env = {}
            for item in config.get("Env") or []:
                name, separator, value = item.partition("=")
                if not name or not separator:
                    raise ValueError(f"Invalid OCI environment entry for {image}: {item!r}")
                env[name] = value
            data = {"image": image, "env": env, "workdir": config.get("WorkingDir") or "/"}
            cached.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            return data
        except (subprocess.SubprocessError, ValueError, KeyError) as exc:
            if attempt == 2:
                detail = getattr(exc, "stderr", None) or str(exc)
                raise RuntimeError(f"Cannot inspect {image}: {detail}") from exc
            time.sleep(2**attempt)
    raise AssertionError("Unreachable")


def prepare_agentenv(
    task_root: str,
    local_save_dir: str,
    remote_task_root: str | None = None,
    workers: int = 8,
) -> Path:
    """Preserve every V2 task and its verifier, making image ENV explicit."""
    import toml

    if workers < 1:
        raise ValueError("workers must be positive")
    destination = Path(local_save_dir).expanduser().resolve()
    output = destination / "swe_bench_pro_v2_agentenv_train.parquet"
    if output.exists() or (destination / "tasks").exists():
        raise FileExistsError(f"Choose a fresh destination for task copies: {destination}")
    dataset = build_swe_bench_pro(dataset_config="default", tasks_dir=task_root)
    rows = list(dataset)
    tasks = []
    for row in rows:
        metadata = row["extra_info"]["tools_kwargs"]["task"]["metadata"]
        source = Path(metadata["task_path"])
        config = toml.loads((source / "task.toml").read_text(encoding="utf-8"))
        image = config["environment"]["docker_image"]
        if not image.startswith("ghcr.io/scaleapi/swe-bench_pro-v2:"):
            raise ValueError(f"Unexpected V2 image: {image}")
        tasks.append((metadata, source, config, image))
    cache = destination / "image-configs"
    cache.mkdir(parents=True, exist_ok=True)
    images = sorted({task[3] for task in tasks})
    configs = {}
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(_image_config, image, cache): image for image in images}
        for count, future in enumerate(as_completed(pending), 1):
            image = pending[future]
            try:
                configs[image] = future.result()
            except Exception as exc:
                errors.append(str(exc))
                print(f"ERROR: {exc}", flush=True)
            if count % 50 == 0 or count == len(images):
                print(f"Inspected {count}/{len(images)} image configurations", flush=True)
    if errors:
        raise RuntimeError(f"Failed to inspect {len(errors)} images; no tasks were dropped or dataset written")
    relocated = Path(remote_task_root) if remote_task_root else destination / "tasks"
    for metadata, source, config, image in tasks:
        target = destination / "tasks" / metadata["instance_id"]
        config["environment"]["env"] = configs[image]["env"] | config["environment"].get("env", {})
        config["environment"]["workdir"] = configs[image]["workdir"]
        config["agent"]["network_mode"] = "public"
        shutil.copytree(source, target)
        (target / "task.toml").write_text(toml.dumps(config), encoding="utf-8")
        metadata["task_path"] = str(relocated / metadata["instance_id"])
        metadata["native_agentenv"] = True
        metadata["image_environment_source"] = image
        metadata["training_network_mode"] = "public"
    Dataset.from_list(rows).to_parquet(output)
    print(f"Wrote all {len(rows)} native AgentENV V2 tasks to {output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", required=True, help="Existing official or online V2 Harbor tasks")
    parser.add_argument("--local-save-dir", required=True)
    parser.add_argument("--remote-task-root", default=None, help="Task path used by remote training workers")
    parser.add_argument("--workers", type=int, default=8)
    prepare_agentenv(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
