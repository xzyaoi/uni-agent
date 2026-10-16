"""Prepare official Harbor Verified tasks for prebuilt native AgentENV images.

The pristine export is retained. This script does not build images, provision
sandboxes, execute agents, or run verifiers. UV provisioning is a separate,
explicit prerequisite because native AgentENV does not run Dockerfile RUNs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

DATASET_REF = "swebench-verified@1.0"
DATASET_COMMIT = "86723674f04e4209ac479d0fb75d9d9f44b4377e"
UV_VERSION = "0.7.13"
UV_ARCHIVE_NAME = "uv-x86_64-unknown-linux-gnu.tar.gz"
UV_SOURCE_URL = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/{UV_ARCHIVE_NAME}"
# Published SHA256 in the official GitHub release asset's digest field.
UV_SHA256 = "909278eb197c5ed0e9b5f16317d1255270d1f9ea4196e7179ce934d48c4c2545"
NATIVE_PATH = "/root/.local/bin:/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_from_dockerfile(path: Path) -> str:
    matches = re.findall(r"^\s*FROM\s+(\S+)", path.read_text(), re.MULTILINE | re.IGNORECASE)
    if len(matches) != 1 or not matches[0].startswith("swebench/sweb.eval.x86_64."):
        raise ValueError(f"Unexpected Verified base image in {path}")
    return "ghcr.io/swe-images/" + matches[0].split("/", 1)[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path,
                        help="Pristine exported swebench-verified directory")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--download", action="store_true",
                        help="Download the official dataset if source-root is absent")
    parser.add_argument("--harbor", default="harbor")
    parser.add_argument("--groups", type=int, default=128)
    parser.add_argument("--responses", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--uv-provisioning", choices=("image", "environment-bootstrap"),
                        required=True, help="Record the independently established UV provisioning method")
    parser.add_argument("--uv-archive", type=Path, help="Pinned UV 0.7.13 Linux x86_64 archive for native bootstrap")
    parser.add_argument("--download-uv", action="store_true", help="Download the pinned official UV archive once")
    args = parser.parse_args()
    source = args.source_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    if not source.exists() and args.download:
        subprocess.run([args.harbor, "download", DATASET_REF, "--export", "--output-dir", str(source.parent)], check=True)
    originals = sorted(source.glob("*/task.toml"))
    if len(originals) != 500:
        raise ValueError(f"Expected all 500 official Verified tasks under {source}, found {len(originals)}")
    if not 0 < args.groups <= 500 or args.responses <= 0:
        raise ValueError("groups must be in 1..500 and responses must be positive")
    if output.exists():
        raise FileExistsError(f"Use a new output-root: {output}")
    uv_asset = None
    if args.download_uv or args.uv_archive:
        if args.uv_provisioning != "environment-bootstrap":
            raise ValueError("A UV archive requires --uv-provisioning environment-bootstrap")
        archive = (args.uv_archive or output.parent / "assets" / f"uv-{UV_VERSION}-{UV_ARCHIVE_NAME}").expanduser().resolve()
        if not archive.exists() and args.download_uv:
            archive.parent.mkdir(parents=True, exist_ok=True)
            temporary = archive.with_suffix(archive.suffix + ".part")
            try:
                with urllib.request.urlopen(UV_SOURCE_URL, timeout=120) as response, temporary.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
                if sha256(temporary) != UV_SHA256:
                    raise ValueError("Downloaded UV archive does not match the official pinned SHA256")
                temporary.replace(archive)
            finally:
                temporary.unlink(missing_ok=True)
        if sha256(archive) != UV_SHA256:
            raise ValueError("UV archive does not match the official UV 0.7.13 Linux x86_64 SHA256")
        uv_asset = {"asset_path": str(archive), "sha256": UV_SHA256,
                    "source_url": UV_SOURCE_URL, "version": UV_VERSION,
                    "archive_bytes": archive.stat().st_size,
                    "architecture": "x86_64", "installed_directory": "/root/.local/bin"}
    output.mkdir(parents=True)
    tasks = output / "tasks"
    tasks.mkdir()
    source_records = []
    for original in originals:
        original_task = original.parent
        destination = tasks / original_task.name
        shutil.copytree(original_task, destination)
        image = image_from_dockerfile(destination / "environment" / "Dockerfile")
        task_file = destination / "task.toml"
        content = task_file.read_text()
        settings = ("[environment]\n"
                    f"docker_image = {json.dumps(image)}\n"
                    'workdir = "/testbed"\n'
                    'network_mode = "public"\n')
        if re.search(r"^\s*(docker_image|workdir|network_mode)\s*=", content, re.MULTILINE):
            raise ValueError(f"Source environment already declares image/workdir/network settings: {original}")
        content, count = re.subn(r"^\[environment\]\s*$", settings.rstrip(), content, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(f"Expected exactly one environment table: {original}")
        if "[environment.env]" in content:
            raise ValueError(f"Source task already declares environment variables: {original}")
        content += "\n[environment.env]\nPATH = " + json.dumps(NATIVE_PATH) + "\n"
        task_file.write_text(content)
        # Record the exact upstream verifier; do not replace its PEP723 parser,
        # pinned SWE-bench dependencies, test patch or reward-file logic.
        test_file = destination / "tests" / "test.sh"
        verifier = test_file.read_text()
        if "uv run parser.py" not in verifier:
            raise ValueError(f"Unexpected official verifier contract: {test_file}")
        source_records.append({
            "task_id": original_task.name,
            "image": image,
            "source_task_toml_sha256": sha256(original),
            "prepared_task_toml_sha256": sha256(task_file),
            "verifier_sha256": sha256(test_file),
            "tests_config_sha256": sha256(destination / "tests" / "config.json"),
            "instruction_sha256": sha256(destination / "instruction.md"),
        })

    from uni_agent.tasks.harbor.preprocess import build_harbor_dataset

    dataset = build_harbor_dataset(tasks, dataset_name="swebench-verified")
    rows = dataset.to_list()
    if len(rows) != 500:
        raise ValueError("Prepared Harbor dataset does not contain 500 tasks")
    records_by_id = {record["task_id"]: record for record in source_records}
    for row in rows:
        metadata = row["extra_info"]["tools_kwargs"]["task"]["metadata"]
        metadata.update({"dataset_revision": DATASET_REF, "tasks_revision": DATASET_COMMIT,
                         "native_agentenv": True, "training_network_mode": "public"})
    # Reconstruct to preserve the added provenance metadata in Arrow's schema.
    from datasets import Dataset

    data_path = output / "swebench_verified_harbor.parquet"
    Dataset.from_list(rows).to_parquet(data_path)
    indices = list(range(len(rows)))
    random.Random(args.seed).shuffle(indices)
    selection = []
    for index in indices[:args.groups]:
        metadata = rows[index]["extra_info"]["tools_kwargs"]["task"]["metadata"]
        task_id = Path(metadata["task_path"]).name
        selection.append({"dataset_index": index, "instance_id": metadata["instance_id"],
                          "task_path": metadata["task_path"], **records_by_id[task_id]})
    manifest = {
        "dataset_ref": DATASET_REF,
        "upstream_task_commit": DATASET_COMMIT,
        "source_root": str(source), "task_root": str(tasks), "data_path": str(data_path),
        "task_count": len(rows), "groups": args.groups, "responses": args.responses,
        "episodes_per_point": args.groups * args.responses, "seed": args.seed,
        "uv_version": UV_VERSION, "uv_provisioning": args.uv_provisioning,
        "uv_provisioning_verified": False,
        "native_bootstrap": uv_asset,
        "native_dockerfile_runs_executed": False,
        "verifier_dependencies": ["swebench==4.0.3", "datasets==2.16.1", "fastcore<1.11"],
        "selected_tasks": selection,
        "selected_images": sorted({item["image"] for item in selection}),
        "all_tasks": source_records,
    }
    manifest["dataset_sha256"] = sha256(data_path)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "selected-images.txt").write_text("\n".join(manifest["selected_images"]) + "\n")
    print(json.dumps({"tasks": len(rows), "groups": args.groups, "episodes": args.groups * args.responses,
                      "data_path": str(data_path), "unique_selected_images": len(manifest["selected_images"])}))


if __name__ == "__main__":
    main()
