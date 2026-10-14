"""Freeze source/config provenance for an instrumented training run."""

import argparse
import hashlib
import json
import subprocess
import tarfile
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--image", type=Path, required=True)
parser.add_argument("--hash-image", action="store_true")
args = parser.parse_args()
output = args.run / "provenance"
output.mkdir(exist_ok=True)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def filter_source(info):
    if "__pycache__" in Path(info.name).parts or info.name.endswith(".pyc"):
        return None
    return info


archive = output / "runtime-source.tar.gz"
with tarfile.open(archive, "w:gz") as bundle:
    for name in ("uni_agent", "verl/verl", "cscs/observability"):
        bundle.add(args.repo / name, arcname=name, filter=filter_source)
manifest = {
    "timestamp": time.time(),
    "source_archive_sha256": digest(archive),
    "image": {
        "path": str(args.image),
        "bytes": args.image.stat().st_size,
        "mtime": args.image.stat().st_mtime,
        "sha256": digest(args.image) if args.hash_image else None,
    },
    "model_path": str(args.model),
    "model_files": {},
    "run_files": {},
}
for path in sorted(args.model.iterdir()):
    if path.is_file():
        stat = path.stat()
        manifest["model_files"][path.name] = {
            "bytes": stat.st_size,
            "mtime": stat.st_mtime,
            "sha256": digest(path) if stat.st_size < 32 * 1024 * 1024 else None,
        }
for path in sorted(args.run.iterdir()):
    if path.is_file() and path.suffix in (".sh", ".yaml", ".env"):
        manifest["run_files"][path.name] = digest(path)
for name, command in {
    "git_commit": ["git", "-C", str(args.repo), "rev-parse", "HEAD"],
    "git_status": ["git", "-C", str(args.repo), "status", "--short"],
    "slurm_config": ["scontrol", "show", "config"],
}.items():
    result = subprocess.run(command, capture_output=True)
    manifest[name] = result.stdout.decode(errors="replace")
(output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Source and configuration provenance:", output)
