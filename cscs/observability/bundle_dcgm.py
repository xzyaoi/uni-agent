#!/usr/bin/env python3
"""Bundle the site's matching dcgmi client/library for use through scratch mounts."""
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binary = shutil.which("dcgmi")
    if not binary:
        raise SystemExit("Host dcgmi is required to enable in-container DCGM")
    linked = subprocess.check_output(["ldd", binary]).decode()
    match = re.search(r"(libdcgm\.so\.\d+)\s+=>\s+(\S+)", linked)
    if not match or match[2] == "not":
        raise SystemExit("Cannot resolve the host dcgmi library")
    output = args.output.resolve()
    manifest = {"architecture": platform.machine(), "version": subprocess.check_output([binary, "--version"]).decode().strip(),
                "transport": "Existing site hostengine, localhost (shared container network)", "files": {}}
    for source, target in [(Path(binary), output / "bin/dcgmi"), (Path(match[2]), output / "lib64" / match[1])]:
        source = source.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp-" + str(os.getpid()))
        shutil.copyfile(source, temporary)
        temporary.chmod(0o555)
        temporary.replace(target)
        manifest["files"][str(target.relative_to(output))] = {
            "source": str(source), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Bundled", manifest["version"], "at", output)


if __name__ == "__main__":
    main()
