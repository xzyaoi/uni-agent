"""Reuse the CSCS experiment generator, replacing only the site launcher."""
import os
import runpy
import shutil
from pathlib import Path

jsc = Path(os.environ["JSC_BUNDLE_ROOT"])
bundle = Path(os.environ["BUNDLE_ROOT"])
run = Path(os.environ["RUN_DIR"])
runpy.run_path(str(bundle / "prepare_run.py"), run_name="__main__")
for source, name in [(jsc / "templates/batch.sh", "launcher.sh"),
                     (jsc / "templates/node.sh", "jsc-node.sh")]:
    shutil.copyfile(source, run / name)
    (run / name).chmod(0o750)
with (run / "launch.env").open("a") as stream:
    import shlex
    stream.write("export APPTAINER_BIN=" + shlex.quote(os.environ.get("APPTAINER_BIN", "apptainer")) + "\n")
