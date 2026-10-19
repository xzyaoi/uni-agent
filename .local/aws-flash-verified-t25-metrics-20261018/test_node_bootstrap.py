"""Exercise the generated head/worker collector hooks without GPUs or Ray."""

import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class BootstrapTests(unittest.TestCase):
    def run_node(self, node_id):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binaries = root / "bin"
            binaries.mkdir()
            python = binaries / "python"
            python.write_text(
                f"#!{sys.executable}\n"
                + """
import os,signal,sys,time
from pathlib import Path
root=Path(os.environ['BOOTSTRAP_TEST_ROOT'])
if len(sys.argv)>1 and sys.argv[1].endswith('collect_usage.py'):
    mode=sys.argv[sys.argv.index('--mode')+1]
    (root/(mode+'.pid')).write_text(str(os.getpid()))
    stopping=False
    def stop(*args):
        global stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop)
    while not stopping:time.sleep(.02)
    (root/(mode+'.stopped')).touch()
else:print('OK mocked import')
"""
            )
            python.chmod(0o755)
            ray = binaries / "ray"
            ray.write_text("""#!/usr/bin/env bash
if [ "$1" = status ]; then
    printf 'node_a\nnode_b\n'
elif [ "$1" = job ] && [ "$2" = submit ]; then
    echo raysubmit_bootstrap_test
elif [ "$1" = start ] && [[ " $* " == *" --block "* ]]; then
    exit 17
fi
""")
            ray.chmod(0o755)
            train = root / "train.sh"
            train.write_text("ray job submit --no-wait -- echo test\n")
            env = dict(
                os.environ,
                PATH=str(binaries) + ":" + os.environ["PATH"],
                BOOTSTRAP_TEST_ROOT=str(root),
                RUN_DIR=str(root),
                TELEMETRY_DIR=str(root / "telemetry"),
                VLLM_PROMETHEUS_CONFIG=str(root / "prometheus.yml"),
                SLURM_NODEID=str(node_id),
                SLURM_JOB_ID="test",
                HEAD_NODE="localhost",
                NNODES_T="2",
                REPO_DIR=str(root),
                TRAIN_SCRIPT=str(train),
                RAY_TMPDIR=str(root / "ray"),
                CPUS_PER_NODE="1",
                GPUS_PER_NODE="0",
            )
            process = subprocess.Popen(
                ["bash", str(ROOT / "run_ray_node.sh")],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                output, _ = process.communicate(timeout=20)
                self.assertEqual(process.returncode, 0 if node_id == 0 else 17, output)
                expected = ["gpu", "prometheus"] if node_id == 0 else ["gpu"]
                for mode in expected:
                    self.assertTrue((root / (mode + ".stopped")).exists(), output)
                    pid = int((root / (mode + ".pid")).read_text())
                    with self.assertRaises(ProcessLookupError):
                        os.kill(pid, 0)
                self.assertEqual((root / "prometheus.pid").exists(), node_id == 0)
            finally:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)

    def test_head_starts_two_collectors_and_cleans_up(self):
        self.run_node(0)

    def test_worker_failure_cleans_up_its_gpu_collector(self):
        self.run_node(1)


if __name__ == "__main__":
    unittest.main()
