"""Validate the matched cohort, then reuse the common finite rollout driver."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# This file is also copied into each run directory.
repository = Path(os.environ.get('REPO_DIR', Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(repository))
from cscs.flash_sandbox_sweep.cohort import digest, validate  # noqa: E402
from cscs.sandbox_sweep import run_point as common  # noqa: E402


def main() -> None:
    args = common.parser().parse_args()
    if (args.groups, args.responses, args.seed, args.nnodes, args.n_gpus_per_node,
        args.tensor_parallel_size, args.prompt_length, args.response_length,
        args.temperature, args.top_p) != (128, 8, 42, 2, 4, 4, 4096, 8192, 1.0, 1.0):
        raise ValueError('Flash comparison settings differ from the matched baseline')
    if args.concurrency not in (8, 16, 32, 64):
        raise ValueError('Expected episode concurrency C=64,128,256,512 (8 responses per group)')
    run = Path(args.run_dir).resolve()
    for relative, expected in json.loads((run / 'source-hashes.json').read_text()).items():
        if digest(repository / relative) != expected:
            raise ValueError('Runner source changed after preparation: ' + relative)
    pin = run / 'cohort.json'
    proof = validate(pin, Path(args.data_path), Path(args.model_path))
    if proof != json.loads((run / 'cohort-frozen.json').read_text()):
        raise ValueError('Frozen cohort/task inputs changed after preparation')
    if json.loads(pin.read_text())['benchmark'] != args.benchmark:
        raise ValueError('Benchmark does not match cohort')
    (run / 'cohort-validation.json').write_text(json.dumps(proof, indent=2) + '\n')
    os.environ['FLASH_SWEEP_RUN_ID'] = str(run)
    os.environ['FLASH_SWEEP_TRACE_DIR'] = str(run / 'flash-traces')
    # Ray and Harbor subprocesses inherit the private backend environment from
    # node.sh. The common driver and group semaphore are reused without edits.
    common.main()


if __name__ == '__main__':
    main()
