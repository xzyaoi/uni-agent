import json
import pathlib
import time

from flash_sandbox import HTTPClient

image = 'ghcr.io/scaleapi/swe-bench_pro-v2:instance_NodeBB__NodeBB-00c70ce7b0541cfc94afe567921d7668cdc8f4ac-vnan'
receipt = {'image': image, 'endpoint': 'http://127.0.0.1:8090', 'backend': 'firecracker'}
with HTTPClient(address=receipt['endpoint'], timeout=900) as client:
    started = time.monotonic()
    sandbox = client.start_sandbox(type='firecracker', image=image,
                                   command=['sleep', 'infinity'], memory_mb=8192,
                                   cpu_cores=4, ttl_seconds=1200, request_timeout=900)
    receipt['create_seconds'] = round(time.monotonic() - started, 3)
    try:
        result = sandbox.exec_command(['bash', '-lc', 'printf "PRO_IMAGE_OK\\n"; pwd; command -v node; node --version; ls -d /app /testbed 2>/dev/null || true'])
        assert result.exit_code == 0 and 'PRO_IMAGE_OK' in result.stdout, result
        receipt['stdout'] = result.stdout
        sandbox.write_file('/tmp/training-preflight', b'pro-v2-ready\n')
        assert sandbox.read_file('/tmp/training-preflight') == b'pro-v2-ready\n'
        result = sandbox.shell_exec('export PRO_TRAINING_PREFLIGHT=ready')
        assert result.exit_code == 0, result
        result = sandbox.shell_exec('printf "%s" "$PRO_TRAINING_PREFLIGHT"')
        assert result.exit_code == 0 and result.stdout == 'ready', result
    finally:
        client.stop_sandbox(sandbox, cleanup=True)
        end = time.monotonic() + 120
        while any(item.id == sandbox.id for item in client.list_sandboxes()):
            assert time.monotonic() < end, 'Pro preflight sandbox cleanup timed out'
            time.sleep(1)
receipt['passed'] = True
pathlib.Path('/sandbox/uni-agent-deploy-20261018/pro-image-preflight.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
