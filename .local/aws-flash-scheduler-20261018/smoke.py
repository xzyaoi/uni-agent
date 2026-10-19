import datetime
import hashlib
import json
import pathlib
import time
import urllib.request

from flash_sandbox import HTTPClient, SandboxHTTPError

root = pathlib.Path('/sandbox/uni-agent-deploy-20261018')
image = pathlib.Path('/sandbox/ami-build/images.txt').read_text().splitlines()[0]
report = {
    'verified_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'commit': '7d9cba5d328a12632e0a0e29297f4a4891a02e37',
    'image': image,
    'backend': 'firecracker',
    'scheduler_address': 'http://127.0.0.1:8090',
}
with HTTPClient(address='http://127.0.0.1:8080', with_scheduler=True,
                scheduler_address='http://127.0.0.1:8090', timeout=300) as client:
    start = time.monotonic()
    sandbox = client.start_sandbox(type='firecracker', image=image,
                                   command=['sleep', 'infinity'], memory_mb=2048,
                                   cpu_cores=1, ttl_seconds=900)
    report['create_seconds'] = round(time.monotonic() - start, 3)
    snapshot = None
    try:
        result = sandbox.exec_command(['sh', '-c', 'printf LATEST_CODE_OK; python --version'])
        assert result.exit_code == 0 and 'LATEST_CODE_OK' in result.stdout, result
        report['exec_stdout'] = result.stdout
        sandbox.write_file('/tmp/uni-agent-smoke', b'hibernation-retains-files\n')
        assert sandbox.read_file('/tmp/uni-agent-smoke') == b'hibernation-retains-files\n'
        result = sandbox.shell_exec('cd /tmp && export UNI_AGENT_SMOKE=persistent')
        assert result.exit_code == 0, result
        result = sandbox.shell_exec('printf "%s:%s" "$PWD" "$UNI_AGENT_SMOKE"')
        assert result.exit_code == 0 and result.stdout == '/tmp:persistent', result
        report['persistent_shell_verified'] = True
        result = sandbox.exec_command(['sha256sum', '/usr/local/bin/flash-envd'])
        expected = hashlib.sha256(pathlib.Path('/sandbox/flash-sandbox/current/bin/flash-envd-linux-amd64').read_bytes()).hexdigest()
        assert result.exit_code == 0 and result.stdout.split()[0] == expected, result
        report['guest_agent_sha256'] = expected
        snapshot = client.hibernate_sandbox(sandbox, timeout=300)
        client.resume_sandbox(sandbox)
        assert sandbox.read_file('/tmp/uni-agent-smoke') == b'hibernation-retains-files\n'
        result = sandbox.shell_exec('printf "%s:%s" "$PWD" "$UNI_AGENT_SMOKE"')
        assert result.exit_code == 0 and result.stdout == '/tmp:persistent', result
        report['hibernate_resume_verified'] = True
    finally:
        client.stop_sandbox(sandbox, cleanup=True)
        deadline = time.monotonic() + 60
        while client.list_sandboxes():
            assert time.monotonic() < deadline, 'Smoke sandbox cleanup timed out'
            time.sleep(1)
        if snapshot is not None:
            while True:
                try:
                    client.delete_snapshot(snapshot)
                    break
                except SandboxHTTPError as error:
                    if error.status_code != 503 or time.monotonic() >= deadline:
                        raise
                    time.sleep(1)
    deadline = time.monotonic() + 60
    while client.list_sandboxes():
        assert time.monotonic() < deadline, 'Smoke sandbox cleanup timed out'
        time.sleep(1)
    report['sandbox_and_snapshot_cleanup_verified'] = True
report['dashboard'] = json.load(urllib.request.urlopen('http://127.0.0.1:8090/api/status', timeout=60))['cluster']
assert report['dashboard']['nodes_available'] == 1, report
assert report['dashboard']['prepared_images'] == 256, report
assert report['dashboard']['sandboxes_total'] == 0, report
report['passed'] = True
(root / 'smoke.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
