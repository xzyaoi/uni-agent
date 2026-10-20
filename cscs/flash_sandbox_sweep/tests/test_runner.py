"""Offline checks for cohort identity, launch isolation and owned cleanup."""
import asyncio
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cscs.flash_sandbox_sweep import backend_preflight, cohort, prepare
from cscs.flash_sandbox_sweep.cleanup_sandboxes import owned_ids


def fixture_rows(tmp_path):
    task = tmp_path / 'task.toml'
    task.write_text('[environment]\ndocker_image="example/image@sha256:' + 'a' * 64 + '"\n')
    rows = [{'extra_info': {'tools_kwargs': {'task': {'metadata': {
        'task_path': str(tmp_path), 'instance_id': f'task-{n}'}}}}} for n in range(140)]
    indices = list(range(140))
    random.Random(42).shuffle(indices)
    selected = [{'dataset_index': index, 'instance_id': f'task-{index}',
                 'task_toml_sha256': cohort.digest(task)} for index in indices[:128]]
    pin = {'rows': 140, 'groups': 128, 'seed': 42, 'selected_tasks': selected,
           'selection_sha256': hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()}
    return rows, pin


def test_exact_order_and_task_hash_are_required(tmp_path):
    rows, pin = fixture_rows(tmp_path)
    assert cohort.validate_rows(rows, pin) == pin['selected_tasks']
    index = pin['selected_tasks'][0]['dataset_index']
    rows[index]['extra_info']['tools_kwargs']['task']['metadata']['instance_id'] = 'changed'
    with pytest.raises(ValueError, match='identity, row order or TOML'):
        cohort.validate_rows(rows, pin)


def test_changed_task_and_unpinned_image_rejected(tmp_path):
    rows, pin = fixture_rows(tmp_path)
    task = tmp_path / 'task.toml'
    task.write_text(task.read_text() + '# changed\n')
    with pytest.raises(ValueError, match='TOML'):
        cohort.validate_rows(rows, pin)
    task.write_text('[environment]\ndocker_image="example/image:latest"\n')
    with pytest.raises(ValueError, match='not pinned'):
        cohort.validate_rows(rows, pin)


def test_cleanup_only_current_run_and_endpoint(tmp_path):
    trace = tmp_path / 'flash-traces'
    trace.mkdir()
    row = {'event': 'sandbox_created', 'run_id': str(tmp_path.resolve()),
           'endpoint_sha256': hashlib.sha256(b'http://fixture').hexdigest(), 'sandbox_id': 'owned'}
    rows = [row, {**row, 'run_id': '/other', 'sandbox_id': 'foreign'},
            {**row, 'endpoint_sha256': 'wrong', 'sandbox_id': 'other-backend'},
            {**row, 'event': 'unknown', 'sandbox_id': 'unknown'}]
    (trace / 'owned-1.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n{broken')
    assert owned_ids(tmp_path, 'http://fixture/') == {'owned'}


def test_prepare_four_points_without_secret_in_artifacts(tmp_path, monkeypatch):
    secret = tmp_path / 'client.env'
    secret.write_text('OTELA_API_KEY=must-not-be-copied\n')
    secret.chmod(0o600)
    sdk = tmp_path / 'sdk'
    for package in ('flash_sandbox', 'httpx2'):
        (sdk / package).mkdir(parents=True)
        (sdk / package / '__init__.py').write_text('# fixture')
    root = tmp_path / 'sweep'
    for name, value in {'BENCHMARK': 'swebench-pro-v2', 'SWEEP_DIR': str(root),
                        'SCRATCH': str(tmp_path), 'MODEL_PATH': '/model',
                        'FLASH_SECRET_FILE': str(secret), 'FLASH_SDK_PATH': str(sdk), 'SWEEP_LEVELS': '8 16 32 64',
                        'FLASH_SANDBOX_ENDPOINT': 'http://fixture'}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv('DATA_PATH', raising=False)
    monkeypatch.setattr(prepare, 'validate', lambda *args: {'state': 'fixture'})
    calls = []
    monkeypatch.setattr(prepare.subprocess, 'run', lambda argv, **kwargs: calls.append(argv))
    assert prepare.prepare() == root
    assert len(calls) == 4
    for group in (8, 16, 32, 64):
        run = root / f'groups-{group}'
        launch = (run / 'launch.env').read_text()
        assert f'GROUP_CONCURRENCY={group}\n' in launch
        assert 'FLASH_SANDBOX_TYPE=firecracker\n' in launch
        assert 'must-not-be-copied' not in launch
        assert 'OTELA_API_KEY' not in launch
        assert 'AGENTENV_ENDPOINT' not in launch
        assert 'HARBOR_VENV=' + str(tmp_path) + '/venvs/uni-agentenv-v2' in launch
        assert '"${HARBOR_VENV}/bin/python"' in (run / 'launcher.sh').read_text()
        assert 'override_cpus: 1' in (run / 'task_config.yaml').read_text()
        assert (run / 'launch.env').stat().st_mode & 0o777 == 0o600
    assert json.loads((root / 'sweep-plan.json').read_text())['episode_concurrency'] == [64, 128, 256, 512]


def test_backend_preflight_requires_available_firecracker_node(monkeypatch):
    monkeypatch.setenv('FLASH_SANDBOX_ENDPOINT', 'http://fixture')
    monkeypatch.setenv('OTELA_API_KEY', 'fixture-key')
    monkeypatch.setenv('FLASH_SANDBOX_TYPE', 'firecracker')
    status = SimpleNamespace(available_node_count=1, node_count=1, unavailable_node_count=0,
                             sandbox_count=0, nodes=[SimpleNamespace(available=True, backend_types=['docker'],
                                                                   running_count=0)])

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def cluster_status(self):
            return status

        async def close(self):
            pass

    monkeypatch.setitem(sys.modules, 'flash_sandbox', SimpleNamespace(AsyncHTTPClient=Client))
    with pytest.raises(RuntimeError):
        asyncio.run(backend_preflight.check())
    status.nodes[0].backend_types = ['firecracker']
    assert asyncio.run(backend_preflight.check())['available_backend_nodes'] == 1


def test_frozen_baseline_manifests_match_selection_pins():
    root = Path(__file__).resolve().parents[3]
    for path in root.joinpath('cscs/flash_sandbox_sweep').glob('cohort_*.json'):
        pin = json.loads(path.read_text())
        baseline = root / pin['baseline_manifest']
        assert cohort.digest(baseline) == pin['baseline_manifest_sha256']
        manifest = json.loads(baseline.read_text())
        assert manifest['selected_tasks'] == pin['selected_tasks']
        assert manifest['selection_sha256'] == pin['selection_sha256']
        assert (pin['groups'], pin['responses'], pin['seed']) == (128, 8, 42)


def test_owned_environment_records_before_bootstrap_and_reuses_uv_helper(tmp_path, monkeypatch):
    # Harbor is compute-only. Stub its two base types, retaining the real
    # Flash/AgentENV adapters and the real pinned-uv bootstrap implementation.
    base = SimpleNamespace(BaseEnvironment=object, ExecResult=SimpleNamespace)
    monkeypatch.setitem(sys.modules, 'harbor.environments.base', base)
    from cscs.flash_sandbox_sweep import environment
    archive = tmp_path / 'uv.tar.gz'
    archive.write_bytes(b'pinned-uv-fixture')
    endpoint = 'http://fixture'
    trace = tmp_path / 'flash-traces'
    monkeypatch.setenv('FLASH_SWEEP_RUN_ID', str(tmp_path))
    monkeypatch.setenv('FLASH_SWEEP_TRACE_DIR', str(trace))
    monkeypatch.setenv('AGENTENV_UV_ARCHIVE', str(archive))
    monkeypatch.setenv('AGENTENV_UV_ARCHIVE_SHA256', cohort.digest(archive))
    calls = []

    class Sandbox:
        async def upload(self, local, remote):
            calls.append(('upload', local, remote))

        async def exec_shell(self, command, timeout):
            calls.append(('exec', command, timeout))
            return SimpleNamespace(exit_code=0, stdout='uv 0.7.13\n', stderr='')

    sandbox = Sandbox()

    def initialize(self, *args, **kwargs):
        self._sandbox = sandbox
        self._started = False

    async def create(self, **kwargs):
        assert kwargs['vcpu_count'] == 1
        assert kwargs['cpu_cores'] == 1 and kwargs['memory_mb'] == 4096
        assert owned_ids(tmp_path, endpoint) == {kwargs['sandbox_id']}
        return SimpleNamespace(id=kwargs['sandbox_id'])

    async def start(self, force_build=False):
        client = environment.flash_sandbox.AsyncHTTPClient(endpoint)
        try:
            result = await client.start_sandbox(type='firecracker', cpu_cores=1, memory_mb=4096)
            assert owned_ids(tmp_path, endpoint) == {result.id}
            calls.append('bootstrap')
        finally:
            await client.close()
        self._started = True

    monkeypatch.setattr(environment.OwnedShapeClient.__bases__[0], 'start_sandbox', create)
    monkeypatch.setattr(environment.FlashSandboxEnvironment, '__init__', initialize)
    monkeypatch.setattr(environment.FlashSandboxEnvironment, 'start', start)
    adapter = environment.OwnedFlashSandboxEnvironment()
    asyncio.run(adapter.start())
    assert calls[0] == 'bootstrap'
    assert calls[1] == ('upload', archive, '/tmp/uni-agent-uv.tar.gz')
    assert calls[2][0] == 'exec' and calls[2][2] == 120
    assert 'uv-x86_64-unknown-linux-gnu/uvx' in calls[2][1]
    assert '/root/.local/bin/uv --version' in calls[2][1]


def test_sdk_substitution_is_restored_when_environment_start_fails(monkeypatch):
    from cscs.flash_sandbox_sweep import environment
    original = environment.flash_sandbox.AsyncHTTPClient
    adapter = object.__new__(environment.OwnedFlashSandboxEnvironment)
    adapter._started = False

    async def fail(self, force_build=False):
        assert environment.flash_sandbox.AsyncHTTPClient is environment.OwnedShapeClient
        raise RuntimeError('fixture start failure')

    monkeypatch.setattr(environment.FlashSandboxEnvironment, 'start', fail)
    with pytest.raises(RuntimeError, match='fixture start failure'):
        asyncio.run(adapter.start())
    assert environment.flash_sandbox.AsyncHTTPClient is original


def test_sdk_rejects_wrong_shape_before_allocating():
    from cscs.flash_sandbox_sweep import environment

    async def scenario():
        client = environment.OwnedShapeClient('http://fixture')
        try:
            with pytest.raises(ValueError, match='1 vCPU'):
                await client.start_sandbox(type='firecracker', cpu_cores=2, memory_mb=4096)
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('failure', ['response_lost', 'bootstrap_failed'])
def test_create_claim_survives_response_loss_and_failed_bootstrap(tmp_path, monkeypatch, failure):
    from uuid import UUID

    from cscs.flash_sandbox_sweep import environment
    endpoint = 'http://fixture'
    monkeypatch.setenv('FLASH_SWEEP_RUN_ID', str(tmp_path))
    monkeypatch.setenv('FLASH_SWEEP_TRACE_DIR', str(tmp_path / 'flash-traces'))
    observed = []

    async def create(self, **kwargs):
        claimed = kwargs['sandbox_id']
        UUID(claimed)
        assert owned_ids(tmp_path, endpoint) == {claimed}
        observed.append(claimed)
        if failure == 'response_lost':
            raise TimeoutError('fixture response lost after server accepted create')
        return SimpleNamespace(id=claimed)

    monkeypatch.setattr(environment.OwnedShapeClient.__bases__[0], 'start_sandbox', create)

    async def scenario():
        client = environment.OwnedShapeClient(endpoint)
        try:
            await client.start_sandbox(type='firecracker', cpu_cores=1, memory_mb=4096)
            raise RuntimeError('fixture failed bootstrap after create')
        finally:
            await client.close()

    with pytest.raises((TimeoutError, RuntimeError)):
        asyncio.run(scenario())
    assert len(observed) == 1
    assert owned_ids(tmp_path, endpoint) == set(observed)


@pytest.mark.parametrize('extra_node,sandbox_count', [(True, 0), (False, 1)])
def test_preflight_rejects_extra_nodes_and_dirty_inventory(monkeypatch, extra_node, sandbox_count):
    monkeypatch.setenv('FLASH_SANDBOX_ENDPOINT', 'http://fixture')
    monkeypatch.setenv('OTELA_API_KEY', 'fixture-key')
    monkeypatch.setenv('FLASH_SANDBOX_TYPE', 'firecracker')
    nodes = [SimpleNamespace(available=True, backend_types=['firecracker'], running_count=0)]
    if extra_node:
        nodes.append(SimpleNamespace(available=True, backend_types=['firecracker'], running_count=0))
    status = SimpleNamespace(available_node_count=len(nodes), node_count=len(nodes), unavailable_node_count=0,
                             sandbox_count=sandbox_count, nodes=nodes)

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def cluster_status(self):
            return status

        async def close(self):
            pass

    monkeypatch.setitem(sys.modules, 'flash_sandbox', SimpleNamespace(AsyncHTTPClient=Client))
    with pytest.raises(RuntimeError, match='one available Firecracker node and zero sandboxes'):
        asyncio.run(backend_preflight.check())


def test_create_rejects_unexpected_returned_id_and_retains_both_claims(tmp_path, monkeypatch):
    from cscs.flash_sandbox_sweep import environment
    endpoint = 'http://fixture'
    monkeypatch.setenv('FLASH_SWEEP_RUN_ID', str(tmp_path))
    monkeypatch.setenv('FLASH_SWEEP_TRACE_DIR', str(tmp_path / 'flash-traces'))
    claimed = []
    stopped = []

    async def create(self, **kwargs):
        claimed.append(kwargs['sandbox_id'])
        return SimpleNamespace(id='unexpected-owned-create-result')

    async def stop(self, sandbox_id, **kwargs):
        stopped.append(sandbox_id)

    base = environment.OwnedShapeClient.__bases__[0]
    monkeypatch.setattr(base, 'start_sandbox', create)
    monkeypatch.setattr(base, 'stop_sandbox', stop)

    async def scenario():
        client = environment.OwnedShapeClient(endpoint)
        try:
            await client.start_sandbox(type='firecracker', cpu_cores=1, memory_mb=4096)
        finally:
            await client.close()

    with pytest.raises(RuntimeError, match='different from the claimed ID'):
        asyncio.run(scenario())
    assert stopped == ['unexpected-owned-create-result']
    assert owned_ids(tmp_path, endpoint) == {*claimed, 'unexpected-owned-create-result'}


@pytest.mark.parametrize('cleanup', [True, False])
@pytest.mark.parametrize('refactored_transport', [True, False])
def test_synchronous_delete_preserves_auth_transport_and_pause_intent(monkeypatch, cleanup, refactored_transport):
    from cscs.flash_sandbox_sweep.client import SynchronousDeleteClient
    calls = []

    async def delete(path, **kwargs):
        calls.append((path, kwargs))

    async def forbidden_sdk_default(*args, **kwargs):
        raise AssertionError('Must not call the SDK asynchronous default stop')

    monkeypatch.setattr(SynchronousDeleteClient.__bases__[0], 'stop_sandbox', forbidden_sdk_default)
    client = object.__new__(SynchronousDeleteClient)
    if refactored_transport:
        client._transport = SimpleNamespace(_delete=delete)
    else:
        client._delete = delete
    asyncio.run(client.stop_sandbox(SimpleNamespace(id='owned-id'), cleanup=cleanup, timeout=120))
    assert calls == [('sandboxes/owned-id', {'params': {'cleanup': str(cleanup).lower(), 'async': 'false'},
                                            'timeout': 120})]


def test_owned_and_cleanup_clients_share_synchronous_delete():
    from cscs.flash_sandbox_sweep import cleanup_sandboxes, environment
    from cscs.flash_sandbox_sweep.client import SynchronousDeleteClient
    assert issubclass(environment.OwnedShapeClient, SynchronousDeleteClient)
    assert cleanup_sandboxes.SynchronousDeleteClient is SynchronousDeleteClient


@pytest.mark.parametrize('status_code', [404, 401, 500])
def test_synchronous_delete_normalizes_only_missing_sandbox(status_code):
    from flash_sandbox import SandboxHTTPError, SandboxNotFoundError

    from cscs.flash_sandbox_sweep.client import SynchronousDeleteClient
    original = SandboxHTTPError(status_code, 'fixture error', method='DELETE', path='sandboxes/owned-id',
                                response_json={'error': 'fixture'})
    original.origin = 'coordinator'

    async def delete(path, **kwargs):
        raise original

    client = object.__new__(SynchronousDeleteClient)
    client._transport = SimpleNamespace(_delete=delete)
    with pytest.raises(SandboxNotFoundError if status_code == 404 else SandboxHTTPError) as caught:
        asyncio.run(client.stop_sandbox('owned-id'))
    if status_code == 404:
        assert caught.value.__cause__ is original
        assert caught.value.method == 'DELETE' and caught.value.path == 'sandboxes/owned-id'
        assert caught.value.origin == 'coordinator' and caught.value.response_json == {'error': 'fixture'}
    else:
        assert caught.value is original


def test_predecessor_cleanup_report_preserves_completed_run(monkeypatch, tmp_path):
    from flash_sandbox import SandboxNotFoundError

    from cscs.flash_sandbox_sweep import cleanup_sandboxes

    previous = tmp_path / 'previous'
    previous.mkdir()
    original = previous / 'sandbox-cleanup.json'
    original.write_text('completed-run evidence\n')
    report = tmp_path / 'current-predecessor-cleanup.json'

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def stop_sandbox(self, sid, **kwargs):
            assert sid == 'owned-predecessor-id'
            raise SandboxNotFoundError(404, 'already gone')

        async def close(self):
            pass

    monkeypatch.setenv('FLASH_SANDBOX_ENDPOINT', 'http://fixture.invalid')
    monkeypatch.setenv('OTELA_API_KEY', 'fixture')
    monkeypatch.setattr(cleanup_sandboxes, 'SynchronousDeleteClient', Client)
    monkeypatch.setattr(cleanup_sandboxes, 'owned_ids', lambda run, endpoint: {'owned-predecessor-id'})
    counts = asyncio.run(cleanup_sandboxes.cleanup(previous, report))
    assert counts == {'recorded_ids': 1, 'deleted': 0, 'already_gone': 1, 'errors': 0}
    assert json.loads(report.read_text()) == counts
    assert original.read_text() == 'completed-run evidence\n'
