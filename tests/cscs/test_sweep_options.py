"""Local fixtures for controlled sweeps; no scheduler, SSH, or AWS access."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import runpy
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


REPO = Path(__file__).resolve().parents[2]
SWEEP = REPO / 'cscs/sandbox_sweep'
SPEC = importlib.util.spec_from_file_location('controlled_campaign', REPO / 'outputs/pro_multinode_2x_20261018/campaign.py')
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


class SweepOptionsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def fixture_submit(self, options, *, succeeds=True):
        source = self.root / 'repo/cscs/sandbox_sweep'
        source.mkdir(parents=True)
        for name in ('submit.sh', 'batch.sh'):
            (source / name).write_text((SWEEP / name).read_text())
        for name in ('run_point.py', 'node.sh', 'cleanup_sandboxes.py', 'gateway_preflight.py'):
            (source / name).write_text('# local fixture\n')
        (source / 'queue_points.py').write_text('# No scheduler access in this fixture.\n')
        native = source.parent / 'native_agentenv'
        native.mkdir()
        (native / 'bind_cpu_memory.sh').write_text('# fixture\n')
        observability = source.parent / 'observability'
        observability.mkdir()
        for name in ('collect.py', 'export_metrics.py', 'bundle_dcgm.py'):
            (observability / name).write_text('# fixture\n')
        config = self.root / 'task.yaml'
        config.write_text('artifacts_dir: /fixture\n')
        env = {key: value for key, value in os.environ.items() if not key.startswith('SWEEP_')}
        env.update(TASK_CONFIG_SOURCE=str(config), SWEEP_DIR=str(self.root / 'run'), BENCHMARK='swebench-pro-v2')
        env.update(options)
        result = subprocess.run(['bash', str(source / 'submit.sh')], env=env,
                                capture_output=True, text=True, timeout=10)
        if succeeds:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return self.root / 'run'

    def test_explicit_levels_time_and_deadline_are_frozen(self):
        run = self.fixture_submit({'SWEEP_LEVELS': '16 128', 'SWEEP_TIME_LIMIT': '01:00:00',
                                   'SWEEP_DEADLINE_EPOCH': '2000000000', 'AGENTENV_EXPECTED_NODE_COUNT': '1'})
        plan = json.loads((run / 'sweep-plan.json').read_text())
        self.assertEqual(plan, {'levels': [16, 128], 'conditional_128': None,
                                'time_limit': '01:00:00', 'submission_deadline_epoch': 2000000000})
        for level in (16, 128):
            point = run / f'groups-{level}'
            self.assertIn('#SBATCH --time=01:00:00\n', (point / 'launcher.sh').read_text())
            self.assertIn('TIME_LIMIT=01:00:00\n', (point / 'launch.env').read_text())
            self.assertIn('AGENTENV_EXPECTED_NODE_COUNT=1\n', (point / 'launch.env').read_text())
        self.assertFalse((run / 'groups-8').exists())

    def test_default_sweep_and_optional_128_keep_original_gate(self):
        run = self.fixture_submit({'SWEEP_INCLUDE_128': 'True'})
        plan = json.loads((run / 'sweep-plan.json').read_text())
        self.assertEqual(plan['levels'], [8, 16, 32, 64, 128])
        self.assertTrue(plan['conditional_128'])
        self.assertNotIn('submission_deadline_epoch', plan)
        self.assertIn('#SBATCH --time=00:45:00\n', (run / 'groups-8/launcher.sh').read_text())

    def test_invalid_options_fail_before_creating_sweep(self):
        for options in ({'SWEEP_LEVELS': '16 16'}, {'SWEEP_LEVELS': '512'},
                        {'SWEEP_TIME_LIMIT': '01:00:00\n#SBATCH --nodes=8'},
                        {'SWEEP_DEADLINE_EPOCH': 'nan'}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory, \
                    patch.object(self, 'root', Path(directory)):
                run = self.fixture_submit(options, succeeds=False)
                self.assertFalse(run.exists())

    def run_queue(self, plan, reply, now=100):
        (self.root / 'sweep-plan.json').write_text(json.dumps(plan))
        for level in plan['levels']:
            (self.root / f'groups-{level}').mkdir(exist_ok=True)
        clock = now if callable(now) else lambda: now
        with patch.object(sys, 'argv', ['queue_points.py', str(self.root), '--once']), \
                patch('subprocess.run', side_effect=reply) as command, patch('time.time', side_effect=clock):
            runpy.run_path(str(SWEEP / 'queue_points.py'), run_name='__main__')
        return command

    def test_expired_deadline_preserves_jobs_and_never_submits(self):
        (self.root / 'groups-8').mkdir()
        (self.root / 'groups-8/job_id').write_text('123\n')
        command = self.run_queue({'levels': [8, 16], 'submission_deadline_epoch': 99},
                                 lambda *args, **kwargs: self.fail('No scheduler calls after deadline'))
        command.assert_not_called()
        self.assertIn('123', (self.root / 'jobs.tsv').read_text())
        state = json.loads((self.root / 'submission-deadline.json').read_text())
        self.assertEqual(state['unsubmitted_levels'], [16])
        self.assertFalse((self.root / 'groups-16/job_id').exists())

    def test_explicit_128_bypasses_default_gate_and_submits_before_deadline(self):
        (self.root / 'groups-64').mkdir()
        (self.root / 'groups-64/job_id').write_text('123\n')
        (self.root / 'groups-64/result.json').write_text(json.dumps({'state': 'completed'}))

        def reply(argv, **kwargs):
            if argv[0] == 'squeue':
                return SimpleNamespace(returncode=0, stdout='', stderr='')
            self.assertEqual(argv[0], 'sbatch')
            return SimpleNamespace(returncode=0, stdout='456\n', stderr='')

        command = self.run_queue({'levels': [64, 128], 'conditional_128': None,
                                  'submission_deadline_epoch': 200}, reply)
        self.assertEqual([call.args[0][0] for call in command.call_args_list], ['squeue', 'sbatch'])
        self.assertEqual((self.root / 'groups-128/job_id').read_text(), '456\n')

    def test_default_128_gate_still_skips_incomplete_predecessor(self):
        (self.root / 'groups-64').mkdir()
        (self.root / 'groups-64/job_id').write_text('123\n')
        (self.root / 'groups-64/result.json').write_text(json.dumps({'state': 'completed'}))

        def reply(argv, **kwargs):
            self.assertIn(argv[0], ('squeue', 'sacct'))
            return SimpleNamespace(returncode=0, stdout='COMPLETED|\n' if argv[0] == 'sacct' else '', stderr='')

        self.run_queue({'levels': [64, 128], 'conditional_128': 'requires clean 64-group completion'}, reply)
        self.assertEqual(json.loads((self.root / 'groups-128/skipped.json').read_text())['state'], 'skipped')

    def test_deadline_is_checked_again_after_predecessor_lookup(self):
        (self.root / 'groups-8').mkdir()
        (self.root / 'groups-8/job_id').write_text('123\n')
        (self.root / 'groups-8/result.json').write_text(json.dumps({'state': 'completed'}))
        now = [100]

        def reply(argv, **kwargs):
            self.assertEqual(argv[0], 'squeue')
            now[0] = 201
            return SimpleNamespace(returncode=0, stdout='', stderr='')

        command = self.run_queue({'levels': [8, 16], 'submission_deadline_epoch': 200}, reply, now=lambda: now[0])
        self.assertEqual(command.call_count, 1)
        self.assertFalse((self.root / 'groups-16/job_id').exists())

    def prepare_campaign(self, count, runtime_count=None):
        deployment = self.root / 'deployment'
        deployment.mkdir(exist_ok=True)
        preparation = self.root / 'preparation'
        preparation.mkdir(exist_ok=True)
        cluster = {'instances': [{'instance_id': f'i-{n}'} for n in range(count)],
                   'gateway_endpoint': 'http://example.invalid:8080'}
        if runtime_count is not None:
            cluster['runtime_count'] = runtime_count
        (deployment / 'cluster.json').write_text(json.dumps(cluster))
        (deployment / 'secrets.json').write_text(json.dumps({'api_key': 'fixture-key'}))
        (preparation / 'cohort.json').write_text(json.dumps({'images': [{}] * 128}))
        with patch.object(campaign, 'ROOT', self.root):
            return campaign.Campaign(SimpleNamespace())

    def test_one_and_two_runtime_manifests_and_legacy_default(self):
        for count in (1, 2):
            with self.subTest(count=count):
                self.assertEqual(len(self.prepare_campaign(count, count).nodes), count)
        self.assertEqual(len(self.prepare_campaign(2).nodes), 2)
        for count, declared in ((1, None), (1, 2), (2, 1), (2, 3), (1, True)):
            with self.subTest(count=count, declared=declared), self.assertRaises(RuntimeError):
                self.prepare_campaign(count, declared)

    def test_root_rebinding_keeps_source_repository(self):
        names = ('ROOT', 'NAME', 'REMOTE_SWEEP', 'REMOTE_CONTROL', 'SECRET_DIR', 'AWS_ROOT')
        with patch.multiple(campaign, **{name: getattr(campaign, name) for name in names}):
            campaign.configure_root(self.root / 'point-1')
            self.assertEqual(campaign.ROOT, self.root / 'point-1')
            self.assertEqual(campaign.REPO, REPO)
            for name in names[1:]:
                self.assertIn('point-1', str(getattr(campaign, name)))

    def test_monitor_prefers_exact_venv_symlink_and_falls_back_without_venv(self):
        for available in (True, False):
            with self.subTest(venv_available=available), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                repo = root / 'repo'
                interpreter = repo / '.venv/bin/python'
                if available:
                    interpreter.parent.mkdir(parents=True)
                    interpreter.symlink_to(Path(sys.executable).resolve())
                controller = campaign.Campaign.__new__(campaign.Campaign)
                controller.args = SimpleNamespace(cluster='mock-cluster', identity='/fixture/key', campaign_deadline=5400)
                controller.nodes = [{'node_id': 'runtime-1', 'public_dns': 'host.invalid'}]
                controller.ssh = Mock(return_value='true')
                controller.mark = Mock()
                with patch.object(campaign, 'ROOT', root), patch.object(campaign, 'REPO', repo), \
                        patch.object(campaign.sys, 'executable', '/base/python'), \
                        patch.object(campaign.subprocess, 'Popen', return_value=SimpleNamespace(pid=12345)) as launch:
                    controller.monitor()
                argv = launch.call_args.args[0]
                self.assertEqual(argv[0], str(interpreter) if available else '/base/python')
                if available:
                    self.assertNotEqual(argv[0], str(interpreter.resolve()))
                self.assertEqual(argv[1], str(repo / 'cscs/agentenv_multinode/monitor_cluster.py'))
                controller.ssh.assert_called_once()

    def test_shared_dataset_is_existing_only_and_reverified_on_resume(self):
        controller = self.prepare_campaign(1, 1)
        controller.args = SimpleNamespace(cluster='mock-cluster', data_root=Path('/remote/shared-data'))
        (self.root / 'preparation/pinned-images.json').write_text('{"images": []}')
        controller.upload = Mock()
        controller.worker = Mock()
        controller.wait_worker = Mock()
        controller.ssh = Mock()
        controller.remote_json = Mock(return_value={'state': 'isolated_selected_tasks_pinned',
                                                    'selected_tasks': [{}] * 128,
                                                    'measured_dataset': '/remote/shared-data/train.parquet'})
        with patch.object(campaign, 'ROOT', self.root):
            controller.dataset()
        command = controller.worker.call_args.args[2]
        self.assertEqual(command[command.index('--output-root') + 1], '/remote/shared-data')
        self.assertIn('--existing-only', command)
        self.assertIn(repr(command), controller.ssh.call_args.args[1])
        self.assertEqual(controller.state['data_path'], '/remote/shared-data/train.parquet')

    def test_shared_dataset_hash_validation_does_not_rewrite_frozen_files(self):
        def sha(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()

        source = self.root / 'source.parquet'
        source.write_bytes(b'original dataset fixture')
        target = self.root / 'measured'
        target.mkdir()
        measured = target / 'swe_bench_pro_v2_agentenv_train.parquet'
        measured.write_bytes(b'frozen measured fixture')
        indices = list(range(642))
        random.Random(42).shuffle(indices)
        rows = [{'extra_info': {'tools_kwargs': {'task': {'metadata': {
            'instance_id': f'task-{index}', 'task_path': str(self.root / 'original' / f'task-{index}')
        }}}}} for index in range(642)]
        selected, measured_tasks, pins = [], [], {}
        for index in indices[:128]:
            original = self.root / 'original' / f'task-{index}' / 'task.toml'
            original.parent.mkdir(parents=True)
            image = f'repository-{index}:tag'
            original.write_text('[environment]\ndocker_image = ' + json.dumps(image) + '\n')
            selected.append({'dataset_index': index, 'instance_id': f'task-{index}',
                             'task_toml_sha256': sha(original), 'image': image})
            pins[image] = f'repository-{index}@sha256:{index:064x}'
            frozen = target / f'task-{index}' / 'task.toml'
            frozen.parent.mkdir()
            frozen.write_text('[environment]\ndocker_image = ' + json.dumps(pins[image]) + '\n')
            measured_tasks.append({'measured_task_path': str(frozen.parent), 'measured_task_toml_sha256': sha(frozen)})
        cohort = self.root / 'cohort.json'
        cohort.write_text(json.dumps({'dataset': str(source), 'dataset_sha256': sha(source),
                                     'seed': 42, 'groups': 128, 'selected_tasks': selected}))
        images = self.root / 'images.json'
        images.write_text(json.dumps({'images': [{'original_ref': key, 'resolved_ref': value} for key, value in pins.items()]}))
        (target / 'measurement-manifest.json').write_text(json.dumps({
            'source_dataset_sha256': sha(source), 'image_mapping': pins, 'cohort_manifest_sha256': sha(cohort),
            'measured_dataset_sha256': sha(measured), 'selected_tasks': measured_tasks, 'measured_dataset': str(measured)}))
        pa, pq = ModuleType('pyarrow'), ModuleType('pyarrow.parquet')
        pa.parquet = pq
        pq.read_table = lambda _: SimpleNamespace(num_rows=642, to_pylist=lambda: rows)

        def verify():
            argv = ['prepare.py', '--cohort', str(cohort), '--images', str(images),
                    '--output-root', str(target), '--existing-only']
            with patch.dict(sys.modules, {'pyarrow': pa, 'pyarrow.parquet': pq}), \
                    patch.object(sys, 'argv', argv), patch('builtins.print'):
                exec(campaign.PIN_DATASET_SCRIPT, {})

        before = {str(path): sha(path) for path in target.rglob('*') if path.is_file()}
        with self.assertRaises(SystemExit) as exited:
            verify()
        self.assertEqual(exited.exception.code, 0)
        self.assertEqual(before, {str(path): sha(path) for path in target.rglob('*') if path.is_file()})
        for path in (measured, Path(measured_tasks[0]['measured_task_path']) / 'task.toml'):
            with self.subTest(tampered=path.name):
                saved = path.read_bytes()
                path.write_bytes(b'tampered')
                with self.assertRaises(ValueError):
                    verify()
                path.write_bytes(saved)


if __name__ == '__main__':
    unittest.main()
