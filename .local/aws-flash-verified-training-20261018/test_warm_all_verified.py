import importlib.util
import json
from pathlib import Path
import stat
import sys
import tempfile
import types
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('warm_verified', Path(__file__).with_name('warm_all_verified.py'))
warm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(warm)


class WarmTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cache = self.root / 'images'
        self.cache.mkdir()
        self.image_id = 'a' * 64
        self.ref = 'ghcr.io/swe-images/example:latest'

    def tearDown(self):
        self.temp.cleanup()

    def template(self, refs=None):
        directory = self.cache / self.image_id
        directory.mkdir(exist_ok=True)
        rootfs = directory / 'scratch-template.ext4'
        with rootfs.open('wb') as stream:
            stream.write(b'\0' * 1080 + b'\x53\xef' + b'\0' * 2048)
        metadata = {'image_id': self.image_id, 'image_ref': 'old:latest',
                    'image_refs': refs or [], 'image_config': {'env': ['A=B']},
                    'size_bytes': rootfs.stat().st_blocks * 512, 'unknown': {'keep': 2}}
        (directory / 'metadata.json').write_text(json.dumps(metadata))
        return directory, metadata

    def test_valid_cache_exact_refs_and_corruption(self):
        directory, metadata = self.template([self.ref])
        self.assertEqual(warm.cache_index(self.cache)[1][self.ref], self.image_id)
        for key, bad_value in [('image_id', 'b' * 64), ('size_bytes', 1), ('image_config', None)]:
            (directory / 'metadata.json').write_text(json.dumps({**metadata, key: bad_value}))
            self.assertEqual(warm.cache_index(self.cache), ({}, {}))
        (directory / 'metadata.json').write_text(json.dumps(metadata))
        with (directory / 'scratch-template.ext4').open('r+b') as stream:
            stream.seek(1080)
            stream.write(b'xx')
        self.assertEqual(warm.cache_index(self.cache), ({}, {}))

    def test_alias_is_atomic_preserves_fields_owner_mode_and_rejects_changed_id(self):
        directory, original = self.template()
        path = directory / 'metadata.json'
        path.chmod(0o640)
        before = path.stat()
        events = []
        def inspect(*args, **kwargs):
            return types.SimpleNamespace(stdout='sha256:' + self.image_id + '\n')
        warm.register_alias(self.cache, self.ref, self.image_id, events.append, inspect)
        after = json.loads(path.read_text())
        self.assertEqual(after, {**original, 'image_refs': [self.ref]})
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o640)
        self.assertEqual((path.stat().st_uid, path.stat().st_gid), (before.st_uid, before.st_gid))
        self.assertFalse(path.with_name(path.name + '.tmp').exists())
        self.assertEqual(events[0]['kind'], 'alias_registration')
        def changed(*args, **kwargs):
            return types.SimpleNamespace(stdout='sha256:' + 'b' * 64)
        with self.assertRaisesRegex(RuntimeError, 'config ID changed'):
            warm.register_alias(self.cache, 'other:latest', self.image_id, events.append, changed)
        self.assertEqual(json.loads(path.read_text()), after)
        path.write_text('{bad')
        with self.assertRaisesRegex(RuntimeError, 'invalid template'):
            warm.register_alias(self.cache, self.ref, self.image_id, events.append, inspect)

    def test_settled_allocation_refresh_requires_clean_read_only_fsck(self):
        directory, original = self.template([self.ref])
        path = directory / 'metadata.json'
        original['size_bytes'] += 4096
        path.write_text(json.dumps(original))
        path.chmod(0o640)
        owner = (path.stat().st_uid, path.stat().st_gid)
        self.assertIsNone(warm.valid_template(directory))
        commands, events = [], []
        def clean_fsck(command, **kwargs):
            commands.append(command)
            return types.SimpleNamespace(returncode=0, stdout='clean', stderr='')
        warm.refresh_allocations(self.cache, events.append, clean_fsck)
        expected = {**original, 'size_bytes': (directory / 'scratch-template.ext4').stat().st_blocks * 512}
        self.assertEqual(json.loads(path.read_text()), expected)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o640)
        self.assertEqual((path.stat().st_uid, path.stat().st_gid), owner)
        self.assertEqual(commands, [['e2fsck', '-f', '-n', str(directory / 'scratch-template.ext4')]])
        self.assertEqual(events[0]['fsck_returncode'], 0)
        self.assertEqual(events[0]['old_size_bytes'], original['size_bytes'])
        self.assertEqual(warm.valid_template(directory), expected)

    def test_bad_read_only_fsck_preserves_mismatched_metadata(self):
        directory, original = self.template([self.ref])
        path = directory / 'metadata.json'
        original['size_bytes'] += 4096
        path.write_text(json.dumps(original))
        before = path.read_bytes()
        events = []
        def bad_fsck(command, **kwargs):
            return types.SimpleNamespace(returncode=4, stdout='filesystem errors', stderr='')
        with self.assertRaisesRegex(RuntimeError, 'read-only e2fsck failed'):
            warm.refresh_allocations(self.cache, events.append, bad_fsck)
        self.assertEqual(path.read_bytes(), before)
        self.assertIsNone(warm.valid_template(directory))
        self.assertFalse(events[0]['ok'])

    def test_cold_existing_config_id_alias_is_durable_before_reclaim(self):
        directory, original = self.template()
        commands, events = [], []
        def inspect(command, **kwargs):
            commands.append(command)
            return types.SimpleNamespace(stdout='sha256:' + self.image_id + '\n')
        outcomes = [{'image': self.ref, 'ok': True}, {'image': 'failed:latest', 'ok': False}]
        warm.persist_pull_aliases(self.cache, outcomes, None, events.append, inspect)
        metadata = json.loads((directory / 'metadata.json').read_text())
        self.assertEqual(metadata, {**original, 'image_refs': [self.ref]})
        self.assertEqual(warm.cache_index(self.cache)[1][self.ref], self.image_id)
        self.assertTrue(commands)
        self.assertTrue(all(command[3] == self.ref for command in commands))
        self.assertEqual(events[0]['kind'], 'alias_registration')
        commands.clear()
        warm.persist_pull_aliases(self.cache, outcomes, None, events.append, inspect)
        self.assertEqual(commands, [], 'already-durable cold refs do not need Docker inspection')

    def test_prune_only_matching_valid_config_ids_and_named_refs(self):
        self.template([self.ref])
        rows = [
            {'ID': 'sha256:' + self.image_id, 'Repository': 'prepared', 'Tag': 'latest', 'Digest': 'sha256:digest'},
            {'ID': 'sha256:' + 'b' * 64, 'Repository': 'unprepared', 'Tag': 'latest'},
            {'ID': 'sha256:' + self.image_id, 'Repository': '<none>', 'Tag': '<none>'},
        ]
        commands = []
        def run(command, **kwargs):
            commands.append(command)
            stdout = '\n'.join(map(json.dumps, rows)) if command[:3] == ['docker', 'image', 'ls'] else ''
            return types.SimpleNamespace(stdout=stdout, stderr='', returncode=0)
        events = []
        warm.reclaim_prepared(self.cache, events.append, run)
        removed = [command for command in commands if command[:3] == ['docker', 'image', 'rm']]
        self.assertEqual(removed, [
            ['docker', 'image', 'rm', '--no-prune', 'prepared:latest'],
            ['docker', 'image', 'rm', '--no-prune', 'prepared@sha256:digest'],
        ])
        self.assertTrue(all(event['ok'] for event in events))

    def test_batch_scoped_cleanup_preserves_legacy_docker_refs(self):
        self.template([self.ref])
        legacy_id = 'b' * 64
        legacy_directory = self.cache / legacy_id
        legacy_directory.mkdir()
        source = self.cache / self.image_id
        (legacy_directory / 'scratch-template.ext4').write_bytes((source / 'scratch-template.ext4').read_bytes())
        legacy_metadata = json.loads((source / 'metadata.json').read_text())
        legacy_metadata.update(image_id=legacy_id, image_ref='legacy:latest', image_refs=[])
        legacy_metadata['size_bytes'] = (legacy_directory / 'scratch-template.ext4').stat().st_blocks * 512
        (legacy_directory / 'metadata.json').write_text(json.dumps(legacy_metadata))
        rows = [{'ID': 'sha256:' + self.image_id, 'Repository': 'batch', 'Tag': 'latest'},
                {'ID': 'sha256:' + legacy_id, 'Repository': 'legacy', 'Tag': 'latest'}]
        commands = []
        def run(command, **kwargs):
            commands.append(command)
            stdout = '\n'.join(map(json.dumps, rows)) if command[:3] == ['docker', 'image', 'ls'] else ''
            return types.SimpleNamespace(stdout=stdout, stderr='', returncode=0)
        warm.reclaim_prepared(self.cache, lambda _: None, run, image_ids={self.image_id})
        removed = [command for command in commands if command[:3] == ['docker', 'image', 'rm']]
        self.assertEqual(removed, [['docker', 'image', 'rm', '--no-prune', 'batch:latest']])

    def test_pressure_cleanup_stops_as_soon_as_headroom_returns(self):
        self.template([self.ref])
        rows = [{'ID': 'sha256:' + self.image_id, 'Repository': name, 'Tag': 'latest'}
                for name in ('a', 'b', 'c')]
        removed = []
        def run(command, **kwargs):
            if command[:3] == ['docker', 'image', 'rm']:
                removed.append(command[-1])
            stdout = '\n'.join(map(json.dumps, rows)) if command[:3] == ['docker', 'image', 'ls'] else ''
            return types.SimpleNamespace(stdout=stdout, stderr='', returncode=0)
        warm.reclaim_prepared(self.cache, lambda _: None, run, stop_when=lambda: bool(removed))
        self.assertEqual(removed, ['a:latest'])

    def test_headroom_blocks_before_evict_threshold(self):
        self.assertTrue(warm.enough_headroom({'volume_used_bytes': 735 * warm.GIB, 'volume_free_bytes': 100 * warm.GIB}))
        self.assertFalse(warm.enough_headroom({'volume_used_bytes': 736 * warm.GIB, 'volume_free_bytes': 100 * warm.GIB}))
        self.assertFalse(warm.enough_headroom({'volume_used_bytes': 500 * warm.GIB, 'volume_free_bytes': 63 * warm.GIB}))

    def status(self, refs):
        return {'sandbox_count': 0, 'nodes': [{'id': 'worker', 'available': True,
                'backend_types': ['firecracker'], 'image_inventory': {
                    'images': list(refs), 'complete': True, 'physical_size_complete': True,
                    'refreshed_at': datetime.now(timezone.utc).isoformat()}}]}

    def test_exact_500_inventory_required_and_fresh(self):
        refs = [f'ghcr.io/swe-images/task-{i}:latest' for i in range(500)]
        status = self.status(refs)
        self.assertTrue(warm.inventory_ready(status, refs)[0])
        status['nodes'][0]['image_inventory']['images'][-1] = 'unrelated:latest'
        self.assertFalse(warm.inventory_ready(status, refs)[0])
        status = self.status(refs)
        status['nodes'][0]['image_inventory']['refreshed_at'] = '2000-01-01T00:00:00Z'
        self.assertFalse(warm.inventory_ready(status, refs)[0])
        status = self.status(refs)
        status['sandbox_count'] = 1
        self.assertFalse(warm.inventory_ready(status, refs)[0])

    def test_capacity_block_does_not_start_native_pull(self):
        refs = [f'ghcr.io/swe-images/task-{i}:latest' for i in range(500)]
        manifest = self.root / 'manifest.json'
        manifest.write_text(json.dumps({'images': refs}))
        fake = types.ModuleType('prewarm_cluster_images')
        fake.ClusterClient = lambda *args: types.SimpleNamespace(status=lambda: self.status([]))
        fake.warm_by_pull = lambda *args: self.fail('capacity-blocked batch must not pull')
        args = types.SimpleNamespace(output_dir=str(self.root / 'output'), manifest=str(manifest),
               coordinator='http://mock', api_key=None, cache=str(self.cache), volume=str(self.root),
               concurrency=4, batch_size=8, prune_prepared_docker=False)
        stats = {'volume_used_bytes': 740 * warm.GIB, 'volume_free_bytes': 100 * warm.GIB}
        with patch.dict(sys.modules, {'prewarm_cluster_images': fake}), \
             patch.object(warm, 'ensure_no_docker_containers'), \
             patch.object(warm, 'docker_rows', return_value=[]), \
             patch.object(warm, 'volume_stats', return_value=stats):
            self.assertEqual(warm.execute(args), 1)
        summary = json.loads((self.root / 'output' / 'summary.json').read_text())
        self.assertFalse(summary['passed'])
        self.assertEqual(summary['phase'], 'blocked')

    def test_resume_uses_valid_cache_not_receipts_with_mock_http(self):
        refs = [f'ghcr.io/swe-images/task-{i}:latest' for i in range(500)]
        self.template(refs)
        manifest = self.root / 'manifest.json'
        manifest.write_text(json.dumps({'images': refs}))
        status = self.status(refs)
        fake = types.ModuleType('prewarm_cluster_images')
        fake.ClusterClient = lambda *args: types.SimpleNamespace(status=lambda: status)
        fake.warm_by_pull = lambda *args: self.fail('valid cached refs must not be pulled again')
        args = types.SimpleNamespace(output_dir=str(self.root / 'output'), manifest=str(manifest),
               coordinator='http://mock', api_key=None, cache=str(self.cache), volume=str(self.root),
               concurrency=4, batch_size=8, prune_prepared_docker=False)
        with patch.dict(sys.modules, {'prewarm_cluster_images': fake}), \
             patch.object(warm, 'ensure_no_docker_containers'), \
             patch.object(warm, 'docker_rows', return_value=[]), \
             patch.object(warm, 'volume_stats', return_value={
                 'volume_used_bytes': 100 * warm.GIB, 'volume_free_bytes': 900 * warm.GIB}):
            self.assertEqual(warm.execute(args), 0)
        summary = json.loads((self.root / 'output' / 'summary.json').read_text())
        self.assertTrue(summary['passed'])
        self.assertEqual(summary['ready_count'], 500)


if __name__ == '__main__':
    unittest.main()
