"""Exercise sweep configuration and evidence writing without allocating models."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('metrics_run_point', REPO / 'cscs/sandbox_sweep/run_point.py')
run_point = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_point)


class AttributeDict(dict):
    def __setattr__(self, name, value):
        self[name] = value


class FakeOmegaConf:
    @staticmethod
    def load(path):
        return [AttributeDict(name='harbor', artifacts_dir='/fixture/original')]

    @staticmethod
    def is_list(value):
        return isinstance(value, list)

    @staticmethod
    def save(value, path):
        Path(path).write_text(json.dumps(value))

    @staticmethod
    def update(config, key, value, **kwargs):
        parts = key.split('.')
        for part in parts[:-1]:
            if not hasattr(config, part):
                setattr(config, part, SimpleNamespace())
            config = getattr(config, part)
        setattr(config, parts[-1], value)


class VllmMetricsConfigTests(unittest.TestCase):
    def exercise(self, version_error=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / 'task'
            task.mkdir()
            (task / 'task.toml').write_text('[environment]\ndocker_image="fixture@sha256:abc"\n')
            model = root / 'model'
            model.mkdir()
            (model / 'config.json').write_text('{}')
            config_file = root / 'source.yaml'
            config_file.write_text('- name: harbor\n  artifacts_dir: /fixture/original\n')
            run = root / 'run'
            config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(
                rollout=SimpleNamespace(disable_log_stats=True)), transfer_queue={})
            ray = ModuleType('ray')
            ray.init, ray.shutdown = Mock(), Mock()
            datasets = ModuleType('datasets')
            datasets.load_dataset = Mock(return_value=[{'extra_info': {'tools_kwargs': {'task': {'metadata': {
                'task_path': str(task), 'instance_id': 'fixture-task'
            }}}}}])
            omega = ModuleType('omegaconf')
            omega.OmegaConf = FakeOmegaConf
            inference = ModuleType('examples.inference.parallel_infer_verl')
            inference.AgentFrameworkRolloutAdapter = Mock()
            inference.LLMServerManager = SimpleNamespace(create=Mock(side_effect=RuntimeError('Stop before model allocation')))
            inference.TaskConfigResolver = SimpleNamespace(from_file=Mock(return_value=SimpleNamespace(defaults_by_name={})))
            inference.init_config = Mock(return_value=config)
            inference._build_prompts, inference._read_rm_scores = Mock(), Mock()
            inference.tq, inference.tu = SimpleNamespace(init=Mock()), Mock()
            framework = ModuleType('cscs.sandbox_sweep.group_framework')
            framework.GroupLimitedGatewayAgentFramework = object
            modules = {'ray': ray, 'datasets': datasets, 'omegaconf': omega,
                       'examples.inference.parallel_infer_verl': inference,
                       'cscs.sandbox_sweep.group_framework': framework}
            argv = ['run_point.py', '--data-path', str(root / 'data.parquet'), '--model-path', str(model),
                    '--task-config', str(config_file), '--run-dir', str(run), '--concurrency', '16',
                    '--groups', '1', '--responses', '8', '--gpu-memory-utilization', '0.45']
            package = Mock(side_effect=run_point.package_metadata.PackageNotFoundError('vllm')) if version_error else Mock(return_value='0.11.2.fixture')
            with patch.dict(sys.modules, modules), patch.object(sys, 'argv', argv), \
                    patch.object(sys, 'path', list(sys.path)), patch.dict(os.environ, {'REPO_DIR': str(REPO)}), \
                    patch.object(run_point.package_metadata, 'version', package), patch.object(run_point.signal, 'signal'), \
                    self.assertRaisesRegex(RuntimeError, 'Stop before model allocation'):
                run_point.main()
            package.assert_called_once_with('vllm')
            self.assertIs(inference.LLMServerManager.create.call_args.kwargs['config'], config)
            self.assertFalse(config.actor_rollout_ref.rollout.disable_log_stats)
            engine = inference.init_config.call_args.args[0]
            self.assertEqual(engine.gpu_memory_utilization, 0.45)
            self.assertEqual(engine.tensor_parallel_size, 4)
            self.assertEqual(engine.concurrency, 128)
            self.assertEqual(config.actor_rollout_ref.rollout.temperature, 1.0)
            self.assertEqual(config.actor_rollout_ref.rollout.prompt_length, 4096)
            self.assertEqual(config.actor_rollout_ref.rollout.response_length, 8192)
            self.assertEqual(config.actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype, 'auto')
            expected = {'actor_rollout_ref.rollout.disable_log_stats': False,
                        'vllm_package_version': None if version_error else '0.11.2.fixture'}
            for name in ('manifest.json', 'settings.json', 'result.json'):
                self.assertEqual(json.loads((run / name).read_text())['observability'], expected)
            ray.shutdown.assert_called_once()

    def test_stats_enabled_in_engine_config_and_recorded_without_changing_capacity_or_sampling(self):
        self.exercise()

    def test_missing_package_metadata_is_recorded_without_blocking_config(self):
        self.exercise(version_error=True)


if __name__ == '__main__':
    unittest.main()
