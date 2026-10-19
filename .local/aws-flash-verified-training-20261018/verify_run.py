#!/usr/bin/env python3
"""Check generated overrides and execute the current trainer/concurrency contracts."""
import ast
import asyncio
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
import tomllib

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent


def load_method(path, name, globals_):
    tree = ast.parse(path.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)
    method.decorator_list = []
    method.returns = None
    for node in ast.walk(method):
        if isinstance(node, ast.arg):
            node.annotation = None
    namespace = dict(globals_)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


with tempfile.TemporaryDirectory() as temp:
    temp = Path(temp)
    ray = temp / "ray"
    ray.write_text('#!/usr/bin/env python3\nimport json,os,sys\nopen(os.environ["MOCK_RAY_ARGS"],"w").write(json.dumps(sys.argv[1:]))\n')
    ray.chmod(0o755)
    env = dict(os.environ, MOCK_RAY_ARGS=str(temp / "args.json"))
    env["PATH"] = f'{temp}:{env["PATH"]}'
    command = f'source {shlex.quote(str(ROOT / "run.env"))}; export DATA_DIR=/tmp RUNTIME_DIR=/tmp NNODES=2; bash {shlex.quote(str(ROOT / "train.sh"))}'
    result = subprocess.run(["bash", "-c", command], env=env, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    overrides = json.loads((temp / "args.json").read_text())

required = [
    "trainer.v1.trainer_mode=colocate_async", "trainer.nnodes=2", "trainer.n_gpus_per_node=4",
    "trainer.total_training_steps=5", "trainer.save_freq=1", "trainer.test_freq=-1",
    "data.train_batch_size=64", "actor_rollout_ref.rollout.n=8",
    "actor_rollout_ref.actor.ppo_mini_batch_size=16",
    "actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4",
    "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2",
    "actor_rollout_ref.actor.megatron.context_parallel_size=1",
    "actor_rollout_ref.rollout.tensor_model_parallel_size=4",
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.45",
    "++actor_rollout_ref.rollout.custom.agent_framework.gateway_count=2",
    "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions=512",
    "data.max_prompt_length=8192", "data.max_response_length=8192",
    "actor_rollout_ref.actor.megatron.param_offload=True",
    "actor_rollout_ref.actor.megatron.grad_offload=True",
    "actor_rollout_ref.actor.megatron.optimizer_offload=True", "trainer.logger=['console']",
]
for item in required:
    assert item in overrides, item
assert "+" not in overrides


class Config(dict):
    __getattr__ = dict.__getitem__


fetches = []
trainer = SimpleNamespace(config=SimpleNamespace(data=Config(train_batch_size=64, gen_batch_size=1)), global_steps=1)
trainer._fetch_one_gen_batch = lambda: fetches.append(1) or [len(fetches)]
tu = SimpleNamespace(concat_tensordict=lambda chunks: sum(chunks, []), assign_non_tensor_data=lambda *args: None)
next_batch = load_method(REPO / "verl/verl/trainer/ppo/v1/trainer_base.py", "_next_train_batch", {"tu": tu})
assert len(next_batch(trainer)) == 64
assert len(fetches) == 64

limit_session = load_method(REPO / "uni_agent/framework/framework.py", "_run_agent_episode_with_concurrency_limit", {"asyncio": asyncio})


async def check_global_cap():
    ready, release = asyncio.Event(), asyncio.Event()
    state = {"active": 0, "peak": 0}

    async def episode(**kwargs):
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        if state["active"] == 512:
            ready.set()
        await release.wait()
        state["active"] -= 1
        return [], {}

    framework = SimpleNamespace(_semaphore_loop=None, _runner_semaphores={}, runner_registry={"task": SimpleNamespace(max_concurrent_sessions=512)}, _run_agent_episode=episode)
    sessions = [asyncio.create_task(limit_session(framework, sample_fields={}, sample_index=prompt, session_index=session,
                global_steps=1, sampling_params={}, num_sessions=8)) for prompt in range(64) for session in range(8)]
    try:
        await asyncio.wait_for(ready.wait(), timeout=3)
        assert state["peak"] == 512
    finally:
        release.set()
        await asyncio.gather(*sessions)


asyncio.run(check_global_cap())
runtime = json.loads((ROOT / "runtime_env.json").read_text())
task = json.loads((ROOT / "task_config.yaml").read_text())[0]
assert task["harbor_env"] == "uni_agent.tasks.harbor.verified_flash_environment:VerifiedFlashEnvironment"
assert runtime["env_vars"]["FLASH_SANDBOX_START_TIMEOUT_S"] == "900"
assert "export FLASH_SANDBOX_START_TIMEOUT_S=900\n" in (ROOT / "run.env").read_text()
for key in ("FLASH_SANDBOX_START_TIMEOUT_S", "FLASH_SANDBOX_UV_ARCHIVE", "FLASH_SANDBOX_UV_ARCHIVE_SHA256"):
    assert key in runtime["env_vars"]
    assert f'{key}="${{{key}}}"' in (ROOT / "launcher.sh").read_text()
assert "OTELA_API_KEY" not in runtime["env_vars"]
tomllib.loads((ROOT / "flash-training.toml").read_text())
for name in ("generate_run.py", "preflight.py"):
    compile((ROOT / name).read_text(), str(ROOT / name), "exec")
for name in ("batch.sh", "launcher.sh", "train.sh", "run_ray_node.sh"):
    subprocess.run(["bash", "-n", str(ROOT / name)], check=True)
evidence = {"verified_overrides": required, "training_prompts_per_dispatch": 64, "rollouts_per_prompt": 8,
            "total_sessions": 512, "observed_global_semaphore_peak": 512, "effective_gen_batch_size": 1,
            "gen_batch_note": "Async trainer coalesces 64 single-prompt dataloader fetches before rollout dispatch; setting gen_batch_size=64 is unnecessary and would be overridden."}
(ROOT / "validation.json").write_text(json.dumps(evidence, indent=2) + "\n")
print("Verified: 64 prompt dispatch, 8 rollouts each, global semaphore permits 512 sessions; generated settings and syntax passed")
