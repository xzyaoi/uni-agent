from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from verl.workers.engine.megatron.transformer_impl import MegatronEngine


def _engine(*, dist_path: str | None) -> MegatronEngine:
    engine = object.__new__(MegatronEngine)
    engine.engine_config = SimpleNamespace(
        use_dist_checkpointing=True,
        dist_checkpointing_path=dist_path,
    )
    engine.vanilla_bridge = False
    engine.bridge = MagicMock()
    engine.model_config = SimpleNamespace(local_path="/models/qwen")
    engine.is_value_model = False
    return engine


def test_dist_checkpoint_saves_can_bootstrap_from_hf_weights():
    engine = _engine(dist_path=None)
    module = [MagicMock()]

    with patch("verl.workers.engine.megatron.transformer_impl.load_mcore_dist_weights") as load_dist:
        engine._load_initial_model_weights(module)

    load_dist.assert_not_called()
    engine.bridge.load_hf_weights.assert_called_once_with(
        module,
        "/models/qwen",
        allowed_mismatched_params=[],
    )


def test_explicit_dist_checkpoint_path_still_loads_native_weights():
    engine = _engine(dist_path="/checkpoints/model/dist_ckpt")
    module = [MagicMock()]

    with patch("verl.workers.engine.megatron.transformer_impl.load_mcore_dist_weights") as load_dist:
        engine._load_initial_model_weights(module)

    load_dist.assert_called_once_with(
        module,
        "/checkpoints/model/dist_ckpt",
        is_value_model=False,
    )
    engine.bridge.load_hf_weights.assert_not_called()


if __name__ == "__main__":
    test_dist_checkpoint_saves_can_bootstrap_from_hf_weights()
    test_explicit_dist_checkpoint_path_still_loads_native_weights()
    print("checkpoint bootstrap tests passed")
