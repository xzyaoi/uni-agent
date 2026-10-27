"""Episode budgets are fixed before loading models or allocating workers."""

import pytest

from cscs.sandbox_sweep.run_point import parser


def arguments(*extra):
    return parser().parse_args(
        [
            "--data-path",
            "/fixture/data",
            "--model-path",
            "/fixture/model",
            "--task-config",
            "/fixture/task.yaml",
            "--run-dir",
            "/fixture/run",
            "--concurrency",
            "16",
            *extra,
        ]
    )


def test_sweep_has_a_finite_whole_episode_budget():
    assert arguments().session_timeout_seconds == 1800


def test_sweep_can_select_a_longer_explicit_budget():
    assert arguments("--session-timeout-seconds", "7200").session_timeout_seconds == 7200


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_invalid_episode_budget_fails_before_runtime_start(value):
    with pytest.raises(SystemExit):
        arguments(f"--session-timeout-seconds={value}")
