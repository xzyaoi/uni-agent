"""Tests for the in-sandbox bridge daemon (examples/browsergym/bgym_bridge/bridge.py).

BrowserGym/playwright are NOT needed: fake modules are injected into
``sys.modules``, then ``serve`` runs in a thread and the stdlib client drives
it. This pins the daemon's step/result/shutdown wiring, the worker-thread
serialization and the result-file contract the task layer depends on.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
import types
from pathlib import Path

import pytest

BRIDGE_PATH = Path(__file__).resolve().parents[3] / "examples" / "browsergym" / "bgym_bridge" / "bridge.py"


def _load_bridge():
    spec = importlib.util.spec_from_file_location("bgym_bridge_under_test", BRIDGE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_browsergym(monkeypatch, tmp_path: Path):
    """Browsergym/playwright/gymnasium stand-ins covering everything serve() touches.

    Mirrors the browsergym >= 0.4 shape: tasks come from ``gym.make`` (fake here),
    the AXTree flattener lives in ``browsergym.utils.obs``.
    """
    monkeypatch.setenv("MINIWOB_URL", "http://127.0.0.1:9999/")  # skip www serving

    class FakeEnv:
        calls: list[tuple[str, str]] = []

        def __init__(self, env_name=None, headless=True, **kwargs):
            self.env_name = env_name
            self.closed = False

        @staticmethod
        def _obs():
            return {
                "goal": "Click the button labelled OK.",
                "axtree_object": {},
                "extra_element_properties": {},
                "url": "http://127.0.0.1:8000/miniwob/click-test",
                "elapsed_time": 0.0,
            }

        def reset(self, seed=None):
            FakeEnv.calls.append(("reset", str(seed)))
            return self._obs(), {}

        def step(self, action):
            FakeEnv.calls.append(("step", action))
            if "bad" in action:
                raise ValueError(f"unknown action {action!r}")
            if action.startswith("send_msg_to_user"):
                return self._obs(), 1.0, True, False, {"send_message_to_user": "done"}
            return self._obs(), 0.0, False, False, {}

        def close(self):
            self.closed = True

    fake_env_instance = FakeEnv()

    gym_mod = types.ModuleType("gymnasium")
    gym_mod.make = lambda env_id, **kwargs: fake_env_instance
    monkeypatch.setitem(sys.modules, "gymnasium", gym_mod)

    browsergym = types.ModuleType("browsergym")
    core = types.ModuleType("browsergym.core")
    env_mod = types.ModuleType("browsergym.core.env")
    env_mod.BrowserEnv = FakeEnv
    obs_mod = types.ModuleType("browsergym.core.observation")
    obs_mod.flatten_axtree_to_str = lambda obs: "[1] button 'OK'"  # legacy location
    utils = types.ModuleType("browsergym.utils")
    utils_obs = types.ModuleType("browsergym.utils.obs")
    utils_obs.flatten_axtree_to_str = lambda obs: "[1] button 'OK'"  # 0.4+ location
    utils.obs = utils_obs
    miniwob = types.ModuleType("browsergym.miniwob")
    miniwob.__file__ = str(tmp_path / "fake_miniwob" / "__init__.py")
    core.env = env_mod
    core.observation = obs_mod
    browsergym.core = core
    browsergym.utils = utils
    browsergym.miniwob = miniwob
    for name, module in {
        "browsergym": browsergym,
        "browsergym.core": core,
        "browsergym.core.env": env_mod,
        "browsergym.core.observation": obs_mod,
        "browsergym.utils": utils,
        "browsergym.utils.obs": utils_obs,
        "browsergym.miniwob": miniwob,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    playwright = types.ModuleType("playwright")
    sync_api = types.ModuleType("playwright.sync_api")

    class BrowserType:  # records the launch kwargs the patch injected
        last_launch_kwargs = None

        def launch(self, **kwargs):
            BrowserType.last_launch_kwargs = kwargs
            return object()

    sync_api.BrowserType = BrowserType
    playwright.sync_api = sync_api
    monkeypatch.setitem(sys.modules, "playwright", playwright)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_api)
    return FakeEnv


@pytest.fixture()
def daemon(monkeypatch, tmp_path):
    """Run ``bridge serve`` in a thread; yield (module, port); stop on teardown."""
    bridge = _load_bridge()
    _install_fake_browsergym(monkeypatch, tmp_path)
    port = 18099
    result_file = tmp_path / "result.json"
    thread = threading.Thread(
        target=lambda: bridge.main(
            [
                "serve",
                "--task",
                "browsergym.miniwob.miniwob_click-test",
                "--port",
                str(port),
                "--seed",
                "7",
                "--result-file",
                str(result_file),
            ]
        ),
        daemon=True,
    )
    thread.start()

    from urllib.request import urlopen

    for _ in range(100):
        try:
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=1.0) as r:
                if json.loads(r.read()).get("ok"):
                    break
        except OSError:
            time.sleep(0.05)
    else:
        raise RuntimeError("fake-env daemon never became healthy")

    yield bridge, port, result_file
    try:
        bridge._request(port, "/shutdown", payload={})
    except Exception:
        pass
    thread.join(timeout=10.0)


def test_serve_reset_info_step_result_shutdown(daemon):
    bridge, port, result_file = daemon

    info = bridge._request(port, "/info")
    assert info["ok"] and "Click the button labelled OK." in info["goal"]
    assert "[1] button 'OK'" in info["obs_text"]

    step = bridge._request(port, "/step", payload={"action": "click(bid='1')"})
    assert step["ok"] and step["n_steps"] == 1 and step["done"] is False

    bad = bridge._request(port, "/step", payload={"action": "click(bid='bad')"})
    assert bad["ok"] is False and "unknown action" in bad["error"]

    final = bridge._request(port, "/step", payload={"action": "send_msg_to_user(msg='done')"})
    assert final["ok"] and final["done"] and final["reward"] == 1.0 and final["success"] is True
    assert final["user_message"] == "done"  # surfaced from the step's info dict

    # result file written at termination; steps after done are rejected softly
    payload = json.loads(result_file.read_text())
    assert payload["reward"] == 1.0 and payload["success"] is True
    post = bridge._request(port, "/step", payload={"action": "noop()"})
    assert post["ok"] and "already finished" in post["obs_text"]

    result = bridge._request(port, "/result")
    assert result["n_steps"] == 2 and result["success"] is True


def test_launch_patch_injects_rootless_chromium_flags(daemon):
    """The patch must fill (not overwrite) rootless-safe launch kwargs."""
    from playwright.sync_api import BrowserType

    browser_type = BrowserType()
    BrowserType.launch(browser_type)  # no user kwargs -> defaults injected
    kwargs = BrowserType.last_launch_kwargs
    assert kwargs["chromium_sandbox"] is False
    assert "--disable-dev-shm-usage" in kwargs["args"]
    assert "--disable-gpu" in kwargs["args"]

    BrowserType.launch(browser_type, chromium_sandbox=True, args=["--foo"])  # user choices win
    kwargs = BrowserType.last_launch_kwargs
    assert kwargs["chromium_sandbox"] is True  # setdefault respected the caller
    assert kwargs["args"][:1] == ["--foo"]
    assert "--disable-dev-shm-usage" in kwargs["args"]
