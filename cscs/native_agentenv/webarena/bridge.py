#!/usr/bin/env python
"""BrowserGym bridge: a stateful env daemon + a stateless client CLI, one file.

BrowserGym's ``BrowserEnv`` is turn-based and holds live state (a headless
Chromium via playwright's *sync* API), while sandbox backends like
flash-sandbox only offer one-shot ``exec`` calls. This bridge splits the
difference:

* ``serve``   -- the daemon: creates + resets a ``BrowserEnv``, then serves
  ``step`` / ``info`` / ``result`` / ``health`` / ``shutdown`` over plain HTTP
  on a sandbox-local port. All env operations are serialized onto the thread
  that created the env (playwright's sync API is not thread-safe); HTTP
  handler threads hand closures to that worker and wait for the answer.
* everything else -- the client: one ``exec`` per call, JSON on stdout,
  exit code 0/1. The host-side ``browser`` tool (``uni_agent/tools/browser.py``)
  shells out to these client subcommands.

Typical episode inside one sandbox::

    python /opt/bgym/bridge.py serve --task browsergym.miniwob.miniwob_click-test \
        --seed 0 --port 8027 --obs-max-chars 30000 --result-file /logs/bgym/result.json &
    python /opt/bgym/bridge.py wait   --port 8027 --timeout 240
    python /opt/bgym/bridge.py info   --port 8027            # goal + initial AXTree
    python /opt/bgym/bridge.py step   --port 8027 --action "click(bid='12')"
    python /opt/bgym/bridge.py result --port 8027            # final reward (after done)
    python /opt/bgym/bridge.py shutdown --port 8027

Only the daemon imports browsergym / playwright, so the client subcommands run
on any image with plain Python 3.

Rootless-sandbox notes (flash-sandbox runs Chromium in a rootless container):

* playwright's launch is patched to default to ``chromium_sandbox=False`` plus
  ``--disable-dev-shm-usage --disable-gpu``; opt back in with
  ``BROWSERGYM_CHROMIUM_SANDBOX=1`` if your runtime supports the sandbox.
* MiniWoB++ tasks are self-hosted: the daemon statically serves the
  ``browsergym.miniwob`` package's ``www/`` tree on 127.0.0.1:8000 (skipped
  when ``MINIWOB_URL`` is already set to a running server).
"""

from __future__ import annotations

import argparse
import inspect
import itertools
import json
import logging
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Queue
from urllib.error import URLError
from urllib.request import Request, urlopen

LOG = logging.getLogger("bgym_bridge")

#: Extra Chromium flags every rootless/container runtime wants. ``--no-zygote``
#: is REQUIRED on flash-sandbox: the zygote process cannot fork under the
#: microVM backend's restrictions, which kills Page.captureScreenshot even
#: though page navigation and AXTree extraction work fine (verified empirically:
#: only no-zygote / single-process flag sets capture screenshots there).
_DEFAULT_LAUNCH_ARGS = ("--disable-dev-shm-usage", "--disable-gpu", "--no-zygote")


# --------------------------------------------------------------------------- #
# env worker: serialize env ops on the creating thread (playwright sync API)
# --------------------------------------------------------------------------- #
class _EnvWorker:
    """Run closures one-at-a-time on the env thread; handlers wait for results."""

    def __init__(self) -> None:
        self._queue: Queue = Queue()
        self._pending: dict[int, threading.Event] = {}
        self._results: dict[int, object] = {}
        self._tokens = itertools.count(1)
        self._lock = threading.Lock()

    def submit(self, fn, timeout: float | None = None):
        """Run ``fn`` on the worker thread and return its result (or exception)."""
        token = next(self._tokens)
        event = threading.Event()
        with self._lock:
            self._pending[token] = event
        self._queue.put((token, fn))
        if not event.wait(timeout):
            with self._lock:
                self._pending.pop(token, None)
            raise TimeoutError(f"env operation did not complete within {timeout}s")
        with self._lock:
            return self._results.pop(token)

    def run_forever(self) -> None:
        """Consume closures until a ``None`` sentinel arrives (call on the env thread)."""
        while True:
            item = self._queue.get()
            if item is None:
                return
            token, fn = item
            try:
                result = fn()
            except BaseException as exc:  # surfaced to the submitter, not the worker
                result = exc
            with self._lock:
                event = self._pending.pop(token, None)
                self._results[token] = result
            if event is not None:
                event.set()

    def stop(self) -> None:
        self._queue.put(None)


# --------------------------------------------------------------------------- #
# daemon state
# --------------------------------------------------------------------------- #
class _EpisodeState:
    """Mutable episode bookkeeping shared by the handlers (worker-serialized)."""

    def __init__(self) -> None:
        self.env: object | None = None
        self.n_steps = 0
        self.cum_reward = 0.0
        self.last_reward = 0.0
        self.done = False
        self.truncated = False
        self.goal_text = ""
        self.obs_text = ""
        self.url = ""
        self.user_message = ""
        self.error: str | None = None
        self.result_file: Path | None = None
        self.recovering = False

    def result_payload(self, ok: bool = True, error: str | None = None) -> dict:
        payload = {
            "ok": ok,
            "reward": self.cum_reward,
            "final_reward": self.last_reward,
            "done": self.done,
            "truncated": self.truncated,
            "n_steps": self.n_steps,
            "url": self.url,
            "user_message": self.user_message,
            # success = the episode terminated on its own AND scored > 0
            # (MiniWoB/WebArena rewards are sparse 0/1 at the terminal step).
            "success": bool(self.done and not self.truncated and self.cum_reward > 0),
        }
        if error is not None:
            payload["error"] = error
        return payload


STATE = _EpisodeState()
WORKER = _EnvWorker()


def _write_result_file() -> None:
    if STATE.result_file is None:
        return
    try:
        STATE.result_file.parent.mkdir(parents=True, exist_ok=True)
        STATE.result_file.write_text(json.dumps(STATE.result_payload(), indent=2), encoding="utf-8")
    except Exception:
        LOG.warning("could not write result file %s", STATE.result_file, exc_info=True)


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    note = f"\n<NOTE>observation clipped at {limit} chars</NOTE>"
    return text[:limit] + note


def _coerce_goal(goal) -> str:
    if goal is None:
        return ""
    if isinstance(goal, str):
        return goal
    if isinstance(goal, (list, tuple)):
        return "\n".join(str(g) for g in goal)
    return json.dumps(goal, ensure_ascii=False, default=str)


# --------------------------------------------------------------------------- #
# env operations (called ON the worker thread only)
# --------------------------------------------------------------------------- #
def _make_env(task: str, kwargs: dict) -> object:
    """Create the env, current API first, legacy fallback second.

    browsergym >= 0.4 registers tasks in the gymnasium registry under ids like
    ``browsergym/miniwob.click-test`` (built via ``gym.make``); older versions
    constructed ``BrowserEnv(env_name=...)`` directly. Unknown kwargs are
    dropped with a warning against whichever constructor we end up calling.
    """
    from browsergym.core.env import BrowserEnv

    params = inspect.signature(BrowserEnv.__init__).parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    env_id = _normalize_env_id(task)
    try:
        import gymnasium as gym

        _import_benchmark_module(env_id)
        safe = {k: v for k, v in kwargs.items() if k in params or accepts_var_kw}
        dropped = sorted(set(kwargs) - set(safe))
        if dropped:
            LOG.warning("BrowserEnv does not accept %s; dropping them", dropped)
        return gym.make(env_id, **safe)
    except Exception as exc:
        LOG.warning("gym.make(%r) failed (%s); trying legacy BrowserEnv(env_name=...)", env_id, exc)
        legacy_params = params
        legacy = {
            k: v
            for k, v in (("env_name", task), *(kwargs.items()))
            if k == "env_name" or k in legacy_params or accepts_var_kw
        }
        return BrowserEnv(**legacy)


def _normalize_env_id(task: str) -> str:
    """Canonical gym id: ``browsergym/miniwob.click-test``.

    Accepts the dotted legacy spelling (``browsergym.miniwob.click-test``) and
    bare family-qualified names, normalizing to the slash form the registry
    uses since browsergym 0.4.
    """
    if task.startswith("browsergym/"):
        return task
    if task.startswith("browsergym."):
        return "browsergym/" + task[len("browsergym.") :]
    return task


def _import_benchmark_module(env_id: str) -> None:
    """Import ``browsergym.<family>`` so its tasks are registered with gym."""
    import importlib

    family = env_id.split("/", 1)[1].split(".", 1)[0]  # "browsergym/miniwob.x" -> "miniwob"
    try:
        importlib.import_module(f"browsergym.{family}")
    except ImportError:
        LOG.warning("could not import browsergym.%s; is the benchmark package in the image?", family)


def _env_reset(seed: int | None) -> None:
    """Create + reset the env, then cache the derived observation text."""
    out = STATE.env.reset(seed=seed)
    if isinstance(out, tuple):  # gym >= 0.26 -> (obs, info)
        obs = out[0]
        info = out[1] if len(out) > 1 else {}
    else:
        obs, info = out, {}
    _cache_obs(obs, info)
    STATE.goal_text = _coerce_goal(obs.get("goal") or obs.get("goal_object"))
    LOG.info("env reset done: goal=%r url=%s obs_chars=%d", STATE.goal_text[:200], STATE.url, len(STATE.obs_text))


def _flatten_obs(obs: dict) -> str:
    """Flatten the AXTree (API moved across browsergym versions).

    >= 0.4: ``flatten_axtree_to_str(axtree_object, extra_properties=...)`` lives
    in ``browsergym.utils.obs``; older versions took the whole obs dict from
    ``browsergym.core.observation``. Dispatch on the signature.
    """
    try:
        from browsergym.utils.obs import flatten_axtree_to_str
    except ImportError:
        from browsergym.core.observation import flatten_axtree_to_str

    params = inspect.signature(flatten_axtree_to_str).parameters
    if "extra_properties" in params and "axtree_object" in obs:
        return flatten_axtree_to_str(
            obs["axtree_object"], extra_properties=obs.get("extra_element_properties")
        )
    return flatten_axtree_to_str(obs)


def _cache_obs(obs: dict, info: dict | None = None) -> None:
    STATE.obs_text = _flatten_obs(obs)
    STATE.url = str(obs.get("url", "") or "")
    info = info or {}
    STATE.user_message = str(
        obs.get("user_message") or info.get("send_message_to_user") or info.get("user_message") or ""
    )


def _env_step(action: str) -> dict:
    """Step the env once; invalid actions return ``ok: False`` for self-correction."""
    if STATE.done:
        payload = STATE.result_payload()
        payload["obs_text"] = "The episode is already finished. No further actions are accepted."
        return payload
    if os.environ.get("BRIDGE_TEST_STUCK") and STATE.n_steps == 0:
        # Deterministic watchdog test: sleep well past the step timeout; the
        # handler must mark the episode truncated and kill Chromium.
        LOG.warning("TEST HOOK: simulating a stuck env (sleeping 400s)")
        time.sleep(400)
    try:
        out = STATE.env.step(action)
        if len(out) == 5:  # gym >= 0.26: (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, _info = out
        else:  # older API: (obs, reward, done, info)
            obs, reward, terminated, _info = out
            truncated = False
    except Exception as exc:  # bad action syntax, unknown bid, hung page, ...
        LOG.warning("step rejected: %s: %s", type(exc).__name__, exc)
        return {
            "ok": False,
            "error": f"Invalid action ({type(exc).__name__}): {exc}",
            "n_steps": STATE.n_steps,
        }
    STATE.n_steps += 1
    STATE.last_reward = float(reward or 0.0)
    STATE.cum_reward += STATE.last_reward
    _cache_obs(obs, _info if isinstance(_info, dict) else {})
    STATE.done = bool(terminated)
    STATE.truncated = bool(truncated)
    LOG.info(
        "step %d: action=%r reward=%.3f done=%s truncated=%s",
        STATE.n_steps,
        action,
        STATE.last_reward,
        STATE.done,
        STATE.truncated,
    )
    payload = STATE.result_payload()
    payload["obs_text"] = STATE.obs_text
    if STATE.done:
        LOG.info("episode finished: n_steps=%d reward=%.3f truncated=%s", STATE.n_steps, STATE.cum_reward, truncated)
        _write_result_file()
    return payload


def _env_close() -> None:
    try:
        STATE.env.close()
    except Exception:
        LOG.warning("env.close() failed", exc_info=True)


# --------------------------------------------------------------------------- #
# stuck-env watchdog
# --------------------------------------------------------------------------- #
#: Measured in production (job 3617068 autopsies): a hung playwright/env call
#: blocks the env worker thread FOREVER -- every later /step queues behind it
#: and times out, so a once-live daemon becomes a zombie (63 of 167 failures:
#: DAEMON-ALIVE with a frozen log). The watchdog converts that hang into a
#: clean truncated episode: mark terminal state first (handler-level
#: short-circuit answers immediately, no worker round-trip), then kill
#: Chromium so the stuck call raises and the worker thread unblocks.


def _kill_chromium() -> None:
    """Best-effort SIGKILL of the Chromium tree from a non-env thread."""
    import subprocess

    for pattern in ("ms-playwright", "headless_shell", "chromium", "chrome"):
        try:
            subprocess.run(["pkill", "-9", "-f", pattern], timeout=10,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except Exception:
            pass
    LOG.warning("watchdog: chromium kill sweep done")


def _recover_stuck_env(reason: str) -> None:
    """Mark the episode truncated, persist the result, then unblock the worker."""
    if STATE.recovering:
        return
    STATE.recovering = True
    LOG.warning("watchdog: env stuck (%s) -- marking episode truncated, killing browser", reason)
    STATE.done = True
    STATE.truncated = True
    STATE.error = f"env stuck: {reason}"
    _write_result_file()
    threading.Thread(target=_kill_chromium, name="bgym-watchdog", daemon=True).start()


# --------------------------------------------------------------------------- #
# HTTP server
# --------------------------------------------------------------------------- #
class _BridgeHandler(BaseHTTPRequestHandler):
    server_version = "bgym-bridge/1"

    def log_message(self, fmt, *args):  # route access logs through logging
        LOG.debug("http %s", fmt % args)

    def _reply(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler API)
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        LOG.info("GET %s (n_steps=%d done=%s)", path, STATE.n_steps, STATE.done)
        if path in ("/health", "/"):
            self._reply({"ok": True, "n_steps": STATE.n_steps, "done": STATE.done})
        elif path == "/info":
            self._reply(
                {
                    "ok": True,
                    "goal": STATE.goal_text,
                    "obs_text": STATE.obs_text,
                    "url": STATE.url,
                    "n_steps": STATE.n_steps,
                    "done": STATE.done,
                }
            )
        elif path == "/result":
            self._reply(STATE.result_payload())
        else:
            self._reply({"ok": False, "error": f"unknown path {path!r}"}, status=404)

    def do_POST(self):  # noqa: N802 (BaseHTTPRequestHandler API)
        path = self.path.split("?", 1)[0].rstrip("/")
        body = self._body()
        LOG.info("POST %s body=%s (n_steps=%d done=%s)", path, str(body)[:80], STATE.n_steps, STATE.done)
        if path == "/step":
            action = str(body.get("action", ""))
            if STATE.done or STATE.truncated:
                # Handler-level short-circuit: the worker may be stuck (see
                # watchdog); answer from state without queuing behind it.
                payload = STATE.result_payload()
                payload["obs_text"] = (
                    f"The episode has finished early (truncated={STATE.truncated}, "
                    f"error={STATE.error}). No further browser actions are accepted."
                )
                self._reply(payload)
                return
            try:
                payload = WORKER.submit(
                    lambda: _env_step(action),
                    timeout=float(os.environ.get("BRIDGE_STEP_TIMEOUT", "300")),
                )
            except TimeoutError as exc:
                _recover_stuck_env(f"env.step did not complete within 300s: {exc}")
                payload = {
                    "ok": False,
                    "error": (
                        "the environment stopped responding (step timeout); the episode "
                        "has been marked truncated -- call submit to finish"
                    ),
                    "truncated": True,
                    "done": True,
                }
            self._reply(payload)
        elif path == "/shutdown":
            self._reply({"ok": True, "bye": True})
            WORKER.stop()  # worker thread exits its loop; main() finalizes
        else:
            self._reply({"ok": False, "error": f"unknown path {path!r}"}, status=404)


# --------------------------------------------------------------------------- #
# serve plumbing
# --------------------------------------------------------------------------- #
def _patch_playwright_launch() -> None:
    """Default Chromium launch to rootless-container-safe settings.

    BrowserGym launches Chromium itself; depending on version it may or may not
    expose launch flags. Patching ``BrowserType.launch`` is version-proof: we
    only fill in kwargs the caller didn't set.
    """
    from playwright.sync_api import BrowserType

    original = BrowserType.launch

    def launch(self, *args, **kwargs):
        if os.environ.get("BROWSERGYM_CHROMIUM_SANDBOX", "0") != "1":
            kwargs.setdefault("chromium_sandbox", False)
        extra = list(kwargs.get("args") or [])
        for flag in _DEFAULT_LAUNCH_ARGS:
            if flag not in extra:
                extra.append(flag)
        kwargs["args"] = extra
        return original(self, *args, **kwargs)

    BrowserType.launch = launch
    LOG.info("patched playwright BrowserType.launch (chromium_sandbox=False, %s)", _DEFAULT_LAUNCH_ARGS)


def _miniwob_site_candidates() -> list[tuple[Path, str]]:
    """(docroot, url_base) candidates for locally serving the MiniWoB sites.

    browsergym <= 0.3 bundled the sites under its ``www/`` tree; newer versions
    expect the standalone Farama ``miniwob`` package, whose HTML lives under
    ``miniwob/html/miniwob/<subdomain>.html`` (hence the ``/miniwob/`` URL base).
    """
    candidates: list[tuple[Path, str]] = []
    try:
        import browsergym.miniwob as bm

        candidates.append((Path(bm.__file__).parent / "www", "/"))
    except ImportError:
        pass
    try:
        import miniwob

        candidates.append((Path(miniwob.__file__).parent / "html", "/miniwob/"))
    except ImportError:
        pass
    return candidates


def _maybe_serve_miniwob(task: str) -> None:
    """Statically serve the MiniWoB HTML tree when the env needs it.

    BrowserGym's miniwob tasks point a local browser at ``MINIWOB_URL`` and
    refuse to construct without it. Serving the tree ourselves is version-
    proof: no external process, no upstream CLI-name churn.
    """
    if "miniwob" not in task or os.environ.get("MINIWOB_URL"):
        return
    from functools import partial
    from http.server import SimpleHTTPRequestHandler

    class _QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, fmt, *args):  # page assets would spam the log
            LOG.debug("miniwob-www " + fmt % args)

    for docroot, url_base in _miniwob_site_candidates():
        if not docroot.is_dir():
            continue
        handler = partial(_QuietHandler, directory=str(docroot))
        server = ThreadingHTTPServer(("127.0.0.1", 8000), handler)
        threading.Thread(target=server.serve_forever, name="miniwob-www", daemon=True).start()
        os.environ.setdefault("MINIWOB_URL", f"http://127.0.0.1:8000{url_base}")
        LOG.info(
            "serving miniwob sites %s on http://127.0.0.1:8000%s (MINIWOB_URL=%s)",
            docroot,
            url_base,
            os.environ["MINIWOB_URL"],
        )
        return
    LOG.warning(
            "no local MiniWoB site tree found (pip install miniwob into the image, or set "
            "MINIWOB_URL to a running server); miniwob tasks will fail to construct"
        )


def cmd_serve(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    STATE.result_file = Path(args.result_file) if args.result_file else None

    _patch_playwright_launch()
    _maybe_serve_miniwob(args.task)

    env_kwargs = {"headless": not args.no_headless}
    if args.slow_mo:
        env_kwargs["slow_mo"] = args.slow_mo
    if args.viewport:
        width, _, height = args.viewport.partition("x")
        env_kwargs["viewport"] = {"width": int(width), "height": int(height)}

    LOG.info("creating BrowserEnv(task=%s, %s)", args.task, env_kwargs)
    STATE.env = _make_env(args.task, env_kwargs)
    _env_reset(args.seed)  # cache goal + first observation
    STATE.obs_text = _truncate(STATE.obs_text, args.obs_max_chars)

    server = ThreadingHTTPServer((args.bind, args.port), _BridgeHandler)
    threading.Thread(target=server.serve_forever, name="bgym-http", daemon=True).start()
    LOG.info("bridge ready on http://%s:%d (task=%s seed=%s)", args.bind, args.port, args.task, args.seed)

    try:
        WORKER.run_forever()  # blocks until /shutdown
    except KeyboardInterrupt:
        pass
    finally:
        if not STATE.done:
            _write_result_file()
        _env_close()
        server.shutdown()
        LOG.info("bridge stopped: %s", json.dumps(STATE.result_payload()))
    return 0


# --------------------------------------------------------------------------- #
# client CLI (stdlib only -- no browsergym import)
# --------------------------------------------------------------------------- #
def _request(port: int, path: str, payload: dict | None = None, timeout: float = 60.0) -> dict:
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def cmd_client(args: argparse.Namespace) -> int:
    if args.command == "step":
        payload = _request(
            args.port,
            "/step",
            {"action": args.action},
            timeout=max(args.timeout, 1.0) if args.timeout else 300.0,
        )
    elif args.command == "info":
        payload = _request(args.port, "/info")
    elif args.command == "result":
        payload = _request(args.port, "/result")
    elif args.command == "health":
        payload = _request(args.port, "/health")
    elif args.command == "shutdown":
        payload = _request(args.port, "/shutdown", payload={})  # empty body => POST
    elif args.command == "wait":
        import time

        deadline = time.monotonic() + max(args.timeout, 1.0)
        while True:
            try:
                payload = _request(args.port, "/health", timeout=5.0)
                if payload.get("ok"):
                    print(json.dumps(payload))
                    return 0
            except (URLError, OSError, json.JSONDecodeError):
                pass
            if time.monotonic() >= deadline:
                print(json.dumps({"ok": False, "error": f"daemon not ready within {args.timeout}s"}))
                return 1
            time.sleep(args.poll)
    else:  # pragma: no cover - argparse restricts choices
        raise SystemExit(f"unknown command {args.command!r}")

    print(json.dumps(payload))
    return 0 if payload.get("ok") else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the env daemon (in the sandbox image)")
    serve.add_argument("--task", required=True, help="BrowserGym env name, e.g. browsergym.miniwob.miniwob_click-test")
    serve.add_argument("--seed", type=int, default=None, help="reset seed (per-episode determinism)")
    serve.add_argument("--port", type=int, default=8027)
    serve.add_argument("--bind", default="127.0.0.1")
    serve.add_argument("--no-headless", action="store_true", help="run Chromium headed (debug; needs X)")
    serve.add_argument("--slow-mo", type=float, default=0, help="playwright slow_mo (ms)")
    serve.add_argument("--viewport", default=None, help="viewport as WxH, e.g. 1280x720")
    serve.add_argument("--obs-max-chars", type=int, default=30000)
    serve.add_argument("--result-file", default=None, help="where the final result JSON is written")
    serve.add_argument("--log-level", default="INFO")
    serve.set_defaults(func=cmd_serve, is_serve=True)

    def client(name: str, help_: str):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--port", type=int, default=8027)
        p.add_argument("--timeout", type=float, default=None)
        p.set_defaults(func=cmd_client, is_serve=False)
        return p

    step = client("step", "send one action; prints the next observation as JSON")
    step.add_argument("--action", required=True, help="BrowserGym action, e.g. click(bid='12')")
    client("info", "goal + current observation")
    client("result", "final episode result (reward, done, n_steps)")
    client("health", "daemon liveness probe")
    client("shutdown", "close the env, write the result file, stop the daemon")
    wait = client("wait", "poll health until the daemon is ready")
    wait.add_argument("--poll", type=float, default=2.0)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
