"""Compute connectivity checks against a local HTTP gateway; no AWS or SSH."""
import contextlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "cscs/sandbox_sweep/gateway_preflight.py"
SPEC = importlib.util.spec_from_file_location("gateway_preflight", HELPER)
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)
KEY = "preflight-test-secret"
READY = [{"id": "runtime-1", "status": "ready"}, {"id": "runtime-2", "status": "ready"}]


@contextlib.contextmanager
def gateway(payload=READY, *, status=200, delay=0):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            requests.append((self.path, self.headers.get("X-API-Key")))
            time.sleep(delay)
            self.send_response(status, KEY if status != 200 else None)
            self.end_headers()
            try:
                self.wfile.write(json.dumps(payload).encode())
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class GatewayPreflightTests(unittest.TestCase):
    def test_authenticated_direct_request_bypasses_environment_proxies(self):
        env = {name: "http://127.0.0.1:1" for name in
               ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")}
        env.update(NO_PROXY="", no_proxy="")
        with gateway() as (endpoint, requests), patch.dict(os.environ, env):
            preflight.check_gateway(endpoint, KEY, 2)
        self.assertEqual(requests, [("/nodes", KEY)])

    def test_unready_or_wrong_count_fails_after_one_request(self):
        for payload, failure in [(READY[:1], "NodeCountMismatch"),
                                 ([*READY, READY[0]], "NodeCountMismatch"),
                                 ([READY[0], {"status": "connecting"}], "NodesNotReady"),
                                 ({"nodes": READY}, "InvalidNodeList")]:
            with self.subTest(payload=payload), gateway(payload) as (endpoint, requests):
                with self.assertRaisesRegex(RuntimeError, failure) as error:
                    preflight.check_gateway(endpoint, KEY, 2)
                self.assertIn(socket.gethostname(), str(error.exception))
                self.assertNotIn(KEY, str(error.exception))
                self.assertEqual(len(requests), 1)

    def test_authentication_error_does_not_print_server_reason_or_key(self):
        with gateway(status=403) as (endpoint, requests):
            with self.assertRaisesRegex(RuntimeError, "HTTP403") as error:
                preflight.check_gateway(endpoint, KEY, 2)
            self.assertNotIn(KEY, str(error.exception))
            self.assertEqual(len(requests), 1)

    def test_timeout_fails_without_retry(self):
        with gateway(delay=0.2) as (endpoint, requests):
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "TimeoutError"):
                preflight.check_gateway(endpoint, KEY, 2, timeout=0.03)
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertEqual(len(requests), 1)

    def test_connectivity_failure_has_hostname_and_class_without_url(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            with self.assertRaisesRegex(RuntimeError, "ConnectionRefusedError") as error:
                preflight.check_gateway(f"http://127.0.0.1:{port}/?{KEY}", KEY, 2)
        self.assertIn(socket.gethostname(), str(error.exception))
        self.assertNotIn(KEY, str(error.exception))

    def test_node_script_stops_before_ray_or_collectors_on_failed_preflight(self):
        with tempfile.TemporaryDirectory() as directory, gateway(status=401) as (endpoint, requests):
            run = Path(directory)
            (run / "gateway_preflight.py").write_text(HELPER.read_text())
            bin_dir = run / "bin"
            bin_dir.mkdir()
            (bin_dir / "python3").symlink_to(sys.executable)
            marker = run / "ray-started"
            ray = bin_dir / "ray"
            ray.write_text(f"#!/bin/sh\ntouch '{marker}'\nexit 99\n")
            ray.chmod(0o755)
            env = dict(os.environ, PATH=str(bin_dir) + ":" + os.environ["PATH"],
                       UNI_AGENT_CPU_MEMORY_BOUND="1", RUN_DIR=str(run), REPO_DIR=str(ROOT),
                       AGENTENV_EXPECTED_NODE_COUNT="2", AGENTENV_ENDPOINT=endpoint,
                       AGENTENV_API_KEY=KEY)
            result = subprocess.run(["bash", str(ROOT / "cscs/sandbox_sweep/node.sh")],
                                    env=env, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 1)
            self.assertIn("HTTP401", result.stderr)
            self.assertNotIn(KEY, result.stdout + result.stderr)
            self.assertFalse(marker.exists())
            self.assertFalse((run / "telemetry").exists())
            self.assertEqual(len(requests), 1)


if __name__ == "__main__":
    unittest.main()
