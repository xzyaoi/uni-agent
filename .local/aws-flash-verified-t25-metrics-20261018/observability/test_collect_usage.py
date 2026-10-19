import argparse
import csv
import importlib.util
import json
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).with_name("collect_usage.py")
spec = importlib.util.spec_from_file_location("collect_usage", SCRIPT)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/fail":
            self.send_error(503)
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'vllm:num_requests_running{model_name="qwen"} 17\n')

    def log_message(self, *args):
        pass


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.endpoint = "127.0.0.1:" + str(self.server.server_port)
        self.discovery = self.root / "file_sd.json"
        self.args = argparse.Namespace(
            discovery_file=self.discovery,
            discovery_format="file-sd",
            max_bytes=1024 * 1024,
            max_targets=8,
            timeout_seconds=1,
        )
        self.context = {"job_id": "42", "node": "head", "hostname": "head"}

    def collect(self):
        path = self.root / "records.jsonl"
        records = collector.Records(path)
        try:
            collector.collect_prometheus(self.args, records, self.context)
        finally:
            records.close()
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_late_discovery_and_endpoint_restart(self):
        rows = self.collect()
        self.assertFalse(rows[-1]["ok"])
        self.discovery.write_text(json.dumps([{"targets": [self.endpoint]}]))
        rows = self.collect()
        self.assertTrue(rows[-1]["ok"])
        self.assertIn("num_requests_running", rows[-1]["prometheus"])
        self.discovery.write_text(json.dumps([{"targets": ["http://" + self.endpoint + "/new-replica"]}]))
        rows = self.collect()
        self.assertTrue(rows[-1]["source_url"].endswith("/new-replica"))
        self.assertEqual(rows[-1]["labels"]["job_id"], "42")

    def test_http_failure_preserves_other_replica_and_labels(self):
        self.discovery.write_text(
            json.dumps(
                [
                    {"targets": ["http://" + self.endpoint + "/fail"], "labels": {"replica": "bad"}},
                    {"targets": [self.endpoint], "labels": {"replica": "healthy"}},
                ]
            )
        )
        rows = self.collect()
        self.assertEqual(len(rows), 2)
        by_replica = {r["labels"]["replica"]: r for r in rows}
        self.assertEqual(by_replica["bad"]["http_status"], 503)
        self.assertTrue(by_replica["healthy"]["ok"])
        self.assertNotEqual(by_replica["bad"]["source_url"], by_replica["healthy"]["source_url"])
        self.assertTrue(all(isinstance(row["timestamp"], float) for row in rows))

    def test_prometheus_yaml_selects_only_rollout_targets(self):
        self.args.discovery_format = "prometheus-yaml"
        self.discovery.write_text(
            "scrape_configs:\n- job_name: ray\n  file_sd_configs: []\n"
            '- job_name: rollout\n  static_configs:\n  - targets: ["' + self.endpoint + '"]\n'
        )
        rows = self.collect()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["labels"]["job"], "rollout")
        self.assertEqual(rows[0]["labels"]["replica_index"], "0")

    def test_partial_yaml_retries_next_cycle(self):
        self.args.discovery_format = "prometheus-yaml"
        self.discovery.write_text("scrape_configs: [")
        self.assertFalse(self.collect()[-1]["ok"])
        self.discovery.write_text(
            'scrape_configs:\n- job_name: rollout\n  static_configs:\n  - targets: ["' + self.endpoint + '"]\n'
        )
        self.assertTrue(self.collect()[-1]["ok"])

    def test_http_response_limit_is_recorded(self):
        result = collector.scrape({"source_url": "http://" + self.endpoint + "/metrics", "labels": {}}, 1, 8)
        self.assertFalse(result["ok"])
        self.assertIn("byte limit", result["error"])
        self.assertNotIn("prometheus", result)

    def test_discovery_rejects_credentials_and_target_overflow(self):
        self.discovery.write_text(json.dumps([{"targets": ["http://secret:key@localhost:8080"]}]))
        with self.assertRaisesRegex(ValueError, "without credentials"):
            collector.discover(self.discovery, 10000, 1)
        self.discovery.write_text(json.dumps([{"targets": ["localhost:1", "localhost:2"]}]))
        with self.assertRaisesRegex(ValueError, "target limit"):
            collector.discover(self.discovery, 10000, 1)

    def test_raw_records_readable_while_open_and_after_reopen(self):
        path = self.root / "durable.jsonl"
        records = collector.Records(path)
        records.append({"timestamp": 1, "prometheus": "a 1\nb 2\n"})
        self.assertEqual(json.loads(path.read_text())["prometheus"], "a 1\nb 2\n")
        records.close()
        records = collector.Records(path)
        records.append({"timestamp": 2})
        records.close()
        self.assertEqual(len(path.read_text().splitlines()), 2)

    def run_child(self):
        process = subprocess.Popen(
            [
                sys.executable,
                str(SCRIPT),
                "--mode",
                "prometheus",
                "--discovery-file",
                str(self.discovery),
                "--output-dir",
                str(self.root),
                "--hostname",
                "testnode",
                "--job-id",
                "42",
                "--interval-seconds",
                "5",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.addCleanup(lambda: process.kill() if process.poll() is None else None)
        path = self.root / "testnode-vllm.jsonl"
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if path.exists() and '"kind":"discovery"' in path.read_text():
                return process, path
            time.sleep(0.01)
        self.fail("collector did not flush initial discovery record")

    def test_sigterm_flushes_and_exits_without_waiting_for_interval(self):
        process, path = self.run_child()
        process.send_signal(signal.SIGTERM)
        _, error = process.communicate(timeout=2)
        self.assertEqual(process.returncode, 0, error)
        self.assertEqual(json.loads(path.read_text().splitlines()[-1])["kind"], "collector_stopped")

    def test_sigkill_keeps_prior_samples_readable(self):
        process, path = self.run_child()
        process.kill()
        process.communicate(timeout=2)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(rows[0]["kind"], "collector_started")
        self.assertEqual(rows[-1]["kind"], "discovery")

    def test_gpu_schema_csv_quoting_and_unavailable_values(self):
        raw = '0, GPU-one, "NVIDIA, H100", 98, N/A, 40000, 80000, 500.0, 72\n'
        rows = collector.parse_gpu_rows(raw)
        self.assertEqual(rows[0]["name"], "NVIDIA, H100")
        self.assertEqual(rows[0]["utilization.memory"], "N/A")
        with self.assertRaisesRegex(ValueError, "field count"):
            collector.parse_gpu_rows("0, GPU-one\n")

    def test_gpu_success_csv_and_error_event(self):
        events = collector.Records(self.root / "gpu.jsonl")
        self.addCleanup(events.close)
        with (self.root / "gpu.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=("job_id", "hostname", "node", "timestamp", *collector.GPU_FIELDS)
            )
            writer.writeheader()
            result = subprocess.CompletedProcess([], 0, "0, GPU-one, H100, 98, 30, 40, 80, 500, 72\n", "")
            with mock.patch.object(collector.subprocess, "run", return_value=result):
                collector.collect_gpu(self.args, events, self.context, writer, stream)
            with (self.root / "gpu.csv").open() as snapshot:
                self.assertEqual(list(csv.DictReader(snapshot))[0]["utilization.gpu"], "98")
            with mock.patch.object(collector.subprocess, "run", side_effect=FileNotFoundError("no nvidia-smi")):
                collector.collect_gpu(self.args, events, self.context, writer, stream)
        self.assertEqual(json.loads((self.root / "gpu.jsonl").read_text())["kind"], "gpu_error")


if __name__ == "__main__":
    unittest.main()
