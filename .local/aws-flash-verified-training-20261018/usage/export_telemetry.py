"""Archive the retained scheduler telemetry without claiming dropped coverage."""
import csv
import gzip
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:18090/api/telemetry/v1/"


def fetch(path):
    with urllib.request.urlopen(BASE + path, timeout=120) as response:
        return json.load(response)


metrics = fetch("metrics")
with gzip.open(ROOT / "scheduler-metrics.json.gz", "wt") as stream:
    json.dump(metrics, stream)
upper = metrics["high_watermark"]
cursor = 0
pages = []
rows = []
count = 0
fields = ["timestamp", "cursor", "node_id", "tracked", "running", "pending_booting",
          "pending_setup", "tearing_down", "memory_bytes", "cpu_cores", "cpu_usage_usec",
          "memory_byte_seconds", "observed_at", "scope_id", "scope_generation",
          "complete", "cpu_complete"]
with gzip.open(ROOT / "scheduler-events.jsonl.gz", "wt") as archive:
    while cursor < upper:
        page = fetch(f"events?after={cursor}&limit=10000")
        assert page["server_instance_id"] == metrics["server_instance_id"], "Server reset"
        pages.append({key: value for key, value in page.items() if key != "events"})
        events = [event for event in page["events"] if event["cursor"] <= upper]
        for event in events:
            assert event["cursor"] > cursor, "Nonmonotonic export"
            archive.write(json.dumps(event, separators=(",", ":")) + "\n")
            count += 1
            if event["kind"] != "resource_sample":
                continue
            sample = event["details"]["sample"]
            lifecycle = sample.get("sandbox_lifecycle", {})
            for node in sample.get("nodes", []):
                usage = node.get("residency_memory", {})
                row = {key: usage.get(key) for key in fields}
                row.update(timestamp=event["timestamp_unix_seconds"], cursor=event["cursor"],
                           node_id=node["id"], tracked=sample.get("sandbox_count"),
                           running=lifecycle.get("running", 0),
                           pending_booting=lifecycle.get("pending_booting", 0),
                           pending_setup=lifecycle.get("pending_setup", 0),
                           tearing_down=lifecycle.get("tearing_down", 0),
                           memory_bytes=usage.get("current_bytes"))
                rows.append(row)
        next_cursor = page["next_cursor"]
        if not events or next_cursor <= cursor:
            break
        cursor = next_cursor
with (ROOT / "resource-samples.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
metadata = {"captured_high_watermark": upper, "events_archived": count,
            "resource_samples": len(rows), "pages": pages}
(ROOT / "telemetry-export.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps({key: value for key, value in metadata.items() if key != "pages"}))
