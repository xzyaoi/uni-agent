"""Build a run-scoped usage report from preserved host logs and Slurm accounting."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
START = datetime(2026, 10, 18, 19, 21, 21, tzinfo=timezone.utc).timestamp()
END = START + 2712


def stamp(text):
    return datetime.strptime(text, "%Y/%m/%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


def iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def write_csv(name, rows):
    with (ROOT / name).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


pattern = re.compile(r"^(\d{4}/\d{2}/\d{2} \d\d:\d\d:\d\d) cluster stats: .*?"
                     r"sandboxes_running=(\d+) sandboxes_tracked=(\d+) "
                     r"cpu=(\d+)m/(\d+)m .*?mem=(\d+)/(\d+) MB")
host = []
for line in (ROOT / "cluster.log").read_text().splitlines():
    match = pattern.search(line)
    if not match:
        continue
    time = stamp(match[1])
    if START - 31 <= time <= END + 31:
        host.append(dict(timestamp=time, timestamp_utc=iso(time), elapsed_seconds=time - START,
                         running=int(match[2]), tracked=int(match[3]),
                         host_cpu_cores=int(match[4]) / 1000, host_cpu_total=int(match[5]) / 1000,
                         host_memory_gib=int(match[6]) / 1024,
                         host_memory_total_gib=int(match[7]) / 1024))
assert host[0]["timestamp"] <= START and host[-1]["timestamp"] >= END
assert all(0 < b["timestamp"] - a["timestamp"] <= 31 for a, b in zip(host, host[1:]))
write_csv("cluster-usage.csv", host)


def integrate(key):
    total = 0
    for a, b in zip(host, host[1:]):
        ta, tb = a["timestamp"], b["timestamp"]
        low, high = max(ta, START), min(tb, END)
        if low >= high:
            continue
        left = a[key] + (b[key] - a[key]) * (low - ta) / (tb - ta)
        right = a[key] + (b[key] - a[key]) * (high - ta) / (tb - ta)
        total += (left + right) / 2 * (high - low)
    return total


residents = set()
events = []
timeline = [dict(timestamp=START, timestamp_utc=iso(START), elapsed_seconds=0,
                 event="baseline", sandbox_id="", resident_vms=0)]
orphan_destroys = []
duplicates = []
for line in (ROOT / "node.log").read_text().splitlines():
    match = re.match(r"^(\d{4}/\d{2}/\d{2} \d\d:\d\d:\d\d) "
                     r"(created|restored|destroyed) (sb-\d+)", line)
    if not match:
        continue
    time, action, identifier = stamp(match[1]), match[2], match[3]
    if not START <= time <= END:
        continue
    if action == "destroyed":
        if identifier not in residents:
            orphan_destroys.append(identifier)
        residents.discard(identifier)
    else:
        if identifier in residents:
            duplicates.append(identifier)
        residents.add(identifier)
    events.append((time, action, identifier))
    timeline.append(dict(timestamp=time, timestamp_utc=iso(time), elapsed_seconds=time - START,
                         event=action, sandbox_id=identifier, resident_vms=len(residents)))
assert not orphan_destroys and not duplicates, (orphan_destroys, duplicates)
assert not residents, "Run left resident VMs"
assert len([event for event in events if event[1] != "destroyed"]) == 1024
assert all(a["timestamp"] <= b["timestamp"] for a, b in zip(timeline, timeline[1:]))
timeline.append(dict(timestamp=END, timestamp_utc=iso(END), elapsed_seconds=END - START,
                     event="end", sandbox_id="", resident_vms=len(residents)))
vm_seconds = sum(a["resident_vms"] * (b["timestamp"] - a["timestamp"])
                 for a, b in zip(timeline, timeline[1:]))
write_csv("sandbox-concurrency.csv", timeline)

with (ROOT / "slurm-accounting.psv").open() as stream:
    accounting = list(csv.DictReader(stream, delimiter="|"))
allocation = next(row for row in accounting if row["JobID"] == "3692513")
training = next(row for row in accounting if row["JobID"] == "3692513.3")
assert int(allocation["ElapsedRaw"]) == END - START
hours = int(allocation["ElapsedRaw"]) / 3600
observed = [row for row in host if START <= row["timestamp"] <= END]
export = json.loads((ROOT / "telemetry-export.json").read_text())
report = dict(
    job_id="3692513", start_utc=iso(START), end_utc=iso(END), elapsed_seconds=END - START,
    state=allocation["State"], configured_global_concurrency=512,
    batch=dict(prompts=64, rollouts_per_prompt=8, samples=512),
    sandbox_usage=dict(created_vms=1024, destroyed_vms=1024,
                       peak_acknowledged_resident_vms=max(row["resident_vms"] for row in timeline),
                       sampled_peak_running_vms=max(row["running"] for row in observed),
                       average_acknowledged_resident_vms=vm_seconds / (END - START),
                       resident_vm_hours=vm_seconds / 3600,
                       first_vm_created_utc=iso(events[0][0]), last_vm_destroyed_utc=iso(events[-1][0]),
                       resolution="Successful VM lifecycle logs, second-resolution timestamps"),
    aws_host_usage=dict(instance_type="m5d.metal", cpu_cores=96,
                        average_cpu_cores=integrate("host_cpu_cores") / (END - START),
                        average_cpu_percent=integrate("host_cpu_cores") / (END - START) / 96 * 100,
                        sampled_peak_cpu_cores=max(row["host_cpu_cores"] for row in observed),
                        average_memory_gib=integrate("host_memory_gib") / (END - START),
                        sampled_peak_memory_gib=max(row["host_memory_gib"] for row in observed),
                        total_memory_gib=observed[0]["host_memory_total_gib"],
                        sample_count=len(observed), sample_interval_seconds=30,
                        average_method="Trapezoidal integration, linearly interpolated run boundaries",
                        scope="Whole AWS host gauges, includes worker and host overhead"),
    clariden_usage=dict(nodes=2, gpus=8, allocated_node_hours=2 * hours,
                       allocated_gpu_hours=8 * hours,
                       allocated_cpu_core_hours=int(allocation["CPUTimeRAW"]) / 3600,
                       total_cpu_time=allocation["TotalCPU"],
                       training_step_max_rss_gib=int(training["MaxRSS"]) / 1024 ** 3,
                       node_energy_joules=int(allocation["ConsumedEnergyRaw"]),
                       node_energy_kwh=int(allocation["ConsumedEnergyRaw"]) / 3_600_000,
                       historical_gpu_utilization_percent=None),
    telemetry_coverage=dict(retained_events=export["events_archived"],
                            retained_resource_samples=export["resource_samples"],
                            first_page=export["pages"][0],
                            note="Scheduler ring lost early samples; full-run host gauges and VM concurrency recovered from durable logs"),
    limitations=["Host CPU/RAM peaks are 30-second sampled peaks, not continuous maxima",
                 "VM hours use acknowledged created/restored/destroyed lifecycle events, not admission queue time",
                 "GPU hours measure allocation; utilization percentages were not retained",
                 "No optimizer step completion or checkpoint was recorded"])
token_path = ROOT / "training-token-usage.json"
if token_path.exists():
    report["training_usage"] = json.loads(token_path.read_text())
    assert report["training_usage"]["trajectory_files"] == 1024
    assert report["training_usage"]["peak_overlapping_intervals"]["trial"] == report["sandbox_usage"]["peak_acknowledged_resident_vms"] == 512
(ROOT / "usage-summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({key: value for key, value in report.items() if key != "telemetry_coverage"}, indent=2))
