"""Render the preserved whole-run measurements with the existing project environment."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
report = json.loads((root / 'usage-summary.json').read_text())
with (root / 'cluster-usage.csv').open() as stream:
    host = list(csv.DictReader(stream))
with (root / 'sandbox-concurrency.csv').open() as stream:
    vms = list(csv.DictReader(stream))
x = [float(row['elapsed_seconds']) / 60 for row in host]
vm_x = [float(row['elapsed_seconds']) / 60 for row in vms]
vm_y = [int(row['resident_vms']) for row in vms]
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, layout='constrained')
fig.suptitle('SWE-bench Verified · Qwen3.5-27B · Job 3692513', fontsize=16, fontweight='bold')

axes[0].step(vm_x, vm_y, where='post', color='#2563eb', linewidth=1.5)
axes[0].fill_between(vm_x, vm_y, step='post', color='#2563eb', alpha=.12)
axes[0].axhline(512, color='#9ca3af', linestyle='--', linewidth=1)
axes[0].set(ylabel='Resident sandboxes', ylim=(0, 560),
            title='Peak 512 · average 214 over the allocation · 161.45 sandbox-hours')
axes[0].annotate('First VM: 8m 56s', xy=(8 + 56 / 60, 0), xytext=(2, 180),
                 arrowprops={'arrowstyle': '->', 'color': '#6b7280'}, color='#4b5563', fontsize=9)

cpu = [float(row['host_cpu_cores']) for row in host]
axes[1].plot(x, cpu, color='#16a34a', linewidth=1.5)
axes[1].fill_between(x, cpu, color='#16a34a', alpha=.12)
axes[1].set(ylabel='AWS host CPU (vCPUs)', ylim=(0, 100),
            title='Average 26.5 / 96 vCPUs (27.6%) · sampled peak 91.0 vCPUs')

memory = [float(row['host_memory_gib']) for row in host]
axes[2].plot(x, memory, color='#9333ea', linewidth=1.5)
axes[2].fill_between(x, memory, color='#9333ea', alpha=.12)
axes[2].axhline(report['aws_host_usage']['total_memory_gib'], color='#9ca3af', linestyle='--', linewidth=1)
axes[2].set(ylabel='AWS host RAM (GiB)', ylim=(0, 400),
            title='Average 115.1 GiB · sampled peak 278.3 GiB · host total 377.5 GiB',
            xlabel='Minutes since start (2026-10-18 19:21:21 UTC)')
for ax in axes:
    ax.set_xlim(0, report['elapsed_seconds'] / 60)
    ax.grid(alpha=.18)
    ax.spines[['top', 'right']].set_visible(False)
    ax.title.set_fontsize(10)
fig.supxlabel('CPU/RAM: whole-host gauges sampled every 30s. Concurrency: successful VM lifecycle logs.\n'
               'Clariden allocation: 2 nodes, 8 GPUs, 45m12s; GPU utilization history unavailable.', fontsize=9)
fig.savefig(root / 'usage.png', dpi=160)
fig.savefig(root / 'usage.svg')
print(str(root / 'usage.png'))
