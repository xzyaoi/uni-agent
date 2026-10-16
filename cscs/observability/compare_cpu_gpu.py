#!/usr/bin/env python3
"""Compare CPU and GPU counters over matching complete one-minute bins."""
import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    args=p.parse_args();root=args.study
    summary=json.loads((root/'summary.json').read_text())
    rows=list(csv.DictReader((root/'metrics_1min.csv').open()))
    end=dt.datetime.fromisoformat(summary['snapshot_end_utc']).timestamp()
    recent=[r for r in rows if end-600<=float(r['epoch']) and float(r['epoch'])+60<=end]
    keys=['sandbox_cpu_busy_pct','sandbox_cpu_guest_pct','sandbox_cpu_psi_pct',
          'trainer_gpu_busy_pct','trainer_sm_active_pct','trainer_tensor_active_pct',
          'rollout_gpu_busy_pct','rollout_sm_active_pct','rollout_tensor_active_pct']
    paired=[r for r in recent if all(r.get(k) for k in keys)]
    if not paired:raise SystemExit('No complete paired CPU/GPU bins')
    means={k:float(np.mean([float(r[k]) for r in paired])) for k in keys}
    auxiliary = {}
    for key in ['refill_inflight_prompts','refill_ready_prompts','refill_ready_threshold','refill_backlog_pause_fraction','rollout_vllm_active_requests','rollout_vllm_kv_cache_pct']:
        values_for_key = [float(row[key]) for row in paired if row.get(key)]
        if values_for_key:
            auxiliary[key] = float(np.mean(values_for_key))
    times=[dt.datetime.fromtimestamp(float(r['epoch']),dt.timezone.utc) for r in rows]
    def values(key):return [float(r[key]) if r.get(key) else np.nan for r in rows]
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.25})
    fig,axes=plt.subplots(3,1,figsize=(13,11))
    for key,label in [('sandbox_cpu_busy_pct','Sandbox CPU busy'),('sandbox_cpu_guest_pct','Guest execution (subset)'),('sandbox_cpu_psi_pct','CPU pressure / stalled time')]:
        axes[0].plot(times,values(key),label=label)
    axes[0].set_ylabel('CPU capacity / stall (%)');axes[0].legend(loc='upper left')
    for role in ['trainer','rollout']:
        axes[1].plot(times,values(role+'_sm_active_pct'),label=role.title()+' SM active')
        axes[1].plot(times,values(role+'_tensor_active_pct'),label=role.title()+' tensor active',linestyle='--')
    axes[1].set_ylabel('GPU activity (%)');axes[1].legend(loc='upper left',ncol=2)
    for ax in axes[:2]:
        ax.set_ylim(0,100);ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M',tz=dt.timezone.utc))
        ax.set_xlim(times[0],times[-1]);ax.set_xlabel('UTC on '+times[0].strftime('%Y-%m-%d'))
    x=np.arange(3);width=.24
    axes[2].bar(x-width,[means[k] for k in ['sandbox_cpu_busy_pct','trainer_gpu_busy_pct','rollout_gpu_busy_pct']],width,label='CPU busy / GPU busy')
    axes[2].bar(x,[np.nan,means['trainer_sm_active_pct'],means['rollout_sm_active_pct']],width,label='SM active')
    axes[2].bar(x+width,[np.nan,means['trainer_tensor_active_pct'],means['rollout_tensor_active_pct']],width,label='Tensor active')
    axes[2].set_xticks(x,['Sandbox CPU (128 threads)','Trainer GPUs (16)','Rollout GPUs (24)'])
    axes[2].set_ylim(0,105);axes[2].set_ylabel('Counter mean (%)');axes[2].legend(loc='upper right')
    first=dt.datetime.fromtimestamp(float(paired[0]['epoch']),dt.timezone.utc)
    last=dt.datetime.fromtimestamp(float(paired[-1]['epoch'])+60,dt.timezone.utc)
    axes[2].set_title('Matched recent window: '+first.strftime('%H:%M')+'–'+last.strftime('%H:%M')+' UTC; '+str(len(paired))+' complete bins')
    fig.suptitle('Job 3668373: sandbox CPU versus trainer and rollout GPUs\nDistinct counters: equal percentages do not imply equal compute efficiency')
    fig.tight_layout(rect=[0,0,1,.95])
    for ext in ['png','pdf']:fig.savefig(root/('10_cpu_gpu_comparison.'+ext),dpi=160)
    plt.close(fig)
    result={'window_start_utc':first.isoformat(),'window_end_utc':last.isoformat(),'paired_minutes':len(paired),'means':means,'feeder_and_serving':auxiliary}
    (root/'cpu_gpu_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    (root/'CPU_GPU.md').write_text('''# CPU/GPU comparison\n\nMatched complete bins: **'''+first.isoformat()+' to '+last.isoformat()+'''**.\n\n| Counter | Sandbox CPU | Trainer GPUs | Rollout GPUs |\n|---|---:|---:|---:|\n| Busy | %.1f%% | %.1f%% | %.1f%% |\n| SM active | — | %.1f%% | %.1f%% |\n| Tensor active | — | %.1f%% | %.1f%% |\n\nCPU pressure averaged **%.1f%%**. Guest execution averaged **%.1f%% of host CPU capacity**, already included in CPU busy.\n\nThe trainer GPUs are substantially more active than rollout GPUs in this window. GPU busy alone overstates how much of the device is active: rollout busy is much higher than rollout SM/tensor activity. CPU and GPU counters have different definitions, so their percentages are not a direct efficiency or FLOPS comparison.\n\nSandbox CPU contention and feeder ready-backlog pauses both occur in this run. This comparison cannot establish that CPU contention alone causes rollout underutilization. GPU SM active includes stalled active warps; tensor activity excludes non-tensor work.\n\n[Comparison figure](10_cpu_gpu_comparison.png) · [PDF](10_cpu_gpu_comparison.pdf) · [Detailed resource findings](FINDINGS.md)\n'''%tuple(means[k] for k in ['sandbox_cpu_busy_pct','trainer_gpu_busy_pct','rollout_gpu_busy_pct','trainer_sm_active_pct','rollout_sm_active_pct','trainer_tensor_active_pct','rollout_tensor_active_pct','sandbox_cpu_psi_pct','sandbox_cpu_guest_pct']))
    if auxiliary:
        with (root/'CPU_GPU.md').open('a') as stream:
            stream.write('\n## Feeder state in the same window\n\n')
            for key,value in auxiliary.items():
                stream.write('- '+key+': %.3f\n' % value)
            if auxiliary.get('refill_backlog_pause_fraction',0)>=0.9:
                stream.write('\nThe ready-data threshold paused admission for nearly the entire window. The trainer continues consuming existing results while rollout work drains. With low contemporaneous CPU pressure, this window is not evidence that additional sandbox CPU capacity is needed to feed generation. Sustained generation requires a different ready-backlog policy; no setting was changed by this comparison.\n')
    print(json.dumps(result))


if __name__=='__main__':main()
