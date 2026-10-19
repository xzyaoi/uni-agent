# Experiment figures

## One versus two CPU hosts at C=128

[SVG](figure/cpu_host_scaling_c128_pro.svg) · [PNG](figure/cpu_host_scaling_c128_pro.png) · [PDF](figure/cpu_host_scaling_c128_pro.pdf) · [Metrics CSV](figure/cpu_host_scaling_c128_pro.csv)

The SWE-bench Pro V2 comparison uses the throughput/resource layout, colors,
fonts, 9 × 3.75-inch canvas, and axis ranges from `campaign_throughput.ipynb`.
The x-axis is active CPU host count; concurrency stays at 128 trajectories
(16 groups × eight responses). Each m5d.metal host has 96 vCPUs; inference
uses the same eight GH200 GPUs. Both runs use the same pinned 128-task cohort,
measurement manifest, and model settings.

| Metric | 1 host | 2 hosts |
| --- | ---: | ---: |
| Rollout duration (min) | 31.87 | 24.10 |
| Completed reward verifications/min | 32.00 | 42.48 |
| Mean CPU busy (%) | 68.35 | 31.04 |
| Mean GPU SM activity (%) | 7.81 | 8.68 |
| Peak fleet anonymous memory (GiB) | 161.16 | 173.96 |
| Incomplete verifications | 4 | 0 |
| Stored usable trajectories | 1,023 | 1,024 |

Both runs finished 1,024 attempts without failed groups. Two hosts show 32.7%
higher completed-verification throughput and 24.4% shorter rollout duration.
Completed verification includes reward-zero outcomes; it does not mean a solved task.
These are single observations without error bars. The one-host result is job
**3695409**, from `pro_overnight_20261019_recovery2`; the two-host result is job
**3694759**, from the preceding `pro_multinode_2x_20261018` sweep. The planned
overnight two-host repetition has no completed result in the local exports.

CPU percentages use each fleet's total capacity (96/192 vCPUs). CPU and GPU
averages use complete one-minute bins within the rollout window. GPU SM
activity is rebuilt from raw DCGM exports for both observations; every included
minute contains all eight GPU identities, although sampling is not synchronized.
Peak memory is the maximum fleet sum of per-host `aenv.service memory.stat anon`
means in five-second bins with every active host present, excluding file cache.
Model initialization is excluded from throughput; native sandbox setup is included.

```bash
python meta/plot/plot_cpu_host_scaling.py
# Optional: refresh frozen inputs from existing local outputs/ exports.
python meta/plot/plot_cpu_host_scaling.py --refresh-data
```

Rendering requires Matplotlib, NumPy, and Seaborn and uses only
`data/cpu_host_scaling_c128_20261019.json`. The adjacent provenance JSON records
source hashes, job IDs, windows, matching checks, and metric definitions.

## SWE-bench Pro and Verified episode timelines

Separate timeline figures:

- **SWE-bench Pro:** [SVG](figure/episode_timelines_pro.svg) · [PNG](figure/episode_timelines_pro.png) · [PDF](figure/episode_timelines_pro.pdf)
- **SWE-bench Verified:** [SVG](figure/episode_timelines_verified.svg) · [PNG](figure/episode_timelines_verified.png) · [PDF](figure/episode_timelines_verified.pdf)

Each benchmark has its own 9 × 5.5-inch figure, with concurrency panels at episode
ceilings 128/256/512 (16/32/64 active groups, eight episodes each).

Reproduce from the small frozen JSON files, without `outputs/` or remote access:

```bash
python meta/plot/plot_episode_timelines.py
python meta/plot/plot_episode_timelines.py \
  --data meta/plot/data/episode_timelines_verified_20261016.json \
  --output meta/plot/figure/episode_timelines_verified
```

Plotting requires Matplotlib. Frozen JSON records source hashes, selected sandbox
IDs, measured intervals, eligibility counts, and exclusion reasons. Selection CSVs
and startup summaries are exported alongside the plots. Every panel selects four
complete, error-free-call episodes nearest the eligible median lifetime; it is a
conditional cohort, not all attempts, and episodes are not task-matched. The Pro
512-episode point had an engine failure. Error-free calls do not prove reward
verification succeeded or the task was solved.

Cold Start measures native start including readiness; Setup ends at the first
model call. Sandbox Interaction measures native calls during agent execution.
Verification is inferred from final agent shell-close completion through stop
invocation and includes patch/artifact work. LLM Wait is residual time, including
model calls and untraced orchestration. Deletion time is excluded.

Re-extract Verified timelines when the original local trace exports are available:

```bash
python meta/plot/extract_episode_timelines.py \
  --campaign outputs/verified_sweep_20261015_235315Z \
  --output meta/plot/data/episode_timelines_verified_20261016.json \
  --benchmark 'SWE-bench Verified'
```

`episode_timelines.ipynb` includes both campaigns and their selection/startup tables,
and displays each benchmark as a separate figure using the shared Python renderer.
The script also generates separate Pro and Verified figures by default.

`campaign_throughput.ipynb` displays a separate **9 × 3.75-inch** bar figure for
each benchmark: throughput, GPU SM activity, CPU utilization, and peak anonymous
memory. Exports are `figure/concurrency_throughput_pro.*` and
`figure/concurrency_throughput_verified.*`. The memory axis uses 0–250 GiB in both.
The Verified memory CSV and provenance JSON record raw-sample hashes and rollout
windows. Both notebooks include rendered outputs and regenerate separate SVG,
PDF, and PNG files from frozen data without cluster access.

## Updated Verified 256-trajectory observation

Both notebooks and the overview use job **3674787** for Verified's 256 point:
**81.89 verified episodes/min**, **19.98% mean GPU SM activity**, **90.26% mean CPU busy**,
and **138.07 GiB peak anonymous memory**. All 1,024 trajectories were stored, with
1,017 completed and 7 incomplete reward verifications. No engine failure was observed.
Verified's other points and all Pro points retain their original measurements.
This is a later repetition, not one uninterrupted sweep or an average of runs.

The same rerun supplies all 256-point metrics and the four median-near timeline episodes.
The original failed 256 observation and timeline selection remain in provenance/JSON metadata.
Refresh frozen data from local exports with `python meta/plot/update_verified_rerun.py`,
then execute the notebooks and overview renderer. Rendering needs no remote access.

## Verified sweep overview

[SVG](figure/verified_sweep_overview.svg) · [PNG](figure/verified_sweep_overview.png) · [PDF](figure/verified_sweep_overview.pdf)

```bash
python meta/plot/plot_verified_sweep.py
```

Frozen data: `data/verified_sweep_20261016.json`; provenance and source SHA256:
`data/verified_sweep_20261016.provenance.json`. The figure compares throughput,
batch time, CPU/GPU activity, and incomplete/missing results. Each point attempted
128 groups × 8 episodes on two GH200 nodes (eight rollout GPUs). Warmup and model
initialization are excluded; native sandbox setup is included. CPU/GPU means
use complete one-minute bins after model readiness and have different counter
definitions. There is one run per point, with no statistical error bars. The
128-active-group point was skipped because 64 did not meet the clean-completion
gate. Reward verification completion includes reward-zero outcomes.
