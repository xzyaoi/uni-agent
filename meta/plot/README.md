# Experiment figures

## One versus two CPU hosts at C=128

### Exploratory CPU / memory-time roofline

Anonymous-memory version (default):
[PNG](figure/cpu_memory_roofline_anon_c128_pro.png) · [SVG](figure/cpu_memory_roofline_anon_c128_pro.svg) · [PDF](figure/cpu_memory_roofline_anon_c128_pro.pdf).
Total charged-memory comparison:
[PNG](figure/cpu_memory_roofline_c128_pro.png) · [SVG](figure/cpu_memory_roofline_c128_pro.svg) · [PDF](figure/cpu_memory_roofline_c128_pro.pdf).

This uses the accounting model studied in the neighboring `infersim` project
(`scripts/analyzer/plot_aws_replay_roofline.py`) and `flash-sandbox`'s
`meta/tools/roofline_profiler/README.md`: `I=C/A`, `Y=C/T`, and the nominal
reference envelope `Y <= min(P, M*I)`. `C` is cumulative cgroup CPU seconds;
`A` is integrated memory-time (GiB-seconds), using `memory.stat anon` by default
or total charged `memory.current` with `--memory-basis charged`.
`P` is nominal logical-CPU count and `M` is host `MemTotal`. There is one panel
per host configuration, using the same Pro C=128 task mixture in both.
Each point represents a whole fleet observation, not an individual task.

The default Y-axis is **useful trajectories per minute**, with `N` stored
usable trajectories credited to group-terminal events within the resource
window. Reward-zero trajectories count. With observed window costs `c=C/N`
(CPU-seconds per useful trajectory) and `d=A/N` (GiB-seconds per useful
trajectory), the plotted conditional CPU/RAM roof is
`R_roof(I) = 60 * min(P, M*I) / c`. At the measured intensity this equals
`60 * min(P/c, M/d)`; the observed point is `R=60*N/T`. Each host configuration
uses its own measured CPU cost per useful trajectory. The curve holds that
CPU cost fixed as intensity varies; it is not a common calibrated operational
roof for both runs. No independent GPU-serving ceiling was measured.

| Sampled-window measurement | 1 host | 2 hosts total |
| --- | ---: | ---: |
| Mean CPU service (CPU-s/s) | 63.34 | 57.04 |
| Mean charged memory (GiB) | 246.47 | 645.41 |
| Mean anonymous memory (GiB) | 86.76 | 74.06 |
| CPU / anon-memory-time intensity (CPU-eq/GiB) | 0.730 | 0.770 |
| CPU / charged-memory-time intensity (CPU-eq/GiB) | 0.257 | 0.088 |
| Nominal CPU / RAM reference knee | 0.254 | 0.254 |
| Useful trajectories credited within sampled window | 1,015 | 1,016 |
| Observed useful trajectories/min | 31.88 | 42.35 |
| Conditional CPU ceiling (trajectories/min) | 48.32 | 142.57 |
| Conditional CPU/anon roof at observed intensity (trajectories/min) | 48.32 | 142.57 |
| Conditional CPU/charged-memory roof at observed intensity (trajectories/min) | 48.32 | 49.55 |

With anonymous memory, both points lie to the right of the nominal knee, on the
horizontal CPU-reference branch. Their measured rates remain below that roof;
this does not demonstrate CPU saturation. This view excludes kernel memory and
file-cache costs, and uses full nominal RAM as an optimistic anonymous-memory
budget. It is a conditional sensitivity view rather than a complete working-set
capacity model.

With total charged memory, the one-host point lies near the nominal reference
knee and the two-host point lies on its memory-side branch.
**This does not demonstrate a memory bottleneck.**
Charged memory includes retained file cache already present at the start:
approximately 131.5 GiB for the one-host run and 316.9 GiB on each host of the
two-host run. There is no baseline subtraction in the charged-memory comparison.
Cache history substantially affects that placement. The anonymous-memory view
instead integrates the captured `memory.stat anon` samples on the same windows.

“Charged” means accounted to the cgroup, not unreclaimable. Clean file cache
can be reclaimed by Linux; dirty pages may require writeback first. In these
windows, mean file memory is 157.17 GiB for one host and 567.94 GiB across two
hosts, while mean anonymous memory is 86.76/74.06 GiB. The two-host fleet has
455.59 GiB of mean inactive file pages. Treating all charged cache as a fixed
per-trajectory memory requirement therefore gives a misleading RAM-capacity
projection when cache can adapt under pressure. The plotted memory branch is
an accounting reference under its fixed-cost assumption; a practical capacity
roof requires working-set/reclaim and refault/latency measurements. Anonymous
memory alone also omits kernel costs and performance-relevant file cache.
See the kernel's [reclaim overview](https://docs.kernel.org/admin-guide/mm/concepts.html#reclaim)
and [cgroup memory definitions](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory).

Hollow markers indicate exploratory placement: the exports retained wall
timestamps and cgroup paths, but not host monotonic timestamps, boot IDs, or
cgroup inodes. Thus they cannot satisfy the strict profiler's identity/clock
checks. Sampling is about five seconds, exceeding the reference renderer's
three-second strict gap criterion; this exploratory integration rejects
nonpositive gaps, gaps over six seconds, path changes, and CPU counter resets.
Nominal capacities are neither calibrated operational capacity nor enforced
safe budgets. Scope exclusivity and effective ancestor quotas are unproven.

For each fleet, CPU and memory-time use exactly the same intersection of its
host sampling windows, with linear interpolation at common boundaries and
trapezoidal memory integration. Unsampled rollout edges are excluded. The
final group of each run terminates just after the last common resource sample,
so its eight trajectories are excluded from the windowed rate. Full-run usable
counts remain 1,023/1,024. Group-terminal timestamps are joined by UID to final
stored-session score counts, whose totals reconcile to the independent usable
counts. Individual trajectory persistence timestamps were not captured, so
availability is attributed at group termination. Resource costs retain all
in-flight and unsuccessful work in that same window. These conditional
per-output costs include boundary effects, shared cache, and idle time.
Both axes are linear, with shared ranges of 0–1.2 CPU-eq/GiB for anonymous
memory (0–0.6 for total charged memory) and
0–160 useful trajectories/min.
It uses a 9 × 3.75-inch canvas with minimum 12 pt visible text.

```bash
python meta/plot/plot_cpu_memory_roofline.py
# Optional: refresh from the existing local raw host exports.
python meta/plot/plot_cpu_memory_roofline.py --refresh-data
# Reproduce the separate charged-memory comparison:
python meta/plot/plot_cpu_memory_roofline.py --memory-basis charged
# Optional original CPU-service view (saved with a separate cpu_service stem):
python meta/plot/plot_cpu_memory_roofline.py --memory-basis charged --y-axis cpu-service
```

Frozen measurements, window coverage, initial cache, limitations, and raw-source
hashes are in `data/cpu_memory_roofline_c128_20261019.json`.

### Modeled pricing at supplied rates

[PNG](figure/cpu_host_pricing_c128_pro.png) · [SVG](figure/cpu_host_pricing_c128_pro.svg) · [PDF](figure/cpu_host_pricing_c128_pro.pdf) · [CSV](figure/cpu_host_pricing_c128_pro.csv)

The pricing figure uses the user's supplied **$5.424/hour per m5d.metal host**
and **$5.970/hour for the entire eight-GPU allocation**, charged concurrently.
Stacked bars compare hourly allocation cost and cost per 1,000 usable
trajectories, with CPU and GPU contributions shown separately. It uses the
same 9 × 3.75-inch canvas, `font_scale=2`, and minimum 12 pt visible text.

| Modeled metric | 1 CPU host | 2 CPU hosts |
| --- | ---: | ---: |
| CPU + GPU allocation (USD/hour) | 11.394 | 16.818 |
| Measured rollout cost (USD) | 6.05 | 6.76 |
| Cost / 1,000 usable trajectories (USD) | 5.92 | 6.60 |

Two hosts cost 47.6% more per hour and 11.5% more per usable trajectory while
delivering 32.4% higher usable-trajectory throughput in the measured runs.
Cost is `(hosts × CPU hourly price + GPU hourly price) × rollout_seconds / 3600`;
divide by stored usable trajectories for unit cost.

This is an active-configuration cost scenario, not an experiment billing
statement: one-host execution excludes the inactive allocated host. Model
initialization, image warmup, provisioning, storage, transfer, and taxes are
excluded. The supplied GPU price is for **8 × H100**, whereas the measured
rollout timings used **8 × GH200**. Applying that price assumes the measured
runtime for this scenario; no H100 throughput equivalence was measured.

```bash
python meta/plot/plot_cpu_host_pricing.py
# Optional alternative rates (GPU price is for the whole eight-GPU allocation):
python meta/plot/plot_cpu_host_pricing.py --cpu-hourly 5.424 --gpu-hourly 5.970
```

Assumptions, inputs, source hash, and computed values are saved in
`data/cpu_host_pricing_c128_20261019.json`.

### Throughput and resource comparison

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

### GPU activity and model demand over time

Compact throughput/request comparison:
[PNG](figure/cpu_host_throughput_requests_c128_pro.png) · [SVG](figure/cpu_host_throughput_requests_c128_pro.svg) · [PDF](figure/cpu_host_throughput_requests_c128_pro.pdf).
The left panel plots usable-trajectory throughput against CPU host count,
matching the original campaign bar style: 1,023/1,024 stored usable trajectories
divided by rollout duration gives 32.10/42.48 trajectories/min for one/two hosts.
The right panel overlays
active client requests for one host (blue) and two hosts (violet), aligned to
model readiness. Lines show exact interval-overlap means in 10-second bins,
including zero-request time through each run's completion. Neither line extends
beyond its observation window. These client counts include gateway/network
waits and are not vLLM running-queue measurements.

```bash
python meta/plot/plot_cpu_host_throughput_requests.py
```

Output-token grouped-bar variant:
[PNG](figure/cpu_host_throughput_output_tokens_c128_pro.png) · [SVG](figure/cpu_host_throughput_output_tokens_c128_pro.svg) · [PDF](figure/cpu_host_throughput_output_tokens_c128_pro.pdf).
The left panel groups mean-rate bars by metric — usable-trajectory
throughput, mean active client requests, mean delivered output tokens/s —
and dodges the one-host (blue) and two-host (violet) bars inside each
group: **32.10 / 42.48** usable trajectories per min, **4.16 / 6.55** mean
active requests, and **350.60 / 453.34** mean output tokens/s. Because the
metrics span two orders of magnitude, the rate axis is log-scaled (ticks
10 / 100 / 1000) and every bar carries its exact value. Violet arrows over
each group mark the one-to-two-host gain for that metric
(**+32.4% / +57.6% / +29.3%**,
computed from the unrounded values). Ten-second-bin peaks (124 / 127 active
requests; 3,557 / 3,685 tokens/s) are validated against the frozen data
but not plotted.
The right panel overlays the aggregate delivered output
tokens/s over time with the host colors. Counts sum
`usage.completion_tokens` across client model-call completions in each
10-second bin, divided by its duration. Each call is counted once across all
rollout replicas; these are completion-time delivery rates rather than
instantaneous GPU token-generation rates, so token peaks are completion bursts,
not generation spikes. Run totals are 670,765 output tokens
for one host and 655,987 for two hosts.
Matching-color dashed lines show duration-weighted mean output rates over each
full telemetry rollout window, including idle time after the last model call.
Each mean line ends at its own run's completion and is labeled with its numeric value.
Arrows annotate the observed gain from one to two CPU hosts: **+32.4%
usable trajectories/min**, **+57.6% mean active requests**, and **+29.3%
mean output tokens/s**, calculated from the
unrounded values. These are descriptive differences between the two runs.

```bash
python meta/plot/plot_cpu_host_throughput_requests.py --metric output-tokens
```

The renderer uses the existing frozen scaling and timeline JSON files;
`characterization.ipynb`'s final cell is an equivalent notebook port.
Both variants use the original fonts, palette families, and legend style on a
9 × 3.75-inch canvas (the log-scaled metric-grouped panel, with a 1.35:1
left-right width split, replaces the original linear throughput panel in the
output-token variant).
The output-token variant omits the bottom caption; metric definitions
and run provenance remain documented here. It uses Seaborn `font_scale=2`,
at least 11 pt for all visible text, and an exact 9 × 3.75-inch export canvas.

Mean resident sandbox VM counts are **84.66** with one host and **74.52 total**
across two hosts, averaging full-node-coverage complete rollout minutes (31/23
bins respectively). The two-host figure is a fleet total, not a per-host value.
These host process counts include resident idle VMs and do not establish the
number of sandboxes actively executing commands. The average includes the
observed post-generation portion of each rollout window.

[PNG](figure/cpu_host_timeline_c128_pro.png) · [SVG](figure/cpu_host_timeline_c128_pro.svg) · [PDF](figure/cpu_host_timeline_c128_pro.pdf) · [10-second metrics CSV](figure/cpu_host_timeline_c128_pro.csv)

The two-column timeline aligns each run to model readiness and shows GPU busy,
SM/tensor activity, client model requests in flight, delivered output tokens/s,
CPU utilization, and resident sandbox VM counts. GPU/request/token series use
10-second bins; CPU/VM series retain complete one-minute bins. Full-run mean
client requests in flight are 4.16/6.55 for one/two hosts, with instantaneous
peaks 124/127. Every client begin event has a matching terminal event
(6,301/6,385 calls). Request means integrate interval overlap, and peaks use
event boundaries rather than periodic snapshots. Last model completion occurs
at 21.93/14.44 minutes; batch completion occurs at 31.89/24.12 minutes on the
wall-clock telemetry window. The throughput denominator uses the independently
recorded monotonic rollout duration (31.87/24.10 minutes).

These are **client requests**, including gateway/network waits. Neither run
captured vLLM running/waiting queue scrapes, so engine batch size, engine queue
length, and KV-cache occupancy are unavailable. Output tokens are assigned to
request completion time. Post-generation elapsed time does not by itself
identify verification or cleanup duration. These rollout-only observations
contain no trainer updates. The separate 27B training job 3692513 has no
retained historical GPU utilization percentages in its local usage export.

```bash
python meta/plot/plot_cpu_host_timeline.py
# Refresh the frozen timeline from local raw exports:
python meta/plot/plot_cpu_host_timeline.py --refresh-data
```

Frozen inputs, source hashes, metric definitions, and summary statistics are in
`data/cpu_host_timeline_c128_20261019.json`. Rendering needs no raw exports or
remote access. CPU/VM minute series are retained in that JSON; the CSV contains
the 10-second GPU/request/token series.

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


## One-host Verified and Pro characterization

`characterization.ipynb` renders a 2×2 figure: one sandbox host running SWE-bench
Verified on the first row, and one host running SWE-bench Pro on the second.
The left column shows stored usable samples/min at C=64/128/256/512; the right
shows mean client model calls in flight, sandbox CPU, and sandbox host RAM.
CPU/RAM bars have solid bases ending at the mean and equal-width hatched
extensions ending at the sampled peak. Client calls use the right count axis,
integrating matched call begin/end/error intervals across both TP4 replicas.
This includes waiting calls; historical Pro lacks actual vLLM running-request
telemetry, so both rows use the same client-side definition.

Verified uses the completed October 19 one-host sweep, with usable counts
1,024 / 1,024 / 1,023 / 1,024. Pro uses the only complete historical one-host
sweep (October 15), with counts 1,024 / 1,020 / 1,020 / 319. **All four historical
Pro points used an unintended 180-second verifier command cap; C=512 also had
an engine failure.** The plot marks these limitations. This is not a clean
matched comparison between benchmarks; the newer clean Pro C=128 run is not
spliced into the old series.

Each sandbox host has 96 vCPUs. Host RAM is `(MemTotal - MemAvailable) / MemTotal`,
including the OS/runtime/VMs. CPU/RAM means are duration-weighted over available
telemetry; peaks use complete aligned five-second bins. Model startup and
cleanup are excluded, while tool and verification waits remain included.

Run all notebook cells from the repository root or `meta/plot`. Inputs are
`data/verified_characterization_1host_20261019.csv`,
`data/pro_characterization_1host_20261015.csv`, and their `.provenance.json`
files (source hashes, windows, aggregation rules, coverage, and limitations).
The final cell ports `plot_cpu_host_throughput_requests.py --metric
output-tokens`, reading `data/cpu_host_scaling_c128_20261019.json` and
`data/cpu_host_timeline_c128_20261019.json` to compare one versus two CPU
hosts at C=128 across usable throughput, mean active client requests,
and mean delivered output tokens/s (mean bars grouped by metric with the
one- and two-host bars dodged inside each group, on a log-scaled rate
axis), beside the output-token timeline.
Rendering needs no cluster
access. Exports: `figure/concurrency_characterization_1host.{png,pdf,svg}`
and `figure/cpu_host_throughput_output_tokens_c128_pro.{png,pdf,svg}`.

The earlier two-host Verified inputs and
`figure/concurrency_characterization_verified.{png,pdf,svg}` remain available
as separate historical artifacts.
