"""Run with the Clariden Harbor Python; print aggregates, never credentials."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path

run = Path('/iopsstor/scratch/cscs/xyao/uni-agent/runs/'
           'qwen35_27b_verified_flash_2n45m_c512_20261018')
totals = Counter()
files = list((run / 'agent-logs').rglob('trajectory.json'))
for path in files:
    for trajectory in json.loads(path.read_text())['trajectories']:
        for field in ('model_token_count', 'response_len', 'prompt_len'):
            totals[field] += trajectory.get(field, 0) or 0

events = {'trial': [], 'agent': []}
for path in (run / 'harbor').glob('*/result.json'):
    result = json.loads(path.read_text())
    for kind, interval in (('trial', result), ('agent', result.get('agent_execution') or {})):
        if interval.get('started_at') and interval.get('finished_at'):
            start = datetime.fromisoformat(interval['started_at'].replace('Z', '+00:00')).timestamp()
            end = datetime.fromisoformat(interval['finished_at'].replace('Z', '+00:00')).timestamp()
            assert end >= start
            events[kind].extend([(start, 1), (end, -1)])
peaks = {}
for kind, intervals in events.items():
    active = peak = 0
    for _, delta in sorted(intervals):
        active += delta
        assert active >= 0
        peak = max(peak, active)
    assert active == 0
    peaks[kind] = peak
print(json.dumps(dict(trajectory_files=len(files), retained_tokens=dict(totals),
                     peak_overlapping_intervals=peaks,
                     total_inference_tokens=None, historical_gpu_utilization_percent=None,
                     notes=['model_token_count is generated tokens retained by response masks',
                            'response_len includes tool and observation tokens',
                            'prompt_len counts retained initial prompts, not every model call',
                            'Harbor agent total_tokens retains only the last call; not cumulative']), indent=2))
