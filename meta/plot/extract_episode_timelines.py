"""Freeze median-near complete episode timelines from the campaign's exported traces.

Run from the repository root: python meta/plot/extract_episode_timelines.py
Only the standard library is needed; plotting consumes the frozen JSON separately.
"""
import collections
import argparse
import gzip
import hashlib
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CAMPAIGN = REPO / 'outputs/sandbox-sweep-20261015_193213Z'
DESTINATION = REPO / 'meta/plot/data/episode_timelines_20261015.json'


def load_point(groups, campaign=CAMPAIGN):
    source = campaign / f'groups-{groups}/inputs/demand.jsonl.gz'
    result = json.loads((source.parent.parent / 'result.json').read_text())
    traces = collections.defaultdict(list)
    with gzip.open(source, 'rt') as stream:
        for line in stream:
            event = json.loads(line)
            kind = event.get('kind')
            if kind == 'native':
                trace = event['trace_file']
            elif kind == 'model' and event.get('sandbox_trace'):
                trace = Path(event['sandbox_trace']).name
            else:
                continue
            traces[trace].append(event)
    successful_starts = [e['duration'] for events in traces.values() for e in events
                         if e['kind'] == 'native' and e.get('operation') == 'start'
                         and e.get('phase') == 'end']
    candidates = []
    excluded = collections.Counter()
    for trace, events in sorted(traces.items()):
        beginnings = {}
        intervals = []
        unmatched = 0
        errors = 0
        for event in sorted(events, key=lambda e: e['timestamp']):
            phase = event.get('phase')
            if phase not in ('begin', 'end', 'error'):
                continue
            key = (event['kind'], event.get('operation_id'))
            if phase == 'begin':
                beginnings[key] = event
            else:
                begin = beginnings.pop(key, None)
                if begin is None:
                    unmatched += 1
                    continue
                errors += phase == 'error'
                intervals.append({
                    'kind': event['kind'], 'operation': event['operation'],
                    'start': begin['timestamp'], 'end': event['timestamp'],
                    'sandbox_id': event.get('sandbox_id'),
                })
        unmatched += len(beginnings)
        starts = [e for e in intervals if e['kind'] == 'native' and e['operation'] == 'start']
        stops = [e for e in intervals if e['kind'] == 'native' and e['operation'] == 'stop']
        closes = [e for e in intervals if e['kind'] == 'native' and e['operation'] == 'shell_close']
        models = [e for e in intervals if e['kind'] == 'model']
        if len(starts) != 1 or len(stops) != 1 or not closes or len(models) < 2:
            excluded['missing_lifecycle_or_fewer_than_two_model_calls'] += 1
            continue
        if unmatched or errors:
            excluded['unmatched_operations_or_recorded_call_errors'] += 1
            continue
        start, end = starts[0]['start'], stops[0]['start']
        close = max(e['end'] for e in closes)
        if not (result['model_ready_at'] <= start < end <= result['finished_at']):
            excluded['outside_rollout_window'] += 1
            continue
        if max(e['end'] for e in models) > close or close > stops[0]['start']:
            excluded['ambiguous_post_agent_boundary'] += 1
            continue
        candidates.append({
            'trace_file': trace, 'sandbox_id': starts[0]['sandbox_id'],
            'start_epoch': start, 'end_epoch': end, 'lifetime_s': end - start,
            'model_calls': len(models), 'native_calls': len(intervals) - len(models),
            'intervals': [e for e in intervals if e['operation'] != 'stop'],
            'first_model_start': min(e['start'] for e in models), 'post_agent_start': close,
            'post_agent_end': stops[0]['start'],
        })
    assert len(candidates) >= 4, (groups, excluded)
    median = statistics.median(e['lifetime_s'] for e in candidates)
    selected = sorted(candidates, key=lambda e: (abs(e['lifetime_s'] - median), e['trace_file']))[:4]
    selected.sort(key=lambda e: (e['lifetime_s'], e['trace_file']))
    for episode in selected:
        start, end = episode['start_epoch'], episode['end_epoch']
        intervals = episode.pop('intervals')
        # Partition by all observed boundaries, avoiding double-counted nested calls.
        boundaries = {start, end, episode['first_model_start'],
                      episode['post_agent_start'], episode['post_agent_end']}
        for interval in intervals:
            assert start <= interval['start'] <= interval['end'] <= end
            boundaries.update((interval['start'], interval['end']))
        segments = []
        ordered = sorted(boundaries)
        for left, right in zip(ordered, ordered[1:]):
            middle = (left + right) / 2
            active = [e for e in intervals if e['start'] <= middle < e['end']]
            if any(e['kind'] == 'native' and e['operation'] == 'start' for e in active):
                category = 'Cold Start'
            elif episode['post_agent_start'] <= middle < episode['post_agent_end']:
                category = 'Verification'
            elif middle < episode['first_model_start']:
                category = 'Setup'
            elif any(e['kind'] == 'native' and e['operation'] != 'stop' for e in active):
                category = 'Sandbox Interaction'
            else:
                category = 'LLM Wait'
            left, right = left - start, right - start
            if segments and segments[-1]['category'] == category:
                segments[-1]['end_s'] = right
            else:
                segments.append({'start_s': left, 'end_s': right, 'category': category})
        episode['segments'] = segments
        episode['phase_seconds'] = {
            category: sum(s['end_s'] - s['start_s'] for s in segments if s['category'] == category)
            for category in sorted({s['category'] for s in segments})
        }
        assert abs(sum(episode['phase_seconds'].values()) - episode['lifetime_s']) < 1e-6
    return {
        'episode_ceiling': groups * 8, 'eligible_episodes': len(candidates),
        'successful_start_calls': len(successful_starts),
        'successful_start_median_s': statistics.median(successful_starts),
        'successful_start_p90_s': statistics.quantiles(successful_starts, n=10, method='inclusive')[8],
        'excluded': dict(excluded), 'eligible_median_lifetime_s': median,
        'source': str(source.relative_to(REPO)),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'episodes': selected,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=CAMPAIGN)
    parser.add_argument('--output', type=Path, default=DESTINATION)
    parser.add_argument('--benchmark', default='SWE-bench Pro V2')
    parser.add_argument('--groups', type=int, nargs='+', default=[16, 32, 64])
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    data = {
        'campaign': campaign.name,
        'benchmark': args.benchmark,
        'selection': 'Four complete, error-free-call episodes nearest the eligible median lifetime per concurrency; ties resolved by trace filename. Each has at least two model calls and fully paired operations.',
        'scope': 'Sandbox start invocation through sandbox stop invocation; deletion/teardown duration is excluded, independently aligned to time zero. Episodes are not task-matched across concurrency levels. Error-free API calls do not establish completed reward verification or task resolution.',
        'classification': 'Original reference legend conventions: Cold Start is the native start interval; Setup is startup completion through first model call; Sandbox Interaction is observed native operations during the agent phase; Verification is inferred from final shell_close completion through stop invocation and includes patch handling/artifact overhead; LLM Wait is the residual, including measured model calls and untraced agent overhead. Verification is a phase proxy, not confirmed successful reward evaluation. Precedence: cold start > verification > setup > native interaction > residual. Stop duration is excluded.',
        'points': [load_point(groups, campaign) for groups in args.groups],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + '\n')
    print('Saved', args.output)
    for point in data['points']:
        print(point['episode_ceiling'], 'eligible', point['eligible_episodes'],
              'median', round(point['eligible_median_lifetime_s'], 2),
              'selected', [round(e['lifetime_s'], 2) for e in point['episodes']])
