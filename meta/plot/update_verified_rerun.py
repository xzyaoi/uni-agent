"""Refresh the Verified 256-point plots from the completed repetition's local exports.

Run before executing the notebooks; rendering itself uses only frozen plot data.
The failed original observation is retained in provenance and timeline metadata.
"""
import csv
import gzip
import hashlib
import json
from pathlib import Path

from extract_episode_timelines import load_point

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / 'meta/plot/data'
RERUN = REPO / 'outputs/verified_rerun_c256_retry_20261016_100619Z'
POINT = RERUN / 'groups-32'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    result = json.loads((POINT / 'result.json').read_text())
    comparison = json.loads((RERUN / 'comparison.json').read_text())
    assert result['state'] == 'completed'
    assert result['progress']['groups_terminal'] == 128
    assert result['progress']['groups_failed'] == 0
    assert result['progress']['episodes_with_usable_trajectory'] == 1024
    assert comparison['same_selected_tasks'] and not comparison['engine_dead_error']
    rows = json.loads((DATA / 'verified_sweep_20261016.json').read_text())
    updated = json.loads((RERUN / 'sweep_results.json').read_text())
    assert len(updated) == 1 and updated[0]['episode_ceiling'] == 256
    replacement = updated[0]
    for count, rate in [('completed_episodes', 'completed_episodes_per_minute'),
                        ('verified_episodes', 'verified_episodes_per_minute')]:
        assert abs(replacement[rate] - replacement[count] * 60 / replacement['wall_seconds']) < 1e-8
    position = next(i for i, row in enumerate(rows) if row['episode_ceiling'] == 256)
    path = DATA / 'verified_sweep_20261016.json'
    provenance_path = path.with_suffix('.provenance.json')
    provenance = json.loads(provenance_path.read_text())
    provenance.setdefault('superseded_256_observation', {
        'source': provenance['source'], 'source_data_sha256': provenance['sha256'],
        'observation': rows[position],
        'reason': 'Original Verified 256 run had a sample_tokens RPC timeout and EngineDeadError.',
    })
    rows[position] = replacement
    write_json(path, rows)
    provenance.update(sha256=digest(path),
                      scope='One plotted observation per concurrency; 256 replaced by successful repetition; 128 groups skipped.')
    provenance['replacement_256'] = {
        'campaign': RERUN.name, 'job_id': '3674787',
        'source': str((RERUN / 'sweep_results.json').relative_to(REPO)),
        'source_sha256': digest(RERUN / 'sweep_results.json'),
        'same_selected_tasks': True, 'engine_dead_error': False,
    }
    provenance['sources_by_episode_ceiling'] = {
        str(row['episode_ceiling']): (
            str((RERUN / 'sweep_results.json').relative_to(REPO)) if row['episode_ceiling'] == 256
            else provenance['source']) for row in rows
    }
    write_json(provenance_path, provenance)

    # Recompute the independent anonymous-memory maximum from the same rerun.
    source = POINT / 'inputs/cloudlab.jsonl.gz'
    samples = []
    with gzip.open(source, 'rt') as stream:
        for line in stream:
            event = json.loads(line)
            if (event.get('kind') == 'host'
                    and result['model_ready_at'] <= event['timestamp'] <= result['finished_at']):
                assert event['cgroup']['path'] == '/system.slice/aenv.service'
                stat = dict(line.split() for line in event['cgroup']['files']['memory.stat'].splitlines())
                samples.append(int(stat['anon']))
    assert len(samples) >= 140
    memory_path = DATA / 'verified_sweep_20261016_memory.csv'
    with memory_path.open() as stream:
        reader = csv.DictReader(stream)
        fields, memory = reader.fieldnames, list(reader)
    memory_index = next(i for i, row in enumerate(memory) if int(row['episode_ceiling']) == 256)
    memory_provenance_path = memory_path.with_suffix('.provenance.json')
    memory_provenance = json.loads(memory_provenance_path.read_text())
    memory_provenance.setdefault('superseded_256_observation', memory[memory_index].copy())
    memory[memory_index] = {'episode_ceiling': 256,
                            'peak_anonymous_memory_gib': max(samples) / 2**30,
                            'host_samples': len(samples)}
    with memory_path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(memory)
    source_index = next(i for i, row in enumerate(memory_provenance['sources'])
                        if row['episode_ceiling'] == 256)
    memory_provenance.setdefault('superseded_256_source', memory_provenance['sources'][source_index])
    memory_provenance['sources'][source_index] = {
        'episode_ceiling': 256, 'campaign': RERUN.name, 'job_id': '3674787',
        'source': str(source.relative_to(REPO)), 'sha256': digest(source),
        'model_ready_at': result['model_ready_at'], 'finished_at': result['finished_at'],
    }
    memory_provenance['sha256'] = digest(memory_path)
    write_json(memory_provenance_path, memory_provenance)

    timeline_path = DATA / 'episode_timelines_verified_20261016.json'
    timeline = json.loads(timeline_path.read_text())
    timeline_index = next(i for i, p in enumerate(timeline['points']) if p['episode_ceiling'] == 256)
    timeline.setdefault('superseded_256_point', timeline['points'][timeline_index])
    point = load_point(32, RERUN)
    point['campaign'] = RERUN.name
    point['job_id'] = '3674787'
    timeline['points'][timeline_index] = point
    timeline['replacement_256'] = provenance['replacement_256']
    write_json(timeline_path, timeline)
    print(f"Updated 256: {replacement['verified_episodes_per_minute']:.2f} verified/min, "
          f"{max(samples) / 2**30:.2f} GiB peak anon, {point['eligible_episodes']} eligible timelines.")


if __name__ == '__main__':
    main()
