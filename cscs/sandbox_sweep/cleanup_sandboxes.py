#!/usr/bin/env python3
"""Delete only sandbox IDs recorded by this sweep point's native client."""
import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run_dir', type=Path)
parser.add_argument('--trace-dir', type=Path)
args = parser.parse_args()
ids = set()
for path in (args.trace_dir or args.run_dir / 'native-traces').glob('*.jsonl'):
    with path.open() as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get('operation') == 'start' and record.get('phase') == 'end' and record.get('sandbox_id'):
                ids.add(record['sandbox_id'])
endpoint = os.environ['AGENTENV_ENDPOINT'].rstrip('/')
key = os.environ['AGENTENV_API_KEY']
counts = {'recorded_ids': len(ids), 'deleted': 0, 'already_gone': 0, 'errors': 0}
for sid in sorted(ids):
    request = urllib.request.Request(endpoint + '/sandboxes/' + quote(sid, safe=''),
        method='DELETE', headers={'X-API-Key': key, 'Authorization': 'Bearer ' + key})
    try:
        with urllib.request.urlopen(request, timeout=15):
            counts['deleted'] += 1
    except urllib.error.HTTPError as exc:
        counts['already_gone' if exc.code == 404 else 'errors'] += 1
    except (OSError, TimeoutError):
        counts['errors'] += 1
(args.run_dir / 'sandbox-cleanup.json').write_text(json.dumps(counts, indent=2) + '\n')
print(json.dumps(counts), flush=True)
if counts['errors']:
    raise SystemExit(1)
