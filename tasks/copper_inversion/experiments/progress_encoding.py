"""Snapshot observed losses and completion time at identical formal round counts."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


def snapshot(base):
    rows = []
    for run in sorted((base / 'full').iterdir()):
        if not run.is_dir():
            continue
        read = lambda p: json.loads(p.read_text())
        status = read(base / 'status' / f'full_{run.name}.json')
        state = read(run / 'checkpoint.json')['state']
        events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()]
        observations = state['observations']
        counts = Counter(o['round_idx'] for o in observations)
        start = status['started_at']
        best = None
        curve = []
        for round_idx, count in sorted(counts.items()):
            losses = [o['evaluation']['metrics']['loss_total'] for o in observations
                      if o['round_idx'] == round_idx and o['evaluation']['status'] == 'succeeded']
            if losses:
                best = min(losses + ([best] if best is not None else []))
            if count != (1 if round_idx == 0 else 4):
                continue
            evaluated = [e for e in events if e['event_type'] == 'candidate_evaluated'
                         and e['iteration'] == round_idx]
            curve.append(dict(round=round_idx, evaluations=1+4*round_idx, best_loss=best,
                              elapsed_seconds=max(e['timestamp_unix'] for e in evaluated)-start))
        failures = []
        for p in (run/'harness/branches').glob('*/turns/*/turn_committed.json'):
            record = read(p)
            if record['submissionStatus'] == 'rejected':
                failures.append(dict(turn=record['turnId'], errors=record['validationErrors']))
        try:
            os.kill(status['pid'], 0)
            alive = True
        except ProcessLookupError:
            alive = False
        encodings = [read(p) for p in (run/'encodings').glob('*.json')]
        rows.append(dict(run=run.name, pid_alive=alive, process_status=status['status'],
            campaign_status=read(run/'status.json')['status'], observations=len(observations),
            successful=sum(o['evaluation']['status']=='succeeded' for o in observations),
            round_counts=dict(counts), curve=curve, rejected_turns=failures,
            encoding_seconds=sum(e['elapsed_seconds'] for e in encodings),
            counters=read(run/'budget.json')['counters']))
    return dict(captured_at=datetime.now(timezone.utc).isoformat(), root=str(base), runs=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.dumps(snapshot(args.root), indent=2)
    if args.output:
        args.output.write_text(result+'\n')
    else:
        print(result)
