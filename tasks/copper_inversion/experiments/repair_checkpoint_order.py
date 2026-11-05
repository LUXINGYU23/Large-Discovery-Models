"""Restore mapping order from committed evidence, without changing checkpoint values.

Legacy copper prompts embed unsorted JSON as a string. Sorted checkpoint writes
change that string on replay. This operational repair preserves canonical state,
all budgets and runtime identity; it never edits a committed turn or its digest.
"""
import argparse
import hashlib
import json
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def repair(root):
    path = root / 'checkpoint.json'
    before = path.read_bytes()
    checkpoint = json.loads(before)
    state = checkpoint['state']
    branch = root / 'harness/branches' / f"round_{state['next_round']:04d}"
    observations = {o['candidate']['candidate_id']: o for o in state['observations']}
    restored = set()
    for source in sorted(branch.glob('sessions/*/turns/*/input.json')):
        turn = json.loads(source.read_text())
        committed = branch / 'turns' / turn['turnId'] / 'turn_committed.json'
        if not committed.exists():
            raise ValueError('Expected a committed turn')
        if json.loads(committed.read_text())['inputDigest'] != turn['inputDigest']:
            raise ValueError('Original input and committed receipt differ')
        for row in json.loads(turn['message'])['new_measurements']:
            evaluation = observations[row['candidate_id']]['evaluation']
            for field in ('metrics', 'resource_usage'):
                if canonical(evaluation[field]) != canonical(row[field]):
                    raise ValueError(f'Cannot repair changed values: {field}')
                evaluation[field] = row[field]
            restored.add(row['candidate_id'])
    if not restored:
        raise ValueError('No committed evidence to restore')
    after = json.dumps(checkpoint, indent=2).encode() + b'\n'
    assert canonical(json.loads(before)) == canonical(json.loads(after))
    backup = root / 'checkpoint.before-order-repair.json'
    with backup.open('xb') as stream:
        stream.write(before)
    receipt = dict(round=state['next_round'], observations=len(restored),
                   before_sha256=hashlib.sha256(before).hexdigest(),
                   after_sha256=hashlib.sha256(after).hexdigest(),
                   canonical_state_sha256=hashlib.sha256(canonical(checkpoint).encode()).hexdigest(),
                   values_unchanged=True, source=str(branch))
    temporary = path.with_suffix('.order-repair.tmp')
    temporary.write_bytes(after)
    temporary.replace(path)
    (root / 'checkpoint-order-repair.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    repair(parser.parse_args().root)
