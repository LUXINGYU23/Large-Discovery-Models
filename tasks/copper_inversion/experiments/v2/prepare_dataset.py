"""Private dataset preparation, with a durable stage record and fail-closed QA."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    args = parser.parse_args()
    stages = [
        ('generation', 'generate_independent', ['--python', str(args.python)]),
        ('population', 'qualify_population', ['--output', str(args.root / 'population-qa.json')]),
        ('confirmation', 'generate_confirmation', ['--python', str(args.python)]),
    ]
    for stage, module, extra in stages:
        (args.root / 'pipeline-status.json').write_text(json.dumps({'stage': stage, 'status': 'running'}))
        result = subprocess.run([sys.executable, '-m',
            'tasks.copper_inversion.experiments.v2.' + module,
            '--root', str(args.root), *extra])
        if result.returncode:
            (args.root / 'pipeline-status.json').write_text(json.dumps(
                {'stage': stage, 'status': 'failed', 'exit_code': result.returncode}))
            raise SystemExit(result.returncode)
    (args.root / 'qa-pipeline-complete.json').write_text(json.dumps(
        {'status': 'population_and_confirmation_generated', 'formal_release': False}))
    (args.root / 'pipeline-status.json').write_text(json.dumps({'status': 'completed_pending_release'}))


if __name__ == '__main__':
    main()
