"""Run ipSAE jobs in isolated artifact directories."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time


def run(args):
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError('Timeout must be positive and finite')
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory must be empty')
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    inputs = {}
    script, pae, structure = args.script.resolve(), args.pae.resolve(), args.structure.resolve()
    if args.pae_cutoff <= 0 or args.distance_cutoff <= 0:
        raise ValueError('Cutoffs must be positive')
    # ipSAE writes beside the structure: copy into the isolated run directory.
    for source in (pae, structure):
        if source.name in ('run.json', 'tool.log'):
            raise ValueError('Reserved input filename')
    if pae.name == structure.name:
        raise ValueError('PAE and structure filenames must differ')
    shutil.copyfile(pae, out / pae.name)
    shutil.copyfile(structure, out / structure.name)
    command = [args.python, str(script), str(out / pae.name), str(out / structure.name),
               str(args.pae_cutoff), str(args.distance_cutoff)]
    inputs = {'script': script, 'pae': pae, 'structure': structure}
    record = {'tool': args.tool, 'command': command, 'status': 'running',
              'inputs': {k: {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in inputs.items()}}
    start = time.monotonic()
    try:
        (out / 'run.json').write_text(json.dumps(record, indent=2))
        with (out / 'tool.log').open('wb') as log:
            subprocess.run(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=args.timeout, check=True)
        artifacts = [p for p in out.rglob('*') if p.is_file() and p.name not in ('run.json', 'tool.log', 'input.json', *(p.name for p in inputs.values()))]
        if not artifacts:
            raise RuntimeError('Tool returned success but produced no artifacts')
        if args.tool == 'ipsae':
            scores = []
            for artifact in artifacts:
                if artifact.suffix != '.txt' or artifact.name.endswith('_byres.txt'):
                    continue
                lines = [line.split() for line in artifact.read_text().splitlines() if line.strip()]
                if not lines or 'ipSAE' not in lines[0]:
                    continue
                header = lines[0]
                for values in lines[1:]:
                    row = dict(zip(header, values))
                    value = float(row['ipSAE'])
                    if not math.isfinite(value):
                        raise ValueError('Non-finite ipSAE output')
                    scores.append({k: row[k] for k in ('Chn1', 'Chn2', 'Type')} | {'ipSAE': value})
            if not scores:
                raise RuntimeError('No valid ipSAE chain-pair scores produced')
            record['chain_pair_scores'] = scores
        record.update(status='succeeded', artifacts=[str(p.relative_to(out)) for p in artifacts])
    except Exception as exc:
        record.update(status='failed', error=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        record['seconds'] = time.monotonic() - start
        (out / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=3600)
    sub = parser.add_subparsers(dest='tool', required=True)
    p = sub.add_parser('ipsae')
    p.add_argument('--python', required=True)
    p.add_argument('--script', type=Path, required=True)
    p.add_argument('--pae', type=Path, required=True)
    p.add_argument('--structure', type=Path, required=True)
    p.add_argument('--pae-cutoff', type=int, default=10)
    p.add_argument('--distance-cutoff', type=int, default=10)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('timeout must be positive')
    print(json.dumps(run(args), indent=2))

if __name__ == '__main__':
    main()
