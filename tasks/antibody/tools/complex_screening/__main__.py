"""Run explicit Protenix or ipSAE jobs in isolated artifact directories."""
import argparse
import hashlib
import json
import math
import os
import re
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
    if args.tool == 'protenix':
        if args.model != 'protenix_base_default_v1.0.0':
            raise ValueError('Only validated public Protenix v1 is supported')
        source = args.input.resolve()
        data = json.loads(source.read_text())
        if not isinstance(data, list) or not data:
            raise ValueError('Expected native Protenix JSON: nonempty list of complexes')
        for item in data:
            if not isinstance(item, dict) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', str(item.get('name', ''))) or len(item.get('sequences', [])) < 2:
                raise ValueError('Each complex needs a name and at least two molecular entities')
            for entry in item['sequences']:
                protein = entry.get('proteinChain', {})
                if not protein.get('sequence') or set(protein['sequence']) - set('ACDEFGHIKLMNPQRSTVWY'):
                    raise ValueError('This adapter supports canonical protein complexes only')
                for field in ('pairedMsaPath', 'unpairedMsaPath'):
                    if protein.get(field):
                        msa = Path(protein[field])
                        if not msa.is_absolute():
                            raise ValueError('Precomputed MSA paths must be absolute')
                        inputs[f"msa_{len(inputs)}"] = msa.resolve()
        if len({item['name'] for item in data}) != len(data):
            raise ValueError('Complex names must be unique')
        seeds = args.seeds or [args.seed]
        if args.samples < 1 or not seeds or len(set(seeds)) != len(seeds) or any(not 0 <= seed < 2**32 for seed in seeds):
            raise ValueError('Samples must be positive and seeds nonnegative')
        if args.gpu < 0 or args.seed < 0:
            raise ValueError('GPU and seed must be nonnegative')
        shutil.copyfile(source, out / 'input.json')
        command = [args.executable, 'pred', '-i', str(source), '-o', str(out / 'prediction'),
                   '-s', ','.join(map(str, seeds)), '-e', str(args.samples), '-n', args.model, '--use_template', 'false',
                   '--use_msa', str(args.use_msa).lower(), '--need_atom_confidence', 'true',
                   '--trimul_kernel', 'torch', '--triatt_kernel', 'torch',
                   '--use_default_params', 'true']
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        executable_dir = str(Path(args.executable).resolve().parent)
        env['PATH'] = executable_dir + os.pathsep + env.get('PATH', '')
        inputs['input'] = source
    else:
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
    if args.tool == 'protenix':
        record.update(validation_scope='experimental_auxiliary', hard_filter_eligible=False,
                      model=args.model, seeds=seeds, samples_per_seed=args.samples, use_msa=args.use_msa)
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
        else:
            predictions = []
            for item in data:
                for seed in seeds:
                    folder = out / 'prediction' / item['name'] / f'seed_{seed}' / 'predictions'
                    structures = sorted(folder.glob('*.cif'))
                    if len(structures) != args.samples:
                        raise RuntimeError(f"Incomplete prediction for {item['name']} seed {seed}")
                    for structure in structures:
                        stem = structure.stem
                        prefix, sample = stem.rsplit('_sample_', 1)
                        confidence = structure.with_name(f'{prefix}_full_data_sample_{sample}.json')
                        summary = structure.with_name(f'{prefix}_summary_confidence_sample_{sample}.json')
                        if not confidence.is_file() or not summary.is_file():
                            raise RuntimeError('Missing native full confidence or summary')
                        predictions.append({'structure': str(structure.relative_to(out)),
                                            'full_confidence': str(confidence.relative_to(out)),
                                            'summary': str(summary.relative_to(out))})
            record['predictions'] = predictions
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
    p = sub.add_parser('protenix')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--executable', required=True, help='Absolute path to isolated environment protenix CLI')
    p.add_argument('--model', default='protenix_base_default_v1.0.0')
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--seeds', type=int, nargs='+', help='Explicit multi-seed inference, overrides --seed')
    p.add_argument('--samples', type=int, default=1)
    p.add_argument('--use-msa', action=argparse.BooleanOptionalAction, default=True, help='Enabled by default; may query configured remote MSA service. Use public or explicitly approved sequences.')
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
