"""Explicit, isolated Boltz-2 complex prediction; confidence is auxiliary only."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


def predict(input_path, output, executable, cache, gpu, seed=19, samples=1, timeout=7200):
    source, output, cache = map(lambda p: Path(p).resolve(), (input_path, output, cache))
    if source.suffix not in ('.yaml', '.yml') or not source.is_file():
        raise ValueError('Provide one native Boltz YAML input')
    if gpu < 0 or not 0 <= seed < 2**32 or samples < 1:
        raise ValueError('Invalid GPU, seed, or sample count')
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Timeout must be positive and finite')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty')
    output.mkdir(parents=True, exist_ok=True)
    # Keep original input location: native MSA/template paths may be relative.
    command = [str(executable), 'predict', str(source), '--out_dir', str(output / 'prediction'),
               '--cache', str(cache), '--model', 'boltz2', '--seed', str(seed),
               '--diffusion_samples', str(samples), '--write_full_pae', '--no_kernels',
               '--num_workers', '0', '--devices', '1']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    record = dict(tool='boltz2', status='running', command=command, seed=seed,
                  samples=samples, validation_scope='experimental_auxiliary',
                  hard_filter_eligible=False, input_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    start = time.monotonic()
    try:
        with (output / 'tool.log').open('wb') as log:
            subprocess.run(command, cwd=source.parent, env=env, stdout=log,
                           stderr=subprocess.STDOUT, timeout=timeout, check=True)
        structures = sorted((output / 'prediction').glob('**/predictions/**/*.cif'))
        # Boltz can exit successfully after per-record failure. Require every requested sample.
        if len(structures) != samples:
            raise RuntimeError(f'Expected {samples} predicted CIFs, received {len(structures)}')
        pairs = []
        for structure in structures:
            pae = structure.with_name('pae_' + structure.stem + '.npz')
            plddt = structure.with_name('plddt_' + structure.stem + '.npz')
            confidence = structure.with_name('confidence_' + structure.stem + '.json')
            if not pae.is_file() or not plddt.is_file() or not confidence.is_file():
                raise RuntimeError('Missing native PAE, pLDDT or confidence artifact')
            data = json.loads(confidence.read_text())
            if 'iptm' not in data or not math.isfinite(float(data['iptm'])):
                raise RuntimeError('Missing or non-finite native ipTM')
            pairs.append(dict(structure=str(structure.relative_to(output)),
                              pae=str(pae.relative_to(output)),
                              plddt=str(plddt.relative_to(output)),
                              confidence=str(confidence.relative_to(output))))
        record.update(status='succeeded', predictions=pairs)
    except Exception as exc:
        record.update(status='failed', error=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        record['seconds'] = time.monotonic() - start
        (output / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--executable', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--seed', type=int, default=19)
    p.add_argument('--samples', type=int, default=1)
    p.add_argument('--timeout', type=float, default=7200)
    a = p.parse_args()
    print(json.dumps(predict(a.input, a.out, a.executable, a.cache, a.gpu,
                             a.seed, a.samples, a.timeout), indent=2))


if __name__ == '__main__':
    main()
