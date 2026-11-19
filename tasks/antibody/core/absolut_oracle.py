"""Isolated Absolut batch evaluation with persistent per-target resources."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

TARGETS = {'HER2': '1N8Z_C', 'PD-1': '5GGR_Y_FuC_5.25'}
ALPHABET = set('ACDEFGHIKLMNPQRSTVWY')

class AbsolutOracle:
    def __init__(self, absolut_path, resource_dirs, run_root, timeout=600):
        self.binary = Path(absolut_path).resolve() / 'src/bin/Absolut'
        self.resources = [Path(p).resolve() for p in resource_dirs]
        self.run_root = Path(run_root).resolve()
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        self.timeout = timeout

    def preflight(self, target, require_cache=False):
        antigen = TARGETS.get(target, target)
        if antigen not in TARGETS.values():
            raise ValueError('Unknown target: ' + target)
        if not self.binary.is_file():
            raise FileNotFoundError(self.binary)
        info = subprocess.run([str(self.binary), 'info_filenames', antigen], capture_output=True, text=True, timeout=30, check=True)
        import re
        match = re.search(r'Pre-calculated structures are in (\S+Structures\.txt)', info.stdout)
        if not match:
            raise RuntimeError('Cannot resolve target resources: ' + info.stdout + info.stderr)
        name = match.group(1)
        source = next((p / name for p in self.resources if (p / name).is_file()), None)
        if source is None:
            raise FileNotFoundError('Target precomputation not ready: ' + name)
        prefix = name.split('-10-11-')[0]
        caches = {}
        for directory in self.resources:
            for cached in directory.glob(prefix + '+*.txt'):
                caches.setdefault(cached.name, cached)
        if require_cache and not any(p.name.endswith('Compact.txt') for p in caches.values()):
            raise FileNotFoundError('Target interaction cache not ready: ' + antigen)
        for path in (source, *caches.values()):
            if not path.is_file() or not os.access(path, os.R_OK) or path.stat().st_size == 0:
                raise FileNotFoundError('Unreadable or empty resource: ' + str(path))
        return antigen, source, tuple(caches.values())

    def evaluate(self, sequences, target):
        if isinstance(sequences, (str, bytes)):
            raise ValueError("sequences must be a collection, not a string")
        sequences = list(sequences)
        if not sequences or any(not isinstance(s, str) or len(s) < 11 or len(s) > 40 or set(s) - ALPHABET for s in sequences):
            raise ValueError('Expected canonical amino-acid sequences of length 11–40')
        antigen, source, caches = self.preflight(target)
        self.run_root.mkdir(parents=True, exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix=antigen + '-', dir=self.run_root))
        (run / source.name).symlink_to(source)
        for cached in caches:
            (run / cached.name).symlink_to(cached)
        (run / 'input.tsv').write_text(''.join(f'{i}\t{s}\n' for i, s in enumerate(sequences, 1)))
        try:
            return self._run(sequences, antigen, run)
        except Exception as exc:
            (run / "error.json").write_text(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc), "antigen": antigen, "run_dir": str(run)}, indent=2) + "\n")
            raise

    def _run(self, sequences, antigen, run):
        with (run / 'evaluation.log').open('wb') as log:
            proc = subprocess.run([str(self.binary), 'repertoire', antigen, 'input.tsv', '1'], cwd=run, stdout=log, stderr=subprocess.STDOUT, timeout=self.timeout)
        if proc.returncode:
            raise RuntimeError(f'Absolut failed ({proc.returncode}); see {run}/evaluation.log')
        output = run / (antigen + 'FinalBindings_Process_1_Of_1.txt')
        energies = {}
        with output.open() as handle:
            next(handle)
            for row in csv.DictReader(handle, delimiter='\t'):
                idx = int(row['ID_slide_Variant'].split('_')[0])
                energy = float(row['Energy'])
                if idx not in range(1, len(sequences)+1) or not math.isfinite(energy) or row['CDR3'] != sequences[idx-1]:
                    raise RuntimeError('Invalid result row')
                energies[idx] = min(energies.get(idx, math.inf), energy)
        if len(energies) != len(sequences):
            raise RuntimeError('Missing sequence results')
        result = {'antigen': antigen, 'oracle': 'Absolut', 'binary_sha256': hashlib.sha256(self.binary.read_bytes()).hexdigest(), 'run_dir': str(run), 'results': [{'sequence': s, 'absolut_energy': energies[i]} for i, s in enumerate(sequences, 1)]}
        (run / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--absolut-path', default=os.environ.get('ABSOLUT_PATH'), required=not os.environ.get('ABSOLUT_PATH'))
    p.add_argument('--resources', action='append', required=True)
    p.add_argument('--run-root', required=True)
    p.add_argument('--target', choices=list(TARGETS), required=True)
    p.add_argument('--sequence', action='append', required=True)
    p.add_argument('--timeout', type=int, default=600)
    a = p.parse_args()
    print(json.dumps(AbsolutOracle(a.absolut_path, a.resources, a.run_root, a.timeout).evaluate(a.sequence, a.target), indent=2))

if __name__ == '__main__':
    main()
