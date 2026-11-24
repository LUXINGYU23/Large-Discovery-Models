"""Optional ipSAE tool for explicitly host-prepared, run-local artifacts."""
import argparse
import json
from pathlib import Path
import os
import sys
from tasks.antibody.tools.complex_screening.__main__ import run


def score_interface(worker, artifact_id):
    if not isinstance(artifact_id, str) or not artifact_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in artifact_id):
        raise ValueError('Invalid interface artifact ID')
    root = worker.jobs.parent.resolve()
    inputs = (root / 'interface_inputs').resolve()
    manifest = inputs / (artifact_id + '.json')
    if manifest.is_symlink():
        raise ValueError('Symlink manifests are not accepted')
    data = json.loads(manifest.read_text())
    paths = {}
    for key in ('pae', 'structure'):
        paths[key] = (inputs / data[key]).resolve()
        if not paths[key].is_relative_to(inputs):
            raise ValueError('Interface input escapes the run-local input directory')
    script = os.environ.get('IPSAE_SCRIPT')
    python = os.environ.get('IPSAE_PYTHON', sys.executable)
    if not script or not Path(script).is_file():
        raise ValueError('Host must configure IPSAE_SCRIPT')
    import tempfile
    output_root = root / 'interface_scores'
    output_root.mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=artifact_id+'-', dir=output_root))
    args = argparse.Namespace(tool='ipsae',out=out,timeout=120,script=Path(script),
        pae=paths['pae'],structure=paths['structure'],python=python,pae_cutoff=10,distance_cutoff=10)
    return run(args)
