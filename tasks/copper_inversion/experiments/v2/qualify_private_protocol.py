"""Real-container protocol acceptance on development geology only.

No formal validation/test inputs are accessed by this command.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import threading

from tasks.copper_inversion.core.execution import digest, source_identity
from tasks.copper_inversion.v2.service import Handler
from tasks.copper_inversion.v2.service_client import ServiceEvaluator
from tasks.copper_inversion.v2.selection import run
from tasks.copper_inversion.v2.suite import file_hash, load_manifest
from .build_suite import build
from .freeze_service_selection import freeze_from_service
from .parallel_runtime import ParallelService, ThreadedServer
from .tune_stochastic import source_for, configurations


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--python', type=Path, required=True)
    p.add_argument('--image', required=True)
    a = p.parse_args()
    if os.geteuid() != 991:
        raise ValueError('private evaluator account required')
    root = a.root.resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    splits = ('train', 'validation', 'test_id', 'test_ood')
    families = ('folded', 'lens', 'multiple', 'faulted_lens')
    parents = [sorted(a.calibration.glob(family + '-*'))[0] for family in families]
    plan = dict(scope='development_protocol_acceptance', parents=list(map(str, parents)),
                splits=splits, seconds=120, slots=4, source_order=['nearest', 'mcmc', 'acq', 'invalid'])
    (root / 'plan.json').write_text(json.dumps(plan, indent=2))

    def observations(item):
        index, kind = item
        target = root / 'packs' / splits[index] / kind
        cache = root / 'cache' / splits[index] / kind
        cache.mkdir(parents=True)
        with (cache / 'generation.log').open('w') as log:
            subprocess.run([str(a.python), '-m', 'tasks.copper_inversion.experiments.v2.observations',
                '--parent', str(parents[index]), '--output', str(target),
                '--seed', str(202610240 + 2 * index + (kind == 'confirmation'))],
                env={**os.environ, 'TMPDIR': str(cache)}, stdout=log,
                stderr=subprocess.STDOUT, check=True, timeout=600)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(observations, [(i, kind) for i in range(4) for kind in ('public', 'confirmation')]))
    for split in splits:
        packs = root / 'packs' / split
        output = root / 'data' / split
        build([packs / 'public'], output, split)
        path = output / 'manifest.json'
        manifest = json.loads(path.read_text())
        confirmation = json.loads((packs / 'confirmation/private-manifest.json').read_text())
        for row in manifest['cases']:
            # Each modality owns a separate confirmation directory.
            old = output / row['truth']
            private = output / 'private-case' / row['case_id']
            private.mkdir(parents=True)
            shutil.copy2(old / 'truth.npy', private / 'truth.npy')
            match = next(c for c in confirmation['cases'] if c['modality'] == row['modality'])
            shutil.copytree(packs / 'confirmation' / match['case_id'], private / 'confirmation')
            row['truth'] = str(private.relative_to(output))
            for name in ('confirmation/problem.json', 'confirmation/problem.npz'):
                row['sha256'][name] = file_hash(private / name)
        path.write_text(json.dumps(manifest, indent=2))
        load_manifest(path)
    config = dict(manifest=str(root / 'data/train/manifest.json'), root=str(root / 'service'),
        image=a.image, seconds=120, slots=4, caller_uid=991, socket=str(root / 'evaluator.sock'))
    backend = ParallelService(config)
    client = ServiceEvaluator(root / 'client', config['socket'], digest(backend.identity), service_uid=991)
    template = Path(__file__).with_name('stochastic_solver.py').read_text()
    sources = ['from tasks.copper_inversion.v2.baselines import initialize\n'
               'def solve(problem,budget,seed):\n    return initialize(problem)\n']
    sources += [source_for(template, configurations(method, 1, 42)[0]) for method in ('mcmc_v2', 'acq_v2')]
    sources += ['def solve(problem,budget,seed):\n    return None\n']
    request = lambda source: {'operation': 'evaluate', 'source': source, 'contract_digest': digest(backend.identity)}
    with ThreadedServer(config['socket'], Handler) as server:
        server.backend = backend
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            # Explicit reservations preserve the chosen source order.
            for source in sources:
                backend.slots.reserve(hashlib.sha256(source.encode()).hexdigest())
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda source: client.request(request(source)), sources))
            assert [r['status'] for r in results] == ['succeeded'] * 3 + ['invalid_interface'], results
            assert all(r['metrics']['legal_rate'] == 1 for r in results[:3]), results
            before = {str(p): file_hash(p) for p in root.rglob('receipt.json')}
            restarted = ParallelService(config)
            assert restarted.dispatch(request(sources[0])) == results[0]
            assert before == {str(p): file_hash(p) for p in root.rglob('receipt.json')}
            overflow = client.request(request(sources[0] + '\n# new source\n'))
            assert overflow['status'] == 'budget_exhausted', overflow
            assert client.request({'operation': 'test'})['status'] == 'rejected_request'
            seal = client.request({'operation': 'seal_training', 'contract_digest': digest(backend.identity)})
            assert seal['used_slots'] == 4 and len(seal['source_sha256s']) == 3
            assert client.request(request(sources[0]))['status'] == 'training_sealed'
        finally:
            server.shutdown()
            thread.join()
    selection = root / 'selection'
    freeze_from_service(root / 'service', selection, root / 'data/validation/manifest.json',
                        root / 'data/test_id/manifest.json', root / 'data/test_ood/manifest.json')
    try:
        run(selection, 'test')
    except ValueError as exc:
        assert 'freeze validation selection' in str(exc)
    else:
        raise AssertionError('test allowed before validation')
    validated = run(selection, 'validation')
    tested = run(selection, 'test')
    assert all(row['metrics']['legal_rate'] == 1 for row in validated['candidates'])
    assert all(row['metrics']['legal_rate'] == 1 for row in tested['results'])
    before = {str(p): file_hash(p) for p in root.rglob('receipt.json')}
    assert run(selection, 'validation') == validated
    assert run(selection, 'test') == tested
    assert before == {str(p): file_hash(p) for p in root.rglob('receipt.json')}
    result = dict(status='passed', scope=plan['scope'], charged_sources=4,
        implementation_sha256=source_identity(), image=a.image,
        baseline_template_sha256=hashlib.sha256(template.encode()).hexdigest(),
        accepted_sources=3, rejected_sources=1, solver_seconds=120,
        replay_receipts_unchanged=True, budget_overflow_rejected=True,
        premature_test_rejected=True, sealed_training_rejected=True,
        formal_holdouts_accessed=False, receipt_count=len(before))
    (root / 'completion.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
