# -*- coding: utf-8 -*-
"""Read frozen seed-42 sources; export algorithm facts and an isolated unit probe.

Run only on the authorized remote host. Never invokes solve(), Docker, an
evaluator, or a model API. Writes only the report's source/data directories.
"""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

O = Path('/Aidata/shared/ldm/copper-baseline-rerun-20261027-ops')
B = O / 'seed42-investor-report-20261031'
R = Path('/Aidata/shared/ldm/copper-measurement-study-20261025-r3')
P = Path('/Aidata/shared/ldm/copper-v2-evaluator/runs/copper-measurement-study-20261025-r3')
SELECTED = '2d2c33ed6a588d9e444a104d135c8d358ca715e6cf061130a92bf5fd0d1c9d0b'
TRAINING_BEST = '776ee0300e9757563c2d5e1b087c3499f111b2bfd9b7dfa979c35f3c8cc979fd'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def inspect_source(source_id):
    p = P / 'selection/ldm-42/candidates' / source_id / 'solver.py'
    assert sha(p) == source_id
    s = p.read_text()
    tree = ast.parse(s)
    cfg = next(ast.literal_eval(n.value) for n in tree.body
               if isinstance(n, ast.Assign) and any(
                   isinstance(t, ast.Name) and t.id == 'CFG' for t in n.targets))
    functions = {n.name: {'start': n.lineno, 'end': n.end_lineno}
                 for n in tree.body if isinstance(n, ast.FunctionDef)}
    dst = B / 'source' / ('ldm42-selected-solver.py' if source_id == SELECTED
                              else 'ldm42-training-best-solver.py')
    dst.write_bytes(p.read_bytes())
    assert sha(dst) == source_id
    return dict(source_sha256=source_id, frozen_path=str(p),
                report_path=str(dst.relative_to(B)), cfg=cfg, functions=functions), s


def probe_best_pair(source):
    """Execute the frozen _polish body with synthetic arrays and a stub score.

    This checks variable bookkeeping, not geological performance. Only one
    loop iteration is allowed. The first cube wins; the second cube is last.
    """
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == '_polish')
    clock = iter([0., 0., 0., 2.])
    score_by_negative_voxels = {8: .4, 64: .8}

    def stub_score(problem, field):
        return score_by_negative_voxels[int(np.sum(field < 0))]

    env = dict(np=np, CFG={'polish_mode': 'best_pair', 'polish_r_choices': (1, 3)},
               time=SimpleNamespace(monotonic=lambda: next(clock)),
               ndimage=SimpleNamespace(binary_dilation=lambda a, structure:
                                        np.ones_like(a, dtype=bool)),
               _ST=None, _score=stub_score)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<frozen _polish>', 'exec'), env)
    f = np.ones((9, 9, 9))
    best, recorded = env['_polish'](
        {}, None, f, 1., 1., SimpleNamespace(integers=lambda n: 0),
        None, None, np.ones_like(f, dtype=bool), f.copy(), 1.)
    actual = stub_score({}, best)
    assert recorded == .4 and actual == .8
    return dict(kind='isolated bookkeeping unit probe', formal_solves=0,
                frozen_function_lines=[node.lineno, node.end_lineno],
                first_proposal_score=.4, last_proposal_score=.8,
                returned_model_stub_score=actual, stored_best_score=recorded,
                mismatch_reproduced=True,
                meaning='best stores final prop rather than selected chosen when radius 1 wins',
                measured_performance_impact='not measured; no rerun or reselection')


before = sha(B / 'data/report.json')
protected = json.loads((O / 'r3-preservation-before.json').read_text())
assert len(protected) == 29
assert all(sha(Path(p)) == expected for p, expected in protected.items())
selected, source = inspect_source(SELECTED)
training_best, _ = inspect_source(TRAINING_BEST)
train_evaluation = None
for p in (R / 'ldm-42/evaluations').glob('*/result.json'):
    s = p.read_text()
    if SELECTED in s:
        train_evaluation = json.loads(s)['metrics']
        break
assert train_evaluation is not None
selected['training_metrics'] = train_evaluation
changes = {k: {'training_best': training_best['cfg'].get(k), 'selected': v}
           for k, v in selected['cfg'].items() if v != training_best['cfg'].get(k)}
assert set(changes) == {'anneal_frac', 'seed_offset'}
# Both variants have identical executable bodies apart from CFG.
def without_cfg(s):
    tree = ast.parse(s)
    # Compare executable function bodies; candidate labels in module docstrings differ.
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    return ast.dump(tree, include_attributes=False)
assert without_cfg(source) == without_cfg((B / training_best['report_path']).read_text())
result = dict(discovery_seed=42, selected=selected, training_best=training_best,
              cfg_differences=changes, identical_function_bodies=True,
              selected_active_path=['shared_observation_initializer', 'sphere', 'anneal', 'best_pair'],
              inactive_paths=['rbf', 'cycle', 'portfolio', 'slab', 'topology', 'ellipsoid'],
              polish_bookkeeping=probe_best_pair(source),
              preserved_r3_files_verified=len(protected),
              data_sha256_before=before, data_sha256_after=sha(B / 'data/report.json'),
              formal_evaluation_rerun=False, winner_changed=False)
assert result['data_sha256_before'] == result['data_sha256_after']
(B / 'data/algorithm-audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps({'selected': SELECTED, 'training_best': TRAINING_BEST,
                  'cfg_differences': changes, 'probe': result['polish_bookkeeping'],
                  'report_data_unchanged': True}, ensure_ascii=False), flush=True)
