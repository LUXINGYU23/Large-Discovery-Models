"""Read-only evidence checks after all fixed development runs finish."""
from pathlib import Path
import ast,hashlib,itertools,json,math
B=Path('/Aidata/shared/ldm/copper-baseline-audit-20261027')
root=B/'actual-time-policy-validation'
plan=json.loads((root/'plan.json').read_text())
rows=json.loads((root/'results.json').read_text())
expected=set(itertools.product(plan['sources'],[c['case_id'] for c in plan['cases']],[47,48,49],[30,90]))
actual=[(r['group'],r['case_id'],r['execution_seed'],r['target_seconds']) for r in rows]
assert len(rows)==432 and len(set(actual))==432 and set(actual)==expected
checks=dict(unique_complete_runs=432,invalid=0,acq_active=0,acq_multiple_candidate_runs=0)
for r in rows:
 assert r['exit_code']==0 and r['reason'] is None
 assert r['target_seconds']-2<r['worker_seconds']<r['target_seconds']+1
 assert all(math.isfinite(r['metrics'][k]) for k in ['loss_total','iou','relative_volume_error'])
 assert r['metrics']['component_count']<=4
 if r['group'].startswith('ldm'):continue
 s=r['search'];checks['invalid']+=s['invalid']
 assert abs(s['best_public_loss']-r['metrics']['loss_total'])<1e-8
 assert r['metrics']['loss_total']<=s['curve'][0]['loss']+1e-8
 assert s['calls']-1-s['unchanged_occupancy']>0
 if r['group'].startswith('acq'):
  checks['acq_active']+=int(s['acquisition_decisions']>0)
  checks['acq_multiple_candidate_runs']+=int(any(int(k)>1 and v>0 for k,v in s['acquisition_batch_sizes'].items()))
assert checks['invalid']==0 and checks['acq_active']==144 and checks['acq_multiple_candidate_runs']==144
def config(path):
 return next(ast.literal_eval(n.value) for n in ast.parse(path.read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONFIG' for t in n.targets))
for name,sha in plan['sources'].items():
 path=root/'sources'/name/'solver.py'
 assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
 if not name.startswith('ldm'):assert config(path)==config(B/(name+'.py'))
original=json.loads((B/'r3-preservation-before.json').read_text())
assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==sha for f,sha in original.items())
checks.update(status='passed',r3_control_hashes_preserved=len(original),frozen_comparison_source_hashes=9,frozen_baseline_configurations=6)
(B/'report/comparison-qa.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
