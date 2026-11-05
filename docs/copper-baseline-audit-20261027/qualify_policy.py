from pathlib import Path
import collections,json,statistics,hashlib
p=Path('/Aidata/shared/ldm/copper-baseline-audit-20261027')
old=json.loads((p/'policy-development/results.json').read_text());new=json.loads((p/'projected-policy/results.json').read_text());assert len(old)==64 and len(new)==32
summary={}
for source,rows in [('policy',old),('projected',new)]:
 for name in sorted({x['variant'] for x in rows}):
  a=[x for x in rows if x['variant']==name];reasons=collections.Counter()
  for x in a:
   assert not x['exit_code'] and x['result']['status']=='succeeded'
   s=x['search'];reasons.update(s['invalid_reasons'])
   assert x['result']['metrics']['loss_total']<=s['curve'][0]['loss']+1e-9
   if source=='projected':
    assert not s['invalid_reasons']
    assert s['calls']-1-s['unchanged_occupancy']>0
    if name.startswith('acq'):
     assert s['acquisition_decisions']>0
     assert sum(v for k,v in s['acquisition_batch_sizes'].items() if int(k)>1)>0
  summary[source+'-'+name]=dict(cases=len(a),invalid_reasons=dict(reasons),proposals=sum(x['search']['proposals'] for x in a),calls_median=statistics.median(x['search']['calls'] for x in a),acquisition_active=sum(x['search']['acquisition_decisions']>0 for x in a),public_loss_mean=statistics.mean(x['result']['metrics']['loss_total'] for x in a))
assert '10 passed' in (p/'pytest-projected.log').read_text()
assert all(x['truth_legal'] for x in json.loads((p/'development-policy/population-qa.json').read_text()))
for f,h in json.loads((p/'r3-preservation-before.json').read_text()).items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
out=dict(status='passed',scope='r3 public hard priors on development copies; 16 train cases; frozen seed43 parameters',groups=summary,r3_control_hashes_unchanged=True,template_sha256=hashlib.sha256((p/'stochastic_solver_projected.py').read_bytes()).hexdigest())
(p/'policy-acceptance.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
