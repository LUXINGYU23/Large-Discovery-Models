from pathlib import Path
import collections,json,statistics,hashlib
p=Path('/Aidata/shared/ldm/copper-baseline-audit-20261027')
a=json.loads((p/'full-development/results.json').read_text());assert len(a)==64
summary={}
for name in sorted({x['variant'] for x in a}):
 rows=[x for x in a if x['variant']==name];reasons=collections.Counter()
 for x in rows:
  assert x['exit_code']==0 and x['result']['status']=='succeeded'
  s=x['search'];reasons.update(s['invalid_reasons'])
  assert x['result']['metrics']['loss_total']<=s['curve'][0]['loss']+1e-9
  if name.startswith('first-order'):
   assert not s['invalid_reasons'].get('distance reinitialization changed validated occupancy',0)
   if 'acq' in name:assert s['acquisition_decisions']>0
 summary[name]=dict(cases=len(rows),invalid_reasons=dict(reasons),proposals=sum(x['search']['proposals'] for x in rows),calls_median=statistics.median(x['search']['calls'] for x in rows),acquisition_active=sum(x['search']['acquisition_decisions']>0 for x in rows),public_loss_mean=statistics.mean(x['result']['metrics']['loss_total'] for x in rows))
assert json.loads((p/'numeric-regression.json').read_text())['status']=='passed'
assert '9 passed' in (p/'pytest.log').read_text()
for f,sha in json.loads((p/'r3-preservation-before.json').read_text()).items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==sha
out=dict(status='passed',scope='16 public development train cases; frozen seed43 configuration; numeric scale and subvoxel regressions; no formal holdout use',groups=summary,r3_control_hashes_unchanged=True)
(p/'development-acceptance.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
