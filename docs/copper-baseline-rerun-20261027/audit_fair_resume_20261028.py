import json,hashlib,time,os
from pathlib import Path
O=Path(__file__).resolve().parent
R=Path('/Aidata/shared/ldm/copper-measurement-study-20261027-r4-baselines')
P=Path('/Aidata/shared/ldm/copper-v2-evaluator/runs')/R.name
E=O/'recovery-fair-20261028'
d=json.loads((R/'recovery-deployment.json').read_text())
assert d['epoch']==str(E)
assert not list(Path(d['queue']).rglob('PAUSED.json'))
rows=[]
before=json.loads((O/'monitor-20261028-1449.json').read_text())
old_rows={r['name']:r for r in before['rows']}
old_failures={r['path'] for r in before['failures']}
failures=[]
for n in d['pids']:
 for pid in [d['pids'][n],d['service_pids'][n]]:
  assert Path(f'/proc/{pid}/cmdline').read_bytes(),('process absent',n,pid)
 cases=len(list((P/n/'evaluations').glob('*/*/*/receipt.json')))
 charged=json.loads((R/n/'budget.json').read_text())['counters']['algorithm_feedback_slots']
 assert old_rows[n]['charged']<=charged<=321
 rows.append(dict(name=n,cases=cases,new_cases=cases-old_rows[n]['cases'],charged=charged,complete=len(list((P/n/'evaluations').glob('*/result.json')))))
 for path in (P/n/'jobs').glob('*/receipt.json'):
  r=json.loads(path.read_text())
  if r['exit_code']!=0 or r.get('reason') or r.get('oom'):failures.append(str(path))
assert set(failures)==old_failures,('failure set changed',set(failures)-old_failures)
checked=0
for path,sha in json.loads((E/'history-before.json').read_text()).items():
 p=Path(path)
 if p.name in ('receipt.json','result.json','identity.json') or p.suffix=='.py':
  assert hashlib.sha256(p.read_bytes()).hexdigest()==sha,('historical result changed',path)
  checked+=1
for row in json.loads((O/'fair-recovery-preflight.json').read_text())['rows']:
 for scope in ('client','private'):
  p=E/(row['group']+'-'+scope+'-budget-before.json')
  assert hashlib.sha256(p.read_bytes()).hexdigest()==row['budgets'][scope]['sha256']
for manifest in ['r3-preservation-before.json','frozen-files.json','ack-frozen-20261028.json','fair-frozen-20261028.json']:
 for path,sha in json.loads((O/manifest).read_text()).items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
out=dict(time=time.time(),rows=rows,historical_result_source_files_unchanged=checked,budget_originals_preserved=12,known_failures=len(failures),new_failures=0,all_groups_new_cases=all(r['new_cases']>0 for r in rows),mcmc43_tail_completed=next(r['complete'] for r in rows if r['name']=='mcmc-43')>=129,manifests_unchanged=True)
(O/'fair-resume-audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
