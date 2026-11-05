"""Read-only post-handoff evidence collection; no training or restart authority."""
import json,os,subprocess,time
from pathlib import Path
O=Path(__file__).resolve().parent
R=Path('/Aidata/shared/ldm/copper-measurement-study-20261027-r4-baselines')
P=Path('/Aidata/shared/ldm/copper-v2-evaluator/runs')/R.name
with (O/'fair-audit-started.json').open('x') as f:json.dump(dict(pid=os.getpid(),time=time.time()),f)
baseline={r['name']:r['cases'] for r in json.loads((O/'monitor-20261028-1449.json').read_text())['rows']}
deadline=time.monotonic()+18000
while True:
    assert time.monotonic()<deadline,'handoff audit deadline exceeded'
    gate=json.loads((O/'gate-fair-20261028.json').read_text());assert gate['stage']!='failed',gate
    d=json.loads((R/'recovery-deployment.json').read_text())
    if d['epoch']==str(O/'recovery-fair-20261028'):
        assert not list(Path(d['queue']).rglob('PAUSED.json')),'new queue paused'
        new=all(len(list((P/n/'evaluations').glob('*/*/*/receipt.json')))>count for n,count in baseline.items())
        tail=len(list((P/'mcmc-43/evaluations').glob('*/result.json')))>=129
        if new and tail:break
    time.sleep(30)
env={**os.environ,'PYTHONPATH':str(R/'source')}
subprocess.run(['/Aidata/shared/ldm/copper-v2-evaluator/code/service-env/bin/python',str(O/'audit_fair_resume_20261028.py')],env=env,check=True,timeout=900)
