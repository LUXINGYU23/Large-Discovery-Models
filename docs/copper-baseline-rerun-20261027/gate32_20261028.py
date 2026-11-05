import os,json,time,subprocess,fcntl,hashlib
from pathlib import Path
O=Path(__file__).resolve().parent
lock=(O/'gate32-20261028.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (O/'gate32-20261028-started.json').exists()
(O/'gate32-20261028-started.json').write_text(json.dumps(dict(pid=os.getpid(),time=time.time())))
assert '10 passed' in (O/'recovery-tests-20261028.log').read_text()
env={**os.environ,'PYTHONPATH':'/Aidata/shared/ldm/copper-measurement-study-20261027-r4-baselines/source','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
try:
 for small in [True,False]:
  root=O/('q32s28' if small else 'q32f28')
  (O/'gate32-20261028.json').write_text(json.dumps(dict(stage='small' if small else 'full',pid=os.getpid(),time=time.time())))
  subprocess.run(['/Aidata/shared/ldm/copper-v2-evaluator/code/service-env/bin/python',str(O/'runtime_load32_20261028.py'),'load','--root',str(root)]+(['--small'] if small else []),env=env,check=True,timeout=7200)
  q=json.loads((root/'completion.json').read_text());assert q['status']=='passed' and q['jobs']==(12 if small else 1308)
  if not small:assert q['peak_active']==32
 (O/'gate32-20261028.json').write_text(json.dumps(dict(stage='passed_pending_recovery_controller',pid=os.getpid(),time=time.time(),production_resumed=False)))
except BaseException as e:
 (O/'gate32-20261028.json').write_text(json.dumps(dict(stage='failed',error=str(e),time=time.time(),production_resumed=False)));raise
