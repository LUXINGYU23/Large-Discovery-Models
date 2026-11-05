from pathlib import Path
import json,time,os,fcntl,hashlib
O=Path('/Aidata/shared/ldm/copper-permanent-repair-20261025');R=Path('/Aidata/shared/ldm/copper-measurement-study-20261025-r3')
lock=(O/'async-continuation-gate.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
p=O/'async-continuation-gate.json';assert not p.exists()
controller=O/'continue_async_20261026.py';expected=hashlib.sha256(controller.read_bytes()).hexdigest()
def status(stage,**kw):p.write_text(json.dumps(dict(pid=os.getpid(),time=time.time(),stage=stage,controller_sha256=expected,**kw),indent=2))
status('waiting_joint_saturated_protocol')
try:
 end=time.monotonic()+3600
 result=O/'async-joint-load-20261026/completion.json'
 while not result.exists():
  assert time.monotonic()<end,'joint qualification stalled'
  time.sleep(15)
 d=json.loads(result.read_text());assert d['status']=='passed' and d['jobs']==1308 and d['peak_active']==96
 assert hashlib.sha256(controller.read_bytes()).hexdigest()==expected
 status('qualified_reconciling')
 os.environ['PYTHONPATH']=str(R/'source')
 os.execv('/Aidata/shared/ldm/copper-v2-evaluator/code/service-env/bin/python',['python',str(controller)])
except BaseException as exc:
 status('failed_formal_remains_paused',error=str(exc));raise
