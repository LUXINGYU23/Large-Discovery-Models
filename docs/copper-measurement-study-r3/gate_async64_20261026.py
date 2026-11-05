from pathlib import Path
import json,time,os,fcntl,hashlib,subprocess
O=Path('/Aidata/shared/ldm/copper-permanent-repair-20261025');R=Path('/Aidata/shared/ldm/copper-measurement-study-20261025-r3');PY='/Aidata/shared/ldm/copper-v2-evaluator/code/service-env/bin/python'
lock=(O/'async64-gate.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
p=O/'async64-gate.json';assert not p.exists()
controller=O/'continue_async64_20261026.py';expected=hashlib.sha256(controller.read_bytes()).hexdigest()
def status(stage,**kw):p.write_text(json.dumps(dict(pid=os.getpid(),time=time.time(),stage=stage,controller_sha256=expected,**kw),indent=2))
def cpu():
 a=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]));return sum(a[:8]),a[3]+a[4]
env={**os.environ,'PYTHONPATH':str(O/'test-source'),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
try:
 for small in (True,False):
  name='async-joint64-small-20261026' if small else 'async-joint64-load-20261026';status('small_load' if small else 'full_joint_load')
  args=[PY,str(O/'async-joint64-source-20261026/qualify_load.py'),'--root',str(O/name)]+(['--small'] if small else [])
  with (O/(name+'.log')).open('wb') as log:
   child=subprocess.Popen(args,cwd=O/'test-source',env=env,stdout=log,stderr=subprocess.STDOUT)
   prev=cpu();deadline=time.monotonic()+3600
   while child.poll() is None:
    assert time.monotonic()<deadline,'qualification stalled'
    time.sleep(10);cur=cpu();total=cur[0]-prev[0];busy=1-(cur[1]-prev[1])/max(1,total);prev=cur
    with (O/'async64-host-resources.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),cpu_busy=busy,loadavg=Path('/proc/loadavg').read_text().strip()))+'\n')
   assert child.returncode==0,'joint load failed; formal remains paused'
  d=json.loads((O/name/'completion.json').read_text());assert d['status']=='passed' and d['jobs']==(12 if small else 1308)
  if not small:assert d['peak_active']==64
 assert hashlib.sha256(controller.read_bytes()).hexdigest()==expected
 status('qualified_reconciling');os.environ['PYTHONPATH']=str(R/'source');os.execv(PY,[PY,str(controller)])
except BaseException as exc:
 status('failed_formal_remains_paused',error=str(exc));raise
