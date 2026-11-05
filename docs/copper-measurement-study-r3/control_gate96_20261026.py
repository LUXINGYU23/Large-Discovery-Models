import os,json,time,subprocess,fcntl
from pathlib import Path
O=Path('/Aidata/shared/ldm/copper-permanent-repair-20261025')
R=Path('/Aidata/shared/ldm/copper-measurement-study-20261025-r3')
PY='/Aidata/shared/ldm/copper-v2-evaluator/code/service-env/bin/python'
lock=(O/'control-gate96-20261026.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
p=O/'control-gate96-20261026.json';assert not p.exists()
def status(stage,**kw):p.write_text(json.dumps(dict(pid=os.getpid(),time=time.time(),stage=stage,**kw),indent=2))
status('full_load_qualification')
try:
 env={**os.environ,'PYTHONPATH':str(O/'test-source'),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
 with (O/'control-load96-20261026.log').open('wb') as f:
  subprocess.run([PY,str(O/'control-load-source96-20261026/qualify_load.py'),'--root',str(O/'load-control96-20261026')],env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=3600)
 status('qualified_continuing')
 os.environ['PYTHONPATH']=str(R/'source')
 os.execv(PY,[PY,str(O/'continue_docker_control96_20261026.py')])
except BaseException as exc:
 status('failed_formal_remains_paused',error=type(exc).__name__+': '+str(exc));raise
