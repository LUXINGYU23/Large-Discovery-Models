"""Gate continuation on complete real-load and private development acceptance."""
from pathlib import Path
import os,json,time,subprocess,hashlib,fcntl
OPS=Path('/Aidata/shared/ldm/copper-permanent-repair-20261025');S=OPS/'test-source'
R=Path('/Aidata/shared/ldm/copper-measurement-study-20261025-r3');E=R/'controller-epochs/admission-deadline-r1'
PR=Path('/Aidata/shared/ldm/copper-v2-evaluator/runs/permanent-repair-protocol-20261025')
PY='/Aidata/shared/ldm/copper-v2-evaluator/code/service-env/bin/python'
lock=(OPS/'qualification-controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (OPS/'qualification-controller.json').exists(),'already launched'
def status(stage,**kwargs):
 (OPS/'qualification-progress.json').write_text(json.dumps(dict(stage=stage,updated_at=time.time(),**kwargs),indent=2))
(OPS/'qualification-controller.json').write_text(json.dumps(dict(pid=os.getpid(),started_at=time.time())))
try:
 for name in ('load-small-v2','load-full-v2','parallel-admission-v2'):
  path=OPS/name/'completion.json';deadline=time.monotonic()+3600
  while not path.exists():
   assert time.monotonic()<deadline,'qualification stalled '+name
   status('waiting_full_acceptance',waiting=name);time.sleep(15)
  assert json.loads(path.read_text())['status']=='passed',name+' failed'
 assert '217 passed, 41 skipped' in (OPS/'full-regressions-v3.log').read_text()
 assert json.loads((OPS/'epoch-regression.json').read_text())['status']=='passed'
 assert '4 passed' in (OPS/'signal-v2-regression-final.log').read_text()
 assert 'pass 14' in (OPS/'sidecar-container-tests.log').read_text() and 'fail 0' in (OPS/'sidecar-container-tests.log').read_text()
 status('private_development_protocol')
 assert not PR.exists(),'private qualification directory already used'
 env={**os.environ,'PYTHONPATH':str(S),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
 args=['runuser','-u','ldm-copper-eval','--',PY,'-m','tasks.copper_inversion.experiments.v2.qualify_private_protocol','--root',str(PR),'--calibration','/Aidata/shared/ldm/copper-v2-evaluator/datasets/prior-calibration-20261024-r2','--python','/Aidata/shared/ldm/copper-v2-evaluator/code/gempy-env/bin/python','--image','sha256:f6856385a0da9444b1e5edb5048b2f6081593fe568c3473c2fda524d0c4489f5']
 with (OPS/'private-protocol.log').open('wb') as log:subprocess.run(args,cwd=S,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
 result=json.loads((PR/'completion.json').read_text());assert result['status']=='passed' and not result['formal_holdouts_accessed']
 epoch=json.loads((E/'epoch.json').read_text());assert result['baseline_template_sha256']==epoch['baseline']['sha256']
 acceptance=dict(status='passed',epoch_sha256=hashlib.sha256((E/'epoch.json').read_bytes()).hexdigest(),private_protocol=result,regressions='217 passed, 41 skipped',sidecar_tests=14,signal_tests=4,load=json.loads((OPS/'load-full-v2/completion.json').read_text()),parallel_admission=json.loads((OPS/'parallel-admission-v2/completion.json').read_text()),created_at=time.time())
 (OPS/'repair-acceptance.json').write_text(json.dumps(acceptance,indent=2));status('qualified_starting_continuation')
 # Replace this gate process with the single continuation supervisor.
 os.environ['PYTHONPATH']=str(R/'source')
 os.execv(PY,[PY,str(OPS/'continue_permanent_repair.py')])
except BaseException as exc:
 status('qualification_failed_formal_stays_paused',error=type(exc).__name__+': '+str(exc));raise
