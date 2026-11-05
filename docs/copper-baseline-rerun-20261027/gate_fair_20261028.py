"""One-shot development gate; never starts or changes formal training."""
import fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
O=Path(__file__).resolve().parent
lock=(O/'gate-fair.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
started=O/'gate-fair-started.json'
with started.open('x') as f:json.dump(dict(pid=os.getpid(),time=time.time()),f)
S=Path('/Aidata/shared/ldm/copper-measurement-study-20261027-r4-baselines/source')
python='/Aidata/shared/ldm/copper-v2-evaluator/code/service-env/bin/python'
env={**os.environ,'PYTHONPATH':str(O)+':/Aidata/shared/ldm/default-research-20261024/testdeps:'+str(S),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
def status(stage,**kw):
    p=O/'gate-fair-20261028.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(stage=stage,time=time.time(),**kw),indent=2));tmp.replace(p)
try:
    files=['fair_queue_20261028.py','runtime_fair_20261028.py','qualify_fair_20261028.py','test_fair_queue_20261028.py']
    frozen={str(O/n):hashlib.sha256((O/n).read_bytes()).hexdigest() for n in files}
    with (O/'fair-frozen-20261028.json').open('x') as f:json.dump(frozen,f,indent=2)
    status('targeted_tests')
    with (O/'fair-tests-final.log').open('x') as log:
        subprocess.run([python,'-m','pytest','-q',str(O/'test_fair_queue_20261028.py')],cwd=S,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    for small,root in [(True,O/'qfs28'),(False,O/'qff28')]:
        status('small' if small else 'full',root=str(root))
        with (O/('fair-small.log' if small else 'fair-full.log')).open('x') as log:
            subprocess.run([python,str(O/'runtime_fair_20261028.py'),'load','--root',str(root),*(['--small'] if small else [])],cwd=S,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=14400)
        result=json.loads((root/'completion.json').read_text())
        assert result['status']=='passed' and not result['failures']
        assert result['jobs']==(12 if small else 1308)
        if not small:assert result['peak_active']==32
    status('passed_pending_recovery_controller',frozen=frozen)
except BaseException as e:
    status('failed',error=repr(e));raise
