"""One-shot development gate; never starts or changes formal training."""
import fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
O=Path(__file__).resolve().parent
lock=(O/'gate-recovery-20261030.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
started=O/'gate-recovery-20261030-started.json'
with started.open('x') as f:json.dump(dict(pid=os.getpid(),time=time.time()),f)
S=Path('/Aidata/shared/ldm/copper-measurement-study-20261027-r4-baselines/source')
python='/Aidata/shared/ldm/copper-v2-evaluator/code/service-env/bin/python'
env={**os.environ,'PYTHONPATH':str(O)+':/Aidata/shared/ldm/default-research-20261024/testdeps:'+str(S),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
def status(stage,**kw):
    p=O/'gate-recovery-20261030.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(stage=stage,time=time.time(),**kw),indent=2));tmp.replace(p)
try:
    frozen=json.loads((O/'fair-frozen-20261028.json').read_text())
    for f,h in frozen.items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
    assert '10 passed' in (O/'recovery-tests-20261030-qualified-env.log').read_text()
    status('targeted_tests')
    with (O/'fair-tests-20261030.log').open('x') as log:
        subprocess.run([python,'-m','pytest','-q',str(O/'test_fair_queue_20261028.py')],cwd=S,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    for small,root in [(True,O/'qfs30'),(False,O/'qff30')]:
        status('small' if small else 'full',root=str(root))
        with (O/('fair-small-20261030.log' if small else 'fair-full-20261030.log')).open('x') as log:
            subprocess.run([python,str(O/'runtime_fair_20261028.py'),'load','--root',str(root),*(['--small'] if small else [])],cwd=S,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=14400)
        result=json.loads((root/'completion.json').read_text())
        assert result['status']=='passed' and not result['failures']
        assert result['jobs']==(12 if small else 1308)
        if not small:assert result['peak_active']==32
    status('passed_pending_recovery_controller',frozen=frozen)
except BaseException as e:
    status('failed',error=repr(e));raise
