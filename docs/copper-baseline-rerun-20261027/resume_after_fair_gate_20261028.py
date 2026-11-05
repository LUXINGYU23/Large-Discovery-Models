"""Single-use conditional handoff: a failed or incomplete gate cannot resume training."""
import fcntl,hashlib,json,os,time
from pathlib import Path
O=Path(__file__).resolve().parent
lock=(O/'fair-handoff.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
controller=O/'recover_fair_20261028.py'
sha=hashlib.sha256(controller.read_bytes()).hexdigest()
with (O/'fair-handoff-started.json').open('x') as f:
    json.dump(dict(pid=os.getpid(),time=time.time(),controller=str(controller),controller_sha256=sha),f,indent=2)
deadline=time.monotonic()+15000
while True:
    state=json.loads((O/'gate-fair-20261028.json').read_text())
    if state['stage']=='passed_pending_recovery_controller':break
    assert state['stage']!='failed',state
    gate_pid=json.loads((O/'gate-fair-started.json').read_text())['pid']
    command=Path(f'/proc/{gate_pid}/cmdline').read_bytes()
    assert str(O/'gate_fair_20261028.py').encode() in command,'gate exited without passing'
    assert time.monotonic()<deadline,'qualification deadline exceeded; leave formal pause intact'
    time.sleep(30)
assert hashlib.sha256(controller.read_bytes()).hexdigest()==sha,'controller changed after authorization'
assert not (O/'recovery-fair-20261028').exists(),'handoff already attempted'
python='/Aidata/shared/ldm/copper-v2-evaluator/code/service-env/bin/python'
env={**os.environ,'PYTHONPATH':str(O)+':/Aidata/shared/ldm/copper-measurement-study-20261027-r4-baselines/source','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
os.execve(python,[python,str(controller)],env)
