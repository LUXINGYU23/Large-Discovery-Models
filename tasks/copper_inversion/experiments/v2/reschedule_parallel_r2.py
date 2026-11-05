"""Trusted one-time scheduling transition; preserves frozen scientific runtime."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT=Path('/Aidata/shared/ldm/copper-formal-study-20261024-r1')
OPS=Path('/Aidata/shared/ldm/copper-formal-scheduling-20261024-r2')
PRIVATE=Path('/Aidata/shared/ldm/copper-v2-evaluator/runs')/ROOT.name
CPY='/Aidata/shared/ldm/copper-encoding-ablation-v3-20261023/source/tasks/copper_inversion/.venv/bin/python'
PPY='/Aidata/shared/ldm/copper-v2-evaluator/code/service-env/bin/python'

def write(path,value,public=False):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.chmod(0o644 if public else 0o600);tmp.replace(path)

def child_command(pid):
    children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
    matches=[]
    for child in children:
        args=Path(f'/proc/{child}/cmdline').read_bytes().decode().strip('\0').split('\0')
        if '-m' in args:matches.append((child,args))
    if len(matches)!=1:raise RuntimeError('ambiguous child '+str(pid))
    return matches[0]

def start(user,python,mode,args,source,log,key=None):
    env={**os.environ,'PYTHONPATH':source,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
    env.pop('LLM_API_KEY',None)
    if key:env['LLM_API_KEY']=key
    with log.open('ab') as stream:
        return subprocess.Popen(['runuser','-u',user,'--',python,str(OPS/('replay_parallel.py' if mode=='ldm' else 'parallel_runtime.py')),'--policy',str(OPS/'policy.json'),mode,*args],cwd=source,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)

def main():
    assert os.geteuid()==0
    prior=Path('/Aidata/shared/ldm/copper-formal-scheduling-20261024-r1')
    transition=json.loads((prior/'transition.json').read_text())
    d=transition['original'];epoch=json.loads((ROOT/'scheduling-epoch.json').read_text())
    d.update({k:epoch[k] for k in ('pids','service_pids','supervisor_pid')})
    assert not (OPS/'transition.json').exists()
    captured=transition['commands'];key=None
    for name,pid in d['pids'].items():
        if name.startswith('ldm-') and Path(f'/proc/{pid}').exists():
            children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
            for child in children:
                for value in Path(f'/proc/{child}/environ').read_bytes().split(b'\0'):
                    if value.startswith(b'LLM_API_KEY='):key=value.split(b'=',1)[1].decode()
    assert key
    assert b'continue_parallel' in Path(f'/proc/{d["supervisor_pid"]}/cmdline').read_bytes()
    # Freeze all controllers atomically at process-group granularity; Docker workloads
    # have independent container deadlines and remain recoverable by stable identity.
    os.kill(d['supervisor_pid'],signal.SIGSTOP)
    for pid in [*d['pids'].values(),*d['service_pids'].values()]:
        if not Path(f"/proc/{pid}").exists():continue
        assert os.getpgid(pid)==pid
        os.killpg(pid,signal.SIGSTOP)
    snapshots={}
    for name in d['pids']:
        for label,base in [('client',ROOT/name),('private',Path(d['services'][name]))]:
            for pattern in ('budget.json','identity.json','plan.json','**/checkpoint*.json','**/receipt.json','tuning-receipts/*.json'):
                for path in base.glob(pattern):
                    snapshots[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    write(OPS/'transition.json',dict(time=time.time(),original=d,commands=captured,files=snapshots))
    # No scientific container is killed or recreated. Existing receipt recovery
    # finalizes in-flight jobs from Docker timestamps and authenticated counters.
    os.kill(d['supervisor_pid'],signal.SIGKILL)
    for pid in [*d['pids'].values(),*d['service_pids'].values()]:
        if Path(f'/proc/{pid}').exists():os.killpg(pid,signal.SIGKILL)
    time.sleep(3)
    services={};workers={}
    for name,config in d['configs'].items():
        socket=Path(config['socket'])
        if socket.exists():socket.rename(socket.with_name('pre-parallel-r2.sock'))
        proc=start('ldm-copper-eval',PPY,'service',['--config',str(PRIVATE/(name+'-config.json'))],d['source'],PRIVATE/(name+'-service.log'));services[name]=proc
        deadline=time.monotonic()+180
        while not socket.exists():
            if proc.poll() is not None or time.monotonic()>deadline:raise RuntimeError('service startup: '+name)
            time.sleep(.25)
        os.chown(socket,991,1001);socket.chmod(0o660)
    for name,args in captured.items():
        if name.startswith('ldm-'):
            arguments=args[args.index('-m')+2:]+['--resume'];mode='ldm'
        else:arguments=['--run',str(ROOT/name)];mode='tune'
        workers[name]=start('zsgpu',CPY,mode,arguments,d['client_source'],ROOT/(name+'.log'),key if mode=='ldm' else None)
    epoch=dict(policy=json.loads((OPS/'policy.json').read_text()),started_at=time.time(),pids={n:p.pid for n,p in workers.items()},service_pids={n:p.pid for n,p in services.items()},supervisor_pid=os.getpid())
    write(ROOT/'scheduling-epoch.json',epoch,True)
    while any(p.poll() is None for p in workers.values()):
        write(ROOT/'progress.json',dict(stage='training',scheduling='parallel-r1',updated_at=time.time(),exit_codes={n:p.poll() for n,p in workers.items()}),True)
        time.sleep(15)
    failed={n:p.returncode for n,p in workers.items() if p.returncode}
    if failed:
        write(ROOT/'needs-reconciliation.json',dict(failed=failed,services_retained=True),True)
        raise RuntimeError('clients require reconciliation')
    for name,config in d['configs'].items():
        script='from tasks.copper_inversion.v2.service_client import ServiceEvaluator; import json; print(json.dumps(ServiceEvaluator(".",'+repr(config['socket'])+','+repr(config['contract'])+',service_uid=991).request({"operation":"seal_training","contract_digest":'+repr(config['contract'])+'})))'
        result=subprocess.run(['runuser','-u','zsgpu','--',CPY,'-c',script],cwd=d['client_source'],env={**os.environ,'PYTHONPATH':d['client_source']},capture_output=True,text=True,check=True)
        seal=json.loads(result.stdout)
        if seal.get('status')!='training_sealed' or seal.get('used_slots')!=321:raise RuntimeError('incomplete seal '+name)
    finish=start('ldm-copper-eval',PPY,'finish',['--plan',str(PRIVATE/'finish-plan.json')],d['source'],PRIVATE/'finish.log')
    write(ROOT/'progress.json',dict(stage='private_validation_test',pid=finish.pid,scheduling='parallel-r1'),True)
    if finish.wait()!=0:raise RuntimeError('private selection requires reconciliation')
    shutil.copy2(PRIVATE/'selection/completion.json',ROOT/'completion.json');os.chown(ROOT/'completion.json',1001,1001)
    for proc in services.values():os.killpg(proc.pid,signal.SIGTERM)
    write(ROOT/'progress.json',dict(stage='completed'),True)

if __name__=='__main__':main()
