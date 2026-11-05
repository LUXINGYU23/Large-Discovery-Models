"""Trusted root deployment supervisor. Secret supplied only through stdin.

Nine independent budgets. Failed clients are retained for explicit reconciliation;
no automatic new run, source mutation, free retries or test-driven reselection.
"""
import argparse
import hashlib
import signal
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

PRIVATE=Path('/Aidata/shared/ldm/copper-v2-evaluator')
CLIENT_PY='/Aidata/shared/ldm/copper-encoding-ablation-v3-20261023/source/tasks/copper_inversion/.venv/bin/python'
PRIVATE_PY=str(PRIVATE/'code/service-env/bin/python')
IMAGE='sha256:f6856385a0da9444b1e5edb5048b2f6081593fe568c3473c2fda524d0c4489f5'
SIDECAR='sha256:8100350c577497d9eec5289f03dfc978f318b4cfc00a6efb9784856c5b95fc74'


def write(path,value,uid=0,gid=991,mode=0o640):
    path.write_text(json.dumps(value,indent=2));os.chown(path,uid,gid);path.chmod(mode)


def run_as(user,python,module,args,source,log,*,key=None):
    env={**os.environ,'PYTHONPATH':str(source),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    env.pop('LLM_API_KEY',None)
    if key:env['LLM_API_KEY']=key
    with log.open('ab') as stream:
        return subprocess.Popen(['runuser','-u',user,'--',python,'-m',module,*map(str,args)],cwd=source,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--release',type=Path,required=True);p.add_argument('--acceptance',type=Path,required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    if os.geteuid()!=0:raise ValueError('trusted deployment requires root')
    from tasks.copper_inversion.v2.suite import load_manifest
    stage=load_manifest(a.release/'train/manifest.json')['stage']
    if stage!=('development' if a.pilot else 'formal'):
        raise ValueError('pilot must use development geology; full study requires formal release')
    rounds=1 if a.pilot else 20
    slots=1+16*rounds
    acceptance=json.loads(a.acceptance.read_text())
    if acceptance.get('status')!='passed' or acceptance.get('formal_holdouts_accessed') is not False:
        raise ValueError('private development protocol acceptance required')
    from tasks.copper_inversion.core.execution import source_identity
    if acceptance.get('implementation_sha256')!=source_identity() or acceptance.get('image')!=IMAGE:
        raise ValueError('protocol acceptance implementation or image differs')
    template=Path(__file__).with_name('stochastic_solver.py')
    if acceptance.get('baseline_template_sha256')!=hashlib.sha256(template.read_bytes()).hexdigest():
        raise ValueError('protocol acceptance baseline template differs')
    if acceptance.get('saturated_load_verified') is not True or acceptance.get('qualified_scheduling') != dict(global_jobs=128,interface_slots=8,jobs_per_process=32,startup_slots=4):
        raise ValueError('saturated container load acceptance required')
    if not a.pilot and acceptance.get('nine_group_pilot_verified') is not True:
        raise ValueError('complete nine-group development pilot required before full study')
    key=sys.stdin.read().strip()
    if not key:raise ValueError('protected remote provider credential required on stdin')
    root=a.root.resolve();root.mkdir(mode=0o755,exist_ok=False);os.chown(root,1001,1001)
    source=root/'source';source.mkdir()
    for name in ('ldm_tts','tasks'):shutil.copytree(a.source/name,source/name,ignore=shutil.ignore_patterns('__pycache__'))
    for path in source.rglob('*'):
        if not path.is_symlink():os.chown(path,0,0);path.chmod(0o755 if path.is_dir() else 0o644)
    from tasks.copper_inversion.core.execution import digest
    from .reconcile_discovery import reconcile
    queue=root/'queue';queue.mkdir(mode=0o777);queue.chmod(0o777)
    for index in range(128):
        lock=queue/f'{index:03}.lock';lock.touch();lock.chmod(0o666)
    (queue/'startup').mkdir(mode=0o777);(queue/'startup').chmod(0o777)
    for index in range(4):
        lock=queue/'startup'/f'{index:03}.lock';lock.touch();lock.chmod(0o666)
    scheduler=source/'tasks/copper_inversion/experiments/v2/parallel_runtime.py'
    policy=dict(queue_root=str(queue),global_jobs=128,interface_slots=8,jobs_per_process=32,startup_slots=4,
                runtime_sha256=hashlib.sha256(scheduler.read_bytes()).hexdigest())
    policy['policy_sha256']=digest(policy)
    policy_path=root/'scheduling-policy.json';write(policy_path,policy,uid=0,gid=0,mode=0o644)
    def parallel(user,python,mode,args,log,secret=None):
        return run_as(user,python,'tasks.copper_inversion.experiments.v2.parallel_runtime',
            ['--policy',policy_path,mode,*args],source,log,key=secret)
    private_run=PRIVATE/'runs'/root.name;private_run.mkdir(mode=0o700);os.chown(private_run,991,991)
    # Publisher runs as the evaluator; it exports training observations only.
    published=root/'publication';published.mkdir();os.chown(published,991,991)
    pub=run_as('ldm-copper-eval',PRIVATE_PY,'tasks.copper_inversion.experiments.v2.publish_training',['--manifest',a.release/'train/manifest.json','--destination',published/'training'],source,private_run/'publish.log')
    if pub.wait()!=0:raise RuntimeError('publication failed')
    services={};workers={};service_procs={};configs={}
    for seed in (42,43,3407):
        for method in ('ldm','mcmc','acq'):
            name=f'{method}-{seed}';service=private_run/name;service.mkdir();os.chown(service,991,991)
            rpc=root/('rpc-'+name);rpc.mkdir(mode=0o750);os.chown(rpc,991,1001)
            config=dict(manifest=str(a.release/'train/manifest.json'),root=str(service),image=IMAGE,seconds=120,slots=slots,caller_uid=1001,socket=str(rpc/'evaluator.sock'))
            config_path=private_run/(name+'-config.json');write(config_path,config)
            proc=parallel('ldm-copper-eval',PRIVATE_PY,'service',['--config',config_path],private_run/(name+'-service.log'));service_procs[name]=proc
            deadline=time.monotonic()+180
            while not Path(config['socket']).exists():
                if proc.poll() is not None or time.monotonic()>deadline:raise RuntimeError('service startup failed: '+name)
                time.sleep(.5)
            os.chown(config['socket'],991,1001);os.chmod(config['socket'],0o660)
            identity=json.loads((service/'identity.json').read_text())
            from tasks.copper_inversion.core.execution import digest
            contract=digest(identity);configs[name]=dict(config,contract=contract)
            args=['--public-catalog',published/'training/catalog.json','--evaluation-socket',config['socket'],'--evaluation-contract',contract,'--evaluation-service-uid','991']
            if method=='ldm':
                args+=['--out-dir',root/name,'--seed',seed,'--iterations',rounds,'--candidates-per-researcher','8','--evaluations-per-round','16','--solver-seconds','120','--harness-seconds','1800','--embedding-model-path','/Aidata/shared/codebo/models/Qwen3-Embedding-0.6B-97b0c614','--embedding-device','cuda:'+str((42,43,3407).index(seed)),'--harness-cache-dir','/Aidata/shared/ldm/cache/gondolin','--sidecar-image',SIDECAR,'--evaluator-image',IMAGE]
                module='tasks.copper_inversion.v2.workflow'
            else:
                args+=['--output',root/name,'--method',method+'_v2','--slots',slots,'--seed',seed,'--seconds','120'];module='tasks.copper_inversion.experiments.v2.tune_stochastic'
            if method=='ldm':
                workers[name]=parallel('zsgpu',CLIENT_PY,'ldm',args,root/(name+'.log'),key)
            else:
                prepared=run_as('zsgpu',CLIENT_PY,module,args+['--prepare-only'],source,root/(name+'.log'))
                if prepared.wait()!=0:raise RuntimeError('baseline preparation failed: '+name)
                workers[name]=parallel('zsgpu',CLIENT_PY,'tune',['--run',root/name],root/(name+'.log'))
            services[name]=str(service)
    plan=dict(release=str(a.release),services=services,selection_root=str(private_run/'selection'),source=str(source),client_source=str(source),case_workers=4,discovery_seeds=[42,43,3407],feedback_slots=slots,rounds=rounds,batch=16,solver_seconds=120,development_only=a.pilot)
    write(private_run/'finish-plan.json',plan)
    write(root/'deployment.json',dict(plan,pids={k:p.pid for k,p in workers.items()},service_pids={k:p.pid for k,p in service_procs.items()},configs=configs,supervisor_pid=os.getpid()),uid=1001,gid=1001,mode=0o644)
    write(root/'scheduling-epoch.json',dict(policy=policy,pids={k:p.pid for k,p in workers.items()},service_pids={k:p.pid for k,p in service_procs.items()},supervisor_pid=os.getpid()),uid=1001,gid=1001,mode=0o644)
    while any(p.poll() is None for p in workers.values()):
        write(root/'progress.json',dict(stage='training',updated_at=time.time(),exit_codes={k:p.poll() for k,p in workers.items()}),uid=1001,gid=1001,mode=0o644)
        time.sleep(15)
    failed={k:p.returncode for k,p in workers.items() if p.returncode}
    if failed:
        write(root/'needs-reconciliation.json',dict(failed=failed,services_retained=True),uid=1001,gid=1001,mode=0o644)
        raise RuntimeError('clients need reconciliation; retained all runs and services')
    reconciled={name:reconcile(root/name,services[name],cap=slots) for name in services}
    write(private_run/'budget-reconciliation.json',reconciled)
    # Only the authorized client UID can seal the training socket.
    for name,config in configs.items():
        script='from tasks.copper_inversion.v2.service_client import ServiceEvaluator; import json; print(json.dumps(ServiceEvaluator(".",'+repr(config['socket'])+','+repr(config['contract'])+',service_uid=991).request({"operation":"seal_training","contract_digest":'+repr(config['contract'])+'})))'
        result=subprocess.run(['runuser','-u','zsgpu','--',CLIENT_PY,'-c',script],cwd=source,env={**os.environ,'PYTHONPATH':str(source)},capture_output=True,text=True,check=True)
        seal=json.loads(result.stdout)
        if seal.get('status')!='training_sealed' or seal.get('used_slots')!=reconciled[name]['private_sources']:raise RuntimeError('incomplete training seal: '+name)
    finish=parallel('ldm-copper-eval',PRIVATE_PY,'finish',['--plan',private_run/'finish-plan.json'],private_run/'finish.log')
    write(root/'progress.json',dict(stage='private_validation_test',pid=finish.pid),uid=1001,gid=1001,mode=0o644)
    if finish.wait()!=0:raise RuntimeError('private selection failed; reconcile existing receipts')
    shutil.copy2(private_run/'selection/completion.json',root/'completion.json');os.chown(root/'completion.json',1001,1001)
    for proc in service_procs.values():os.killpg(proc.pid,signal.SIGTERM)
    write(root/'progress.json',dict(stage='completed'),uid=1001,gid=1001,mode=0o644)


if __name__=='__main__':main()
