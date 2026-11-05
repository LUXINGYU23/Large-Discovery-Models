"""One-shot qualified six-group rerun. No LDM execution or old-run mutation."""
import fcntl,hashlib,json,os,shutil,signal,subprocess,time
from pathlib import Path
from tasks.copper_inversion.experiments.v2.launch_formal import CLIENT_PY,PRIVATE_PY,IMAGE,write,run_as
from tasks.copper_inversion.core.execution import digest,source_identity
from tasks.copper_inversion.experiments.v2.reconcile_discovery import reconcile
O=Path(__file__).resolve().parent
R=Path('/Aidata/shared/ldm/copper-measurement-study-20261027-r4-baselines');S=R/'source'
P=Path('/Aidata/shared/ldm/copper-v2-evaluator/runs')/R.name
RELEASE=Path('/Aidata/shared/ldm/copper-v2-evaluator/datasets/measurement-formal-20261024-r2')
assert os.geteuid()==0
lock=(O/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (O/'started.json').exists(),'one-shot controller already started'
frozen=json.loads((O/'frozen-files.json').read_text())
for path,sha in frozen.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
old=json.loads((O/'r3-preservation-before.json').read_text())
def preserve():
 for path,sha in old.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,'old r3 changed'
preserve()
def status(stage,**kw):write(R/'progress.json',dict(stage=stage,updated_at=time.time(),supervisor_pid=os.getpid(),**kw),uid=1001,gid=1001,mode=0o644)
write(O/'started.json',dict(pid=os.getpid(),time=time.time(),approved=True,source_identity=source_identity()),uid=1001,gid=1001,mode=0o644)
env={**os.environ,'PYTHONPATH':str(S),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'};env.pop('LLM_API_KEY',None)
queue=R/'queue';queue.mkdir();queue.chmod(0o777)
for directory,count in [(queue,64),(queue/'startup',1)]:
 directory.mkdir(exist_ok=True);directory.chmod(0o777)
 for i in range(count):(directory/f'{i:03}.lock').touch();(directory/f'{i:03}.lock').chmod(0o666)
runtime=S/'tasks/copper_inversion/experiments/v2/parallel_runtime.py'
policy=dict(queue_root=str(queue),global_jobs=64,interface_slots=8,jobs_per_process=32,startup_slots=1,runtime_sha256=hashlib.sha256(runtime.read_bytes()).hexdigest())
policy['policy_sha256']=digest(policy);policy_path=R/'scheduling-policy.json';write(policy_path,policy,uid=0,gid=0,mode=0o644)
def checked(args,log,timeout=3600):
 with log.open('ab') as f:subprocess.run(args,cwd=S,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=timeout)
def start(user,python,mode,args,log):
 with log.open('ab') as f:return subprocess.Popen(['runuser','-u',user,'--',python,str(O/'runtime.py'),'--policy',str(policy_path),mode,*map(str,args)],cwd=S,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
services={};workers={};configs={}
try:
 # Saturated acceptance covers the actual repaired template, fixed configs,
 # Unix submit/status, fixture legality, and the same 64-slot control policy.
 for small in [True,False]:
  stage='qualification_small' if small else 'qualification_full'
  status(stage)
  target=O/('load-small' if small else 'load-full')
  checked([PRIVATE_PY,str(O/'runtime.py'),'load','--root',str(target)]+(['--small'] if small else []),O/(stage+'.log'))
  q=json.loads((target/'completion.json').read_text())
  assert q['status']=='passed' and q['jobs']==(12 if small else 1308)
  if not small:assert q['peak_active']==64
 status('qualification_private_protocol')
 dev=Path('/Aidata/shared/ldm/copper-v2-evaluator/runs/r4-baselines-development-protocol-20261027')
 checked(['runuser','-u','ldm-copper-eval','--',PRIVATE_PY,str(O/'runtime.py'),'qualify',str(policy_path),'--root',str(dev),'--calibration','/Aidata/shared/ldm/copper-v2-evaluator/datasets/prior-calibration-20261024-r2','--python','/Aidata/shared/ldm/copper-v2-evaluator/code/gempy-env/bin/python','--image',IMAGE],O/'qualification-private.log')
 q=json.loads((dev/'completion.json').read_text())
 assert q['status']=='passed' and not q['formal_holdouts_accessed'] and q['budget_overflow_rejected'] and q['premature_test_rejected'] and q['sealed_training_rejected']
 assert q['baseline_template_sha256']==hashlib.sha256((S/'tasks/copper_inversion/experiments/v2/stochastic_solver.py').read_bytes()).hexdigest()
 write(O/'qualification.json',dict(status='passed',private=q,load_jobs=1308,peak=64,template=q['baseline_template_sha256']),uid=1001,gid=1001,mode=0o644)
 preserve();status('deploying_training')
 P.mkdir(mode=0o700);os.chown(P,991,991)
 publication=R/'publication';publication.mkdir();os.chown(publication,991,991)
 pub=run_as('ldm-copper-eval',PRIVATE_PY,'tasks.copper_inversion.experiments.v2.publish_training',['--manifest',RELEASE/'train/manifest.json','--destination',publication/'training'],S,P/'publish.log')
 assert pub.wait()==0,'training publication failed'
 for seed in [42,43,3407]:
  for method in ['mcmc','acq']:
   name=f'{method}-{seed}';private=P/name;private.mkdir();os.chown(private,991,991)
   rpc=R/('rpc-'+name);rpc.mkdir(mode=0o750);os.chown(rpc,991,1001)
   config=dict(manifest=str(RELEASE/'train/manifest.json'),root=str(private),image=IMAGE,seconds=120,slots=321,caller_uid=1001,socket=str(rpc/'evaluator.sock'))
   write(P/(name+'-config.json'),config)
   proc=start('ldm-copper-eval',PRIVATE_PY,'service',['--config',P/(name+'-config.json')],P/(name+'-service.log'));services[name]=proc
   deadline=time.monotonic()+180
   while not Path(config['socket']).exists():
    assert proc.poll() is None and time.monotonic()<deadline,'service startup failed: '+name
    time.sleep(.2)
   os.chown(config['socket'],991,1001);os.chmod(config['socket'],0o660)
   config['contract']=digest(json.loads((private/'identity.json').read_text()));configs[name]=config
   args=['--public-catalog',publication/'training/catalog.json','--output',R/name,'--evaluation-socket',config['socket'],'--evaluation-contract',config['contract'],'--evaluation-service-uid',991,'--method',method+'_v2','--slots',321,'--seed',seed,'--seconds',120,'--prepare-only']
   prep=run_as('zsgpu',CLIENT_PY,'tasks.copper_inversion.experiments.v2.tune_stochastic',args,S,R/(name+'.log'))
   assert prep.wait()==0,'baseline preparation failed: '+name
   workers[name]=start('zsgpu',CLIENT_PY,'tune',['--run',R/name],R/(name+'.log'))
 plan=dict(release=str(RELEASE),services={n:str(P/n) for n in services},selection_root=str(P/'selection'),source=str(S),client_source=str(S),case_workers=4,discovery_seeds=[42,43,3407],feedback_slots=321,rounds=20,batch=16,solver_seconds=120,development_only=False)
 write(P/'finish-plan.json',plan)
 write(R/'deployment.json',dict(plan,configs=configs,pids={n:p.pid for n,p in workers.items()},service_pids={n:p.pid for n,p in services.items()},supervisor_pid=os.getpid(),ldm_reused_from='/Aidata/shared/ldm/copper-measurement-study-20261025-r3'),uid=1001,gid=1001,mode=0o644)
 while any(p.poll() is None for p in workers.values()):
  assert all(p.poll() in (None,0) for p in workers.values()),'training client failed; reconciliation required'
  status('training',exit_codes={n:p.poll() for n,p in workers.items()});time.sleep(15)
 assert all(p.returncode==0 for p in workers.values())
 rows={n:reconcile(R/n,P/n,cap=321) for n in services};write(P/'budget-reconciliation.json',rows)
 for n,c in configs.items():
  script='from tasks.copper_inversion.v2.service_client import ServiceEvaluator; import json; print(json.dumps(ServiceEvaluator(".",'+repr(c['socket'])+','+repr(c['contract'])+',service_uid=991).request({"operation":"seal_training","contract_digest":'+repr(c['contract'])+'})))'
  result=subprocess.run(['runuser','-u','zsgpu','--',CLIENT_PY,'-c',script],cwd=S,env=env,capture_output=True,text=True,check=True)
  seal=json.loads(result.stdout);assert seal['status']=='training_sealed' and seal['used_slots']==rows[n]['private_sources']
 write(R/'all-training-sealed.json',dict(time=time.time(),groups=list(configs)),uid=1001,gid=1001,mode=0o644)
 finish=start('ldm-copper-eval',PRIVATE_PY,'finish',['--plan',P/'finish-plan.json'],P/'finish.log')
 status('private_validation_then_test',finish_pid=finish.pid)
 assert finish.wait()==0,'private evaluation failed; retain state'
 shutil.copy2(P/'selection/completion.json',R/'completion.json');os.chown(R/'completion.json',1001,1001)
 preserve();status('evaluation_completed_report_pending')
 for proc in services.values():os.killpg(proc.pid,signal.SIGTERM)
except BaseException as exc:
 if not (queue/'PAUSED.json').exists():write(queue/'PAUSED.json',dict(error=str(exc),time=time.time()),uid=1001,gid=1001,mode=0o644)
 status('needs_attention',error=type(exc).__name__+': '+str(exc));raise
