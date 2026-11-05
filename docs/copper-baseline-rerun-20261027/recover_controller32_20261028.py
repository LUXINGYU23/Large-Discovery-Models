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
E=O/'recovery-20261028-0748';E.mkdir()
write(E/'started.json',dict(pid=os.getpid(),time=time.time()))
frozen=json.loads((O/'frozen-files.json').read_text())
for path,sha in frozen.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
old=json.loads((O/'r3-preservation-before.json').read_text())
def preserve():
 for path,sha in old.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
preserve()
old_queue=R/'queue';queue=R/'queue32-20261028';policy_path=R/'scheduling-policy32-20261028.json'
env={**os.environ,'PYTHONPATH':str(S),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'};env.pop('LLM_API_KEY',None)
def status(stage,**kw):write(R/'progress.json',dict(stage=stage,updated_at=time.time(),supervisor_pid=os.getpid(),recovery=str(E),**kw),uid=1001,gid=1001,mode=0o644)
def checked(args,log,timeout=3600):
 with log.open('ab') as f:subprocess.run(args,cwd=S,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=timeout)
def start(user,python,mode,args,log):
 with log.open('ab') as f:return subprocess.Popen(['runuser','-u',user,'--',python,str(O/'runtime.py'),'--policy',str(policy_path),mode,*map(str,args)],cwd=S,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
services={};workers={}
try:
 assert '10 passed' in (O/'recovery-tests-20261028.log').read_text()
 deployment=json.loads((R/'recovery-deployment.json').read_text());configs=json.loads((R/'deployment.json').read_text())['configs']
 for pid in [deployment['supervisor_pid'],*deployment['pids'].values()]:assert not Path(f'/proc/{pid}').exists(),('old writer still exists',pid)
 assert (old_queue/'PAUSED.json').exists() and (old_queue/'startup/PAUSED.json').exists()
 # Only a separate development load root is touched during qualification.
 status('recovery_qualification')
 # Reuse the just-passed rs1621 qualification; no runtime changes.
 assert json.loads((O/'gate32-20261028.json').read_text())['stage']=='passed_pending_recovery_controller'
 q=json.loads((O/'q32f28/completion.json').read_text());assert q['status']=='passed' and q['jobs']==1308 and q['peak_active']==32 and not q['failures']
 # Confirm every formal orphan is either never dispatched or terminal with authenticated counters.
 from tasks.copper_inversion.core.execution import authenticated_resources
 inventory=[]
 for jpath in P.glob('*/jobs/*/job.json'):
  if (jpath.parent/'receipt.json').exists():continue
  j=json.loads(jpath.read_text());r=subprocess.run(['docker','inspect',j['name']],capture_output=True,text=True,timeout=40)
  if r.returncode:
   assert 'no such object' in r.stderr.lower() and not j['started'] and not j.get('container_id')
   inventory.append(dict(path=str(jpath),state='never_started_missing'))
  else:
   d=json.loads(r.stdout)[0]
   assert any(m['Source']==str(jpath.parent/'output') for m in d['Mounts'])
   if d['State']['Status']=='created':
    assert not j['started'] and not j.get('container_id')
    inventory.append(dict(path=str(jpath),state='created_reuse',container_id=d['Id']))
   else:
    assert d['Id']==j['container_id'] and d['State']['Status']=='exited' and d['State']['ExitCode'] in (0,137)
    log=subprocess.run(['docker','logs',j['name']],capture_output=True,text=True,timeout=40,check=True)
    a=authenticated_resources(log.stdout,j['resource_key']);assert a and a['workload_complete'] and a['exit_code']==d['State']['ExitCode']
    inventory.append(dict(path=str(jpath),state='terminal_reuse',container_id=d['Id'],resources=a))
 assert len(inventory)==22 and sum(x['state']=='terminal_reuse' for x in inventory)==20
 assert sum(x['state']=='created_reuse' for x in inventory)==1
 assert sum(x['state']=='never_started_missing' for x in inventory)==1
 # Revalidate the preflight bytes before stopping any formal service.
 before=json.loads((O/'recovery-preflight-20261028-0718/history.json').read_text())
 for path,sha in before.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
 write(E/'orphan-inventory.json',inventory)
 # Stop only the six recorded service process groups, after command and group ownership checks.
 for n,pid in deployment['service_pids'].items():
  cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
  assert str(O/'runtime.py') in cmd and str(P/(n+'-config.json')) in cmd and os.getpgid(pid)==pid
 for n,pid in deployment['service_pids'].items():os.killpg(pid,signal.SIGTERM)
 deadline=time.monotonic()+30
 while any(Path(f'/proc/{pid}').exists() and Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z' for pid in deployment['service_pids'].values()):
  assert time.monotonic()<deadline;time.sleep(.2)
 # No process still references the exact formal runtime/policy.
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:cmd=(proc/'cmdline').read_bytes()
  except OSError:continue
  assert not (str(O/'runtime.py').encode() in cmd and str(R).encode() in cmd),('remaining writer',proc.name)
 # Preserve all existing scientific bytes, including original ledgers and incomplete jobs.
 hashes={}
 for root in [*(R/n for n in configs),*(P/n for n in configs)]:
  for p in root.rglob('*'):
   if p.is_file() and not p.is_symlink():hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 write(E/'history-before.json',hashes)
 for n in configs:
  for scope,root in [('client',R),('private',P)]:
   src=root/n/'budget.json';shutil.copy2(src,E/(n+'-'+scope+'-budget-before.json'))
   assert json.loads(src.read_text())['counters']['algorithm_feedback_slots']==(129 if n.endswith('3407') else 113)
 for path,sha in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
 preserve()
 # Pause files are archived, never discarded. Existing receipts and source identities remain in place.
 for label,path in [('main',old_queue/'PAUSED.json'),('startup',old_queue/'startup/PAUSED.json')]:shutil.copy2(path,E/(label+'-PAUSED.json'))
 shutil.copy2(R/'scheduling-policy.json',E/'previous-policy.json')
 queue.mkdir();queue.chmod(0o777)
 for directory,count in [(queue,32),(queue/'startup',1)]:
  directory.mkdir(exist_ok=True);directory.chmod(0o777)
  for i in range(count):(directory/f'{i:03}.lock').touch();(directory/f'{i:03}.lock').chmod(0o666)
 runtime=S/'tasks/copper_inversion/experiments/v2/parallel_runtime.py'
 policy=dict(queue_root=str(queue),global_jobs=32,interface_slots=8,jobs_per_process=32,startup_slots=1,runtime_sha256=hashlib.sha256(runtime.read_bytes()).hexdigest())
 policy['policy_sha256']=digest(policy);write(policy_path,policy,uid=0,gid=0,mode=0o644)
 for n,c in configs.items():
  sock=Path(c['socket'])
  if sock.exists():sock.unlink()
  proc=start('ldm-copper-eval',PRIVATE_PY,'service',['--config',P/(n+'-config.json')],P/(n+'-service.log'));services[n]=proc
  deadline=time.monotonic()+180
  while not sock.exists():
   assert proc.poll() is None and time.monotonic()<deadline;time.sleep(.2)
  os.chown(sock,991,1001);os.chmod(sock,0o660)
 # All six services are ready before any client resumes.
 for n in configs:workers[n]=start('zsgpu',CLIENT_PY,'tune',['--run',R/n],R/(n+'.log'))
 write(R/'recovery-deployment.json',dict(supervisor_pid=os.getpid(),pids={n:p.pid for n,p in workers.items()},service_pids={n:p.pid for n,p in services.items()},epoch=str(E),history_files=len(hashes),policy=str(policy_path),queue=str(queue)),uid=1001,gid=1001,mode=0o644)
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
