"""One audited continuation after permanent repair qualification. Never reset budgets."""
import os,json,time,subprocess,fcntl,hashlib,shutil,signal
from pathlib import Path
R=Path('/Aidata/shared/ldm/copper-measurement-study-20261025-r3'); S=R/'source'
P=Path('/Aidata/shared/ldm/copper-v2-evaluator/runs')/R.name
OPS=Path('/Aidata/shared/ldm/copper-permanent-repair-20261025')
E=R/'controller-epochs/admission-deadline-r1'; epoch=json.loads((E/'epoch.json').read_text())
from tasks.copper_inversion.experiments.v2.launch_formal import CLIENT_PY,PRIVATE_PY,write
from tasks.copper_inversion.experiments.v2.reconcile_discovery import reconcile
from tasks.copper_inversion.core.execution import digest,source_identity
lock=(E/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (E/'started.json').exists(),'one-time entry already used'
assert (R/'queue/PAUSED.json').exists()
assert source_identity()==epoch['scientific_identity']
for key in ('interface','client','runtime','baseline'):
 c=epoch[key];assert hashlib.sha256(Path(c['path']).read_bytes()).hexdigest()==c['sha256']
accept=json.loads((OPS/'repair-acceptance.json').read_text());assert accept['status']=='passed' and accept['epoch_sha256']==hashlib.sha256((E/'epoch.json').read_bytes()).hexdigest()
for name,count in [('load-small-v2',12),('load-full-v2',1308)]:
 result=json.loads((OPS/name/'completion.json').read_text());assert result['status']=='passed' and result['jobs']==count
 plan=json.loads((OPS/name/'plan.json').read_text());assert plan['template_sha256']==epoch['baseline']['sha256']
retained=json.loads((OPS/'retained-results-before.json').read_text())['files']
for path,expected in retained.items():
 assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,'retained evidence changed: '+path
d=json.loads((E/'deployment-before.json').read_text())
for pid in [*d['pids'].values(),d['supervisor_pid']]:assert not Path('/proc/'+str(pid)).exists()
prefixes=['ldm-copper-'+digest(str(root/'jobs'))[:10]+'-' for root in [*(R/n for n in d['services']),*(P/n for n in d['services'])]]
active=subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()
assert not [n for n in active if any(n.startswith(p) for p in prefixes)]
# Stop only verified old service process groups; do not touch shared Docker.
for name,pid in d['service_pids'].items():
 proc=Path('/proc')/str(pid)
 if proc.exists():
  cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  assert str(P/(name+'-config.json')) in cmd and 'parallel_runtime' in cmd
  os.killpg(pid,signal.SIGTERM)
for pid in d['service_pids'].values():
 deadline=time.monotonic()+30
 while Path('/proc/'+str(pid)).exists():
  assert time.monotonic()<deadline,'old service did not stop'
  time.sleep(.2)
write(E/'started.json',dict(pid=os.getpid(),time=time.time(),controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),retained_files_verified=len(retained),epoch_sha256=hashlib.sha256((E/'epoch.json').read_bytes()).hexdigest()),uid=0,gid=0,mode=0o644)
services={};workers={}
def start(user,python,mode,args,log,key=None):
 env={**os.environ,'PYTHONPATH':str(S),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'};env.pop('LLM_API_KEY',None)
 if key:env['LLM_API_KEY']=key
 with log.open('ab') as f:
  return subprocess.Popen(['runuser','-u',user,'--',python,epoch['runtime']['path'],'--epoch',str(E/'epoch.json'),'--policy',str(R/'scheduling-policy.json'),mode,*map(str,args)],cwd=S,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
def status(stage,**kw):
 write(R/'progress.json',dict(stage=stage,updated_at=time.time(),epoch=epoch['id'],supervisor_pid=os.getpid(),exit_codes={n:p.poll() for n,p in workers.items()},**kw),uid=1001,gid=1001,mode=0o644)
try:
 for name,c in d['configs'].items():
  socket=Path(c['socket']);socket.unlink(missing_ok=True)
  p=start('ldm-copper-eval',PRIVATE_PY,'service',['--config',P/(name+'-config.json')],P/(name+'-service.log'));services[name]=p
  deadline=time.monotonic()+120
  while not socket.exists():
   assert p.poll() is None and time.monotonic()<deadline,'service startup failed '+name
   time.sleep(.2)
  os.chown(socket,991,1001);socket.chmod(0o660)
  assert digest(json.loads((P/name/'identity.json').read_text()))==c['contract']
 # Archive the pause only after all services and immutable identities are checked.
 shutil.move(R/'queue/PAUSED.json',E/'pause-before-continuation.json')
 key=Path('/Aidata/shared/ldm/nucleobench-one-plus-three-20261023/assets/deepseek.key').read_text().strip();assert key
 names=('public_catalog','evaluation_socket','evaluation_contract','evaluation_service_uid','embedding_model_path','embedding_device','iterations','seed','candidates_per_researcher','evaluations_per_round','solver_seconds','harness_seconds','evaluator_image','sidecar_image','harness_cache_dir')
 for name in d['services']:
  if name.startswith('ldm'):
   c=json.loads((R/name/'config.json').read_text());args=[]
   for k in names:args+=['--'+k.replace('_','-'),str(c[k])]
   args+=['--out-dir',R/name,'--resume'];mode='ldm'
  else:mode='tune';args=['--run',R/name]
  workers[name]=start('zsgpu',CLIENT_PY,mode,args,R/(name+'.log'),key if mode=='ldm' else None)
 del key
 write(R/'scheduling-epoch.json',dict(epoch=str(E/'epoch.json'),policy=json.loads((R/'scheduling-policy.json').read_text()),pids={n:p.pid for n,p in workers.items()},service_pids={n:p.pid for n,p in services.items()},supervisor_pid=os.getpid()),uid=1001,gid=1001,mode=0o644)
 while any(p.poll() is None for p in workers.values()):
  if any(p.poll() not in (None,0) for p in workers.values()):raise RuntimeError('client failure; stop new dispatch for diagnosis')
  status('training');time.sleep(15)
 assert all(p.returncode==0 for p in workers.values())
 rows={n:reconcile(R/n,s,cap=321) for n,s in d['services'].items()};write(P/'budget-reconciliation.json',rows)
 for n,c in d['configs'].items():
  script='from tasks.copper_inversion.v2.service_client import ServiceEvaluator; import json; print(json.dumps(ServiceEvaluator(".",'+repr(c['socket'])+','+repr(c['contract'])+',service_uid=991).request({"operation":"seal_training","contract_digest":'+repr(c['contract'])+'})))'
  result=subprocess.run(['runuser','-u','zsgpu','--',CLIENT_PY,'-c',script],cwd=S,env={**os.environ,'PYTHONPATH':str(S)},capture_output=True,text=True,check=True)
  seal=json.loads(result.stdout);assert seal['status']=='training_sealed' and seal['used_slots']==rows[n]['private_sources']
 f=start('ldm-copper-eval',PRIVATE_PY,'finish',['--plan',P/'finish-plan.json'],P/'finish.log');status('private_validation_test',finish_pid=f.pid)
 assert f.wait()==0,'private evaluation requires reconciliation'
 shutil.copy2(P/'selection/completion.json',R/'completion.json');os.chown(R/'completion.json',1001,1001);status('completed')
 for p in services.values():os.killpg(p.pid,signal.SIGTERM)
except BaseException as exc:
 write(R/'queue/PAUSED.json',dict(reason=type(exc).__name__+': '+str(exc),epoch=epoch['id'],time=time.time()),uid=1001,gid=1001,mode=0o644)
 status('repair_continuation_requires_attention',error=type(exc).__name__+': '+str(exc));raise
