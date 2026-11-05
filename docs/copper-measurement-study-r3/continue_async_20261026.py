"""Audited Docker-control-plane continuation; preserve science and paid sources."""
import os,json,time,subprocess,fcntl,hashlib,shutil,signal
from pathlib import Path
R=Path('/Aidata/shared/ldm/copper-measurement-study-20261025-r3');S=R/'source'
P=Path('/Aidata/shared/ldm/copper-v2-evaluator/runs')/R.name
OPS=Path('/Aidata/shared/ldm/copper-permanent-repair-20261025')
E=R/'controller-epochs/admission-deadline-r1';D=R/'controller-epochs/async-transport-20261026'
from tasks.copper_inversion.experiments.v2.launch_formal import CLIENT_PY,PRIVATE_PY,write
from tasks.copper_inversion.experiments.v2.reconcile_discovery import reconcile
from tasks.copper_inversion.core.execution import digest,source_identity
D.mkdir(exist_ok=True)
lock=(E/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (D/'started.json').exists(),'one-time continuation already used'
old=json.loads((R/'scheduling-epoch.json').read_text())
assert all(not Path('/proc/'+str(pid)).exists() for pid in [old['supervisor_pid'],*old['pids'].values()])
assert (R/'queue/PAUSED.json').exists()
pauses={str(p.relative_to(R/'queue')):p.read_bytes() for p in (R/'queue').rglob('PAUSED.json')}
assert set(pauses)=={'PAUSED.json','startup/PAUSED.json'}
startup_fault=json.loads(pauses['startup/PAUSED.json'])
assert startup_fault['pid']==3921616 and startup_fault['time']==1792957458.3824847
from pause_reconciliation import archive_pauses
epoch=json.loads((E/'epoch.json').read_text());assert source_identity()==epoch['scientific_identity']
for key in ('interface','client','runtime','baseline'):
 c=epoch[key];assert hashlib.sha256(Path(c['path']).read_bytes()).hexdigest()==c['sha256']
assert json.loads((OPS/'control-probe-20261026.json').read_text())['status']=='passed'
load=json.loads((OPS/'load-control96-20261026/completion.json').read_text());assert load['status']=='passed' and load['jobs']==1308 and load['peak_active']==96 and load['plan']['startup_slots']==1
qualified=json.loads((OPS/'async-qualification-20261026.json').read_text())
assert qualified['stage']=='passed_pending_formal_reconciliation' and qualified['private']['receipt_count']==102
joint=json.loads((OPS/'async-joint-load-20261026/completion.json').read_text())
assert joint['status']=='passed' and joint['jobs']==1308 and joint['peak_active']==96
for component,expected in {'/Aidata/shared/ldm/copper-permanent-repair-20261025/async_transport_qualified.py': 'cc295e18f1723bb314226925d4b6cbe2534a6c5bda2ad9e3a43cbf5fde0c70f5', '/Aidata/shared/ldm/copper-permanent-repair-20261025/async_repaired_runtime.py': '1f4a265477dd00bcb3b0448fb3f66002943c4e74fd7dd57163772347201043dd'}.items():
 assert hashlib.sha256(Path(component).read_bytes()).hexdigest()==expected
write(D/'transport-version.json',{'module': '/Aidata/shared/ldm/copper-permanent-repair-20261025/async_transport_qualified.py', 'sha256': 'cc295e18f1723bb314226925d4b6cbe2534a6c5bda2ad9e3a43cbf5fde0c70f5', 'runtime': '/Aidata/shared/ldm/copper-permanent-repair-20261025/async_repaired_runtime.py', 'runtime_sha256': '1f4a265477dd00bcb3b0448fb3f66002943c4e74fd7dd57163772347201043dd'},uid=0,gid=0,mode=0o644)
d=json.loads((E/'deployment-before.json').read_text())
# Every unfinished job is reconciled against Docker before permitting dispatch.
incident=json.loads((OPS/'async-recovery-inventory.json').read_text())
for item in incident:
 record=json.loads(Path(item['job']).read_text())
 q=subprocess.run(['docker','inspect','--format','{{json .State}}',item['name']],capture_output=True,text=True,timeout=40)
 if q.returncode:
  assert not record['started'],'previously dispatched container disappeared'
 else:
  state=json.loads(q.stdout);assert state['Status'] in ('created','exited')
  assert not state['Running']
# Snapshot all existing ledger/receipt/result bytes before service takeover.
files={}
for root in [R,P]:
 for name in d['services']:
  for pattern in ['budget.json','jobs/*/receipt.json','evaluations/*/result.json']:
   for p in (root/name).glob(pattern):files[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
write(D/'retained-before.json',dict(files=files),uid=0,gid=0,mode=0o644)
for name,pid in old['service_pids'].items():
 proc=Path('/proc')/str(pid)
 if proc.exists():
  cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  assert str(P/(name+'-config.json')) in cmd and 'repaired_runtime.py' in cmd
  os.killpg(pid,signal.SIGTERM)
for pid in old['service_pids'].values():
 deadline=time.monotonic()+30
 while Path('/proc/'+str(pid)).exists():
  assert time.monotonic()<deadline,'old service did not exit'
  time.sleep(.2)
for path,expected in files.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
policy=json.loads((R/'scheduling-policy.json').read_text());write(D/'policy-before.json',policy,uid=0,gid=0,mode=0o644)
policy['startup_slots']=1
policy['global_jobs']=96
policy['policy_sha256']=digest({k:v for k,v in policy.items() if k!='policy_sha256'})
write(R/'scheduling-policy.json',policy,uid=1001,gid=1001,mode=0o644)
write(D/'started.json',dict(pid=os.getpid(),time=time.time(),controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),retained_files=len(files),old_epoch=old,policy=policy),uid=0,gid=0,mode=0o644)
services={};workers={}
def start(user,python,mode,args,log,key=None):
 env={**os.environ,'PYTHONPATH':str(S),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'};env.pop('LLM_API_KEY',None)
 if key:env['LLM_API_KEY']=key
 with log.open('ab') as f:
  return subprocess.Popen(['runuser','-u',user,'--',python,str(OPS/'async_repaired_runtime.py'),'--epoch',str(E/'epoch.json'),'--policy',str(R/'scheduling-policy.json'),mode,*map(str,args)],cwd=S,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
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
 archive_pauses(R/'queue',D/'reconciled-pauses',pauses)
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
 if not (R/'queue/PAUSED.json').exists():
  write(R/'queue/PAUSED.json',dict(reason=type(exc).__name__+': '+str(exc),epoch=epoch['id'],time=time.time()),uid=1001,gid=1001,mode=0o644)
 status('repair_continuation_requires_attention',error=type(exc).__name__+': '+str(exc));raise
