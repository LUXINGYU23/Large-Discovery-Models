"""Single-use, audited r3 takeover; retain original workers and private services."""
import os,json,time,subprocess,fcntl,hashlib,shutil,signal
from pathlib import Path
R=Path('/Aidata/shared/ldm/copper-measurement-study-20261025-r3')
P=Path('/Aidata/shared/ldm/copper-v2-evaluator/runs')/R.name
S=R/'source'
from tasks.copper_inversion.experiments.v2.launch_formal import run_as,CLIENT_PY,PRIVATE_PY,write
from tasks.copper_inversion.experiments.v2.reconcile_discovery import reconcile
lock=(R/'monitoring/ldm43-recovery.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
d=json.loads((R/'deployment.json').read_text());c=json.loads((R/'ldm-43/config.json').read_text())
assert not Path('/proc/'+str(d['pids']['ldm-43'])).exists()
prior=json.loads((R/'monitoring/ldm43-recovery-controller.json').read_text())
assert not Path('/proc/'+str(prior['supervisor_pid'])).exists() and not Path('/proc/'+str(prior['replacement_client_pid'])).exists()
assert not (R/'monitoring/ldm43-recovery-attempt2.json').exists()
write(R/'monitoring/ldm43-recovery-attempt1.json',prior,uid=1001,gid=1001,mode=0o644)
assert not (R/'queue/PAUSED.json').exists()
key=Path('/Aidata/shared/ldm/nucleobench-one-plus-three-20261023/assets/deepseek.key').read_text().strip();assert key
names=('public_catalog','evaluation_socket','evaluation_contract','evaluation_service_uid','embedding_model_path','embedding_device','iterations','seed','candidates_per_researcher','evaluations_per_round','solver_seconds','harness_seconds','evaluator_image','sidecar_image','harness_cache_dir')
a=[]
for name in names:a+=['--'+name.replace('_','-'),str(c[name])]
a+=['--out-dir',str(R/'ldm-43'),'--resume']
env={**os.environ,'PYTHONPATH':str(S),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','LLM_API_KEY':key}
with (R/'ldm-43.log').open('ab') as log:
 p=subprocess.Popen(['runuser','-u','zsgpu','--',CLIENT_PY,'/Aidata/shared/ldm/copper-audit-fixes-20261024/replay_r3_exact_inputs.py','--policy',str(R/'scheduling-policy.json'),'ldm',*a],cwd=S,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
del key;del env
state=dict(supervisor_pid=os.getpid(),replacement_client_pid=p.pid,original_supervisor=d['supervisor_pid'],source_unchanged=True,budget_reset=False,started_at=time.time())
write(R/'monitoring/ldm43-recovery-attempt2.json',state,uid=1001,gid=1001,mode=0o644)
write(R/'monitoring/ldm43-recovery-controller.json',state,uid=1001,gid=1001,mode=0o644)
def status(stage,**kw):write(R/'monitoring/recovery-progress.json',dict(stage=stage,updated_at=time.time(),**state,**kw),uid=1001,gid=1001,mode=0o644)
status('resuming_original_submission')
while p.poll() is None:
 status('training_recovery',client_exit=None);time.sleep(15)
if p.returncode:
 status('needs_reconciliation',client_exit=p.returncode);raise SystemExit(p.returncode)
# Original supervisor keeps ownership of the eight unaffected children. It exits
# with reconciliation required because its old LDM43 child failed. Never race it.
while Path('/proc/'+str(d['supervisor_pid'])).exists():
 status('waiting_original_workers',client_exit=0);time.sleep(15)
for name,pid in d['pids'].items():assert not Path('/proc/'+str(pid)).exists(), name+' still running'
rows={name:reconcile(R/name,service,cap=321) for name,service in d['services'].items()}
write(P/'budget-reconciliation.json',rows)
for name,config in d['configs'].items():
 script='from tasks.copper_inversion.v2.service_client import ServiceEvaluator; import json; print(json.dumps(ServiceEvaluator(".",'+repr(config['socket'])+','+repr(config['contract'])+',service_uid=991).request({"operation":"seal_training","contract_digest":'+repr(config['contract'])+'})))'
 v=subprocess.run(['runuser','-u','zsgpu','--',CLIENT_PY,'-c',script],cwd=S,env={**os.environ,'PYTHONPATH':str(S)},capture_output=True,text=True,check=True)
 seal=json.loads(v.stdout);assert seal['status']=='training_sealed' and seal['used_slots']==rows[name]['private_sources']
f=run_as('ldm-copper-eval',PRIVATE_PY,'tasks.copper_inversion.experiments.v2.parallel_runtime',['--policy',R/'scheduling-policy.json','finish','--plan',P/'finish-plan.json'],S,P/'finish.log')
status('private_validation_test',finish_pid=f.pid)
if f.wait():status('private_evaluation_requires_reconciliation',finish_exit=f.returncode);raise SystemExit(1)
shutil.copy2(P/'selection/completion.json',R/'completion.json');os.chown(R/'completion.json',1001,1001)
for pid in d['service_pids'].values():os.killpg(pid,signal.SIGTERM)
write(R/'progress.json',dict(stage='completed',recovery_supervisor=os.getpid()),uid=1001,gid=1001,mode=0o644)
status('completed')
