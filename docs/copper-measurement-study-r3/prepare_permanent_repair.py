"""Prepare an explicit continuation epoch; never relabel historical source bytes."""
from pathlib import Path
import json,hashlib,os,signal,subprocess,sys,shutil,time
R=Path('/Aidata/shared/ldm/copper-measurement-study-20261025-r3');OPS=Path('/Aidata/shared/ldm/copper-permanent-repair-20261025')
P=Path('/Aidata/shared/ldm/copper-v2-evaluator/runs')/R.name
sys.path.insert(0,str(R/'source'))
from tasks.copper_inversion.core.execution import digest,source_identity
from tasks.copper_inversion.experiments.v2.tune_stochastic import source_for
D=json.loads((R/'deployment.json').read_text());rec=json.loads((R/'monitoring/ldm43-recovery-controller.json').read_text())
assert (R/'queue/PAUSED.json').exists()
for pid in [*D['pids'].values(),D['supervisor_pid'],rec['supervisor_pid'],rec['replacement_client_pid']]:assert not Path('/proc/'+str(pid)).exists(),f'writer still alive {pid}'
# Only this study's job roots; never inspect or stop other users' containers.
prefixes=['ldm-copper-'+digest(str(root/'jobs'))[:10]+'-' for root in [*(R/n for n in D['services']),*(P/n for n in D['services'])]]
active=subprocess.run(['docker','ps','--format','{{.Names}}'],capture_output=True,text=True,check=True).stdout.splitlines()
assert not [n for n in active if any(n.startswith(p) for p in prefixes)],'existing study jobs must drain'
E=R/'controller-epochs/admission-deadline-r1';E.mkdir(parents=True,exist_ok=False)
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,x):p.write_text(json.dumps(x,indent=2));p.chmod(0o644)
components={}
for key,name in [('interface','interface_checks.py'),('client','client.py'),('runtime','repaired_runtime.py'),('baseline','stochastic_solver.py')]:
 target=E/name;shutil.copy2(OPS/name,target);os.chown(target,0,0);target.chmod(0o644);components[key]=dict(path=str(target),sha256=h(target))
old=(R/'source/tasks/copper_inversion/experiments/v2/stochastic_solver.py').read_text();new=(E/'stochastic_solver.py').read_text();maps={};source_dir=E/'baseline-sources';source_dir.mkdir()
for name in D['services']:
 if name.startswith('ldm'):continue
 root=R/name;plan=json.loads((root/'plan.json').read_text());ledger=json.loads((root/'budget.json').read_text())
 tag='copper:v2:scientific-source:';charged={k[len(tag):] for k in ledger['metadata']['cumulative_usage'] if k.startswith(tag)}
 rows=[];retained=set()
 for i,params in enumerate(plan['parameters']):
  original=source_for(old,params).encode();original_sha=hashlib.sha256(original).hexdigest();paid=original_sha in charged
  body=original if paid else source_for(new,params).encode();sha=hashlib.sha256(body).hexdigest()
  if paid:retained.add(sha)
  receipt=root/'tuning-receipts'/f'{i:03}.json'
  if receipt.exists():assert json.loads(receipt.read_text())['source_sha256']==sha
  target=source_dir/(sha+'.py')
  if target.exists():assert target.read_bytes()==body
  else:target.write_bytes(body);target.chmod(0o444)
  rows.append(dict(index=i,parameters=params,source_file=str(target),source_sha256=sha,prior_source_sha256=original_sha,already_charged=paid,template_version='original' if paid else 'signal-cleanup-r2'))
 assert retained==charged and len(rows)==321
 f=E/(name+'-sources.json');put(f,rows);maps[name]=dict(path=str(f),sha256=h(f),retained_charged=len(retained))
 put(E/(name+'-budget-before.json'),ledger)
epoch=dict(id='admission-deadline-r1',created_at=time.time(),user_authorized_continuation=True,scientific_identity=source_identity(),sidecar_image='sha256:6c0892c0df8178399c0fec16456720f489493b32c135561c18399f8290c81bd3',baseline_maps=maps,**components)
put(E/'epoch.json',epoch)
put(E/'deployment-before.json',D)
print(json.dumps(dict(epoch=str(E/'epoch.json'),maps={n:x['retained_charged'] for n,x in maps.items()})))
