"""Frozen development comparison, disjoint from the diagnostic training parents."""
import ast,hashlib,json,random,shutil,threading,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tasks.copper_inversion.core import execution
from tasks.copper_inversion.core.execution import Execution,freeze_model
from tasks.copper_inversion.v2.data import read_problem
BASE=Path('/Aidata/shared/ldm/copper-baseline-audit-20261027')
DATA=BASE/'development-policy/id/validation'
ROOT=BASE/'actual-time-policy-validation'
# Serialize Docker control operations, not solver execution. Never stop shared containers.
original_docker=execution.docker
control=threading.Lock()
def docker(*a,**k):
 with control:return original_docker(*a,**k)
execution.docker=docker

def main():
 assert json.loads((BASE/'numeric-regression.json').read_text())['status']=='passed'
 assert json.loads((BASE/'policy-acceptance.json').read_text())['status']=='passed'
 for row in json.loads((BASE/'projected-policy/results.json').read_text()):
  assert row['result']['status']=='succeeded' and not row['exit_code']
  assert not row['search']['invalid_reasons'].get('distance reinitialization changed validated occupancy',0)
 ROOT.mkdir(exist_ok=False)
 cases=json.loads((DATA/'manifest.json').read_text())['cases'];assert len(cases)==8
 diagnostic=json.loads((BASE/'diagnostic/plan.json').read_text())['cases']
 assert not set(c['parent_id'] for c in cases)&set(c['parent_id'] for c in diagnostic)
 variants={}
 fixed=(BASE/'stochastic_solver_projected.py').read_text()
 for method in ['ldm','mcmc','acq']:
  for seed in [42,43,3407]:
   name=f'{method}-{seed}';old=(BASE/(name+'.py')).read_text()
   if method=='ldm':
    source=old;fraction=.95*.97 if seed==43 else .94
    if seed==42:source=source.replace('small = int(np.prod(problem["shape"])) < 5000','small = False')
   else:
    config=next(ast.literal_eval(n.value) for n in ast.parse(old).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONFIG' for t in n.targets))
    source='\n'.join('CONFIG='+repr(config) if line.startswith('CONFIG=') else line for line in fixed.splitlines())+'\n';fraction=.75
   # Timer-only adapter: same worker-entry target including imports, one second
   # reserved for output. Cancel each program's old safety fraction. No sleeping.
   source+='\n_native_solve = solve\ndef solve(problem, budget, seed):\n    import os\n    target=float(os.environ["COPPER_WORKLOAD_DEADLINE_MONOTONIC"])-5.0-1.0\n    remaining=max(.01,target-time.monotonic())\n    adjusted=dict(budget);adjusted["wall_time_seconds"]=remaining/'+repr(fraction)+'\n    return _native_solve(problem,adjusted,seed)\n'
   d=ROOT/'sources'/name;d.mkdir(parents=True);(d/'solver.py').write_text(source);variants[name]=hashlib.sha256(source.encode()).hexdigest()
 tasks=[(name,c,seed,seconds) for name in variants for c in cases for seed in [47,48,49] for seconds in [30,90]]
 random.Random(20261027).shuffle(tasks)
 plan={'role':'development_validation_comparison_not_unseen_test','dataset':str(DATA),'cases':cases,'sources':variants,'target_seconds':[30,90],'hard_margin_seconds':5,'output_reserve_seconds':1,'execution_seeds':[47,48,49],'workers':32,'tasks':len(tasks),'time_adapter':'cancel native fractions; shared worker-entry deadline; no padding; same 2CPU/4GiB','tuning':'no parameter tuning; frozen r3 configs, first-order plus public-prior projection; no parameter tuning'}
 (ROOT/'plan.json').write_text(json.dumps(plan,indent=2))
 for c in cases:
  for seed in [47,48,49]:
   for seconds in [30,90]:
    d=ROOT/'inputs'/c['case_id']/str(seed)/str(seconds);d.mkdir(parents=True)
    for n in ['problem.json','problem.npz']:shutil.copy2(DATA/c['public']/n,d/n)
    (d/'budget.json').write_text(json.dumps(dict(seed=seed,wall_time_seconds=seconds+5,cpu_count=2,memory_mb=4096)))
 ex=Execution(ROOT/'jobs','ldm-copper-baselines:20261024')
 def run(task):
  name,c,seed,seconds=task;cid=c['case_id'];inputs=ROOT/'inputs'/cid/str(seed)/str(seconds)
  identity=dict(source=variants[name],case=cid,seed=seed,target_seconds=seconds)
  job,rc=ex.run(identity,phase='solver',inputs=inputs,source=ROOT/'sources'/name,timeout=seconds+5,cpus=2,memory_mb=4096,science_module='v2')
  row=dict(group=name,case_id=cid,parent_id=c['parent_id'],modality=c['modality'],execution_seed=seed,target_seconds=seconds,wall_seconds=rc['wall_seconds'],worker_seconds=rc['workload_resources']['worker_elapsed_seconds'],cpu_seconds=rc['workload_resources']['cpu_seconds_through_reap'],exit_code=rc['exit_code'],reason=rc['reason'])
  if rc['exit_code'] or rc['reason'] or rc['oom']:raise RuntimeError(str(row))
  model=ROOT/'models'/name/cid/str(seed)/str(seconds);sha=freeze_model(job/'output/model.npy',model,read_problem(inputs)['shape'])
  # Truth is development-only and mounted solely in source-free scorer.
  sj,sr=ex.run(dict(identity,model=sha),phase='scorer',inputs=inputs,model=model,truth=DATA/c['truth'],timeout=30,science_module='v2')
  result=json.loads((sj/'output/result.json').read_text());assert result['status']=='succeeded',(name,result)
  row['metrics']=result['metrics'];f=job/'output/search.json';row['search']=json.loads(f.read_text()) if f.exists() else None
  d=ROOT/'rows';d.mkdir(exist_ok=True);(d/(name+'-'+cid+'-'+str(seed)+'-'+str(seconds)+'.json')).write_text(json.dumps(row))
  print(json.dumps({k:v for k,v in row.items() if k not in ['search','metrics']}),flush=True);return row
 with ThreadPoolExecutor(max_workers=32) as pool:rows=list(pool.map(run,tasks))
 (ROOT/'results.json').write_text(json.dumps(rows));print('COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
