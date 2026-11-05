"""Isolated public development audit. Never reads r3 evaluation inputs or truth."""
from pathlib import Path
import json, argparse, shutil, hashlib
from concurrent.futures import ThreadPoolExecutor
from tasks.copper_inversion.core.execution import Execution,freeze_model
from tasks.copper_inversion.v2.data import read_problem
BASE=Path('/Aidata/shared/ldm/copper-baseline-audit-20261027')
DATA=BASE/'development-policy/id/train'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage');ap.add_argument('--cases',type=int,default=4);ap.add_argument('--seconds',type=int,default=120);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
 root=BASE/args.stage;root.mkdir(exist_ok=False)
 cases=sorted(json.loads((DATA/'manifest.json').read_text())['cases'],key=lambda x:(x['parent_id'],x['modality']))[:args.cases]
 variants=sorted((BASE/'variants'/args.stage).glob('*.py'))
 plan={'role':'public_development_only','dataset':str(DATA),'cases':cases,'sources':{f.stem:hashlib.sha256(f.read_bytes()).hexdigest() for f in variants},'seconds':args.seconds,'seeds':[47],'workers':args.workers}
 (root/'plan.json').write_text(json.dumps(plan,indent=2))
 ex=Execution(root/'jobs','ldm-copper-baselines:20261024')
 for c in cases:
  d=root/'inputs'/c['case_id'];d.mkdir(parents=True)
  for name in ['problem.json','problem.npz']:shutil.copy2(DATA/c['public']/name,d/name)
  (d/'budget.json').write_text(json.dumps(dict(seed=47,wall_time_seconds=args.seconds,cpu_count=2,memory_mb=4096)))
 for f in variants:
  d=root/'sources'/f.stem;d.mkdir(parents=True);shutil.copy2(f,d/'solver.py')
 def one(task):
  f,c=task;cid=c['case_id'];inputs=root/'inputs'/cid;name=f.stem
  job,rc=ex.run(dict(stage=args.stage,source=plan['sources'][name],case=cid,seed=47),phase='solver',inputs=inputs,source=root/'sources'/name,timeout=args.seconds,cpus=2,memory_mb=4096,science_module='v2')
  row={'variant':name,'case':cid,'exit_code':rc['exit_code'],'reason':rc['reason'],'solver_seconds':rc['wall_seconds']}
  if not(rc['exit_code'] or rc['reason'] or rc['oom']):
   model=root/'models'/name/cid;sha=freeze_model(job/'output/model.npy',model,read_problem(inputs)['shape'])
   sj,sr=ex.run(dict(model=sha,case=cid,stage=args.stage),phase='scorer',inputs=inputs,model=model,timeout=30,science_module='v2')
   row['result']=json.loads((sj/'output/result.json').read_text())
   row['search']=json.loads((job/'output/search.json').read_text()) if (job/'output/search.json').exists() else None
  (root/(name+'-'+cid+'.json')).write_text(json.dumps(row,indent=2));print(json.dumps({k:v for k,v in row.items() if k not in ['search','result']}),flush=True);return row
 with ThreadPoolExecutor(max_workers=args.workers) as pool: rows=list(pool.map(one,[(f,c) for f in variants for c in cases]))
 (root/'results.json').write_text(json.dumps(rows,indent=2));print('completed',len(rows),flush=True)
if __name__=='__main__':main()
