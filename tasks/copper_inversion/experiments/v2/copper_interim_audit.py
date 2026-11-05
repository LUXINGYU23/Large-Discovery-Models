import json,time,sys,statistics,hashlib
from pathlib import Path
R=Path('/Aidata/shared/ldm/copper-v2-evaluator/runs/copper-formal-study-20261024-r1/interim-input-20261024')
P=Path('/Aidata/shared/ldm/copper-v2-evaluator/runs/copper-formal-study-20261024-r1')
D=Path('/Aidata/shared/ldm/copper-v2-evaluator/datasets/formal-20261024-r1')
O=P/'interim-audit-20261024'
from tasks.copper_inversion.v2.training_receipts import eligible
from tasks.copper_inversion.v2.selection import freeze,run
IMAGE='sha256:f6856385a0da9444b1e5edb5048b2f6081593fe568c3473c2fda524d0c4489f5'
def main():
 if sys.argv[1]=='freeze':
  O.mkdir(exist_ok=False); series={}; latest={}
  for method in ['ldm','mcmc','acq']:
   for seed in [42,43,3407]:
    name=f'{method}-{seed}';r=R/name;rows=[]
    if method=='ldm':
     b=json.loads((r/'budget.json').read_text());assert b['counters'].get('interface_rejected_sources',0)==(1 if seed==42 else 0)
     for i,o in enumerate(json.loads((r/'checkpoint.json').read_text())['state']['observations']):
      if seed==42 and i==17:rows.append(None)
      rows.append({'source_sha256':o['candidate']['canonical_key'],'metrics':o['evaluation']['metrics']})
    else:
     receipts={int(f.stem):json.loads(f.read_text()) for f in (r/'tuning-receipts').glob('*.json')};i=0
     while i in receipts:
      rows.append(receipts[i])
      if i==0:
       files=list((r/'serialization-interrupted-recovery').glob('*.json'));assert len(files)==1
       extra=json.loads(files[0].read_text());assert extra['status']=='succeeded';rows.append(extra)
      i+=1
    series[name]=rows;latest[name]=min((x for x in rows if x and x.get('metrics')),key=lambda x:x['metrics']['loss_total'])
  budget=min(map(len,series.values()));plan={'created':time.time(),'common_charged_budget':budget,'scope':'interim snapshot; active search receives no heldout feedback; test now consumed','latest_training':latest,'groups':{}}
  for name,rows in series.items():
   valid=[x for x in rows[:budget] if x and x.get('metrics')];valid.sort(key=lambda x:(x['metrics']['loss_total'],x['metrics']['mean_solver_wall_seconds'],x['source_sha256']))
   top=valid[:3];sources=[P/name/'candidates'/x['source_sha256']/'solver.py' for x in top]
   freeze(O/name,sources,D/'validation/manifest.json',[D/'test_id/manifest.json',D/'test_ood/manifest.json'],seconds=120,image=IMAGE,training_run=P/name,training_manifest=D/'train/manifest.json')
   plan['groups'][name]={'prefix_length':len(rows),'top':top}
  (O/'plan.json').write_text(json.dumps(plan,indent=2));print(json.dumps(plan,indent=2),flush=True)
 elif sys.argv[1]=='run':
  sys.path.insert(0,'/Aidata/shared/ldm/copper-formal-scheduling-20261024-r2');from parallel_runtime import install
  policy=json.loads(Path('/Aidata/shared/ldm/copper-formal-scheduling-20261024-r2/policy.json').read_text());install(policy)
  name=sys.argv[2]
  for phase in ['validation','test']:
   result=run(O/name,phase);print(json.dumps({'group':name,'phase':phase,'result':result}),flush=True)
if __name__=='__main__':main()
