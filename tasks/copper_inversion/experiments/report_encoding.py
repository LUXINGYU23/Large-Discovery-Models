"""Read-only live summary; never invent a score for unselected programs."""
import json
from pathlib import Path
import sys
import time

base=Path(sys.argv[1]);rows=[]
for stage in ('tiny','full'):
 for run in sorted((base/stage).glob('*')):
  if not run.is_dir():continue
  def load(p,default):return json.loads(p.read_text()) if p.exists() else default
  checkpoint=load(run/'checkpoint.json',{}).get('state',{})
  obs=checkpoint.get('observations',[])
  successful=[o for o in obs if o['evaluation']['status']=='succeeded']
  enc=[load(p,{}) for p in (run/'encodings').glob('*.json')]
  state=load(base/'status'/f'{stage}_{run.name}.json',{})
  rows.append({'stage':stage,'run':run.name,'status':load(run/'status.json',{}).get('status'),
    'observations':len(obs),'successful':len(successful),'round':max((o['round_idx'] for o in obs),default=0),
    'best_loss':min((o['evaluation']['metrics']['loss_total'] for o in successful),default=None),
    'elapsed_seconds':state.get('elapsed_seconds',time.time()-state.get('started_at',time.time())),
    'encodings':len(enc),'encoding_seconds':sum(r['elapsed_seconds'] for r in enc),
    'counters':load(run/'budget.json',{}).get('counters',{})})
print(json.dumps({'updated_at':time.time(),'runs':rows},indent=2))
