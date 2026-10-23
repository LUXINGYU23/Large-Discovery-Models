"""Export actual selected measurements and all pool predictions for offline analysis."""
import csv
import json
from pathlib import Path
import sys

base=Path(sys.argv[1]);summary=[]
for stage in ('tiny','full'):
 for run in sorted((base/stage).glob('*')):
  if not run.is_dir() or not (run/'checkpoint.json').exists():continue
  state=json.loads((run/'checkpoint.json').read_text())['state']
  observations=state['observations']
  observed={(o['candidate']['candidate_id'],o['round_idx']):o for o in observations}
  events=[json.loads(l) for l in (run/'events.jsonl').read_text().splitlines()]
  rows=[]
  for event in events:
   if event['event_type']!='candidates_selected':continue
   r=event['iteration'];selection=event['payload'];selected=set(selection['selected_candidate_ids'])
   for pred in selection['predictions']:
    key=pred['candidate_id'];obs=observed.get((key,r))
    feature_path=run/'encodings'/f'{key}.json'
    feature=json.loads(feature_path.read_text())
    rows.append({'round':r,'candidate_id':key,'selected':key in selected,
      'status':obs['evaluation']['status'] if obs else 'unmeasured',
      'loss_total':obs['evaluation']['metrics'].get('loss_total') if obs else None,
      'predicted_mean':pred['mean'][0],'predicted_std':pred['std'][0],
      'acquisition_score':pred['acquisition_score'],
      'source_path':f'candidates/{key}/solver.py','feature_path':str(feature_path.relative_to(run)),
      'normalized_sha256':feature['normalized_sha256'],'feature_version':feature['version']})
  with (run/'offline-samples.jsonl').open('w') as f:
   for row in rows:f.write(json.dumps(row)+'\n')
  curve=[];best=None
  for o in observations:
   value=o['evaluation']['metrics'].get('loss_total') if o['evaluation']['status']=='succeeded' else None
   if value is not None:best=value if best is None else min(value,best)
   curve.append({'round':o['round_idx'],'evaluation':len(curve)+1,'candidate':o['candidate']['candidate_id'],
                 'status':o['evaluation']['status'],'loss_total':value,'best_so_far':best})
  if curve:
   with (run/'comparison-curve.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(curve[0]));writer.writeheader();writer.writerows(curve)
  summary.append({'stage':stage,'run':run.name,'proposal_occurrences':len(rows),
                  'unique_proposals':len({r['candidate_id'] for r in rows}),
                  'selected_occurrences':sum(r['selected'] for r in rows),'observations':len(observations),'best_loss':best})
(base/'offline-export.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary))
