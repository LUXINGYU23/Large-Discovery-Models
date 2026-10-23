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
  config=json.loads((run/'config.json').read_text())
  campaign=json.loads((run/'campaign.json').read_text())
  provenance={'run_id':campaign['run_id'],'case_id':config['case_id'],
              'encoding':config['encoding'],'stage':stage}
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
    rows.append({**provenance,'round':r,'candidate_id':key,'selected':key in selected,
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
  status_path=base/'status'/f'{stage}_{run.name}.json'
  status=json.loads(status_path.read_text()) if status_path.exists() else {}
  counters=json.loads((run/'budget.json').read_text())['counters']
  features=[json.loads(p.read_text()) for p in (run/'encodings').glob('*.json')]
  summary.append({**provenance,'run':run.name,'status':status.get('status'),
                  'verification':status.get('verification'),
                  'wall_seconds':status.get('elapsed_seconds'),
                  'encoding_seconds':sum(f['elapsed_seconds'] for f in features),
                  'model_load_seconds':sum(f['model_load_seconds'] for f in features),
                  'research_seconds':counters.get('harness_wall_time_seconds'),
                  'solver_seconds':counters.get('solver_wall_seconds'),
                  'scorer_seconds':counters.get('scorer_wall_seconds'),
                  'llm_requests':counters.get('llm_requests'),
                  'proposal_occurrences':len(rows),
                  'unique_proposals':len({r['candidate_id'] for r in rows}),
                  'selected_occurrences':sum(r['selected'] for r in rows),'observations':len(observations),
                  'successful_observations':sum(o['evaluation']['status']=='succeeded' for o in observations),
                  'best_loss':best})
(base/'offline-export.json').write_text(json.dumps(summary,indent=2))
lines=['# Copper encoding comparison', '',
       'One seed (47). Preflights and full campaigns are separate. Unselected programs have no measured label.', '',
       'Wall time includes process startup, readiness, research, encoding and scoring; it excludes audit, completed resume and export.', '']
for stage in ('tiny','full'):
 lines += [f'## {stage}', '', '| Case | Encoding | Status | Observations | Best loss | Wall min | Encoding sec (includes load) | LLM requests |',
           '|---|---|---|---:|---:|---:|---:|---:|']
 for r in summary:
  if r['stage']!=stage:continue
  loss=f"{r['best_loss']:.8f}" if r['best_loss'] is not None else 'unknown'
  wall=f"{r['wall_seconds']/60:.2f}" if r['wall_seconds'] is not None else 'running'
  lines.append(f"| {r['case_id']} | {r['encoding']} | {r['status']} / {r['verification']} | {r['observations']} | {loss} | {wall} | {r['encoding_seconds']:.2f} | {r['llm_requests']} |")
 lines.append('')
(base/'encoding-comparison.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary))
