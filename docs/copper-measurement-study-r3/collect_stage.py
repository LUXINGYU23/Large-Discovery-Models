"""Trusted, read-only training snapshot. Run remotely; never read formal holdouts."""
from pathlib import Path
import json,time,statistics,sys
R=Path('/Aidata/shared/ldm/copper-measurement-study-20261025-r3');P=Path('/Aidata/shared/ldm/copper-v2-evaluator/runs')/R.name
E=R/'controller-epochs/admission-deadline-r1';epoch=json.loads((R/'scheduling-epoch.json').read_text());started=json.loads((E/'started.json').read_text())['time'];now=time.time()
local43=['2bef02f3c62ef00a269f982352fc0a1c29d72bb655ed694da093f09b0d69fdf8','77123ab016cd66aa5c061bcb870f60e6c13db422b27e8fcac62277a2f327a279']
orders={};outcomes={};health={};prefixes={};costs={}
tag='copper:v2:scientific-source:'
for name,pid in epoch['pids'].items():
 root=R/name;private=P/name;b=json.loads((root/'budget.json').read_text());pb=json.loads((private/'budget.json').read_text());usage=b['metadata']['cumulative_usage'];pu=pb['metadata']['cumulative_usage'];charged={k[len(tag):] for k in usage if k.startswith(tag)};private_charged={k[len(tag):] for k in pu if k.startswith(tag)}
 rejected={s for s in charged if usage[tag+s].get('interface_rejected_sources')}|{s for s in private_charged if pu[tag+s].get('interface_rejected_sources')}
 results={}
 for f in (private/'evaluations').glob('*/result.json'):
  d=json.loads(f.read_text());results[d['candidate_id']]=d
 if name.startswith('ldm'):
  events=[json.loads(l) for l in (root/'events.jsonl').read_text().splitlines() if l.strip()];selected=[e for e in events if e['event_type']=='candidates_selected'];order=[]
  for e in selected:
   for s in e['payload']['selected_candidate_ids']:
    if s not in order:order.append(s)
  if name=='ldm-43':
   assert set(local43)<=rejected
   order[33:33]=local43
  if name=='ldm-42':
   # Round 11 validation journals precede selection; both rejected sources
   # consumed budget after the completed 161-source prefix, in this order.
   local42=['dacffa762f151024236c38a836cb39b2bea54f326b5b97fd3a4ecdcaf8d941bb','b5a8b7ecda1a64a6097dacc6f373c3948639fe365908cea8c20575ad0bc8f6cc']
   assert set(local42)<=rejected
   order[161:161]=local42
   # Round 13 rejected sources: original validation errors identify the artifact,
   # journal completion order precedes the round's candidate selection.
   journals=sorted((root/'harness/branches/round_0013').rglob('validation.json'),key=lambda p:p.stat().st_mtime)
   round13=[]
   for journal in journals:
    validation=json.loads(journal.read_text())
    for error in validation['errors']:
     for artifact in validation.get('artifacts',[]):
      source=artifact['sha256']
      if artifact['pathPointer']==error['path'] and source in rejected and source not in round13:
       assert source not in order
       assert next(e['timestamp_unix'] for e in selected if e['iteration']==12)<journal.stat().st_mtime
       if any(e['iteration']==13 for e in selected):assert journal.stat().st_mtime<next(e['timestamp_unix'] for e in selected if e['iteration']==13)
       round13.append(source)
   order[195:195]=round13
   # Paid research rejections ordered by validation evidence before selection.
   for round_number, insertion in [(19,298),(20,317)]:
    round_rejections=[]
    for journal in sorted((root/f'harness/branches/round_{round_number:04d}').rglob('validation.json'),key=lambda p:p.stat().st_mtime):
     validation=json.loads(journal.read_text())
     for error in validation['errors']:
      for artifact in validation.get('artifacts',[]):
       source=artifact['sha256']
       if artifact['pathPointer']==error['path'] and source in rejected and source not in round_rejections:
        assert source not in order
        assert next(e['timestamp_unix'] for e in selected if e['iteration']==round_number-1)<journal.stat().st_mtime
        if any(e['iteration']==round_number for e in selected):assert journal.stat().st_mtime<next(e['timestamp_unix'] for e in selected if e['iteration']==round_number)
        round_rejections.append(source)
    order[insertion:insertion]=round_rejections
  cp=json.loads((root/'checkpoint.json').read_text())['state'];rounds=cp['next_round']-1;selected_round=max(e['iteration'] for e in selected)
 else:
  mapping=json.loads((E/(name+'-sources.json')).read_text());order=[x['source_sha256'] for x in mapping if x['source_sha256'] in charged];rounds=None;selected_round=None
 assert len(order)==len(set(order)) and set(order)==charged,(name,'charge order mismatch')
 assert private_charged<=charged
 complete=0
 for s in order:
  if s not in results and s not in rejected:break
  complete+=1
 prefixes[name]=complete;orders[name]=order;outcomes[name]=(results,rejected)
 new=[]
 for base in (root,private):
  for f in (base/'jobs').glob('*/receipt.json'):
   if f.stat().st_mtime>started:
    d=json.loads(f.read_text());new.append((f,d))
 failed=[]
 for f,d in new:
  if d.get('exit_code') or d.get('reason') or d.get('oom'):
   failed.append(dict(path=str(f),reason=d.get('reason'),exit_code=d.get('exit_code')))
 response=list((root/'harness').rglob('*.response.bin')) if name.startswith('ldm') else []
 fresh=[p for p in response if p.stat().st_mtime>started]
 def provider_failed(path):
  text=path.read_text(errors='replace');events=[]
  for line in text.splitlines():
   if line.startswith('data: {'):
    try:events.append(json.loads(line[6:]))
    except json.JSONDecodeError:pass
  if not events:
   try:events=[json.loads(text)]
   except json.JSONDecodeError:return False
  return any(isinstance(e,dict) and (e.get('type') in ('error','response.failed','response.incomplete') or e.get('error')) for e in events)
 transport=sum(provider_failed(p) for p in fresh)
 cmd=Path(f'/proc/{pid}/cmdline');health[name]=dict(charged=len(charged),private_charged=len(private_charged),complete_results=len(results),complete_prefix=complete,rejections=len(rejected),completed_rounds=rounds,selected_round=selected_round,alive=cmd.exists(),expected_command=cmd.exists() and any(v in cmd.read_bytes() for v in (b'repaired_runtime.py', b'turn_repaired_runtime_20261027.py')),new_job_receipts=len(new),new_failed_jobs=failed,new_provider_responses=len(fresh),provider_error_responses=transport,latest_provider_age=round(time.time()-max([f.stat().st_mtime for f in response],default=time.time())),latest_new_receipt_age=round(time.time()-max([f.stat().st_mtime for f,d in new],default=time.time())))
 costs[name]=dict(client_counters=b['counters'],private_counters=pb['counters'])
n=min(prefixes.values());groups={};curves={};cost_curves={}
for name,order in orders.items():
 results,rejected=outcomes[name];curve=[];cost_curve=[];best=1.;winner=None;solve_seconds=0.;rejects=0
 for s in order[:n]:
  r=results.get(s);rejects+=s in rejected
  if r:
   solve_seconds+=r.get('resource_usage',{}).get('solver_wall_seconds',0.)
   loss=r['metrics']['loss_total']
   if loss<best:best=loss;winner=(s,r)
  curve.append(best);cost_curve.append(solve_seconds/3600)
 assert winner
 groups[name]=dict(best_loss=best,best_source_sha256=winner[0],best_mean_solver_seconds=winner[1]['metrics']['mean_solver_wall_seconds'],successful_sources=n-rejects,rejected_sources=rejects,cumulative_completed_solver_seconds=solve_seconds)
 curves[name]=curve;cost_curves[name]=cost_curve
summary={}
for m in ['ldm','mcmc','acq']:
 g=[groups[f'{m}-{s}'] for s in [42,43,3407]];loss=[v['best_loss'] for v in g];secs=[v['best_mean_solver_seconds'] for v in g]
 summary[m]=dict(mean=statistics.mean(loss),sample_sd=statistics.stdev(loss),min=min(loss),max=max(loss),mean_solver_seconds=statistics.mean(secs),solver_sample_sd=statistics.stdev(secs),mean_cumulative_completed_solver_hours=statistics.mean(v['cumulative_completed_solver_seconds']/3600 for v in g))
d=dict(time=now,charged_prefix=n,complete_prefixes=prefixes,groups=groups,curves=curves,solver_cost_curves=cost_curves,method_summary=summary,health=health,cost_counters_at_snapshot=costs,epoch_start=started,paused=(R/'queue/PAUSED.json').exists(),note='Training only; original charged order and verified paid rejections. Costs of completed solver jobs exclude interfaces/API/queue/recovery; full-run counters are not prefix-aligned.')
Path(sys.argv[1]).write_text(json.dumps(d,indent=2));print(json.dumps({k:d[k] for k in ['time','charged_prefix','method_summary','health','paused']}))
