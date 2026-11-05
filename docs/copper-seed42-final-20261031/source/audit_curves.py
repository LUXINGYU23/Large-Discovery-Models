# -*- coding: utf-8 -*-
"""Read-only forensic audit of seed-42 baseline tuning and solver histories."""
import json,hashlib,ast,collections,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path('/Aidata/shared/ldm/copper-baseline-rerun-20261027-ops');R=Path('/Aidata/shared/ldm/copper-measurement-study-20261027-r4-baselines');P=Path('/Aidata/shared/ldm/copper-v2-evaluator/runs/copper-measurement-study-20261027-r4-baselines');A=O/'seed42-curve-audit-20261031';A.mkdir(exist_ok=True)
def dig(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(v):
 a=np.array(v,float);return dict(n=len(a),min=float(a.min()),p10=float(np.quantile(a,.1)),median=float(np.median(a)),p90=float(np.quantile(a,.9)),max=float(a.max()),mean=float(a.mean())) if len(a) else None
protected={}
for m in ['mcmc','acq']:
 for f in [R/(m+'-42')/'plan.json',R/(m+'-42')/'results.json',R/(m+'-42')/'budget.json',P/(m+'-42')/'budget.json',P/'selection'/(m+'-42')/'selection.json',P/'selection'/(m+'-42')/'test-result.json']:protected[str(f)]=sha(f)
out=dict(scope='discovery seed 42; training records only; read-only; no new solve',methods={},protected_before=protected)
traces=[]
for method in ['mcmc','acq']:
 g=method+'-42';plan=json.loads((R/g/'plan.json').read_text());rows=json.loads((R/g/'results.json').read_text())['results'];frozen=json.loads((P/'selection'/g/'frozen.json').read_text());man=json.loads(Path(frozen['training']['manifest_path']).read_text())
 assert len(rows)==321 and len({r['source_sha256'] for r in rows})==321
 ctx=dict(manifest=dig(man),implementation=plan['implementation_sha256'],case_workers=4,image=plan['image'],seconds=120,cpus=2,memory_mb=4096,aggregation='parent-macro-confirmation-utility-v1')
 best=float('inf');updates=[];candidate_rows=[];inner=[];missing=[];failures=[];badcurves=[]
 for row in rows:
  index=row['index'];loss=row['metrics']['loss_total']
  if loss<best:best=loss;updates.append(dict(index=index,charged_slot=index+1,loss=loss))
  source=P/g/'candidates'/row['source_sha256']/'solver.py';assert sha(source)==row['source_sha256'];tree=ast.parse(source.read_text());config=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONFIG' for t in n.targets));assert config==plan['parameters'][index]
  evaluation=dig(dict(context=ctx,source=row['source_sha256']));ev=P/g/'evaluations'/evaluation;assert ev.exists(),str(ev)
  cases=json.loads((ev/'private-case-results.json').read_text());parents=collections.defaultdict(list)
  for cr in cases:parents[cr['parent_id']].append(1/(1+cr['metrics']['confirmation_loss_total']) if cr['status']=='succeeded' else 0.)
  independent=1-float(np.mean([np.mean(v) for v in parents.values()]));assert abs(independent-loss)<1e-12
  these=[]
  for case in man['cases']:
   cid=case['case_id'];identity=dict(evaluation=evaluation,case=cid,seed=47);key=dig(dict(identity=identity,science_module='v2',truth_digest=None,phase='solver',image=plan['image'],timeout=120,cpus=2,memory_mb=4096,implementation=plan['implementation_sha256']));job=P/g/'jobs'/key;rec=json.loads((job/'receipt.json').read_text());sp=job/'output/search.json'
   if rec['exit_code'] or rec['reason'] or rec['oom']:failures.append(dict(index=index,case_id=cid,exit_code=rec['exit_code'],reason=rec['reason']))
   if not sp.exists():missing.append(dict(index=index,case_id=cid,job=key));continue
   s=json.loads(sp.read_text());assert s['config']==config;assert s['seed']==47;curve=s['curve'];assert s['calls']==len(curve)
   if not curve:missing.append(dict(index=index,case_id=cid,empty_curve=True));continue
   initial=curve[0]['loss'];final=s['best_public_loss'];assert all(curve[j]['best_loss']<=curve[j-1]['best_loss']+1e-12 for j in range(1,len(curve)));assert abs(min(r['loss'] for r in curve)-final)<1e-10
   x={k:s[k] for k in ['calls','proposals','accepted','invalid','acquisition_decisions','projected_proposals','projected_voxels','unchanged_occupancy','seconds']};x.update(index=index,case_id=cid,initial_public_loss=initial,final_public_loss=final,improved=final<initial-1e-12,relative_public_improvement=(initial-final)/initial if initial>0 else None,acquisition_batch_sizes=s['acquisition_batch_sizes'],invalid_reasons=s['invalid_reasons']);inner.append(x);these.append(x)
   # Fixed training cases and prespecified candidate indices for internal traces.
   if cid in [man['cases'][0]['case_id'],man['cases'][1]['case_id']] and index in [0,7,32,223]:traces.append(dict(method=method,index=index,case_id=cid,modality=case['modality'],curve=curve,search_sha256=sha(sp)))
  candidate_rows.append(dict(index=index,loss=loss,best_so_far=best,mean_iou=row['metrics']['mean_iou'],mean_solver_seconds=row['metrics']['mean_solver_wall_seconds'],config=config,inner_completed=len(these),median_calls=float(np.median([v['calls'] for v in these])) if these else None,inner_improved=sum(v['improved'] for v in these)))
 print(method,'read',len(inner),'training solver traces',flush=True)
 totals={k:sum(r[k] for r in inner) for k in ['calls','proposals','accepted','invalid','acquisition_decisions','projected_proposals','projected_voxels','unchanged_occupancy']};scored=sum(r['calls']-1 for r in inner)
 first_batch=min(r['metrics']['loss_total'] for r in rows[:17]);final=min(r['metrics']['loss_total'] for r in rows)
 out['methods'][method]=dict(plan_search=plan['search'],unique_parameters=len({dig(c) for c in plan['parameters']}),unique_scores=len({r['metrics']['loss_total'] for r in rows}),first_batch_best=first_batch,final_best=final,improvement_after_first_batch_absolute=first_batch-final,improvement_after_first_batch_percent=(first_batch-final)/first_batch*100,best_updates=updates,candidates=candidate_rows,inner=dict(count=len(inner),missing=missing,existing_failed_jobs=failures,totals=totals,scored_proposals=scored,acceptance_rate=totals['accepted']/scored if scored else 0,invalid_proposal_rate=totals['invalid']/totals['proposals'],unchanged_occupancy_rate=totals['unchanged_occupancy']/scored,projection_rate=totals['projected_proposals']/totals['proposals'],improved_cases=sum(r['improved'] for r in inner),cases_with_acquisition=sum(r['acquisition_decisions']>0 for r in inner),calls=stats([r['calls'] for r in inner]),seconds=stats([r['seconds'] for r in inner]),relative_public_improvement=stats([r['relative_public_improvement'] for r in inner if r['relative_public_improvement'] is not None]),acquisition_decisions=stats([r['acquisition_decisions'] for r in inner]),per_candidate=[dict(index=c['index'],median_calls=c['median_calls'],improved_count=c['inner_improved']) for c in candidate_rows]))
 # Candidate raw scores, global incumbent, and zoomed incumbent.
 fig,axs=plt.subplots(1,2,figsize=(13,4.2),layout='constrained');xx=[r['index']+1 for r in candidate_rows];yy=[r['loss'] for r in candidate_rows];bb=[r['best_so_far'] for r in candidate_rows]
 axs[0].scatter(xx,yy,s=10,alpha=.45,c='#d38b26' if method=='mcmc' else '#9c4d9c',label='Each candidate');axs[0].step(xx,bb,where='post',c='#173e56',label='Best so far');axs[0].set(xlabel='Charged source slot',ylabel='Independent confirmation loss',title=method.upper()+' / raw scores and incumbent');axs[0].legend()
 axs[1].step(xx,bb,where='post',c='#173e56');axs[1].set(xlim=(17,321),ylim=(final-.0001,first_batch+.0001),xlabel='Charged source slot',ylabel='Best loss (zoomed)',title='After first 17 candidates / zoom')
 for ax in axs:ax.grid(alpha=.2)
 fig.savefig(A/(method+'-outer-audit.png'),dpi=160);plt.close(fig)
protected_after={p:sha(p) for p in protected};assert protected_after==protected;out['protected_after']=protected_after
pres=json.loads((O/'r3-preservation-before.json').read_text());assert all(sha(p)==h for p,h in pres.items());out['r3_protected_hashes_verified']=len(pres);out['trace_examples']=traces;out['script_sha256']=sha(__file__)
(A/'audit.json').write_text(json.dumps(out,indent=2));
fig,axs=plt.subplots(2,2,figsize=(12,7.4),layout='constrained')
for j,m in enumerate(['mcmc','acq']):
 for i,cid in enumerate([man['cases'][0]['case_id'],man['cases'][1]['case_id']]):
  for tr in traces:
   if tr['method']==m and tr['case_id']==cid:
    curve=tr['curve'];initial=curve[0]['loss'];axs[i,j].plot([r['seconds'] for r in curve],[r['best_loss']/initial if initial else 1 for r in curve],label='candidate '+str(tr['index']))
  axs[i,j].set(title=m.upper()+' / '+cid[:10],xlabel='Solver elapsed seconds',ylabel='Best public loss / initial loss');axs[i,j].grid(alpha=.2);axs[i,j].legend(fontsize=8)
fig.savefig(A/'inner-search-traces.png',dpi=160);plt.close(fig)
print(json.dumps({m:{k:v for k,v in a.items() if k not in ['candidates','inner']}|{'inner_summary':{k:v for k,v in a['inner'].items() if k not in ['per_candidate','missing','existing_failed_jobs']}} for m,a in out['methods'].items()}),flush=True)
