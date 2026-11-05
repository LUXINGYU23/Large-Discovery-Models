from pathlib import Path
import json,csv,statistics,collections,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path('/Aidata/shared/ldm/copper-baseline-audit-20261027')
root=BASE/'actual-time-policy-validation';a=json.loads((root/'results.json').read_text());assert len(a)==432
out=BASE/'report';out.mkdir(exist_ok=True)
flat=[]
for r in a:
 m=r['metrics'];s=r['search'] or {};f={k:v for k,v in r.items() if k not in ['search','metrics']}
 f.update(public_loss=m['loss_total'],bounded_public_loss=m['loss_total']/(1+m['loss_total']),iou=m['iou'],relative_volume_error=m['relative_volume_error'],calls=s.get('calls'),invalid=s.get('invalid'),proposals=s.get('proposals'),acquisition_decisions=s.get('acquisition_decisions'),first_acquisition_seconds=s.get('first_acquisition_seconds'),projected_proposals=s.get('projected_proposals'),unchanged_occupancy=s.get('unchanged_occupancy'))
 f['initial_public_loss']=s['curve'][0]['loss'] if s.get('curve') else None
 flat.append(f)
with (out/'actual-time-cases.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
groups=[]
for g in sorted({r['group'] for r in flat}):
 for t in [30,90]:
  rows=[r for r in flat if r['group']==g and r['target_seconds']==t];assert len(rows)==24
  assert set(r['execution_seed'] for r in rows)=={47,48,49}
  item=dict(group=g,target_seconds=t,solves=len(rows))
  for k in ['bounded_public_loss','iou','relative_volume_error','worker_seconds','wall_seconds','cpu_seconds']:
   parents=collections.defaultdict(list)
   for r in rows:parents[r['parent_id']].append(r[k])
   assert len(parents)==4
   item[k]=statistics.mean(statistics.mean(v) for v in parents.values())
  item['worker_min']=min(r['worker_seconds'] for r in rows);item['worker_max']=max(r['worker_seconds'] for r in rows)
  if not g.startswith('ldm'):
   item['calls_median']=statistics.median(r['calls'] for r in rows)
   item['invalid_ratio']=sum(r['invalid'] for r in rows)/sum(r['proposals'] for r in rows)
   item['acquisition_active']=sum(r['acquisition_decisions']>0 for r in rows)
   item['projected_ratio']=sum(r['projected_proposals'] for r in rows)/sum(r['proposals'] for r in rows)
   item['unchanged_scored_ratio']=sum(r['unchanged_occupancy'] for r in rows)/sum(r['calls']-1 for r in rows)
  groups.append(item)
summary=[]
for method in ['ldm','mcmc','acq']:
 for t in [30,90]:
  rows=[r for r in groups if r['group'].startswith(method+'-') and r['target_seconds']==t]
  item=dict(method=method,target_seconds=t)
  for k in ['bounded_public_loss','iou','worker_seconds','wall_seconds','cpu_seconds']:
   vals=[r[k] for r in rows];item[k]=dict(mean=statistics.mean(vals),sd=statistics.stdev(vals))
  summary.append(item)
preservation=json.loads((BASE/'r3-preservation-before.json').read_text())
assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in preservation.items())
paired=[]
for t in [30,90]:
 for parent in sorted({r['parent_id'] for r in flat}):
  item=dict(target_seconds=t,parent_id=parent)
  for method in ['ldm','mcmc','acq']:
   rr=[r for r in flat if r['target_seconds']==t and r['parent_id']==parent and r['group'].startswith(method+'-')];assert len(rr)==18
   item[method]=dict(loss=statistics.mean(r['bounded_public_loss'] for r in rr),iou=statistics.mean(r['iou'] for r in rr))
  paired.append(item)
result={'scope':'development validation, 4 parents/8 cases; public objective, not independent confirmation; no tuning on validation','groups':groups,'summary':summary,'paired_parents':paired,'solves':len(a),'failed':sum(bool(r['exit_code'] or r['reason']) for r in a),'r3_control_hashes_unchanged':True,'actual_worker_seconds_range':[min(r['worker_seconds'] for r in a),max(r['worker_seconds'] for r in a)]}
(out/'actual-time-summary.json').write_text(json.dumps(result,indent=2))
fig,axs=plt.subplots(1,2,figsize=(11,4.5))
for method,color,label in [('ldm','#087e8b','Frozen LDM'),('mcmc','#d47732','Projected MCMC adaptation'),('acq','#684f9b','Projected ACQ adaptation')]:
 rows=[r for r in summary if r['method']==method]
 for ax,k,title in zip(axs,['bounded_public_loss','iou'],['Public-observation loss (lower better)','Geometry IoU (higher better)']):
  x=[r['worker_seconds']['mean'] for r in rows];y=[r[k]['mean'] for r in rows];err=[r[k]['sd'] for r in rows]
  ax.errorbar(x,y,yerr=err,marker='o',capsize=4,label=label,color=color);ax.set_xlabel('Measured worker seconds');ax.set_title(title);ax.grid(alpha=.2)
axs[0].legend(fontsize=8);fig.suptitle('Development only: matched actual-time targets; mean +/- discovery-seed SD')
fig.tight_layout();fig.savefig(out/'actual-time-quality.png',dpi=180);fig.savefig(out/'actual-time-quality.pdf');plt.close(fig)
print(json.dumps(result,indent=2))
