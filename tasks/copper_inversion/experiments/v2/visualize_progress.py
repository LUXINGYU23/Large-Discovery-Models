import json,time,statistics,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=Path('/Aidata/shared/ldm/copper-formal-study-20261024-r1');O=Path('/Aidata/shared/ldm/copper-formal-preparation-20261024/visualization-2115');O.mkdir(exist_ok=True)
seeds=[42,43,3407];methods=['ldm','mcmc','acq'];curves={};progress={}
for m in methods:
 for s in seeds:
  n=f'{m}-{s}';r=R/n;b=json.loads((r/'budget.json').read_text())['counters'];rows=[]
  if m=='ldm':
   obs=json.loads((r/'checkpoint.json').read_text())['state']['observations'];assert b.get('interface_rejected_sources',0)==(1 if s==42 else 0)
   for i,o in enumerate(obs):
    if s==42 and i==17:rows.append({'kind':'interface_rejected','metrics':None})
    rows.append({'kind':'evaluation','metrics':o['evaluation']['metrics']})
   done=len(obs);rounds=max(o['round_idx'] for o in obs)
  else:
   receipts={int(f.stem):json.loads(f.read_text()) for f in (r/'tuning-receipts').glob('*.json')};i=0
   while i in receipts:
    row=receipts[i];rows.append({'kind':row['status'],'metrics':row.get('metrics')})
    if i==0:
     recovery=list((r/'serialization-interrupted-recovery').glob('*.json'));assert len(recovery)==1
     extra=json.loads(recovery[0].read_text());assert extra['status']=='succeeded';rows.append({'kind':'historical_duplicate','metrics':extra['metrics']})
    i+=1
   done=len(receipts);rounds=None
  best=None;series=[]
  for row in rows:
   metrics=row['metrics']
   if metrics and (best is None or metrics['loss_total']<best['loss_total']):best=metrics
   series.append({'loss':best['loss_total'],'solver_seconds':best['mean_solver_wall_seconds'],'kind':row['kind']})
  curves[n]=series;progress[n]={'committed_records':done,'complete_charged_prefix':len(rows),'reserved_slots':b['algorithm_feedback_slots'],'interface_rejections':b.get('interface_rejected_sources',0),'completed_rounds':rounds}
N=min(map(len,curves.values()));summary={}
for m in methods:
 loss=[curves[f'{m}-{s}'][N-1]['loss'] for s in seeds];sec=[curves[f'{m}-{s}'][N-1]['solver_seconds'] for s in seeds]
 summary[m]={'losses':loss,'mean_loss':statistics.mean(loss),'sd_loss':statistics.stdev(loss),'solver_seconds':sec,'mean_solver_seconds':statistics.mean(sec)}
data={'timestamp':time.time(),'common_budget':N,'summary':summary,'progress':progress,'curves':curves,'limitations':['Training only; private validation/test pending','Actual charged budget includes historical duplicate baseline source and interface rejections','Retrospective submission-order curve, not wall-clock feedback availability','Solver time excludes discovery, queue, interface and scoring costs','3 seeds: descriptive spread, not confidence interval']}
(O/'results.json').write_text(json.dumps(data,indent=2))
colors={'ldm':'#2563eb','mcmc':'#ea580c','acq':'#15946c'};labels={'ldm':'LDM','mcmc':'MCMC','acq':'ACQ'}
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,3,figsize=(15,4.6),gridspec_kw={'width_ratios':[1.6,1,1]})
x=np.arange(1,N+1)
for j,m in enumerate(methods):
 a=np.array([[curves[f'{m}-{s}'][i]['loss'] for i in range(N)] for s in seeds]);mean=a.mean(0);sd=a.std(0,ddof=1)
 ax[0].step(x,mean,where='post',color=colors[m],label=labels[m],lw=2);ax[0].fill_between(x,mean-sd,mean+sd,step='post',color=colors[m],alpha=.12)
 v=summary[m]['losses'];ax[1].scatter([j-.12,j,j+.12],v,color=colors[m],s=45)
 ax[1].plot([j-.24,j+.24],[np.mean(v)]*2,color=colors[m],lw=3)
 for k,s in enumerate(seeds):ax[1].annotate(str(s),(j+(k-1)*.12,v[k]),xytext=(0,6+11*k),textcoords='offset points',ha='center',fontsize=8)
 sec=summary[m]['solver_seconds'];ax[2].scatter([j-.12,j,j+.12],sec,color=colors[m],s=45);ax[2].plot([j-.24,j+.24],[np.mean(sec)]*2,color=colors[m],lw=3)
ax[0].set(title='Best training loss vs. charged budget',xlabel='Charged sources (including rejection / duplicate)',ylabel='Loss = 1 - mean IoU; lower is better');ax[0].legend();ax[0].grid(alpha=.18)
ax[1].set(title=f'Seed variation at budget {N}',ylabel='Best training loss',xticks=range(3),xticklabels=['LDM','MCMC','ACQ']);ax[1].grid(axis='y',alpha=.18)
ax[2].set(title='Selected solver runtime',ylabel='Mean seconds per case (log scale)',xticks=range(3),xticklabels=['LDM','MCMC','ACQ']);ax[2].set_yscale('log');ax[2].axhline(120,color='#888',ls=':',lw=1);ax[2].text(0,130,'120 s limit',fontsize=9);ax[2].grid(axis='y',alpha=.18)
fig.suptitle(f'Copper inversion | {N} matched charged sources | seeds 42 / 43 / 3407',fontsize=16,y=1.02)
fig.text(.5,-.02,'Training only. Shading = sample SD (3 seeds). Baseline duplicate cost retained. Runtime excludes algorithm discovery.',ha='center',fontsize=10,color='#555')
fig.tight_layout();fig.savefig(O/'matched-comparison.png',dpi=180,bbox_inches='tight');fig.savefig(O/'matched-comparison.pdf',bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(1,3,figsize=(14,4),sharey=True)
for a,s in zip(ax,seeds):
 for m in methods:
  rows=curves[f'{m}-{s}'];a.step(range(1,len(rows)+1),[v['loss'] for v in rows],where='post',color=colors[m],label=labels[m],lw=1.8)
 a.axvline(N,color='#777',ls='--',lw=1);a.set(title=f'Seed {s}',xlabel='Charged sources');a.grid(alpha=.18)
ax[0].set_ylabel('Best training loss');ax[0].legend();fig.suptitle('Available progress — compare methods only left of the dashed line',fontsize=14);fig.tight_layout();fig.savefig(O/'seed-progress.png',dpi=180,bbox_inches='tight');plt.close(fig)
print(json.dumps({'common_budget':N,'summary':summary,'progress':progress},indent=2))
