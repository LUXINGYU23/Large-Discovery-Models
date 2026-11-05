# -*- coding: utf-8 -*-
"""Read-only reporting of existing, frozen discovery-seed-42 results."""
import os,json,hashlib,collections,shutil,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage.measure import marching_cubes
from plotly.offline import get_plotlyjs
O=Path('/Aidata/shared/ldm/copper-baseline-rerun-20261027-ops')
B=O/'seed42-investor-report-20261031';B.mkdir(exist_ok=True)
for sub in ['assets','cases','geometry','data']: (B/sub).mkdir(exist_ok=True)
base=Path('/Aidata/shared/ldm/copper-v2-evaluator/datasets/measurement-formal-20261024-r2')
methods=['ldm','mcmc','acq'];colors=['#2563a6','#d38b26','#9c4d9c'];names=['LDM','MCMC','ACQ'];hashes={}
def read(p):
 p=Path(p);raw=p.read_bytes();hashes[str(p)]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
def aggregate(rows):
 parents=collections.defaultdict(list)
 for r in rows:parents[r['parent_id']].append(r)
 return dict(loss_total=1-float(np.mean([np.mean([1/(1+r['metrics']['confirmation_loss_total']) if r['status']=='succeeded' else 0 for r in rs]) for rs in parents.values()])),mean_iou=float(np.mean([np.mean([r['metrics']['iou'] if r['status']=='succeeded' else 0 for r in rs]) for rs in parents.values()])),mean_solver_wall_seconds=float(np.mean([r['solver_wall_seconds'] for r in rows])),solve_count=len(rows),parent_count=len(parents),failed_solves=sum(r['status']!='succeeded' for r in rows))
idx=read(O/'all-case-index-draft-20261031.json');rows=[r.copy() for r in idx['rows'] if r['discovery_seed']==42];assert len(rows)==432
manifest={s:read(base/s/'manifest.json') for s in ['test_id','test_ood']}
meta={c['case_id']:dict(c,split=s,order=i) for s,m in manifest.items() for i,c in enumerate(m['cases'])}
for r in rows:
 rec=read(r['receipt_path']);assert hashes[r['receipt_path']]==r['receipt_sha256'];assert rec['status']=='succeeded'
 r.update(parent_id=rec['parent_id'],modality=meta[r['case_id']]['modality']);assert r['parent_id']==meta[r['case_id']]['parent_id']
tr=read(O/'training-curves-source-draft-20261031.json');training=[g for g in tr['groups'] if g['seed']==42]
validation=read(O/'validation-resource-audit-draft-20261031.json');groups=[g for g in validation['groups'] if g['seed']==42]
cost=read(O/'training-cost-draft-20261031.json');cost=[g for g in cost['groups'] if g['seed']==42]
provider=read(O/'final-delivery-20261031/copper-baseline-rerun-20261027/ldm-preserved-cost-20261031.json')['ldm-42']['provider_usage']
summary=[];execution=[];parents=[]
for method in methods:
 for split in manifest:
  rr=[r for r in rows if r['method']==method and r['split']==split];a=aggregate(rr);summary.append(dict(method=method,split=split,metrics=a))
  for es in [47,48,49]:execution.append(dict(method=method,split=split,execution_seed=es,metrics=aggregate([r for r in rr if r['execution_seed']==es])))
  for pid in sorted({r['parent_id'] for r in rr}):parents.append(dict(method=method,split=split,parent_id=pid,metrics=aggregate([r for r in rr if r['parent_id']==pid])))
old=read(O/'combined-test-summary-draft-20261031.json')
for r in summary:
 expected=next(x['metrics'] for x in old['rows'] if x['method']==r['method'] and x['split']==r['split'] and x['discovery_seed']==42)
 for k in ['loss_total','mean_iou','mean_solver_wall_seconds']:assert abs(expected[k]-r['metrics'][k])<1e-12
comparisons=[]
for split in manifest:
 a=next(r['metrics'] for r in summary if r['method']=='ldm' and r['split']==split)
 for method in ['mcmc','acq']:
  b=next(r['metrics'] for r in summary if r['method']==method and r['split']==split)
  comparisons.append(dict(split=split,baseline=method,loss_reduction_percent=(1-a['loss_total']/b['loss_total'])*100,iou_absolute_gain=a['mean_iou']-b['mean_iou'],iou_relative_gain_percent=(a['mean_iou']/b['mean_iou']-1)*100,extra_seconds=a['mean_solver_wall_seconds']-b['mean_solver_wall_seconds'],extra_time_percent=(a['mean_solver_wall_seconds']/b['mean_solver_wall_seconds']-1)*100))
curves=[]
for g in training:
 vals=[];best=0
 for rnd in range(21):
  rr=[r for r in g['evaluations'] if (r['round_idx'] if g['method']=='ldm' else (0 if r['index']==0 else (r['index']-1)//16+1))==rnd]
  best=max([best]+[r['metrics']['research_utility'] for r in rr]);vals.append(best)
 curves.append(dict(method=g['method'],slots=[1+16*i for i in range(21)],best_utility=vals,evaluated_sources=g['count'],charged_sources=321,training_best=min(g['evaluations'],key=lambda r:r['metrics']['loss_total'])))
# All charts use the same palette, exact metrics, and readable publication axes.
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','savefig.facecolor':'white'})
fig,axs=plt.subplots(1,3,figsize=(14,4.1),layout='constrained')
for ax,k,title in zip(axs,['loss_total','mean_iou','mean_solver_wall_seconds'],['Independent confirmation loss (lower)','Geometry IoU (higher)','Actual solve time / case (seconds)']):
 for j,s in enumerate(manifest):
  vs=[next(r['metrics'][k] for r in summary if r['method']==m and r['split']==s) for m in methods];xx=np.arange(3)+(j-.5)*.35
  ax.bar(xx,vs,width=.32,color=colors,alpha=1 if j==0 else .5,label='ID' if j==0 else 'OOD')
  for x,v in zip(xx,vs):ax.text(x,v+max(vs)*.025,f'{v:.3f}' if k!='mean_solver_wall_seconds' else f'{v:.1f}',ha='center',fontsize=8)
 ax.set(xticks=np.arange(3),xticklabels=names,title=title);ax.set_ylim(0,ax.get_ylim()[1]*1.15);ax.grid(axis='y',alpha=.15);ax.legend()
fig.savefig(B/'assets/effects.png',dpi=170);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
for m,c in zip(methods,colors):
 r=next(x for x in curves if x['method']==m);axs[0].step(r['slots'],1-np.array(r['best_utility']),where='post',color=c,label=m.upper())
axs[0].set(xlabel='Charged candidate slots (batch end)',ylabel='Best training loss (lower)',title='Search progress / discovery seed 42',xlim=(1,321));axs[0].legend();axs[0].grid(alpha=.2)
for j,m in enumerate(methods):
 g=next(g for g in groups if g['method']==m);vv=g['validation_candidates']
 for i,r in enumerate(vv):axs[1].scatter(j+(i-1)*.09,r['metrics']['loss_total'],color=colors[j],s=180 if r['source_sha256']==g['selected_sha256'] else 55,marker='*' if r['source_sha256']==g['selected_sha256'] else 'o',edgecolor='white',zorder=3)
axs[1].set(xticks=range(3),xticklabels=names,ylabel='Validation loss (lower)',title='Frozen validation / star = selected winner');axs[1].grid(axis='y',alpha=.2)
fig.savefig(B/'assets/training-validation.png',dpi=170);plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(11.5,7.2),layout='constrained')
for j,split in enumerate(manifest):
 for i,b in enumerate(['mcmc','acq']):
  ax=axs[j,i];rr=[r for r in parents if r['split']==split and r['method']==b];xx=[r['metrics']['loss_total'] for r in rr];yy=[next(v['metrics']['loss_total'] for v in parents if v['method']=='ldm' and v['split']==split and v['parent_id']==r['parent_id']) for r in rr]
  ax.scatter(xx,yy,c='#2563a6',s=45,alpha=.85);low=min(xx+yy)-.02;high=max(xx+yy)+.02;ax.plot([low,high],[low,high],c='#777',ls='--',lw=1);win=sum(y<x for x,y in zip(xx,yy));ax.set(xlabel=b.upper()+' parent loss',ylabel='LDM parent loss',title=f'{split.upper()} / LDM wins {win}/{len(xx)} parents');ax.grid(alpha=.15)
fig.savefig(B/'assets/parent-comparison.png',dpi=170);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(11.5,4),layout='constrained')
for ax,split in zip(axs,manifest):
 for m,c in zip(methods,colors):
  rr=[r for r in execution if r['method']==m and r['split']==split];ax.plot([r['execution_seed'] for r in rr],[r['metrics']['loss_total'] for r in rr],marker='o',label=m.upper(),color=c)
 ax.set(xticks=[47,48,49],xlabel='Execution seed',ylabel='Independent confirmation loss',title=split.upper());ax.grid(alpha=.2);ax.legend()
fig.savefig(B/'assets/execution-seeds.png',dpi=170);plt.close(fig)
# Dataset, source and ledger bytes remain untouched. Export only report material.
sources=[]
for g in groups:
 m=g['method'];run='copper-measurement-study-20261025-r3' if m=='ldm' else 'copper-measurement-study-20261027-r4-baselines';root=Path('/Aidata/shared/ldm/copper-v2-evaluator/runs')/run/'selection'/f'{m}-42';frozen=read(root/'frozen.json')
 for c in frozen['candidates']:
  f=root/c['source_path'];h=hashlib.sha256(f.read_bytes()).hexdigest();assert h==c['sha256'];sources.append(dict(method=m,sha256=h,selected=h==g['selected_sha256']))
report=dict(discovery_seed=42,test_execution_seeds=[47,48,49],validation_execution_seeds=[47],summary=summary,comparisons=comparisons,execution=execution,parents=parents,training=curves,validation=groups,cost=cost,provider_usage=provider,case_metadata=[dict(case_id=k,parent_id=v['parent_id'],modality=v['modality'],split=v['split'],order=v['order']) for k,v in meta.items()],test_rows=rows,source_identities=sources)
(B/'data/report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));(B/'data/report.js').write_text('window.REPORT='+json.dumps(report,ensure_ascii=False,separators=(',',':'))+';');(B/'assets/plotly.min.js').write_text(get_plotlyjs())
# Re-render all 144 comparisons, using exact binary occupancy and common physical coordinates.
lookup={(r['case_id'],r['execution_seed'],r['method']):r for r in rows};provenance=[]
for split,man in manifest.items():
 for c in man['cases']:
  cid=c['case_id'];inp=base/split/c['public'];prob=read(inp/'problem.json');z=np.load(inp/'problem.npz');tf=base/split/c['truth']/'truth.npy';truth=np.load(tf)>0
  origin=np.array(prob['origin']);sp=np.array(prob['spacing']);shape=np.array(prob['shape']);lo=origin;hi=origin+shape*sp;active=z['domain_mask'] & ~z['forbidden_mask'];cut=np.rint(np.argwhere(truth).mean(axis=0)).astype(int)
  ob=[]
  for k in [0,1]:
   if f'points_{k}' in z:ob.append(dict(points=np.round(z[f'points_{k}'],4).tolist(),labels=z[f'labels_{k}'].tolist()))
  for es in [47,48,49]:
   key=f'{split}-{cid}-e{es}';models=[truth];receipts=[];inputhash={str(tf):hashlib.sha256(tf.read_bytes()).hexdigest()}
   for method in methods:
    r=lookup[cid,es,method];f=Path(r['model_path']);assert hashlib.sha256(f.read_bytes()).hexdigest()==r['model_sha256'];arr=np.load(f)>0;assert arr.shape==truth.shape;assert abs((arr&truth).sum()/(arr|truth).sum()-r['metrics']['iou'])<1e-8;models.append(arr);receipts.append(r);inputhash[str(f)]=r['model_sha256']
   geometry=dict(key=key,origin=lo.tolist(),upper=hi.tolist(),observations=ob,meshes=[])
   fig=plt.figure(figsize=(16,13),layout='constrained');gs=fig.add_gridspec(4,4,height_ratios=[1.5,1,1,1]);allcolors=['#384559']+colors
   for j,arr in enumerate(models):
    ax=fig.add_subplot(gs[0,j],projection='3d');mesh=dict(name=['Truth']+names,color=allcolors[j],vertices=[],faces=[]);mesh['name']=(['Truth']+names)[j]
    if arr.any() and not arr.all():
     vertices,faces,_,_=marching_cubes(arr.astype(float),.5,spacing=sp);vertices+=origin+sp/2;ax.plot_trisurf(*vertices.T,triangles=faces,color=allcolors[j],alpha=.8,linewidth=0,rasterized=True);mesh.update(vertices=np.round(vertices,3).tolist(),faces=faces.tolist())
    geometry['meshes'].append(mesh)
    for obs in ob:
     pts=np.array(obs['points']);lab=np.array(obs['labels']);ax.scatter(*pts.T,s=2,c=np.where(lab==1,'#d95f4a','#3896b4'),alpha=.3)
    ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)');ax.set_box_aspect(hi-lo);ax.view_init(23,-55);ax.tick_params(labelsize=7)
    title=(['Truth']+names)[j]
    if j:title+=f"\nIoU {receipts[j-1]['metrics']['iou']:.3f} | {receipts[j-1]['solver_wall_seconds']:.1f}s"
    ax.set_title(title,fontsize=12,fontweight='bold')
    for row,(axis,a,b,label) in enumerate([(2,0,1,'XY'),(1,0,2,'XZ'),(0,1,2,'YZ')],1):
     ax=fig.add_subplot(gs[row,j]);plane=np.take(arr,cut[axis],axis=axis).T;tp=np.take(truth,cut[axis],axis=axis).T;ap=np.take(active,cut[axis],axis=axis).T;img=np.zeros(plane.shape);img[plane]=1;img[~ap]=2
     ax.imshow(img,origin='lower',extent=[lo[a],hi[a],lo[b],hi[b]],cmap=ListedColormap(['#f7f8fa',allcolors[j],'#ccc']),vmin=0,vmax=2,interpolation='nearest',aspect='equal')
     xs=lo[a]+(np.arange(shape[a])+.5)*sp[a];ys=lo[b]+(np.arange(shape[b])+.5)*sp[b]
     if tp.any() and not tp.all():ax.contour(xs,ys,tp,levels=[.5],colors=['#202020'],linewidths=.8)
     ax.set_title(f'{label}: {"XYZ"[axis]}={origin[axis]+(cut[axis]+.5)*sp[axis]:.1f}m',fontsize=9);ax.set_xlabel('XYZ'[a]+' (m)',fontsize=8);ax.set_ylabel('XYZ'[b]+' (m)',fontsize=8);ax.tick_params(labelsize=7)
   fig.suptitle(f'DISCOVERY SEED 42 | {split.upper()} | case {cid} | execution {es}\nSame coordinates and slice planes. Black contour = truth; gray = inactive; dots = public observations.',fontsize=13)
   fig.savefig(B/'cases'/f'{key}.png',dpi=110);plt.close(fig)
   (B/'geometry'/f'{key}.js').write_text('window.COPPER_GEOMETRY='+json.dumps(geometry,separators=(',',':'))+';')
   provenance.append(dict(key=key,case_id=cid,split=split,execution_seed=es,slice_index=cut.tolist(),model_hashes=inputhash,image_sha256=hashlib.sha256((B/'cases'/f'{key}.png').read_bytes()).hexdigest(),geometry_sha256=hashlib.sha256((B/'geometry'/f'{key}.js').read_bytes()).hexdigest()))
   (B/'progress.json').write_text(json.dumps(dict(completed=len(provenance),total=144)));print(len(provenance),key,flush=True)
assert len(provenance)==144
(B/'data/provenance.json').write_text(json.dumps(dict(renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_hashes=hashes,comparisons=provenance),indent=2));print('DONE',flush=True)
