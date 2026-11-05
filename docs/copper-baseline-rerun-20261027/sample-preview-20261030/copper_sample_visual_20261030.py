import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage.measure import marching_cubes
O=Path('/Aidata/shared/ldm/copper-baseline-rerun-20261027-ops');out=O/'sample-preview-20261030';out.mkdir(exist_ok=True)
base=Path('/Aidata/shared/ldm');private=base/'copper-v2-evaluator/runs'
r3='copper-measurement-study-20261025-r3';r4='copper-measurement-study-20261027-r4-baselines'
best=json.loads((base/r3/'ldm-42/summary.json').read_text())['best']
roots=[private/r3/'ldm-42/evaluations'/'eead2bbb9d1440e51e178ca536b83d5ddd1c91848ecbf2a0221eb5090cdad246']
names=['Truth','LDM / seed 42','MCMC / seed 3407','ACQ / seed 43'];sources=[best['candidate']['canonical_key']]
for group in ['mcmc-3407','acq-43']:
 rows=json.loads((base/r4/group/'results.json').read_text())['results'];b=max(rows,key=lambda x:x['metrics']['research_utility']);sources.append(b['source_sha256'])
 matches=[]
 for f in (private/r4/group/'evaluations').glob('*/result.json'):
  d=json.loads(f.read_text())
  if abs(d['metrics']['research_utility']-b['metrics']['research_utility'])<1e-14:matches.append(f.parent)
 assert len(matches)==1;roots.append(matches[0])
release=base/'copper-v2-evaluator/datasets/measurement-formal-20261024-r2/train'
manifest=json.loads((release/'manifest.json').read_text());chosen=[];seen=set()
for c in manifest['cases']:
 if c['parent_id'] not in seen:chosen.append(c);seen.add(c['parent_id'])
 if len(chosen)==2:break
record={'selection':'first two distinct parents in frozen training manifest; first modality; execution seed 47','source_sha256':sources,'cases':[]}
for n,c in enumerate(chosen,1):
 cid=c['case_id'];inp=release/c['public'];prob=json.loads((inp/'problem.json').read_text());z=np.load(inp/'problem.npz');truthpath=release/c['truth']/'truth.npy';truth=np.load(truthpath)>0
 origin=np.array(prob['origin']);sp=np.array(prob['spacing']);shape=np.array(prob['shape']);lo=origin;hi=origin+shape*sp
 models=[truth];receipts=[];hashes={str(truthpath):hashlib.sha256(truthpath.read_bytes()).hexdigest()}
 for root in roots:
  p=root/cid/'47';f=p/'model/model.npy';models.append(np.load(f)>0);receipts.append(json.loads((p/'receipt.json').read_text()));hashes[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
 for arr,rec in zip(models[1:],receipts):
  iou=np.logical_and(arr,truth).sum()/np.logical_or(arr,truth).sum();assert abs(iou-rec['metrics']['iou'])<1e-8
 cut=np.rint(np.argwhere(truth).mean(axis=0)).astype(int);active=z['domain_mask'] & ~z['forbidden_mask']
 fig=plt.figure(figsize=(16,13),layout='constrained');gs=fig.add_gridspec(4,4,height_ratios=[1.5,1,1,1]);colors=['#384559','#2166ac','#e38b22','#a444a4']
 for j,arr in enumerate(models):
  ax=fig.add_subplot(gs[0,j],projection='3d')
  if arr.any() and not arr.all():
   verts,faces,_,_=marching_cubes(arr.astype(float),.5,spacing=sp);verts+=origin+sp/2
   ax.plot_trisurf(verts[:,0],verts[:,1],verts[:,2],triangles=faces,color=colors[j],alpha=.75,linewidth=0,rasterized=True)
  for k in [0,1]:
   if f'points_{k}' in z:
    pts=z[f'points_{k}'];ax.scatter(*pts.T,s=2,c='#777777',alpha=.35)
  ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)');ax.set_box_aspect(hi-lo);ax.view_init(23,-55);ax.tick_params(labelsize=7)
  title=names[j]
  if j:title+=f"\nIoU {receipts[j-1]['metrics']['iou']:.3f} | {receipts[j-1]['solver_wall_seconds']:.1f}s"
  ax.set_title(title,fontsize=12,fontweight='bold')
  for row,(axis,a,b,label) in enumerate([(2,0,1,'XY'),(1,0,2,'XZ'),(0,1,2,'YZ')],1):
   ax=fig.add_subplot(gs[row,j]);plane=np.take(arr,cut[axis],axis=axis).T;tp=np.take(truth,cut[axis],axis=axis).T;ap=np.take(active,cut[axis],axis=axis).T
   image=np.zeros(plane.shape);image[plane]=1;image[~ap]=2
   extent=[lo[a],hi[a],lo[b],hi[b]];ax.imshow(image,origin='lower',extent=extent,cmap=ListedColormap(['#f7f8fa',colors[j],'#cccccc']),vmin=0,vmax=2,interpolation='nearest',aspect='equal')
   xs=lo[a]+(np.arange(shape[a])+.5)*sp[a];ys=lo[b]+(np.arange(shape[b])+.5)*sp[b]
   if tp.any() and not tp.all():ax.contour(xs,ys,tp,levels=[.5],colors=['#202020'],linewidths=.8)
   ax.set_title(f'{label}: {"XYZ"[axis]}={origin[axis]+(cut[axis]+.5)*sp[axis]:.1f}m',fontsize=9);ax.set_xlabel('XYZ'[a]+' (m)',fontsize=8);ax.set_ylabel('XYZ'[b]+' (m)',fontsize=8);ax.tick_params(labelsize=7)
 fig.suptitle(f'TRAINING PREVIEW {n} | case {cid} | {c["modality"]} | execution seed 47\nFixed manifest order; same coordinates and slice planes. Black contour = truth; gray = inactive; 3D dots = observations.',fontsize=13)
 fig.savefig(out/f'sample-{n}.png',dpi=140);plt.close(fig)
 record['cases'].append({'case_id':cid,'slice_index':cut.tolist(),'receipts':receipts,'hashes':hashes})
record['renderer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();(out/'provenance.json').write_text(json.dumps(record,indent=2));print(out)
