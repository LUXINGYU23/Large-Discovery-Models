import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage.measure import marching_cubes
O=Path('/Aidata/shared/ldm/copper-baseline-rerun-20261027-ops');out=O/'all-case-visual-draft-20261031';out.mkdir(exist_ok=True)
base=Path('/Aidata/shared/ldm/copper-v2-evaluator/datasets/measurement-formal-20261024-r2')
index=O/'all-case-index-draft-20261031.json';rows=json.loads(index.read_text())['rows']
record={'status':'draft_pending_validation_protocol_decision','renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'index_sha256':hashlib.sha256(index.read_bytes()).hexdigest(),'cases':[]}
lookup={(r['case_id'],r['discovery_seed'],r['execution_seed'],r['method']):r for r in rows}
for split,c,ds,es in [(split,c,ds,es) for split in ['test_id','test_ood'] for c in json.loads((base/split/'manifest.json').read_text())['cases'] for ds in [42,43,3407] for es in [47,48,49]]:
 cid=c['case_id'];filename=f'{split}-{cid}-d{ds}-e{es}.png';release=base/split
 inp=release/c['public'];prob=json.loads((inp/'problem.json').read_text());z=np.load(inp/'problem.npz');truthpath=release/c['truth']/'truth.npy';truth=np.load(truthpath)>0
 origin=np.array(prob['origin']);sp=np.array(prob['spacing']);shape=np.array(prob['shape']);lo=origin;hi=origin+shape*sp
 models=[truth];receipts=[];hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [truthpath,inp/'problem.json',inp/'problem.npz',release/'manifest.json']}
 names=['Truth',f'LDM / seed {ds}',f'MCMC / seed {ds}',f'ACQ / seed {ds}']
 for method in ['ldm','mcmc','acq']:
  r=lookup[(cid,ds,es,method)];f=Path(r['model_path']);rec=Path(r['receipt_path'])
  assert hashlib.sha256(f.read_bytes()).hexdigest()==r['model_sha256']
  assert hashlib.sha256(rec.read_bytes()).hexdigest()==r['receipt_sha256']
  arr=np.load(f)>0;assert arr.shape==truth.shape
  iou=np.logical_and(arr,truth).sum()/np.logical_or(arr,truth).sum();assert abs(iou-r['metrics']['iou'])<1e-8
  models.append(arr);receipts.append(r)
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
 fig.suptitle(f'TEST DRAFT | {split} | case {cid} | discovery {ds} | execution {es}\nProtocol deviation pending acceptance. Same coordinates and slice planes. Black contour = truth; gray = inactive; 3D dots = observations.',fontsize=13)
 fig.savefig(out/filename,dpi=105);plt.close(fig)
 record['cases'].append({'case_id':cid,'split':split,'discovery_seed':ds,'execution_seed':es,'image':filename,'slice_index':cut.tolist(),'input_hashes':hashes,'image_sha256':hashlib.sha256((out/filename).read_bytes()).hexdigest()})
 (out/'progress.json').write_text(json.dumps({'completed':len(record['cases']),'total':432,'last_image':filename}))
 print(len(record['cases']),filename,flush=True)
assert len(record['cases'])==432
(out/'provenance.json').write_text(json.dumps(record,indent=2))
