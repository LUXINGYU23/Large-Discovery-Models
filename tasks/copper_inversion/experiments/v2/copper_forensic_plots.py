import json,numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
P=Path('/Aidata/shared/ldm/copper-v2-evaluator/runs/copper-formal-study-20261024-r1/interim-audit-20261024');r=json.loads((P/'forensics.json').read_text());O=P/'figures';O.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10})
fig=plt.figure(figsize=(16,12));sfig,sax=plt.subplots(3,4,figsize=(15,10))
for i,rank in enumerate(['worst','median','best']):
 d=np.load(P/(rank+'.npz'));truth=d['truth'];sp=d['spacing'];origin=d['origin'];shape=truth.shape
 coords=np.indices(tuple(n+1 for n in shape)).astype(float)
 for k in range(3):coords[k]=origin[k]+coords[k]*sp[k]
 z=int(np.argmax(truth.sum(axis=(0,1))))
 for j,name in enumerate(['truth','ldm-43','mcmc-43','acq-43']):
  m=d[name];iou=(m&truth).sum()/max(1,(m|truth).sum());a=fig.add_subplot(3,4,i*4+j+1,projection='3d')
  a.voxels(*coords,m,facecolors='#d49742' if j==0 else '#2979b8',edgecolor=None,alpha=.78,shade=True)
  a.set_box_aspect(np.array(shape)*sp);a.view_init(elev=22,azim=-60);a.set(xlim=(coords[0].min(),coords[0].max()),ylim=(coords[1].min(),coords[1].max()),zlim=(coords[2].min(),coords[2].max()))
  a.set_title(f'{rank.title()} | {name}\n'+('Ground truth' if j==0 else f'IoU {iou:.3f}'));a.set_xlabel('x (m)');a.set_ylabel('y (m)');a.set_zlabel('z (m)');a.tick_params(labelsize=7)
  ax=sax[i,j];code=np.zeros(shape,int);code[truth&m]=1;code[m&~truth]=2;code[truth&~m]=3
  ax.imshow(code[:,:,z].T,origin='lower',interpolation='nearest',cmap=ListedColormap(['#f3f3f3','#2689ad','#e76f51','#e5be4a']),vmin=0,vmax=3,extent=[origin[0],origin[0]+shape[0]*sp[0],origin[1],origin[1]+shape[1]*sp[1]])
  ax.set_title(f'{rank.title()} | {name} | z={origin[2]+(z+.5)*sp[2]:.1f}m');ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
fig.suptitle('Training geometry audit | latest snapshot winners (unequal search progress)\nDrill-only cases: worst / median / best LDM seed 43, selected by training IoU',fontsize=15);fig.tight_layout(rect=(0,0,1,.94));fig.savefig(O/'orebodies-3d.png',dpi=160);plt.close(fig)
sfig.suptitle('Horizontal slice at maximum true-ore area\nBlue: overlap | Red: false positive | Gold: missed ore | Gray: neither',fontsize=14);sfig.tight_layout(rect=(0,0,1,.94));sfig.savefig(O/'orebodies-errors.png',dpi=170);plt.close(sfig)
fig,axs=plt.subplots(1,3,figsize=(15,4.8));colors={'ldm':'#2563eb','mcmc':'#ea580c','acq':'#15946c'}
for name,g in r['groups'].items():
 cs=g['cases'];m=name.split('-')[0];x=np.mean([c['metrics']['loss_total'] for c in cs]);y=g['training']['mean_iou'];axs[0].scatter(x,y,color=colors[m],s=50);axs[0].annotate(name,(x,y),fontsize=8,xytext=(3,3),textcoords='offset points')
 if m!='ldm':
  for c in cs:
   curve=c.get('best_curve',[])
   if curve:
    axs[1].plot([v['seconds'] for v in curve],[v['best_loss']-curve[0]['loss'] for v in curve],color=colors[m],alpha=.08)
for j,m in enumerate(['ldm','mcmc','acq']):
 vals=[np.mean([c['metrics']['confirmation_loss_total'] for c in g['cases']]) for n,g in r['groups'].items() if n.startswith(m+'-')]
 axs[2].scatter([j-.12,j,j+.12],vals,color=colors[m],s=45)
axs[0].set(xlabel='Public observation loss (lower better)',ylabel='Hidden training IoU (higher better)',title='Different objectives: no universal winner')
axs[1].set(xlabel='Solver search seconds',ylabel='Best public loss minus initial loss',title='Baseline search actually improves its objective');axs[1].set_ylim(-.15,.005)
axs[2].set(xticks=range(3),xticklabels=['LDM','MCMC','ACQ'],ylabel='Independent confirmation loss (lower better)',title='Unseen measurements, same training geology')
for ax in axs:ax.grid(alpha=.2)
fig.suptitle('Audit diagnostics; latest training winners, not a matched discovery-budget claim',fontsize=14);fig.tight_layout();fig.savefig(O/'objective-audit.png',dpi=170);plt.close(fig)
print('plots complete')
