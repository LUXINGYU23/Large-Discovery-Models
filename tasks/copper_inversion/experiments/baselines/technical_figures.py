"""Remote scientific plots and geometry audit of frozen copper outputs."""
import itertools
import json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes, mesh_surface_area
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tasks.copper_inversion.core.data import load_problem
from tasks.copper_inversion.core.geometry import signed_distance
from tasks.copper_inversion.core.objective import score

ROOT=Path('/Aidata/shared/ldm/copper-technical-report-20261024')
BASE=Path('/Aidata/shared/ldm/copper-baselines-20261024')
FINAL=BASE/'matched-120s-final-search'
SEARCH=Path('/Aidata/shared/ldm/copper-encoding-ablation-v3-20261023/full')
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
CASES=['drillholes_3d','drillholes_outcrops_3d']
SELECT={'drillholes_3d':'ast','drillholes_outcrops_3d':'qwen'}
LABELS=['Drillholes','Drillholes + outcrops']
COLORS={'Initial':'#8996a2','MCMC 120s':'#dc9650','MCMC 1000 calls':'#8565ab','LDM':'#087e8b'}


def read_model(case,folder,method,seed=47):
    result=json.loads((BASE/folder/case/method/f'result-{seed}.json').read_text())
    return np.load(result['model']),result


def mesh(field,p):
    v,f,_,_=marching_cubes(np.asarray(field,dtype=np.float32),level=0,spacing=tuple(p['spacing']))
    v+=np.array(p['origin'])+.5*np.array(p['spacing'])
    return v,f


def boundary_faces(mask):
    return [f'{"xyz"[a]}{sign}' for a in range(3) for sign,idx in [('-',0),('+',-1)] if np.take(mask,idx,axis=a).any()]


def diagnostics(field,p):
    phi=signed_distance(field,p['shape'],p['spacing']);mask=phi>0
    labels,n=ndi.label(mask,ndi.generate_binary_structure(3,1));sizes=np.bincount(labels.ravel())[1:]
    verts,faces=mesh(phi,p)
    return dict(score=score(p,field),components_6=n,largest_component_fraction=float(sizes.max()/sizes.sum()),
                boundary_faces=boundary_faces(mask),isosurface_area=float(mesh_surface_area(verts,faces)),
                active_positive_voxels=int((mask&p['active_mask']).sum()),
                volume_fraction=float(mask.mean()))


data={};geo={}
for case in CASES:
    p=load_problem(case)
    models={'Initial':p['initial_level_set']}
    for label,folder,method in [('MCMC 120s','matched-120s-v2','paper_mcmc'),
                                ('MCMC 1000 calls','paper-1000-v2','paper_mcmc'),
                                ('LDM','matched-120s-final-search',SELECT[case])]:
        models[label]=read_model(case,folder,method)[0]
    data[case]=(p,models)
    geo[case]={label:diagnostics(field,p) for label,field in models.items()}
    selected=[read_model(case,'matched-120s-final-search',SELECT[case],s)[0]>0 for s in [47,48,49]]
    geo[case]['seed_geometry']={'pairwise_iou':[float((a&b).sum()/(a|b).sum()) for a,b in itertools.combinations(selected,2)],
                                'varying_voxels':int((np.min(selected,axis=0)!=np.max(selected,axis=0)).sum())}
    geo[case]['observations']=[{'name':o['name'],'counts':{str(v):int((o['values']==v).sum()) for v in [0,.5,1]}} for o in p['observations']]

# Same-budget performance, and frozen-solver repeatability.
rows=json.loads((FINAL/'baseline-comparison.json').read_text())
fig,axes=plt.subplots(2,2,figsize=(12,8))
methods=['paper_mcmc','MCMC','ACQ_warmup4','LDM']
for c,case in enumerate(CASES):
    rr=[next(r for r in rows if r['case']==case and r['method']==(SELECT[case] if m=='LDM' else m)) for m in methods]
    for k,key in enumerate(['losses','wall_seconds']):
        ax=axes[k,c]
        for i,(m,r) in enumerate(zip(methods,rr)):
            values=r[key];color=COLORS['LDM'] if m=='LDM' else '#c0824c'
            ax.bar(i,np.mean(values),color=color,alpha=.8,width=.6)
            ax.errorbar(i,np.mean(values),yerr=np.std(values,ddof=1),fmt='none',color='#222',capsize=4)
            ax.scatter(i+np.linspace(-.12,.12,3),values,color='#222',s=16,zorder=4)
        ax.set_xticks(range(4),['Paper\nMCMC','Package\nMCMC','ACQ\nwarmup4','LDM\nselected'])
        ax.set_ylabel('Loss (log scale)' if k==0 else 'Solver wall time (s)')
        if k==0:ax.set_yscale('log');ax.set_title(LABELS[c])
        else:ax.axhline(120,color='#b33',ls='--',lw=1,label='120 s cap');ax.set_ylim(0,130)
fig.suptitle('Frozen solver comparison: equal 120 s / 2 CPU / 4 GiB limits',fontsize=14)
fig.tight_layout();fig.savefig(OUT/'performance.png');plt.close(fig)

# Search curves: discovery and baseline oracle calls deliberately use separate axes.
fig,axes=plt.subplots(2,2,figsize=(12,8))
curves={}
for c,case in enumerate(CASES):
    campaigns=[]
    for mode in ['ast','qwen','fusion']:
        obs=json.loads((SEARCH/f'{case}_{mode}'/'checkpoint.json').read_text())['state']['observations']
        campaigns.append(obs)
    pooled=[]
    for rnd in range(21):
        observed=[o for obs in campaigns for o in obs if o['round_idx']<=rnd]
        pooled.append({'round':rnd,'total_evaluations':len(observed),'best':min(o['evaluation']['metrics']['loss_total'] for o in observed if o['evaluation']['status']=='succeeded')})
    curves[case]=pooled
    ax=axes[0,c];ax.step([x['total_evaluations'] for x in pooled],[x['best'] for x in pooled],where='post',color=COLORS['LDM'],lw=2)
    ax.set_yscale('log');ax.set_title(LABELS[c]+' | LDM discovery');ax.set_xlabel('Total formal solver evaluations across 3 searches');ax.set_ylabel('Best measured loss')
    inset=ax.inset_axes([.42,.43,.53,.48]);inset.plot([x['round'] for x in pooled[1:]],[x['best'] for x in pooled[1:]],color=COLORS['LDM']);inset.set_xlabel('Round',fontsize=8);inset.tick_params(labelsize=7)
    ax=axes[1,c]
    for seed in [47,48,49]:
        d=json.loads((BASE/'matched-120s-v2'/case/'paper_mcmc'/f'result-{seed}.json').read_text())
        arr=d['search']['curve'];ax.step([x['elapsed_seconds'] for x in arr],[x['best_loss'] for x in arr],where='post',label=f'MCMC seed {seed}',alpha=.8)
    ax.set_xlabel('MCMC internal search time (s)');ax.set_ylabel('Best aligned objective');ax.set_title('MCMC within 120 s solver cap');ax.legend(fontsize=8)
fig.suptitle('Discovery budget and per-solver search are different cost axes',fontsize=14)
fig.tight_layout();fig.savefig(OUT/'search_curves.png');plt.close(fig)
(ROOT/'search-curves.json').write_text(json.dumps(curves,indent=2))

# Full zero surfaces, no active-mask clipping; all plots use same extents/view.
fig=plt.figure(figsize=(17,9))
interactive=make_subplots(rows=2,cols=4,specs=[[{'type':'scene'}]*4]*2,
                          subplot_titles=[f'{title}: {label}' for title in LABELS for label in COLORS])
for row,case in enumerate(CASES):
    p,models=data[case];origin=np.array(p['origin']);end=origin+np.array(p['shape'])*p['spacing']
    for col,(label,fld) in enumerate(models.items()):
        phi=signed_distance(fld,p['shape'],p['spacing']);v,f=mesh(phi,p)
        ax=fig.add_subplot(2,4,row*4+col+1,projection='3d')
        surface=Poly3DCollection(v[f],alpha=.45,facecolor=COLORS[label],edgecolor='none',rasterized=True);ax.add_collection3d(surface)
        air=np.argwhere((phi>0)&~p['active_mask']);air=air[::max(1,len(air)//700)];air=origin+(air+.5)*p['spacing']
        if len(air):ax.scatter(*air.T,c='#c9303e',s=2,alpha=.5)
        obs=p['observations'][0];idx=np.argwhere(obs['mask']);xyz=origin+(idx+.5)*p['spacing'];values=obs['values'][obs['mask']]
        ax.scatter(*xyz.T,c=values,cmap=ListedColormap(['#294571','#f4d35e','#a44b06']),s=9,edgecolors='black',linewidths=.2)
        ax.set(xlim=(origin[0],end[0]),ylim=(origin[1],end[1]),zlim=(origin[2],end[2]),xlabel='x',ylabel='y',zlabel='z')
        ax.set_box_aspect(np.array(p['shape'])*p['spacing']);ax.view_init(elev=22,azim=-60)
        met=geo[case][label]['score'];ax.set_title(f'{LABELS[row]}\n{label}\nL={met["loss_total"]:.3f}; air={100*met["air_fraction_of_positive"]:.1f}%',fontsize=10)
        interactive.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],color=COLORS[label],opacity=.65,name=label,showlegend=False),row=row+1,col=col+1)
        interactive.add_trace(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='markers',marker=dict(size=2,color=values,colorscale=[[0,'#294571'],[.5,'#f4d35e'],[1,'#a44b06']]),name='Drill labels',showlegend=False),row=row+1,col=col+1)
        if len(air):interactive.add_trace(go.Scatter3d(x=air[:,0],y=air[:,1],z=air[:,2],mode='markers',marker=dict(size=2,color='#c9303e'),name='Positive outside active mask',showlegend=False),row=row+1,col=col+1)
fig.suptitle('Zero isosurfaces | seed 47 fixed in advance | red dots: positive cells outside active mask',fontsize=14)
fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(OUT/'orebodies_3d.png');plt.close(fig)
interactive.update_layout(height=1000,width=1700,title='Frozen orebody surfaces — rotate/zoom each panel. Red = outside active mask. No ground truth available.')
for i in range(1,9):interactive.update_layout(**{f'scene{i if i>1 else ""}':dict(aspectmode='data',xaxis=dict(range=[0,40]),yaxis=dict(range=[0,50]),zaxis=dict(range=[-25,5]))})
interactive.write_html(ROOT/'orebodies-interactive.html',include_plotlyjs=True,full_html=True)

# Cross-sections show solid occupancy and active-mask boundary without hiding air.
for case in CASES:
    p,models=data[case];fig,axes=plt.subplots(3,4,figsize=(15,10))
    for c,(label,fld) in enumerate(models.items()):
        phi=signed_distance(fld,p['shape'],p['spacing'])
        for row,(axis,index,axesnames) in enumerate([(1,27,('x','z')),(0,25,('y','z')),(2,19,('x','y'))]):
            ax=axes[row,c];sl=[slice(None)]*3;sl[axis]=index;sl=tuple(sl)
            occupancy=(phi[sl]>0).T;active=p['active_mask'][sl].T;display=occupancy.astype(int);display[occupancy&~active]=2
            dims=[i for i in range(3) if i!=axis];extent=[p['origin'][dims[0]],p['origin'][dims[0]]+p['shape'][dims[0]],p['origin'][dims[1]],p['origin'][dims[1]]+p['shape'][dims[1]]]
            ax.imshow(display,origin='lower',extent=extent,cmap=ListedColormap(['#eef1f4','#61b6b8','#e26872']),vmin=0,vmax=2,interpolation='nearest')
            gx=p['origin'][dims[0]]+np.arange(occupancy.shape[1])+.5;gy=p['origin'][dims[1]]+np.arange(occupancy.shape[0])+.5
            if active.min()!=active.max():ax.contour(gx,gy,active,levels=[.5],colors='black',linewidths=.8)
            for obs in p['observations']:
                vals=obs['values'][sl];idx=np.argwhere(np.isfinite(vals));v=vals[np.isfinite(vals)]
                if len(idx):ax.scatter(p['origin'][dims[0]]+idx[:,0]+.5,p['origin'][dims[1]]+idx[:,1]+.5,c=v,cmap=ListedColormap(['#294571','#f4d35e','#a44b06']),s=8,edgecolors='black',linewidths=.2,vmin=0,vmax=1)
            ax.set_xlabel(axesnames[0]);ax.set_ylabel(axesnames[1]);ax.set_title(f'{label}; {"xyz"[axis]} index={index}')
    fig.suptitle(f'{case}: solid orebody (teal), outside-active orebody (red), active boundary (black)',fontsize=13)
    fig.tight_layout();fig.savefig(OUT/f'sections_{case}.png');plt.close(fig)

(ROOT/'geometry.json').write_text(json.dumps(geo,indent=2))
print(json.dumps(geo,indent=2))
