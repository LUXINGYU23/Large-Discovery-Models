"""Plot post-hoc spatial/whole-hole sensitivity; retain all folds."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tasks.copper_inversion.core.data import read_problem
from tasks.copper_inversion.core.geometry import signed_distance

ROOT=Path('/Aidata/shared/ldm/copper-technical-report-20261024')
FINAL=Path('/Aidata/shared/ldm/copper-baselines-20261024/matched-120s-final-search')
rows=json.loads((ROOT/'holdout-summary.json').read_text());summary=[]
for d in rows:
    work=Path(d['model']).parents[1];p=read_problem(work/'score-input-held')
    r=json.loads((FINAL/d['case']/d['method']/'result-47.json').read_text())
    phi=signed_distance(np.load(r['model']),p['shape'],p['spacing'])
    binary=sum(int(((o['values']==0)|(o['values']==1)).sum()) for o in p['observations'])
    wrong=sum(int(((phi[o['values']==0]>0)).sum())+int(((phi[o['values']==1]<=0)).sum()) for o in p['observations'])
    contacts=np.concatenate([phi[o['values']==.5] for o in p['observations']])
    h=d['held']['metrics']
    summary.append(dict(case=d['case'],split=d['split'],binary_count=binary,held_count=sum(d['held_counts']),
                        wrong=h['label_mismatches'],error_rate=h['label_mismatches']/binary,
                        full_fit_wrong=wrong,full_fit_error_rate=wrong/binary,
                        contact_count=len(contacts),contact_rmse=h['contact_rmse'],
                        full_fit_contact_rmse=float(np.sqrt(np.mean(contacts**2))),held_loss=h['loss_total'],
                        full_loss=d['full']['metrics']['loss_total']))
(ROOT/'holdout-metrics.json').write_text(json.dumps(summary,indent=2))
fig,axes=plt.subplots(2,3,figsize=(14,8))
groups=[('Drillholes | leave one hole',[r for r in summary if r['case']=='drillholes_3d']),
        ('Drillholes + outcrops | leave one hole',[r for r in summary if r['case']=='drillholes_outcrops_3d' and r['split'].startswith('hole')]),
        ('Drillholes + outcrops | spatial blocks',[r for r in summary if r['split'].startswith('spatial')])]
for col,(title,rr) in enumerate(groups):
    x=np.arange(len(rr))
    for row,(key,refkey,scale,label) in enumerate([('error_rate','full_fit_error_rate',100,'Held 0/1 classification error (%)'),('contact_rmse','full_fit_contact_rmse',1,'Held contact RMSE (grid units)')]):
        ax=axes[row,col];ax.bar(x-.18,[r[refkey]*scale for r in rr],width=.35,color='#9dbbc0',label='Full-data fit (in-sample)')
        ax.bar(x+.18,[r[key]*scale for r in rr],width=.35,color='#c85d55',label='Labels removed from solver input')
        ax.set_xticks(x,[f'{"Q" if col==2 else "H"}{i}' for i in x]);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
        if row==0:ax.set_title(title)
axes[0,0].legend(fontsize=8)
fig.suptitle('Post-hoc sensitivity only: the frozen code was designed on the full cases',fontsize=14)
fig.tight_layout();fig.savefig(ROOT/'figures/holdout_sensitivity.png',dpi=160)
print(json.dumps(summary,indent=2))
