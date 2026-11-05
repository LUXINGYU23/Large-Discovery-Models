import json,ast,hashlib
from pathlib import Path
import numpy as np
from tasks.copper_inversion.v2.data import read_problem
from tasks.copper_inversion.v2.baselines import initialize
from tasks.copper_inversion.v2.objective import score
from tasks.copper_inversion.core.execution import digest,source_identity
P=Path('/Aidata/shared/ldm/copper-v2-evaluator/runs/copper-formal-study-20261024-r1');O=P/'interim-audit-20261024';D=Path('/Aidata/shared/ldm/copper-v2-evaluator/datasets/formal-20261024-r1/train')
plan=json.loads((O/'plan.json').read_text());manifest=json.loads((D/'manifest.json').read_text());identity=source_identity();image='sha256:f6856385a0da9444b1e5edb5048b2f6081593fe568c3473c2fda524d0c4489f5'
report={'common_budget':plan['common_charged_budget'],'groups':{},'priors':[]};models={}
for name in plan['groups']:
 sha=plan['latest_training'][name]['source_sha256'];r=P/name
 result_path=next(p for p in (r/'evaluations').glob('*/result.json') if json.loads(p.read_text())['candidate_id']==sha)
 rows=json.loads((result_path.parent/'private-case-results.json').read_text());bycase={x['case_id']:x for x in rows}
 source=(r/'candidates'/sha/'solver.py').read_text();tree=ast.parse(source)
 item={'sha':sha,'training':plan['latest_training'][name]['metrics'],'imports':[ast.unparse(n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))],'cases':[]}
 if not name.startswith('ldm'):
  item['config']=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONFIG' for t in n.targets))
 for case in manifest['cases']:
  cid=case['case_id'];row=bycase[cid];p=read_problem(D/case['public']);truth=np.load(D/case['truth']/'truth.npy');field=np.load(result_path.parent/cid/'47/model/model.npy');mask=field>0
  initial=initialize(p);im=initial>0
  iou=lambda a:float((a&truth).sum()/max(1,(a|truth).sum()))
  entry={'case_id':cid,'modality':case['modality'],'metrics':row['metrics'],'solver_seconds':row['solver_wall_seconds'],'initial_iou':iou(im),'initial_public_loss':score(p,initial)['loss_total'],'inactive_positive':int((mask&(~p['domain_mask']|p['forbidden_mask'])).sum()),'changed_voxels_from_initial':int((mask!=im).sum())}
  if not name.startswith('ldm'):
   key=digest(dict(identity=dict(evaluation=result_path.parent.name,case=cid,seed=47),science_module='v2',truth_digest=None,phase='solver',image=image,timeout=120,cpus=2,memory_mb=4096,implementation=identity))
   f=r/'jobs'/key/'output/search.json'
   if not f.exists():
    key=digest(dict(identity=dict(evaluation=result_path.parent.name,case=cid,seed=47),science_module='v2',truth_digest=None,phase='solver',image=image,timeout=120.0,cpus=2,memory_mb=4096,implementation=identity));f=r/'jobs'/key/'output/search.json'
   if f.exists():
    search=json.loads(f.read_text());entry['search']={k:v for k,v in search.items() if k!='curve'};entry['start_loss']=search['curve'][0]['loss'];entry['best_curve']=search['curve']
   else:entry['search_missing']=str(f)
  item['cases'].append(entry)
  if name=='ldm-42':report['priors'].append({'case':cid,'priors':p['priors']})
  if name in ['ldm-43','mcmc-43','acq-43']:
   models[(name,cid)]=mask
 report['groups'][name]=item
 print(name,sha,item['training'],flush=True)
# Predefined selection: drill-only cases at worst / median / best LDM43 IoU.
cases=[c for c in manifest['cases'] if c['modality']=='drill'];scores={c['case_id']:next(x for x in report['groups']['ldm-43']['cases'] if x['case_id']==c['case_id'])['metrics']['iou'] for c in cases};cases.sort(key=lambda c:(scores[c['case_id']],c['case_id']))
selected=[cases[0],cases[len(cases)//2],cases[-1]];report['visual_cases']=[]
for label,c in zip(['worst','median','best'],selected):
 cid=c['case_id'];p=read_problem(D/c['public']);data={'truth':np.load(D/c['truth']/'truth.npy'),'feasible':p['domain_mask']&~p['forbidden_mask'],'spacing':np.asarray(p['spacing']),'origin':np.asarray(p['origin']),'points':p['observations'][0]['points'],'labels':p['observations'][0]['labels']}
 for name in ['ldm-43','mcmc-43','acq-43']:data[name]=models[(name,cid)]
 np.savez_compressed(O/(label+'.npz'),**data);report['visual_cases'].append({'rank':label,'case_id':cid,'ldm_iou':scores[cid]})
(O/'forensics.json').write_text(json.dumps(report,indent=2))
