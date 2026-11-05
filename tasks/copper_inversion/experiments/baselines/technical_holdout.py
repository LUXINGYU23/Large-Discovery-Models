"""Post-hoc frozen-solver sensitivity audit, not an unseen discovery benchmark."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import numpy as np
from tasks.copper_inversion.core.data import load_problem, save_problem
from tasks.copper_inversion.core.execution import Execution, freeze_model
from ldm_tts.engine.run_store import atomic_json_write

ROOT = Path('/Aidata/shared/ldm/copper-technical-report-20261024')
FINAL = Path('/Aidata/shared/ldm/copper-baselines-20261024/matched-120s-final-search')
manifest = json.loads((FINAL/'manifest.json').read_text())
specs = []
for case, method in [('drillholes_3d','ast'), ('drillholes_outcrops_3d','qwen')]:
    p = load_problem(case)
    holes = np.unique(np.argwhere(p['observations'][0]['mask'])[:,:2], axis=0)
    for i, (x,y) in enumerate(holes):
        masks = [np.zeros(p['shape'], bool) for _ in p['observations']]
        masks[0][x,y,:] = True
        specs.append((case,method,f'hole_{i}',masks))
    if 'outcrops' in case:
        xx,yy,_ = np.indices(p['shape'])
        for i in range(4):
            block = ((xx < 20) if i//2==0 else (xx >= 20)) & ((yy < 25) if i%2==0 else (yy >= 25))
            specs.append((case,method,f'spatial_quadrant_{i}',[block]*len(p['observations'])))


def run(spec):
    case,method,name,masks = spec
    work=ROOT/'holdout'/case/name
    work.mkdir(parents=True,exist_ok=False)
    original=load_problem(case)
    train,held=copy.deepcopy(original),copy.deepcopy(original)
    counts=[]
    for i,mask in enumerate(masks):
        actual=mask & original['observations'][i]['mask']
        counts.append(int(actual.sum()))
        train['observations'][i]['values'][actual]=np.nan
        held['observations'][i]['values'][~actual]=np.nan
        for p in (train,held):
            p['observations'][i]['mask']=np.isfinite(p['observations'][i]['values'])
    assert sum(counts)>0
    save_problem(train,work/'inputs')
    atomic_json_write(work/'inputs/budget.json',dict(seed=47,wall_time_seconds=120,cpu_count=2,memory_mb=4096))
    origin=next(s for s in manifest['sources'] if s['case']==case and s['method']==method)
    source=(Path(origin['work'])/origin['payload']['source_path']).parent
    solver=Execution(work/'jobs',manifest['solver_image'])
    job,receipt=solver.run(dict(case=case,held=name,source=origin['payload']['sha256']),phase='solver',
                           inputs=work/'inputs',source=source,timeout=120,cpus=2,memory_mb=4096)
    result=dict(case=case,method=method,split=name,held_counts=counts,solver_receipt=receipt,
                caveat='Post-hoc: frozen code was designed on the full original case; not unbiased validation')
    if receipt['exit_code'] or receipt['reason'] or receipt['oom']:
        result['status']='failed'
    else:
        sha=freeze_model(job/'output/model.npy',work/'model',original['shape'])
        scorer=Execution(work/'score-jobs',manifest['scorer_image'])
        result.update(status='succeeded',model=str(work/'model/model.npy'))
        for label,p in [('train',train),('held',held),('full',original)]:
            inputs=work/f'score-input-{label}';save_problem(p,inputs)
            sj,sr=scorer.run(dict(model=sha,split=label),phase='scorer',inputs=inputs,model=work/'model',timeout=30)
            assert not sr['exit_code'] and not sr['reason']
            result[label]=json.loads((sj/'output/result.json').read_text())
    atomic_json_write(work/'result.json',result)
    return result


with ThreadPoolExecutor(max_workers=2) as pool:
    results=list(pool.map(run,specs))
atomic_json_write(ROOT/'holdout-summary.json',results)
print('Completed',len(results),'post-hoc sensitivity runs',flush=True)
