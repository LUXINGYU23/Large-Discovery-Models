"""Bounded diagnostic of old vs fixed transitions on public interface fixtures.

No research, training feedback slots or held-out data. All variants get the same
120-second/2 CPU/4 GiB solve limit and independent source-free scoring jobs.
"""
import argparse,json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .tune_stochastic import configurations,source_for
from tasks.copper_inversion.core.candidate import seal
from tasks.copper_inversion.core.execution import Execution,freeze_model
from tasks.copper_inversion.v2.fixtures import fixtures
from tasks.copper_inversion.v2.data import save_problem


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--old-template',type=Path,required=True);parser.add_argument('--image',required=True);a=parser.parse_args()
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    templates={'old':a.old_template.read_text(),'fixed':Path(__file__).with_name('stochastic_solver.py').read_text()}
    problems=list(fixtures());variants={}
    for version,template in templates.items():
        for method in ['mcmc_v2','acq_v2']:
            name=version+'-'+method
            variants[name]=seal(root,source_for(template,configurations(method,1,42)[0]).encode())
    for i,p in enumerate(problems):
        d=root/'inputs'/str(i);save_problem(p,d)
        (d/'budget.json').write_text(json.dumps(dict(seed=47,wall_time_seconds=120,cpu_count=2,memory_mb=4096)))
    execution=Execution(root/'jobs',a.image)
    (root/'plan.json').write_text(json.dumps(dict(role='public_development_diagnostic',variants=variants,
        cases=[p['case_id'] for p in problems],seconds=120,cpus=2,memory_mb=4096,seeds=[47],image=execution.image),indent=2))
    def one(item):
        name,i=item;payload=variants[name];identity={'variant':payload['sha256'],'fixture':problems[i]['case_id'],'seed':47}
        inputs=root/'inputs'/str(i)
        job,receipt=execution.run(identity,phase='solver',inputs=inputs,source=(root/payload['source_path']).parent,
            timeout=120,cpus=2,memory_mb=4096,science_module='v2')
        row=dict(name=name,fixture=i,solver_seconds=receipt['wall_seconds'],reason=receipt['reason'],exit_code=receipt['exit_code'])
        if not (receipt['exit_code'] or receipt['reason'] or receipt['oom']):
            model=root/'models'/name/str(i);sha=freeze_model(job/'output/model.npy',model,problems[i]['shape'])
            scored,s=execution.run(dict(identity=identity,model=sha),phase='scorer',inputs=inputs,model=model,timeout=30,science_module='v2')
            if s['exit_code'] or s['reason'] or s['oom']:raise RuntimeError('independent scorer failed')
            row.update(json.loads((scored/'output/result.json').read_text()))
            search=json.loads((job/'output/search.json').read_text());row['search']={k:v for k,v in search.items() if k!='curve'}
            row['initial_public_loss']=search['curve'][0]['loss']
        (root/f'{name}-{i}.json').write_text(json.dumps(row,indent=2));return row
    with ThreadPoolExecutor(max_workers=16) as pool:rows=list(pool.map(one,[(n,i) for n in variants for i in range(len(problems))]))
    report=dict(status='passed' if all(r.get('status')=='succeeded' and r['metrics']['loss_total']<=r['initial_public_loss']+1e-9 for r in rows) else 'failed',
        scope='transition legality, no worse than initial public objective; not a full benchmark comparison',rows=rows)
    (root/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps({'status':report['status'],'completed':len(rows)}))
    if report['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
