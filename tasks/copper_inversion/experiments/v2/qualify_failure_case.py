"""Three-seed regression of an identified training failure, never held-out data."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from tasks.copper_inversion.core.execution import Execution,freeze_model,source_identity
from tasks.copper_inversion.core.candidate import seal
from tasks.copper_inversion.v2.data import read_problem,save_problem
from tasks.copper_inversion.v2.suite import load_manifest
from tasks.copper_inversion.experiments.v2.tune_stochastic import source_for
from tasks.copper_inversion.experiments.v2.qualify_load import IMAGE


def main():
    p=argparse.ArgumentParser()
    for name in ('root','manifest','failure'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();manifest=load_manifest(a.manifest)
    if manifest['split']!='train':raise ValueError('regression must not access held-out data')
    failure=json.loads(a.failure.read_text());case=next(c for c in manifest['cases'] if c['case_id']==failure['case_id'])
    root=a.root.resolve();root.mkdir(parents=True,exist_ok=False)
    problem=read_problem(a.manifest.parent/case['public'])
    template=Path(__file__).with_name('stochastic_solver.py').read_text()
    payload=seal(root,source_for(template,failure['configuration']).encode())
    execution=Execution(root/'jobs',IMAGE)
    (root/'plan.json').write_text(json.dumps(dict(scope='training failure regression only',failure=failure,
        implementation=source_identity(),source=payload['sha256'],seeds=[47,48,49],seconds=120),indent=2))
    def one(seed):
        inputs=root/f'input-{seed}';save_problem(problem,inputs)
        (inputs/'budget.json').write_text(json.dumps(dict(seed=seed,wall_time_seconds=120,cpu_count=2,memory_mb=4096)))
        job,receipt=execution.run(dict(source=payload['sha256'],case=case['case_id'],seed=seed),phase='solver',inputs=inputs,
            source=(root/payload['source_path']).parent,timeout=120,cpus=2,memory_mb=4096,science_module='v2')
        if receipt['exit_code'] or receipt['reason']:return dict(seed=seed,status='failed',receipt=receipt)
        model=root/f'model-{seed}';sha=freeze_model(job/'output/model.npy',model,problem['shape'])
        scored,score_receipt=execution.run(dict(model=sha,seed=seed),phase='scorer',inputs=inputs,model=model,
            truth=a.manifest.parent/case['truth'],timeout=30,science_module='v2')
        if score_receipt['exit_code'] or score_receipt['reason']:raise RuntimeError('independent scorer failed')
        result=json.loads((scored/'output/result.json').read_text())
        search=json.loads((job/'output/search.json').read_text())
        return dict(seed=seed,status=result['status'],receipt=receipt,metrics=result['metrics'],
                    calls=search['calls'],invalid_proposals=search['invalid'])
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(one,[47,48,49]))
    passed=all(r['status']=='succeeded' for r in rows)
    result=dict(status='passed' if passed else 'failed',scope='training failure regression',rows=rows)
    (root/'completion.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(status=result['status'],seeds=3)))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
