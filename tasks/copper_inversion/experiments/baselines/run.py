"""Frozen-solver comparison, original execution sandbox and independent scorer."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from tasks.copper_inversion.core.data import load_problem, save_problem
from tasks.copper_inversion.core.candidate import seal
from tasks.copper_inversion.core.interface_checks import InterfaceChecks
from tasks.copper_inversion.core.execution import Execution, freeze_model, digest
from ldm_tts.engine.run_store import atomic_json_write

EVALUATOR='sha256:9c0c855e43120f0000e3e5dcf79002763c65e55839e4908306cac29d1ff71062'
CASES=('drillholes_3d','drillholes_outcrops_3d')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--image',required=True)
    parser.add_argument('--acq-warmup',type=int,default=30)
    parser.add_argument('--objective',choices=['aligned','original'],default='aligned')
    parser.add_argument('--seconds',type=int,default=120);parser.add_argument('--calls',type=int,default=1000)
    parser.add_argument('--methods',nargs='+',choices=['paper_mcmc','MCMC','ACQ_MCMC','ast','qwen','fusion'],default=['paper_mcmc','MCMC','ACQ_MCMC','ast','qwen','fusion'])
    parser.add_argument('--seeds',type=int,nargs='+',default=[47,48,49]);parser.add_argument('--round',type=int,default=6)
    parser.add_argument('--final-search',action='store_true',help='Freeze the best solver through the last fully completed search round')
    args=parser.parse_args();root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    manifest_path=root/'manifest.json'
    if manifest_path.exists():
        raise ValueError('new immutable comparison root required')
    specifications=[]
    template=Path(__file__).with_name('solver.py').read_text()
    for case in CASES:
        for method in args.methods:
            work=root/case/method;work.mkdir(parents=True,exist_ok=True)
            if method in ('ast','qwen','fusion'):
                old=args.campaign/f'{case}_{method}'
                state=json.loads((old/'checkpoint.json').read_text())['state']
                obs=state['observations']
                selection_round=state['next_round']-1 if args.final_search else args.round
                assert sum(o['round_idx']<=selection_round for o in obs)==1+4*selection_round
                if args.final_search:
                    assert len(obs)==1+4*selection_round, 'final search has a partially evaluated round'
                eligible=[o for o in obs if o['round_idx']<=selection_round and o['evaluation']['status']=='succeeded']
                winner=min(eligible,key=lambda o:o['evaluation']['metrics']['loss_total'])
                source=(old/winner['candidate']['payload']['source_path']).read_bytes()
                origin=dict(run=str(old),selection_round=selection_round, selection_evaluations=1+4*selection_round,
                            selected=winner['candidate']['candidate_id'],historical_loss=winner['evaluation']['metrics']['loss_total'])
            else:
                source=template.replace("METHOD = 'paper_mcmc'",f'METHOD = {method!r}').replace('MAX_CALLS = 1000',f'MAX_CALLS = {args.calls}').replace("OBJECTIVE = 'aligned'",f'OBJECTIVE = {args.objective!r}').replace('ACQ_WARMUP = 30',f'ACQ_WARMUP = {args.acq_warmup}').encode()
                origin=dict(adapter=method, query_cap=args.calls)
            payload=seal(work,source)
            specifications.append(dict(case=case,method=method,work=str(work),payload=payload,origin=origin))
    atomic_json_write(manifest_path,dict(created_at=time.time(),seconds=args.seconds,seeds=args.seeds,objective=args.objective,acq_warmup=args.acq_warmup,
        solver_image=args.image,scorer_image=EVALUATOR,cpus=2,memory_mb=4096,
        scientific_comparison='frozen solver resource budget, NOT matched end-to-end design cost or oracle calls',sources=specifications))
    def run(spec):
        work=Path(spec['work']);problem=load_problem(spec['case'])
        solver=Execution(work/'jobs',args.image);scorer=Execution(work/'score-jobs',EVALUATOR)
        InterfaceChecks(work,solver,spec['case'])(spec['payload'])
        for seed in args.seeds:
            inputs=work/f'inputs-{seed}';save_problem(problem,inputs)
            atomic_json_write(inputs/'budget.json',dict(seed=seed,wall_time_seconds=args.seconds,cpu_count=2,memory_mb=4096))
            job,receipt=solver.run(dict(source=spec['payload']['sha256'],seed=seed,seconds=args.seconds),phase='solver',
                inputs=inputs,source=(work/spec['payload']['source_path']).parent,timeout=args.seconds,cpus=2,memory_mb=4096)
            result=dict(case=spec['case'],method=spec['method'],seed=seed,solver_receipt=receipt,status='failed')
            if not receipt['exit_code'] and not receipt['reason'] and not receipt['oom']:
                model=work/f'model-{seed}';sha=freeze_model(job/'output/model.npy',model,problem['shape'])
                sj,sr=scorer.run(dict(seed=seed,model=sha),phase='scorer',inputs=inputs,model=model,timeout=30)
                if sr['exit_code'] or sr['reason']:raise RuntimeError('trusted scoring failed')
                result.update(json.loads((sj/'output/result.json').read_text()),scorer_receipt=sr,model=str(model/'model.npy'))
                if (job/'output/search.json').exists():
                    result['search']=json.loads((job/'output/search.json').read_text())
            atomic_json_write(work/f'result-{seed}.json',result)
        return dict(case=spec['case'],method=spec['method'],status='completed')
    def guarded(spec):
        try:return run(spec)
        except Exception as exc:
            result=dict(case=spec['case'],method=spec['method'],status='failed',error=str(exc))
            atomic_json_write(Path(spec['work'])/'failure.json',result);return result
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(guarded,specifications))
    atomic_json_write(root/'status.json',dict(finished_at=time.time(),jobs=results))

if __name__=='__main__':main()
