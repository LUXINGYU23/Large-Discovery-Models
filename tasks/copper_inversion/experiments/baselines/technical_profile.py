"""Instrumented copies for mechanism analysis; never replace official results."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from tasks.copper_inversion.core.candidate import seal
from tasks.copper_inversion.core.data import load_problem, save_problem
from tasks.copper_inversion.core.execution import Execution, freeze_model
from ldm_tts.engine.run_store import atomic_json_write

ROOT=Path('/Aidata/shared/ldm/copper-technical-report-20261024')
FINAL=Path('/Aidata/shared/ldm/copper-baselines-20261024/matched-120s-final-search')
m=json.loads((FINAL/'manifest.json').read_text())
WRAPPER='''
_original_solve_for_profile = solve
def solve(problem, budget, seed):
    import json as _j, time as _t
    from pathlib import Path as _P
    stats = dict(full_distance_calls=0,narrow_distance_calls=0,distance_seconds=0.0,first_full_losses=[])
    _distance = skfmm.distance
    _loss_before = _loss
    def counted_distance(*args, **kwargs):
        key = 'narrow_distance_calls' if kwargs.get('narrow') is not None else 'full_distance_calls'
        stats[key] += 1
        t = _t.perf_counter()
        try: return _distance(*args, **kwargs)
        finally: stats['distance_seconds'] += _t.perf_counter()-t
    def counted_loss(*args, **kwargs):
        value = _loss_before(*args, **kwargs)
        if len(stats['first_full_losses']) < 8: stats['first_full_losses'].append(float(value))
        return value
    skfmm.distance = counted_distance
    globals()['_loss'] = counted_loss
    started = _t.perf_counter()
    model = _original_solve_for_profile(problem,budget,seed)
    stats['elapsed_seconds'] = _t.perf_counter()-started
    _P('/output/profile.json').write_text(_j.dumps(stats))
    return model
'''


def run(pair):
    case,method=pair;work=ROOT/'profile'/case;work.mkdir(parents=True,exist_ok=False)
    s=next(s for s in m['sources'] if s['case']==case and s['method']==method)
    source=(Path(s['work'])/s['payload']['source_path']).read_text()
    payload=seal(work,(source+WRAPPER).encode());p=load_problem(case);save_problem(p,work/'inputs')
    atomic_json_write(work/'inputs/budget.json',dict(seed=47,wall_time_seconds=120,cpu_count=2,memory_mb=4096))
    ex=Execution(work/'jobs',m['solver_image']);job,receipt=ex.run(dict(profile=payload['sha256']),phase='solver',
        inputs=work/'inputs',source=(work/payload['source_path']).parent,timeout=120,cpus=2,memory_mb=4096)
    assert not receipt['exit_code'] and not receipt['reason']
    profile=json.loads((job/'output/profile.json').read_text())
    sha=freeze_model(job/'output/model.npy',work/'model',p['shape'])
    sc=Execution(work/'scoring',m['scorer_image']);sj,sr=sc.run(dict(model=sha),phase='scorer',inputs=work/'inputs',model=work/'model',timeout=30)
    assert not sr['exit_code'] and not sr['reason']
    out=dict(case=case,method=method,original_sha256=s['payload']['sha256'],instrumented_sha256=payload['sha256'],
             profile=profile,score=json.loads((sj/'output/result.json').read_text()),receipt=receipt,
             caveat='Instrumentation changes code identity and timing; diagnostic only, not a replacement performance run')
    atomic_json_write(work/'result.json',out);return out


with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,[('drillholes_3d','ast'),('drillholes_outcrops_3d','qwen')]))
atomic_json_write(ROOT/'profile-summary.json',results)
