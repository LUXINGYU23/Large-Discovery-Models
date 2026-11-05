"""Independent development check of fixed synthetic population priors.

No evaluation truth is read and no threshold is fitted to these outcomes. A
failed prespecified parent is retained as a calibration failure, not replaced.
"""
import argparse,json,subprocess,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.ndimage import label
from tasks.copper_inversion.v2.geological_policy import public_priors,POLICY_VERSION,MAX_COMPONENTS
from tasks.copper_inversion.v2.data import read_problem
from tasks.copper_inversion.v2.baselines import initialize
from tasks.copper_inversion.v2.geometry import canonicalize
from tasks.copper_inversion.v2.fixtures import fixtures


def check_numerics():
    rows=[]
    for index,p in enumerate(fixtures()):
        p['priors']=public_priors(p['domain_mask'] & ~p['forbidden_mask'],p['spacing'])
        initial=initialize(p);mask,_=canonicalize(p,initial)
        for name,f in [('empty',np.full(p['shape'],-1.)),('full_feasible',np.where(p['domain_mask']&~p['forbidden_mask'],1.,-1.))]:
            try:canonicalize(p,f)
            except ValueError:rejected=True
            else:rejected=False
            rows.append(dict(fixture=index,probe=name,passed=rejected))
        q={**p,'spacing':(np.asarray(p['spacing'])*1000).tolist(),'origin':(np.asarray(p['origin'])*1000).tolist(),
           'observations':[{**o,'points':o['points']*1000,'transition_scale':o['transition_scale']*1000,'contact_sigma':o['contact_sigma']*1000} for o in p['observations']]}
        q['priors']=public_priors(q['domain_mask']&~q['forbidden_mask'],q['spacing'])
        canonicalize(q,initial)
        rows.append(dict(fixture=index,probe='legal_initial_and_unit_conversion',passed=True,components=int(label(mask)[1])))
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--generator-python',type=Path,required=True);args=parser.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    families=['folded','lens','multiple','faulted','faulted_lens']
    plan=[dict(family=family,seed=int(np.random.SeedSequence([2026102407,i,j]).generate_state(1)[0])) for i,family in enumerate(families) for j in range(2)]
    (root/'plan.json').write_text(json.dumps(dict(role='independent_development',policy=POLICY_VERSION,parents=plan,no_replacements=True),indent=2))
    def generate(item):
        destination=root/f"{item['family']}-{item['seed']}"
        cache=root/'cache'/destination.name;cache.mkdir(parents=True,mode=0o700)
        with destination.with_suffix('.log').open('w') as log:
            result=subprocess.run([str(args.generator_python),'-m','tasks.copper_inversion.experiments.v2.geology','--family',item['family'],'--seed',str(item['seed']),'--output',str(destination)],env={**os.environ,'TMPDIR':str(cache)},stdout=log,stderr=subprocess.STDOUT,timeout=300)
        if result.returncode:return dict(**item,passed=False,reason='generator failure; retained without replacement')
        meta=json.loads((destination/'private-parent.json').read_text())
        with np.load(destination/'geology.npz') as d:truth=d['truth'];feasible=~d['forbidden_mask']
        priors=public_priors(feasible,meta['spacing']);volume=float(truth.sum()*np.prod(meta['spacing']));components=int(label(truth)[1])
        return dict(**item,geometry_sha256=meta['geometry_sha256'],volume_fraction=float(truth.sum()/feasible.sum()),components=components,
                    passed=bool(priors['hard_volume_bounds'][0]<=volume<=priors['hard_volume_bounds'][1] and components<=MAX_COMPONENTS and not np.any(truth&~feasible)))
    with ThreadPoolExecutor(max_workers=5) as pool:rows=list(pool.map(generate,plan))
    numeric=check_numerics()
    report=dict(status='passed' if all(r['passed'] for r in rows+numeric) else 'failed',policy=POLICY_VERSION,
                calibration_scope='independent_development',used_evaluation_truth=False,parents=rows,numerical_checks=numeric,
                limitations=['Fixed synthetic population assumptions; not field-geology validation','No fitted area or fragment penalty','At most four connected bodies; 2-65% of feasible domain; same public policy for all methods'])
    (root/'geological_constraints.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if report['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
