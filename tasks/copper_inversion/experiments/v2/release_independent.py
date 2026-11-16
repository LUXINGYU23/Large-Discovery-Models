"""Trusted, one-time synthetic release after numerical and access qualification.

The functions are shared with `experiments/rsi/release.py`; `main` is the legacy
two-pack release. An RSI generation plan must use the RSI release tool instead.
"""
import argparse
import json
import os
import shutil
from pathlib import Path
import numpy as np
from tasks.copper_inversion.v2.data import read_problem
from tasks.copper_inversion.v2.suite import file_hash,load_manifest
from tasks.copper_inversion.v2.qualification import verify_geological_constraints
from tasks.copper_inversion.v2.geometry import canonicalize
from .split_audit import shape_fingerprint


def qualify_evidence(root,calibration,isolation,geological_constraints):
    """Fail closed unless population, calibration, isolation and prior evidence passed."""
    if not (root/'qa-pipeline-complete.json').exists():raise ValueError('population pipeline incomplete')
    evidence={name:json.loads(Path(path).read_text()) for name,path in [('calibration',calibration),('isolation',isolation),('population',root/'population-qa.json')]}
    if any(v['status']!='passed' for v in evidence.values()):raise ValueError('qualification failed')
    constraints=json.loads(Path(geological_constraints).read_text())
    if (constraints.get('status')!='passed' or constraints.get('calibration_scope')!='independent_development'
            or constraints.get('used_evaluation_truth') is not False):
        raise ValueError('independent geological-prior calibration is required')
    evidence['geological_constraints']=constraints
    from tasks.copper_inversion.v2.geological_policy import POLICY_VERSION
    if constraints.get('policy')!=POLICY_VERSION:raise ValueError('geological calibration policy differs')
    return evidence


def historical_fingerprints(historical):
    found=set()
    for path in Path(historical).rglob('truth.npy'):
        found.add(shape_fingerprint(np.load(path,allow_pickle=False)))
    # Calibration parents retain truth in the original generator archive.
    for path in Path(historical).rglob('geology.npz'):
        with np.load(path,allow_pickle=False) as data:
            found.add(shape_fingerprint(data['truth']))
    return found


def historical_exclusion(historical,current):
    """`historical` maps a label to a prior release or generator archive."""
    historic=set();counts={}
    for label,path in sorted(historical.items()):
        found=historical_fingerprints(path)
        if not found:raise ValueError('historical exclusion set missing: '+label)
        counts[label]=len(found);historic|=found
    if set(current)&historic:raise ValueError('new population collides with previously seen geometry')
    return dict(status='passed',historical_fingerprints=len(historic),historical_sources=counts,new_fingerprints=len(set(current)),
        scope='coarse normalized rotation/reflection fingerprint exclusion against prior private suites')


def pair_confirmation(public_path,hidden_path,truth_path):
    """Confirmation shares geometry, masks and priors but no public measurement."""
    public=read_problem(public_path);confirm=read_problem(hidden_path)
    truth=np.load(truth_path,allow_pickle=False)
    canonicalize(public,np.where(truth,1.,-1.))
    for key in ('shape','origin','spacing'):
        if public[key]!=confirm[key]:raise ValueError('confirmation geometry mismatch')
    for key in ('domain_mask','forbidden_mask'):
        if not np.array_equal(public[key],confirm[key]):raise ValueError('confirmation domain mismatch')
    if public['priors']!=confirm['priors']:raise ValueError('confirmation geological priors differ')
    if all(np.array_equal(x['points'],y['points']) for x,y in zip(public['observations'],confirm['observations'])):
        raise ValueError('confirmation reuses public measurements')
    return public,confirm


def expected_cases(root):
    """Two packs per planned parent; the legacy population had 56 parents."""
    path=root/'generation-plan.json'
    plan=json.loads(path.read_text()) if path.is_file() else {}
    if plan.get('schema')=='rsi-generation-plan-v1':
        raise ValueError('one-layout RSI populations are released by experiments.rsi.release')
    counts=plan.get('counts')
    if isinstance(counts,dict) and counts and all(type(v) is int and v>0 for v in counts.values()):
        return 2*sum(counts.values())
    return 112


def main():
    p=argparse.ArgumentParser()
    for name in ('dataset','output','calibration','isolation','geological-constraints'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--historical',type=Path,nargs='+',required=True)
    a=p.parse_args();root=a.dataset
    if os.geteuid()!=991:raise ValueError('requires private evaluator account')
    expected=expected_cases(root)
    evidence=qualify_evidence(root,a.calibration,a.isolation,a.geological_constraints)
    current={r['fingerprint'] for r in evidence['population']['parents']}
    evidence['historical_disjointness']=historical_exclusion({str(i):path for i,path in enumerate(a.historical)},current)
    source_manifests=[root/'id/suites'/s/'manifest.json' for s in ('train','validation','test_id')]+[root/'ood/suites/test_ood/manifest.json']
    pending=[]
    for path in source_manifests:
        verify_geological_constraints(path)
        m=load_manifest(path);part='ood' if m['split']=='test_ood' else 'id'
        parentpacks={}
        for confirmation in (root/part/'confirmation').glob('*/private-manifest.json'):
            c=json.loads(confirmation.read_text());parentpacks[c['parent_digest']]=(confirmation.parent,c)
        for row in m['cases']:
            folder,c=parentpacks[row['parent_id']]
            hidden=folder/next(x['case_id'] for x in c['cases'] if x['modality']==row['modality'])
            pair_confirmation(path.parent/row['public'],hidden,path.parent/row['truth']/'truth.npy')
            pending.append((path,m['split'],row,hidden))
    if len(pending)!=expected:raise ValueError('incomplete prespecified population')
    evidence['confirmation']=dict(status='passed',cases=len(pending),scope='independent whole-hole/location and noise streams; geometry/masks identical; not a new geology parent')
    out=a.output.resolve();out.mkdir(mode=0o700,parents=True,exist_ok=False)
    manifests={s:dict(schema='copper-suite-v2',stage='formal',split=s,seeds=[47,48,49] if s.startswith('test') else [47],cases=[]) for s in ('train','validation','test_id','test_ood')}
    for path,split,row,hidden in pending:
        public=out/split/'public'/row['case_id'];truth=out/split/'private'/row['case_id']
        shutil.copytree(path.parent/row['public'],public);truth.mkdir(parents=True)
        shutil.copy2(path.parent/row['truth']/'truth.npy',truth/'truth.npy')
        shutil.copytree(hidden,truth/'confirmation')
        names=['problem.json','problem.npz','truth.npy','confirmation/problem.json','confirmation/problem.npz']
        manifests[split]['cases'].append({**row,'public':str(public.relative_to(out/split)),'truth':str(truth.relative_to(out/split)),
            'sha256':{name:file_hash((public if name in ('problem.json','problem.npz') else truth)/name) for name in names}})
    hashes={}
    for split,manifest in manifests.items():
        path=out/split/'manifest.json';path.write_text(json.dumps(manifest,sort_keys=True,indent=2));load_manifest(path);hashes[str(path.relative_to(out))]=file_hash(path)
    (out/'evidence').mkdir()
    for name,value in evidence.items():(out/'evidence'/f'{name}.json').write_text(json.dumps(value,sort_keys=True,indent=2))
    receipt=dict(schema='copper-qualified-synthetic-v1',status='qualified',evaluator_uid=os.geteuid(),manifest_sha256=hashes,evidence_sha256={str(p.relative_to(out)):file_hash(p) for p in (out/'evidence').glob('*.json')},limitations=['Synthetic benchmark only; no real mine validation','Empirical discretization calibration, not an error-bound theorem','Not secure against host administrators or trusted Docker operator'])
    (out/'qualification.json').write_text(json.dumps(receipt,sort_keys=True,indent=2))
    # Host parent remains 0700; nested files must be readable by isolated nobody containers.
    for path in out.rglob('*'):path.chmod(0o755 if path.is_dir() else 0o444)
    print(json.dumps(dict(status='qualified_synthetic_release',cases=len(pending),root=str(out))))


if __name__=='__main__':main()
