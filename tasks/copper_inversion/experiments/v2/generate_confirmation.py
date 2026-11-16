"""Separate whole-hole/location/noise streams, kept in the private account."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import shutil
import subprocess
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);p.add_argument('--root',type=Path,required=True);p.add_argument('--python',type=Path,required=True);p.add_argument('--rsi-separated',action='store_true');a=p.parse_args()
    if not 1<=a.workers<=16:raise ValueError("workers must be in [1,16]")
    jobs=[]
    for part in ('id','ood'):
        dataset=a.root/part
        plan=json.loads((dataset/'plan.json').read_text())
        if a.rsi_separated and plan.get('confirmation_policy',{}).get('rule')!='first of 16 presampled streams with no overlapping whole holes':
            raise ValueError('confirmation separation must be fixed in the generation plan')
        entries={f"{e['split']}-{e['index']:02}":e for e in plan['parents']}
        for row in json.loads((dataset/'progress.json').read_text())['records']:
            if row['status']!='accepted':continue
            name=row['parent'];entry=entries[name]
            jobs.append((dataset,row,entry,name))
    def generate(job):
        dataset,row,entry,name=job
        seeds=entry['confirmation_seeds'] if a.rsi_separated else [int(np.random.SeedSequence([entry['observation_seed'],920381]).generate_state(1)[0])]
        if a.rsi_separated and (len(seeds)!=16 or len(set(seeds))!=16):
            raise ValueError('invalid presampled confirmation streams')
        destination=dataset/'confirmation'/name
        audit_path=dataset/'confirmation-audits'/(name+'.json')
        if a.rsi_separated and destination.exists():
            if not audit_path.is_file():raise ValueError('existing confirmation lacks an audit receipt')
            return json.loads(audit_path.read_text())
        attempts=[]
        for attempt,seed in enumerate(seeds):
            candidate=(dataset/'confirmation-candidates'/f'{name}-{attempt:02}' if a.rsi_separated else destination)
            command=[str(a.python),'-m','tasks.copper_inversion.experiments.v2.observations','--parent',str(dataset/row['parent_directory']),'--output',str(candidate),'--seed',str(seed),'--coverage',entry['coverage']]
            if entry['noisy']:command+=['--noisy']
            if not (candidate/'private-manifest.json').exists():
                cache=dataset/'cache'/f'{name}-confirmation-{attempt:02}';cache.mkdir(parents=True,exist_ok=True,mode=0o700)
                with (dataset/f'{name}-confirmation-{attempt:02}.log').open('w') as log:
                    subprocess.run(command,env={**os.environ,'TMPDIR':str(cache.resolve())},stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
            if a.rsi_separated:
                from tasks.copper_inversion.experiments.rsi.measurements import audit_holes
                from tasks.copper_inversion.v2.data import read_problem
                public=dataset/'packs'/name
                public_manifest=json.loads((public/'private-manifest.json').read_text())
                hidden_manifest=json.loads((candidate/'private-manifest.json').read_text())
                public_id=next(c['case_id'] for c in public_manifest['cases'] if c['modality']=='drill')
                hidden_id=next(c['case_id'] for c in hidden_manifest['cases'] if c['modality']=='drill')
                audit=audit_holes(read_problem(public/public_id),read_problem(candidate/hidden_id))
                attempts.append(dict(seed=seed,**audit))
                if audit['status']!='passed':continue
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copytree(candidate,destination)
                result=dict(parent=name,split=entry['split'],selected_stream=seed,path=str(destination),attempts=attempts,
                            policy='coordinate-conditioned hole separation; no truth used')
                audit_path.parent.mkdir(parents=True,exist_ok=True)
                with audit_path.open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2)
                return result
            return dict(parent=name,split=entry['split'],independent_stream=seed,path=str(destination))
        raise ValueError('all presampled confirmation streams overlapped: '+name)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        records=list(pool.map(generate,jobs))
    (a.root/'confirmation-complete.json').write_text(json.dumps(dict(status='generated',records=records),indent=2))


if __name__=='__main__':main()
