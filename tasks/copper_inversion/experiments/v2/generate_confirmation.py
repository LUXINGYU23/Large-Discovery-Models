"""Separate whole-hole/location/noise streams, kept in the private account."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);p.add_argument('--root',type=Path,required=True);p.add_argument('--python',type=Path,required=True);a=p.parse_args()
    if not 1<=a.workers<=16:raise ValueError("workers must be in [1,16]")
    jobs=[]
    for part in ('id','ood'):
        dataset=a.root/part
        entries={f"{e['split']}-{e['index']:02}":e for e in json.loads((dataset/'plan.json').read_text())['parents']}
        for row in json.loads((dataset/'progress.json').read_text())['records']:
            if row['status']!='accepted':continue
            name=row['parent'];entry=entries[name]
            jobs.append((dataset,row,entry,name))
    def generate(job):
        dataset,row,entry,name=job
        seed=int(np.random.SeedSequence([entry['observation_seed'],920381]).generate_state(1)[0])
        destination=dataset/'confirmation'/name
        command=[str(a.python),'-m','tasks.copper_inversion.experiments.v2.observations','--parent',str(dataset/row['parent_directory']),'--output',str(destination),'--seed',str(seed),'--coverage',entry['coverage']]
        if entry['noisy']:command+=['--noisy']
        if not (destination/'private-manifest.json').exists():
            cache=dataset/'cache'/(name+'-confirmation');cache.mkdir(parents=True,exist_ok=True,mode=0o700)
            with (dataset/(name+'-confirmation.log')).open('w') as log:
                subprocess.run(command,env={**os.environ,'TMPDIR':str(cache.resolve())},stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        return dict(parent=name,split=entry['split'],independent_stream=seed,path=str(destination))
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        records=list(pool.map(generate,jobs))
    (a.root/'confirmation-complete.json').write_text(json.dumps(dict(status='generated',records=records),indent=2))


if __name__=='__main__':main()
