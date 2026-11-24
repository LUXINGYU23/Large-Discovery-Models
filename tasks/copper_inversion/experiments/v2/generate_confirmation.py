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
    jobs=[];modes={};plans={}
    for part in ('id','ood'):
        dataset=a.root/part
        if not (dataset/'plan.json').is_file() or not (dataset/'progress.json').is_file():continue
        plan=json.loads((dataset/'plan.json').read_text())
        plans[dataset]=plan
        if a.rsi_separated or 'confirmation_policy' in plan:
            from tasks.copper_inversion.experiments.rsi.measurements import confirmation_mode
            modes[dataset]=confirmation_mode(plan)
        entries={f"{e['split']}-{e['index']:02}":e for e in plan['parents']}
        for row in json.loads((dataset/'progress.json').read_text())['records']:
            if row['status']!='accepted':continue
            name=row['parent'];entry=entries[name]
            jobs.append((dataset,row,entry,name))
    def generate(job):
        dataset,row,entry,name=job
        mode=modes.get(dataset);rsi=mode is not None
        plan=plans[dataset]
        if rsi:
            from tasks.copper_inversion.experiments.rsi.measurements import (
                audit_holes, audit_pack_pair, confirmation_min_ore_holes, confirmation_pack_count,
                confirmation_streams, ore_hole_count)
            packs_required=confirmation_pack_count(plan)
            min_ore=confirmation_min_ore_holes(plan,entry['split'])
            seeds=confirmation_streams(entry,mode,packs_required)
        else:
            packs_required=1;min_ore=0
            seeds=[int(np.random.SeedSequence([entry['observation_seed'],920381]).generate_state(1)[0])]
        destination=dataset/'confirmation'/name
        audit_path=dataset/'confirmation-audits'/(name+'.json')
        if rsi and destination.exists():
            if not audit_path.is_file():raise ValueError('existing confirmation lacks an audit receipt')
            return json.loads(audit_path.read_text())
        attempts=[];selected=[];cursor=0
        while cursor<len(seeds):
            if rsi and packs_required>1:
                need=packs_required-len(selected)
                batch_end=min(len(seeds),cursor+(need if mode=='fail_release' else max(need+2,6)))
                batch_items=[dict(seed=int(seeds[idx]),
                                  output=str(dataset/'confirmation-candidates'/f'{name}-{idx:02}'))
                             for idx in range(cursor,batch_end)
                             if not (dataset/'confirmation-candidates'/f'{name}-{idx:02}'/'private-manifest.json').exists()]
                if batch_items:
                    cache=dataset/'cache'/f'{name}-confirmation-batch-{cursor:02}';cache.mkdir(parents=True,exist_ok=True,mode=0o700)
                    batch_file=cache/'batch.json';batch_file.write_text(json.dumps(batch_items))
                    bcmd=[str(a.python),'-m','tasks.copper_inversion.experiments.v2.observations',
                          '--parent',str(dataset/row['parent_directory']),'--batch',str(batch_file),'--coverage',entry['coverage']]
                    if entry['noisy']:bcmd+=['--noisy']
                    with (dataset/f'{name}-confirmation-batch-{cursor:02}.log').open('w') as log:
                        subprocess.run(bcmd,env={**os.environ,'TMPDIR':str(cache.resolve())},stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
            else:
                batch_end=cursor+1
            for attempt in range(cursor,batch_end):
                seed=seeds[attempt]
                candidate=(dataset/'confirmation-candidates'/f'{name}-{attempt:02}' if rsi else destination)
                command=[str(a.python),'-m','tasks.copper_inversion.experiments.v2.observations','--parent',str(dataset/row['parent_directory']),'--output',str(candidate),'--seed',str(seed),'--coverage',entry['coverage']]
                if entry['noisy']:command+=['--noisy']
                if not (candidate/'private-manifest.json').exists():
                    cache=dataset/'cache'/f'{name}-confirmation-{attempt:02}';cache.mkdir(parents=True,exist_ok=True,mode=0o700)
                    with (dataset/f'{name}-confirmation-{attempt:02}.log').open('w') as log:
                        subprocess.run(command,env={**os.environ,'TMPDIR':str(cache.resolve())},stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
                if not rsi:
                    return dict(parent=name,split=entry['split'],independent_stream=seed,path=str(destination))
                from tasks.copper_inversion.v2.data import read_problem
                public=dataset/'packs'/name
                public_manifest=json.loads((public/'private-manifest.json').read_text())
                hidden_manifest=json.loads((candidate/'private-manifest.json').read_text())
                public_id=next(c['case_id'] for c in public_manifest['cases'] if c['modality']=='drill')
                hidden_id=next(c['case_id'] for c in hidden_manifest['cases'] if c['modality']=='drill')
                cand_drill=read_problem(candidate/hidden_id)
                audit=audit_holes(read_problem(public/public_id),cand_drill)
                ore_holes=ore_hole_count(cand_drill,hidden_manifest)
                pack_audits=[audit_pack_pair(prev_drill,cand_drill) for _,_,prev_drill,_ in selected]
                pack_overlap=sum(pa['overlapping_holes'] for pa in pack_audits)
                passed=(audit['status']=='passed') and (pack_overlap==0) and (ore_holes>=min_ore)
                if packs_required==1 and min_ore==0:
                    attempts.append({"seed":seed,**audit})
                else:
                    reason=('passed' if passed else 'public_overlap' if audit['status']!='passed'
                            else 'pack_overlap' if pack_overlap else 'undiscovered')
                    attempts.append({"seed":seed,**audit,"status":'passed' if passed else 'failed',
                                     "reason":reason,"pack_overlapping_holes":pack_overlap,"ore_holes":ore_holes})
                if not passed and mode=='resample_until_separated':continue
                selected.append((seed,candidate,cand_drill,hidden_manifest))
                if len(selected)<packs_required:continue
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copytree(selected[0][1],destination)
                if packs_required>1:
                    pack_rows=[]
                    for k,(s_k,cand_k,_,man_k) in enumerate(selected):
                        shutil.copytree(cand_k,destination/f'c{k}')
                        pack_rows.append(dict(pack=f'c{k}',observation_seed=s_k,
                                              hole_count=man_k.get('hole_count'),ore_holes=man_k.get('ore_holes'),
                                              cases=man_k['cases']))
                    top_man=json.loads((destination/'private-manifest.json').read_text())
                    top_man['packs']=pack_rows
                    (destination/'private-manifest.json').write_text(json.dumps(top_man,sort_keys=True,indent=2))
                all_passed=all(a['status']=='passed' for a in attempts[-packs_required:]) if mode=='fail_release' else True
                result=dict(parent=name,split=entry['split'],selected_stream=selected[0][0],
                            path=str(destination),attempts=attempts,
                            status='passed' if all_passed else 'failed',policy=mode,
                            scope='coordinate-conditioned hole separation; no truth used' if mode=='resample_until_separated'
                            else 'single presampled stream; overlap fails the release; no truth used')
                if packs_required>1:
                    result['selected_streams']=[s_k for s_k,_,_,_ in selected]
                    result['packs']=packs_required
                    result['min_ore_holes']=min_ore
                audit_path.parent.mkdir(parents=True,exist_ok=True)
                with audit_path.open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2)
                return result
            cursor=batch_end
        raise ValueError(f'insufficient separated confirmation streams ({len(selected)}/{packs_required}): {name}')
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        records=list(pool.map(generate,jobs))
    (a.root/'confirmation-complete.json').write_text(json.dumps(dict(status='generated',records=records,overlapping=[r['parent'] for r in records if r.get('status')=='failed']),indent=2))


if __name__=='__main__':main()
