"""Private population diagnostics, without any candidate algorithm evaluation."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
from tasks.copper_inversion.v2.data import read_problem
from tasks.copper_inversion.v2.geometry import canonicalize
from .split_audit import shape_fingerprint


def audit(root):
    records=[];arrays=[]
    for part in ('id','ood'):
        dataset=root/part
        entries={f"{e['split']}-{e['index']:02}":e for e in json.loads((dataset/'plan.json').read_text())['parents']}
        for row in json.loads((dataset/'progress.json').read_text())['records']:
            if row['status']!='accepted':continue
            name=row['parent'];entry=entries[name]
            pack=dataset/'packs'/name
            manifest=json.loads((pack/'private-manifest.json').read_text())
            truth=np.load(pack/'private/truth.npy',allow_pickle=False)
            p=read_problem(pack/manifest['cases'][0]['case_id'])
            _,distance=canonicalize(p,np.where(truth,1.,-1.))
            residuals=[]
            for obs in p['observations']:
                contact=obs['labels']==.5
                if contact.any():
                    coords=((obs['points'][contact]-p['origin'])/p['spacing']-.5).T
                    residuals.extend(abs(map_coordinates(distance,coords,order=1,mode='nearest'))/obs['contact_sigma'])
            axes=[np.linspace(0,n-1,24) for n in truth.shape]
            coarse=map_coordinates(truth.astype(np.uint8),np.stack(np.meshgrid(*axes,indexing='ij')),order=0).astype(bool)
            arrays.append(coarse)
            records.append(dict(name=name,split=entry['split'],family=entry['family'],noisy=entry['noisy'],fingerprint=shape_fingerprint(truth),contact_count=len(residuals),contact_p95_in_scale=float(np.quantile(residuals,.95)) if residuals else None,contact_max_in_scale=float(max(residuals)) if residuals else None,components=row['component_count'],air_violations=int(np.sum(truth&p['forbidden_mask']))))
    pairs=[]
    for i,a in enumerate(arrays):
        for j in range(i):
            if records[i]['split']==records[j]['split']:continue
            b=arrays[j];best=0.
            for axes in itertools.permutations(range(3)):
                rotated=b.transpose(axes)
                for flips in itertools.product((False,True),repeat=3):
                    v=rotated[tuple(slice(None,None,-1) if f else slice(None) for f in flips)]
                    overlap=np.count_nonzero(a&v);union=np.count_nonzero(a|v)
                    best=max(best,overlap/union if union else 1.)
            pairs.append(dict(a=records[i]['name'],b=records[j]['name'],normalized_symmetry_iou=best))
    pairs.sort(key=lambda p:-p['normalized_symmetry_iou'])
    collisions=[p for p in pairs if p['normalized_symmetry_iou']>=.98]
    errors=[r['name'] for r in records if r['air_violations'] or (r['contact_p95_in_scale'] is not None and r['contact_p95_in_scale']>(3. if r['noisy'] else 2.))]
    return dict(status='passed' if not collisions and not errors else 'failed',scope='numerical consistency and normalized symmetry near-duplicate screen; geological review remains required',criteria=dict(max_cross_split_symmetry_iou=.98,clean_contact_p95=2.,noisy_contact_p95=3.),parent_count=len(records),parents=records,nearest_pairs=pairs[:20],near_duplicates=collisions,contact_or_air_failures=errors)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.root)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:result[k] for k in ('status','parent_count','contact_or_air_failures')}))
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
