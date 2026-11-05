"""Trusted development review: rank similar geometries, never discard by score.

Domain-normalized Dice maximized over axis permutations/reflections is a screen,
not proof of independence under arbitrary continuous deformations.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
from tasks.copper_inversion.v2.suite import load_manifest


def coarse(mask):
    axes=[np.linspace(0,n-1,24) for n in mask.shape]
    return map_coordinates(mask.astype(np.uint8),np.stack(np.meshgrid(*axes,indexing="ij")),
                           order=0,mode="nearest").astype(bool)


def orientations(mask):
    for axes in itertools.permutations(range(3)):
        rotated=np.transpose(mask,axes)
        for flips in itertools.product((False,True),repeat=3):
            yield rotated[tuple(slice(None,None,-1) if f else slice(None) for f in flips)]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifests",nargs="+",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();parents=[]
    for path in args.manifests:
        manifest=load_manifest(path)
        if manifest["stage"]!="development":raise ValueError("development review only")
        seen=set()
        for row in manifest["cases"]:
            if row["parent_id"] in seen:continue
            seen.add(row["parent_id"])
            mask=coarse(np.load(path.parent/row["truth"]/"truth.npy",allow_pickle=False))
            parents.append((dict(parent_id=row["parent_id"],split=manifest["split"]),mask))
    pairs=[]
    for i,(left,a) in enumerate(parents):
        for right,b in parents[i+1:]:
            if left["split"]==right["split"]:continue
            denominator=int(a.sum()+b.sum())
            if not denominator:raise ValueError("empty geometry cannot pass population review")
            dice=max(2*int(np.count_nonzero(a&v))/denominator for v in orientations(b))
            pairs.append(dict(left=left,right=right,max_coarse_dice=dice))
    pairs.sort(key=lambda r:r["max_coarse_dice"],reverse=True)
    result=dict(status="ranked_for_review",parents=len(parents),cross_split_pairs=len(pairs),
        method="24-cubed domain normalization; 48 axis permutations/reflections; binary Dice",
        limitations="No arbitrary rotation, translation or deformation alignment; no automatic exclusions or independence claim",
        pairs=pairs)
    with args.output.open("x") as f:json.dump(result,f,indent=2)
    print(json.dumps({**{k:v for k,v in result.items() if k!="pairs"},"top_pairs":pairs[:5]}))


if __name__=="__main__":main()
