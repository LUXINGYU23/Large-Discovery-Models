"""Pilot observations sampled from the continuous implicit GemPy parent.

Geological structure and measurement randomness use independent seeds. No labels
or contact locations are calculated from the scoring-resolution occupancy/SDF.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .geology import model_from_metadata, query_ore, terrain_height
from tasks.copper_inversion.v2.data import save_problem
from .measurement_policy import scales
from tasks.copper_inversion.v2.geological_policy import public_priors,POLICY_VERSION


def build(parent, output, seed, noisy=False, coverage="uniform"):
    if coverage not in {"uniform", "clustered"}:
        raise ValueError("unknown predeclared observation coverage")
    meta=json.loads((parent/"private-parent.json").read_text())
    encoded=(parent/"geology.npz").read_bytes()
    if hashlib.sha256(encoded).hexdigest()!=meta["geometry_sha256"]:
        raise ValueError("parent geometry digest changed")
    with np.load(parent/"geology.npz",allow_pickle=False) as data:
        truth,forbidden=data["truth"],data["forbidden_mask"]
    shape=truth.shape; spacing=np.asarray(meta["spacing"]); origin=np.asarray(meta["origin"])
    rng=np.random.default_rng(seed)
    model=model_from_metadata(meta)
    points=[]; labels=[]; groups=[]
    hole_count=int(rng.integers(3,13))
    origin=np.asarray(meta["origin"]); spacing=np.asarray(meta["spacing"])
    lower=origin+.5*spacing; upper=origin+(np.asarray(shape)-.5)*spacing
    # OOD coverage is selected before observations, independently of ore. Both
    # holes and outcrop samples cover the same random quarter-width window.
    coverage_rng=np.random.default_rng(np.random.SeedSequence([seed,8191]))
    window_lower=lower[:2]+coverage_rng.uniform(.1,.65,2)*(upper[:2]-lower[:2])
    window_upper=window_lower+.25*(upper[:2]-lower[:2])
    # Resolve paths more finely than the output grid, then locate crossings by
    # queries to the implicit geology. This does not interpolate a coarse SDF.
    paths=[]
    for group in range(hole_count):
        top=(rng.uniform(lower[:2]+spacing[:2],upper[:2]-spacing[:2]) if coverage=="uniform"
             else rng.uniform(window_lower,window_upper))
        bottom=np.clip(top+rng.normal(0,3,2)*spacing[:2],lower[:2],upper[:2])
        fraction=np.linspace(0,1,8*shape[2])
        xy=top[None,:]+fraction[:,None]*(bottom-top)[None,:]
        z=upper[2]+fraction*(lower[2]-upper[2])
        path=np.column_stack([xy,z])
        path=path[z<=terrain_height(meta,xy)]
        paths.append(path)
    all_points=np.concatenate(paths)
    classes=query_ore(model,all_points,target_id=meta["ore_lithology_id"])
    cursor=0; left=[];right=[];left_labels=[];contact_groups=[]
    for group,path in enumerate(paths):
        ore=classes[cursor:cursor+len(path)];cursor+=len(path)
        retain=rng.random(len(path))>.12
        selected=np.arange(len(path))[retain][::4]
        points.extend(path[selected]);labels.extend(ore[selected].astype(float));groups.extend([group]*len(selected))
        for j in np.flatnonzero(ore[:-1]!=ore[1:]):
            left.append(path[j]);right.append(path[j+1]);left_labels.append(ore[j]);contact_groups.append(group)
    if left:
        left,right,left_labels=np.asarray(left),np.asarray(right),np.asarray(left_labels)
        for _ in range(16):
            midpoint=(left+right)/2
            same=query_ore(model,midpoint,target_id=meta["ore_lithology_id"])==left_labels
            left[same]=midpoint[same];right[~same]=midpoint[~same]
        points.extend((left+right)/2);labels.extend([.5]*len(left));groups.extend(contact_groups)
    points=np.asarray(points);labels=np.asarray(labels);groups=np.asarray(groups,dtype=int)
    scale_policy=scales(spacing,noisy)
    physical_sigma=scale_policy["measurement_sigma"]
    effective_scale=scale_policy["effective_scale"]
    noise_rate=.02 if noisy else 0.
    if noisy:
        points=np.clip(points+rng.normal(0,physical_sigma,points.shape),lower,upper)
        binary=labels!=.5
        flips=binary&(rng.random(len(labels))<noise_rate)
        labels[flips]=1-labels[flips]
    observation=dict(name="drill",points=points,labels=labels,groups=groups,weight=1.,
        transition_scale=effective_scale,contact_sigma=effective_scale,label_error=noise_rate)
    # Sample the actual continuous terrain, not a dense grid top face. A tiny
    # inward offset gives the rock-side limiting classification.
    xy=(rng.uniform(lower[:2],upper[:2],size=(120,2)) if coverage=="uniform"
        else rng.uniform(window_lower,window_upper,size=(120,2)))
    z=terrain_height(meta,xy)
    keep=(z>=lower[2])&(z<=upper[2])
    xyz=np.column_stack([xy[keep],z[keep]-1e-5*np.min(spacing)])
    surface_labels=query_ore(model,xyz,target_id=meta["ore_lithology_id"]).astype(float)
    if noisy:
        flip=rng.random(len(xyz))<noise_rate;surface_labels[flip]=1-surface_labels[flip]
        xyz=np.clip(xyz+rng.normal(0,physical_sigma,xyz.shape),lower,upper)
    normalized=(xyz[:,:2]-origin[:2])/(np.asarray(shape[:2])*spacing[:2])
    spatial_groups=(normalized[:,0]*4).astype(int)*4+(normalized[:,1]*4).astype(int)
    outcrop=dict(name="outcrop",points=xyz,labels=surface_labels,groups=spatial_groups,
        weight=1.,transition_scale=effective_scale,contact_sigma=effective_scale,label_error=noise_rate)
    output.mkdir(parents=True,exist_ok=False)
    truth_dir=output/"private";truth_dir.mkdir();np.save(truth_dir/"truth.npy",truth)
    manifest=dict(stage="development-only",parent_digest=meta["geometry_sha256"],observation_seed=seed,
        observation_recipe="implicit-query-v4",coverage=coverage,noise="noisy" if noisy else "clean",contact_bisection_steps=16,probe_spacing_fraction=.125,
        scale_policy=scale_policy,area_prior_enabled=False,geological_policy=POLICY_VERSION,cases=[])
    for name,observations in (("drill",[observation]),("drill_outcrop",[observation,outcrop])):
        identifier=hashlib.sha256((meta["geometry_sha256"]+name+str(seed)+manifest["observation_recipe"]+manifest["noise"]+coverage+POLICY_VERSION).encode()).hexdigest()[:20]
        problem=dict(schema_version="problem_schema_v2",geometry_version="occupancy_v2",objective_version="objective_v2",
            case_id=identifier,shape=list(shape),origin=origin.tolist(),spacing=spacing.tolist(),
            domain_mask=np.ones(shape,bool),forbidden_mask=forbidden,
            open_boundary_faces=["x-","x+","y-","y+","z-","z+"],observations=observations,
            priors=public_priors(~forbidden,spacing))
        save_problem(problem,output/identifier)
        manifest["cases"].append(dict(case_id=identifier,modality=name,truth="private/truth.npy"))
    (output/"private-manifest.json").write_text(json.dumps(manifest,sort_keys=True,indent=2))
    print(json.dumps(dict(stage=manifest["stage"],hole_count=hole_count,drill_labels=len(labels),contacts=int(np.count_nonzero(labels==.5)),cases=len(manifest["cases"]))))


def main():
    p=argparse.ArgumentParser();p.add_argument("--parent",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--seed",type=int,required=True)
    p.add_argument("--noisy",action="store_true")
    p.add_argument("--coverage",choices=["uniform","clustered"],default="uniform")
    a=p.parse_args();build(a.parent,a.output,a.seed,a.noisy,a.coverage)


if __name__=="__main__":main()
