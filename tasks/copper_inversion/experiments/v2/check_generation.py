"""Remote reconstruction/measurement checks on public development data only."""
import argparse
import json
from pathlib import Path
import numpy as np
from .geology import model_from_metadata,query_ore
from tasks.copper_inversion.v2.data import read_problem


def main():
    p=argparse.ArgumentParser();p.add_argument("--parent",type=Path,required=True);p.add_argument("--packs",type=Path,required=True)
    a=p.parse_args();meta=json.loads((a.parent/"private-parent.json").read_text())
    manifest=json.loads((a.packs/"private-manifest.json").read_text())
    if manifest["stage"]!="development-only" or manifest["noise"]!="clean":
        raise ValueError("this check uses clean development observations")
    model=model_from_metadata(meta);target=meta["ore_lithology_id"]
    shape=meta["shape"];origin=np.asarray(meta["origin"]);spacing=np.asarray(meta["spacing"])
    xyz=np.stack(np.meshgrid(*[origin[i]+(np.arange(n)+.5)*spacing[i] for i,n in enumerate(shape)],indexing="ij"),axis=-1)
    with np.load(a.parent/"geology.npz") as data:
        truth=data["truth"];forbidden=data["forbidden_mask"]
    rebuilt=query_ore(model,xyz.reshape(-1,3),target_id=target).reshape(shape)&~forbidden
    if not np.array_equal(truth,rebuilt):raise AssertionError("parent reconstruction changed geology")
    rows=[]
    for case in manifest["cases"]:
        problem=read_problem(a.packs/case["case_id"])
        for obs in problem["observations"]:
            binary=obs["labels"]!=.5
            predicted=query_ore(model,obs["points"][binary],target_id=target)
            if not np.array_equal(predicted,obs["labels"][binary]):raise AssertionError("clean binary labels do not match implicit parent")
            contacts=obs["points"][~binary]
            if len(contacts):
                delta=np.concatenate([np.eye(3),-np.eye(3)])*float(spacing.min())*1e-4
                near=contacts[:,None,:]+delta[None,:,:]
                classes=query_ore(model,near.reshape(-1,3),target_id=target).reshape(-1,6)
                if not np.all(classes.any(1)&~classes.all(1)):raise AssertionError("contact does not bracket an implicit boundary")
            rows.append(dict(case=case["case_id"],modality=obs["name"],binary_count=int(binary.sum()),contact_count=len(contacts)))
    print(json.dumps(dict(status="passed",reconstructed_voxels=int(truth.size),checks=rows)))


if __name__=="__main__":main()
