"""Trusted cross-split provenance audit, before any formal method evaluation."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
from tasks.copper_inversion.v2.suite import load_manifest,file_hash


def shape_fingerprint(mask):
    """Normalized-domain exact duplicate screen including axis rotations/reflections.

    This is a coarse screen, not proof of independent geology. Lineage metadata
    remains mandatory and visually similar populations require separate review.
    """
    axes=[np.linspace(0,n-1,24) for n in mask.shape]
    coords=np.stack(np.meshgrid(*axes,indexing="ij"),axis=0)
    coarse=map_coordinates(mask.astype(np.uint8),coords,order=0,mode="nearest").astype(bool)
    hashes=[]
    for axes in itertools.permutations(range(3)):
        rotated=np.transpose(coarse,axes)
        for flips in itertools.product((False,True),repeat=3):
            view=rotated[tuple(slice(None,None,-1) if f else slice(None) for f in flips)]
            hashes.append(hashlib.sha256(np.packbits(view).tobytes()).hexdigest())
    return min(hashes)


def check_records(records):
    ownership={key:{} for key in ("parent_id","lineage_id","truth_sha256","shape_fingerprint")}
    for row in records:
        for key,seen in ownership.items():
            old=seen.setdefault(row[key],row["split"])
            if old!=row["split"]:
                raise ValueError(f"cross-split collision: {key}; {old} and {row['split']}")
    return {split:sum(r["split"]==split for r in records) for split in sorted({r["split"] for r in records})}


def audit(manifests,lineage_path):
    lineage=json.loads(Path(lineage_path).read_text())
    if set(lineage)!={"schema","parents"} or lineage["schema"]!="copper-lineage-v2":
        raise ValueError("explicit lineage registry required")
    if len({r["parent_id"] for r in lineage["parents"]})!=len(lineage["parents"]):
        raise ValueError("duplicate parent provenance")
    provenance={row["parent_id"]:row for row in lineage["parents"]}
    records=[];seen=set();split_names=set();stages=set()
    for path in manifests:
        path=Path(path).resolve();manifest=load_manifest(path)
        split=manifest["split"]
        if split in split_names:raise ValueError("multiple manifests for one split")
        split_names.add(split);stages.add(manifest["stage"])
        for case in manifest["cases"]:
            identity=(split,case["parent_id"])
            if identity in seen:continue
            seen.add(identity)
            if case["parent_id"] not in provenance:raise ValueError("parent has no declared lineage")
            entry=provenance[case["parent_id"]]
            if set(entry)!={"parent_id","lineage_id","recipe","generation_seed","scope"}:
                raise ValueError("incomplete parent provenance")
            if not isinstance(entry["lineage_id"],str) or not entry["lineage_id"]:
                raise ValueError("missing ancestral lineage")
            mask=np.load(path.parent/case["truth"]/"truth.npy",allow_pickle=False)
            if mask.dtype!=bool or mask.ndim!=3:raise ValueError("invalid truth mask")
            records.append(dict(split=split,parent_id=case["parent_id"],lineage_id=entry["lineage_id"],
                truth_sha256=case["sha256"]["truth.npy"],shape_fingerprint=shape_fingerprint(mask)))
    if len(stages)!=1:raise ValueError("cannot mix development and formal suites")
    if set(provenance)!={r["parent_id"] for r in records}:raise ValueError("lineage registry membership differs")
    counts=check_records(records)
    return dict(status="passed",scope="lineage and exact/coarse geometry collision checks only",parents=counts,
        manifest_sha256={str(p):file_hash(p) for p in manifests},lineage_sha256=file_hash(lineage_path),
        remaining_checks=["near-duplicate retrieval and geological review","independent hidden-access isolation"])


def main():
    p=argparse.ArgumentParser();p.add_argument("--manifests",nargs="+",type=Path,required=True)
    p.add_argument("--lineage",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();result=audit(a.manifests,a.lineage)
    with a.output.open("x") as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=="__main__":main()
