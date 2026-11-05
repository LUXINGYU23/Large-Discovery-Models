"""Freeze development packs into the trusted suite manifest; no formal release."""
import argparse
import json
import shutil
from pathlib import Path
from tasks.copper_inversion.v2.suite import file_hash,load_manifest


def build(packs,output,split="train"):
    parents=set();ids=set();plans=[]
    for pack in packs:
        pack=Path(pack).resolve()
        meta=json.loads((pack/"private-manifest.json").read_text())
        if meta["stage"]!="development-only":raise ValueError("only inspected development packs accepted")
        parent=meta["parent_digest"]
        if parent in parents:raise ValueError("duplicate parent or observation variant in suite")
        parents.add(parent)
        for row in meta["cases"]:
            if row["case_id"] in ids:raise ValueError("duplicate case identity")
            ids.add(row["case_id"])
            plans.append((pack,parent,row))
    if not plans:raise ValueError("empty development suite")
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for pack,parent,row in plans:
        public=output/"public"/row["case_id"]
        public.mkdir(parents=True)
        for name in ("problem.json","problem.npz"):shutil.copy2(pack/row["case_id"]/name,public/name)
        truth=output/"private"/parent;truth.mkdir(parents=True,exist_ok=True)
        shutil.copy2(pack/row["truth"],truth/"truth.npy")
        rows.append(dict(case_id=row["case_id"],parent_id=parent,modality=row["modality"],
            public=str(public.relative_to(output)),truth=str(truth.relative_to(output)),
            sha256={name:file_hash((truth if name=="truth.npy" else public)/name) for name in ("problem.json","problem.npz","truth.npy")}))
    path=output/"manifest.json"
    path.write_text(json.dumps(dict(schema="copper-suite-v2",stage="development",split=split,seeds=[47],cases=rows),sort_keys=True,indent=2))
    load_manifest(path)
    print(json.dumps(dict(manifest=str(path),parents=len(parents),cases=len(rows),stage="development")))


def main():
    p=argparse.ArgumentParser();p.add_argument("--packs",nargs="+",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--split",choices=["train","validation","test_id","test_ood"],default="train")
    a=p.parse_args();build(a.packs,a.output,a.split)


if __name__=="__main__":main()
