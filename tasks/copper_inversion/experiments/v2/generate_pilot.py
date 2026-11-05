"""Preassigned independent pilot parents, generated before method comparison.

ID and held-out OOD development recipes. No formal data are published here.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import label
from .build_suite import build
from .split_audit import audit
from tasks.copper_inversion.v2.geological_policy import MAX_COMPONENTS,POLICY_VERSION

FAMILIES=("folded","lens","multiple","faulted")


def quality(parent,family):
    with np.load(parent/"geology.npz",allow_pickle=False) as data:
        mask,forbidden=data["truth"],data["forbidden_mask"]
    components,count=label(mask)
    if count>MAX_COMPONENTS:raise ValueError('predeclared public component limit exceeded')
    touches=any(np.take(mask,side,axis=axis).any() for axis in range(3) for side in (0,-1))
    if family in {"lens","multiple"} and (touches or count!=(1 if family=="lens" else 2)):
        raise ValueError("closed-body topology quality gate failed")
    if family=="faulted_lens" and (touches or not 1<=count<=4):
        raise ValueError("faulted closed-body topology quality gate failed")
    if np.any(mask&forbidden):raise ValueError("geology in air")
    return dict(component_count=count,touches_domain=bool(touches),positive_voxels=int(mask.sum()),
        component_voxels=np.bincount(components.ravel())[1:].tolist())


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    p.add_argument("--generator-python",type=Path,required=True);p.add_argument("--seed",type=int,required=True)
    p.add_argument("--ood-for",type=Path,help="completed ID pilot; generate four independent OOD parents and audit jointly")
    p.add_argument("--workers",type=int,default=8)
    p.add_argument("--full-study",action="store_true",help="Generate the prespecified 24/8/16 ID or 8 OOD parent population for qualification; does not grant formal status")
    a=p.parse_args()
    if not 1<=a.workers<=16:raise ValueError("workers must be in [1,16]")
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).parent
    splits=((("test_ood",8),) if a.ood_for else (("train",24),("validation",8),("test_id",16))) if a.full_study else ((("test_ood",4),) if a.ood_for else (("train",8),("validation",4),("test_id",4)))
    plan=dict(schema="copper-pilot-plan-v4",stage="development",master_seed=a.seed,
        geological_policy=POLICY_VERSION,max_components=MAX_COMPONENTS,
        quality_gates="generator fraction [0.02,0.65]; no air; lens exactly 1 closed component; multiple exactly 2 closed components; faulted_lens 1 to 4 closed components",
        replacements="up to 3 presampled seeds per parent; only geology quality failures may advance",
        missing=[] if a.ood_for else ["test_ood"],source_sha256={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ("geology.py","observations.py","generate_pilot.py")},parents=[])
    for split_index,(split,count) in enumerate(splits):
        for i in range(count):
            seed_stream=np.random.SeedSequence([a.seed,split_index,i]).generate_state(4)
            lineage=hashlib.sha256(f"pilot:{a.seed}:{split}:{i}".encode()).hexdigest()
            family=("faulted_lens" if i<count//2 else FAMILIES[(i-count//2)%len(FAMILIES)]) if a.ood_for else FAMILIES[i%4]
            plan["parents"].append(dict(split=split,index=i,family=family,lineage_id=lineage,
                coverage="clustered" if a.ood_for and i>=count//2 else "uniform",
                shift=("structure" if i<count//2 else "coverage") if a.ood_for else "id",
                generation_seeds=[int(s) for s in seed_stream[:3]],observation_seed=int(seed_stream[3]),noisy=(i//4+i%2)%2==1))
    for split,_ in splits:
        members=[e for e in plan["parents"] if e["split"]==split]
        if sum(e["noisy"] for e in members)*2!=len(members):
            raise ValueError("ID splits require the same 50/50 noise mixture")
    (root/"plan.json").write_text(json.dumps(plan,sort_keys=True,indent=2))
    def generate(entry):
        records=[]
        name=f"{entry['split']}-{entry['index']:02}"
        accepted=None
        for attempt,seed in enumerate(entry["generation_seeds"]):
            parent=root/"parents"/f"{name}-{attempt}"
            log=root/f"{name}-{attempt}.log"
            cache=root/'cache'/f'{name}-{attempt}';cache.mkdir(parents=True,mode=0o700)
            with log.open("w") as out:
                result=subprocess.run([str(a.generator_python),"-m","tasks.copper_inversion.experiments.v2.geology",
                    "--family",entry["family"],"--seed",str(seed),"--output",str(parent)],env={**os.environ,'TMPDIR':str(cache)},stdout=out,stderr=subprocess.STDOUT,timeout=300)
            if result.returncode:
                # An infrastructure or numerical bug is not a valid rejection;
                # only the explicit predeclared scientific quality gate retries.
                if "predeclared geology quality gate" not in log.read_text():
                    raise RuntimeError(f"generator failed; inspect {log}")
                records.append(dict(parent=name,attempt=attempt,status="rejected_fraction"))
                continue
            try:q=quality(parent,entry["family"])
            except ValueError as exc:
                records.append(dict(parent=name,attempt=attempt,status="rejected_topology",reason=str(exc)))
                continue
            accepted=(parent,seed,q);break
        if accepted is None:raise RuntimeError(f"all predeclared geometry attempts rejected: {name}")
        parent,seed,q=accepted
        packs=root/"packs"/name
        command=[str(a.generator_python),"-m","tasks.copper_inversion.experiments.v2.observations",
            "--parent",str(parent),"--output",str(packs),"--seed",str(entry["observation_seed"]),"--coverage",entry["coverage"]]
        if entry["noisy"]:command.append("--noisy")
        with (root/f"{name}-observations.log").open("w") as out:
            subprocess.run(command,env={**os.environ,'TMPDIR':str(cache)},stdout=out,stderr=subprocess.STDOUT,timeout=600,check=True)
        metadata=json.loads((parent/"private-parent.json").read_text())
        lineage=dict(parent_id=metadata["geometry_sha256"],lineage_id=entry["lineage_id"],
            recipe=metadata["recipe"],generation_seed=seed,scope="development")
        records.append(dict(parent=name,parent_directory=str(parent.relative_to(root)),status="accepted",family=entry["family"],noisy=entry["noisy"],shift=entry["shift"],coverage=entry["coverage"],**q))
        return entry["split"],packs,lineage,records
    records=[];lineages=[];packs_by_split={s:[] for s,_ in splits}
    # Each parent has its own paths/cache; publish progress in the fixed plan order.
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for split,packs,lineage,parent_records in pool.map(generate,plan["parents"]):
            lineages.append(lineage);packs_by_split[split].append(packs);records.extend(parent_records)
            (root/"progress.json").write_text(json.dumps(dict(records=records,accepted=len(lineages),planned=len(plan["parents"])),indent=2))
            print(json.dumps(records[-1]),flush=True)
    manifests=[]
    for split,packs in packs_by_split.items():
        destination=root/"suites"/split;build(packs,destination,split)
        manifests.append(destination/"manifest.json")
    if a.ood_for:
        original=a.ood_for.resolve()
        if not (original/"completion.json").is_file():raise ValueError("ID pilot must be completed before OOD audit")
        lineages.extend(json.loads((original/"lineage.json").read_text())["parents"])
        manifests.extend(original/"suites"/s/"manifest.json" for s in ("train","validation","test_id"))
    lineage=root/"lineage.json";lineage.write_text(json.dumps(dict(schema="copper-lineage-v2",parents=lineages),indent=2))
    (root/"split-audit.json").write_text(json.dumps(audit(manifests,lineage),indent=2))
    (root/"completion.json").write_text(json.dumps(dict(status="ood_pilot_generated" if a.ood_for else "id_pilot_generated",parents=len(lineages),
        cases=2*len(lineages),missing=plan["missing"]+["visual geology review","population near-duplicate review"]),indent=2))


if __name__=="__main__":main()
