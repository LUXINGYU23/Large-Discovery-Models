"""Preassigned independent pilot parents, generated before method comparison.

ID and held-out OOD development recipes. No formal data are published here.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
import numpy as np
from scipy.ndimage import label
from .build_suite import build
from .split_audit import audit
from tasks.copper_inversion.experiments.rsi.measurements import parse_cli_holes as _parse_cli_holes, parse_holes
from tasks.copper_inversion.v2.geological_policy import MAX_COMPONENTS,POLICY_VERSION

FAMILIES=("folded","lens","multiple","faulted")


def vertical_ore_runs(mask):
    """Lengths of contiguous vertical (axis-2) ore runs across all columns."""
    mask=np.asarray(mask,dtype=bool)
    if not mask.any():return np.empty(0,dtype=int)
    padded=np.pad(mask,((0,0),(0,0),(1,1)),constant_values=False)
    diff=np.diff(padded.astype(np.int8),axis=2)
    return np.flatnonzero(diff==-1)-np.flatnonzero(diff==1)


def quality(parent,family,*,volume_bounds=None,min_vertical_thickness=None):
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
    feasible=~forbidden
    volume_fraction=float(mask.sum()/feasible.sum())
    runs=vertical_ore_runs(mask)
    median_thickness=float(np.median(runs)) if len(runs) else 0.
    if volume_bounds is not None and not (volume_bounds[0]<=volume_fraction<=volume_bounds[1]):
        raise ValueError(f"predeclared volume fraction quality gate failed: {volume_fraction:.4f}")
    if min_vertical_thickness is not None and median_thickness<min_vertical_thickness:
        raise ValueError(f"predeclared vertical thickness quality gate failed: {median_thickness:.2f}")
    return dict(component_count=count,touches_domain=bool(touches),positive_voxels=int(mask.sum()),
        component_voxels=np.bincount(components.ravel())[1:].tolist(),
        volume_fraction=volume_fraction,median_vertical_thickness=median_thickness)


def make_plan(seed,splits,ood,source,*,generation_attempts=3,density_shift_splits=()):
    """Parent rows fixed from the master seed alone, before any geology exists."""
    plan=dict(schema="copper-pilot-plan-v4",stage="development",master_seed=seed,
        geological_policy=POLICY_VERSION,max_components=MAX_COMPONENTS,
        quality_gates="generator fraction [0.02,0.65]; no air; lens exactly 1 closed component; multiple exactly 2 closed components; faulted_lens 1 to 4 closed components",
        replacements=f"up to {generation_attempts} presampled seeds per parent; only geology quality failures may advance",
        missing=[] if ood else ["test_ood"],source_sha256={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ("geology.py","observations.py","generate_pilot.py")},parents=[])
    density_set=set(density_shift_splits or ())
    for split_index,(split,count) in enumerate(splits):
        three_shifts=ood and split in density_set
        if three_shifts and count%3!=0:
            raise ValueError(f"density-shift OOD split {split} requires count divisible by 3")
        for i in range(count):
            seed_stream=np.random.SeedSequence([seed,split_index,i]).generate_state(generation_attempts+1)
            lineage=hashlib.sha256(f"pilot:{seed}:{split}:{i}".encode()).hexdigest()
            if not ood:
                family=FAMILIES[i%4];coverage="uniform";shift="id"
            elif three_shifts:
                third=count//3
                if i<third:
                    family="faulted_lens";coverage="uniform";shift="structure"
                elif i<2*third:
                    family=FAMILIES[(i-third)%len(FAMILIES)];coverage="clustered";shift="coverage"
                else:
                    family=FAMILIES[(i-2*third)%len(FAMILIES)];coverage="uniform";shift="density"
            else:
                family="faulted_lens" if i<count//2 else FAMILIES[(i-count//2)%len(FAMILIES)]
                coverage="clustered" if i>=count//2 else "uniform"
                shift="structure" if i<count//2 else "coverage"
            plan["parents"].append(dict(split=split,index=i,family=family,lineage_id=lineage,
                coverage=coverage,shift=shift,
                generation_seeds=[int(s) for s in seed_stream[:generation_attempts]],
                observation_seed=int(seed_stream[generation_attempts]),noisy=(i//4+i%2)%2==1))
    for split,_ in splits:
        members=[e for e in plan["parents"] if e["split"]==split]
        if sum(e["noisy"] for e in members)*2!=len(members):
            raise ValueError("splits require a 50/50 noise mixture")
    return plan


def _copy_parent(src,dst):
    dst.mkdir(parents=True,exist_ok=False)
    for name in ("geology.npz","private-parent.json"):
        shutil.copyfile(src/name,dst/name)


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    p.add_argument("--generator-python",type=Path,required=True);p.add_argument("--seed",type=int,required=True)
    p.add_argument("--ood-for",type=Path,help="completed ID pilot; generate independent OOD parents and audit jointly")
    p.add_argument("--workers",type=int,default=8)
    p.add_argument("--holes",nargs="+",help="public borehole count N, bounds LO HI, or lo|mid|hi (default: 3 12)")
    p.add_argument("--ood-holes",nargs="+",help="borehole bounds for shift=='density' OOD parents")
    p.add_argument("--splits",nargs="+",help="subset of planned splits to generate now (plan still commits all splits)")
    p.add_argument("--resume-planned",action="store_true",help="generate remaining planned splits into an existing part directory")
    p.add_argument("--reuse-parents",type=Path,help="reuse verified accepted parents from another part directory with the same plan")
    p.add_argument("--rsi-plan",type=Path,help="RSI release generation-plan.json (default: OUTPUT/../generation-plan.json when it is one); fixes counts, one layout per parent, confirmation policy and short tier")
    p.add_argument("--full-study",action="store_true",help="Without an RSI plan: generate the legacy 24/8/16 ID or 8 OOD parent population; does not grant formal status")
    a=p.parse_args()
    if not 1<=a.workers<=16:raise ValueError("workers must be in [1,16]")
    holes=_parse_cli_holes(a.holes) if a.holes else (3,12)
    ood_holes=_parse_cli_holes(a.ood_holes) if a.ood_holes else ((3,12) if holes!=(3,12) else (24,48))
    root=a.output.resolve()
    rsi_root=a.rsi_plan.resolve().parent if a.rsi_plan else root.parent
    rsi=None
    if a.rsi_plan or ((rsi_root/"generation-plan.json").is_file() and
            json.loads((rsi_root/"generation-plan.json").read_text()).get("schema")=="rsi-generation-plan-v1"):
        from tasks.copper_inversion.experiments.rsi import plan as rsiplan
        if a.rsi_plan and a.rsi_plan.name!="generation-plan.json":raise ValueError("RSI plan must be a generation-plan.json")
        rsi,rsi_sha=rsiplan.load_generation_plan(rsi_root)
        part="ood" if a.ood_for else "id"
        if a.seed!=rsi[part+"_seed"]:raise ValueError("seed differs from the RSI generation plan")
    source=Path(__file__).parent
    if rsi is not None:
        splits=rsiplan.part_splits(rsi,part)
        gen_attempts=rsi.get("size_gates",{}).get("generation_attempts",3)
        density_shifts=rsi.get("density_shift_splits",())
    else:
        splits=((("test_ood",8),) if a.ood_for else (("train",24),("validation",8),("test_id",16))) if a.full_study else ((("test_ood",4),) if a.ood_for else (("train",8),("validation",4),("test_id",4)))
        gen_attempts=3;density_shifts=()
    if a.resume_planned:
        if not (root/"plan.json").is_file():raise ValueError("--resume-planned requires an existing plan.json")
        plan=json.loads((root/"plan.json").read_text())
        if rsi is not None and rsiplan.replay_part_plan(plan,rsi,rsi_sha,part)!=plan:
            raise ValueError("existing part plan does not match RSI generation plan")
        existing_progress=json.loads((root/"progress.json").read_text())["records"] if (root/"progress.json").is_file() else []
        existing_lineages=json.loads((root/"lineage.json").read_text())["parents"] if (root/"lineage.json").is_file() else []
    else:
        root.mkdir(parents=True,exist_ok=False)
        plan=make_plan(a.seed,splits,bool(a.ood_for),source,generation_attempts=gen_attempts,density_shift_splits=density_shifts)
        if rsi is not None:
            rsiplan.part_plan(plan,rsi,rsi_sha,part)
            rsiplan.write_part_plan(root,plan)
        else:(root/"plan.json").write_text(json.dumps(plan,sort_keys=True,indent=2))
        existing_progress=[];existing_lineages=[]
    all_split_names=[s for s,_ in splits]
    target_splits=list(a.splits) if a.splits else all_split_names
    if any(s not in all_split_names for s in target_splits):
        raise ValueError("unknown target split in --splits")
    done_parents={r["parent"] for r in existing_progress if r.get("status")=="accepted"}
    reusable={}
    if a.reuse_parents:
        reuse_root=a.reuse_parents.resolve()
        if json.loads((reuse_root/"plan.json").read_text())!=plan:
            raise ValueError("--reuse-parents requires an identical part plan")
        for row in json.loads((reuse_root/"progress.json").read_text())["records"]:
            if row.get("status")=="accepted":
                reusable[row["parent"]]=(reuse_root/row["parent_directory"],row)
    size_gates=plan.get("size_gates")
    disc_policy=plan.get("discoverability_policy")
    def generate(entry):
        records=[]
        name=f"{entry['split']}-{entry['index']:02}"
        use_size=(size_gates is not None and entry["split"] in size_gates.get("id_splits",()))
        vol_bounds=tuple(size_gates["volume_bounds"]) if use_size else None
        min_thick=size_gates["min_vertical_thickness"] if use_size else None
        accepted=None
        if name in reusable:
            src_parent,src_row=reusable[name]
            rel_dir=src_row["parent_directory"]
            parent=root/rel_dir
            _copy_parent(src_parent,parent)
            q=quality(parent,entry["family"],volume_bounds=vol_bounds,min_vertical_thickness=min_thick)
            meta_check=json.loads((parent/"private-parent.json").read_text())
            accepted=(parent,int(meta_check["seed"]),q)
        else:
            for attempt,seed in enumerate(entry["generation_seeds"]):
                parent=root/"parents"/f"{name}-{attempt}"
                log=root/f"{name}-{attempt}.log"
                cache=root/'cache'/f'{name}-{attempt}';cache.mkdir(parents=True,mode=0o700)
                with log.open("w") as out:
                    result=subprocess.run([str(a.generator_python),"-m","tasks.copper_inversion.experiments.v2.geology",
                        "--family",entry["family"],"--seed",str(seed),"--output",str(parent)],env={**os.environ,'TMPDIR':str(cache)},stdout=out,stderr=subprocess.STDOUT,timeout=300)
                if result.returncode:
                    if "predeclared geology quality gate" not in log.read_text():
                        raise RuntimeError(f"generator failed; inspect {log}")
                    records.append(dict(parent=name,attempt=attempt,status="rejected_fraction"))
                    continue
                try:q=quality(parent,entry["family"],volume_bounds=vol_bounds,min_vertical_thickness=min_thick)
                except ValueError as exc:
                    records.append(dict(parent=name,attempt=attempt,status="rejected_topology",reason=str(exc)))
                    continue
                accepted=(parent,seed,q);break
        if accepted is None:raise RuntimeError(f"all predeclared geometry attempts rejected: {name}")
        parent,seed,q=accepted
        packs=root/"packs"/name
        entry_holes=ood_holes if entry.get("shift")=="density" else parse_holes(holes)
        obs_seeds=entry.get("observation_seeds",[entry["observation_seed"]])
        min_ore=(int(disc_policy["min_public_ore_holes"])
                 if disc_policy is not None and entry["split"] in disc_policy.get("id_splits",()) else 0)
        obs_attempts=[]
        selected_obs_seed=None
        obs_cursor=0
        while obs_cursor<len(obs_seeds):
            batch_end=(obs_cursor+1) if (obs_cursor==0 or "observation_seeds" not in entry) else min(len(obs_seeds),obs_cursor+4)
            if batch_end>obs_cursor+1:
                bcache=root/'cache'/f'{name}-obs-batch-{obs_cursor:02}';bcache.mkdir(parents=True,exist_ok=True,mode=0o700)
                bfile=bcache/'batch.json'
                bfile.write_text(json.dumps([dict(seed=int(obs_seeds[k]),output=str(root/"observation-candidates"/f"{name}-{k:02}"))
                                             for k in range(obs_cursor,batch_end)
                                             if not (root/"observation-candidates"/f"{name}-{k:02}"/"private-manifest.json").exists()]))
                bcmd=[str(a.generator_python),"-m","tasks.copper_inversion.experiments.v2.observations",
                      "--parent",str(parent),"--batch",str(bfile),"--coverage",entry["coverage"],
                      "--holes",str(entry_holes[0]),str(entry_holes[1])]
                if entry["noisy"]:bcmd.append("--noisy")
                with (root/f"{name}-observations-batch-{obs_cursor:02}.log").open("w") as out:
                    subprocess.run(bcmd,env={**os.environ,'TMPDIR':str(bcache)},stdout=out,stderr=subprocess.STDOUT,timeout=600,check=True)
            for obs_idx in range(obs_cursor,batch_end):
                obs_seed=obs_seeds[obs_idx]
                cand_dir=(root/"observation-candidates"/f"{name}-{obs_idx:02}") if "observation_seeds" in entry else packs
                if not (cand_dir/"private-manifest.json").exists():
                    cache=root/'cache'/f'{name}-obs-{obs_idx:02}';cache.mkdir(parents=True,exist_ok=True,mode=0o700)
                    command=[str(a.generator_python),"-m","tasks.copper_inversion.experiments.v2.observations",
                        "--parent",str(parent),"--output",str(cand_dir),"--seed",str(obs_seed),
                        "--coverage",entry["coverage"],"--holes",str(entry_holes[0]),str(entry_holes[1])]
                    if entry["noisy"]:command.append("--noisy")
                    with (root/f"{name}-observations-{obs_idx:02}.log" if "observation_seeds" in entry else root/f"{name}-observations.log").open("w") as out:
                        subprocess.run(command,env={**os.environ,'TMPDIR':str(cache)},stdout=out,stderr=subprocess.STDOUT,timeout=600,check=True)
                obs_manifest=json.loads((cand_dir/"private-manifest.json").read_text())
                ore_holes=int(obs_manifest.get("ore_holes",0))
                hole_count=int(obs_manifest.get("hole_count",0))
                passed=ore_holes>=min_ore
                obs_attempts.append(dict(seed=int(obs_seed),attempt=obs_idx,hole_count=hole_count,ore_holes=ore_holes,
                                         status="passed" if passed else "failed"))
                if passed:
                    if cand_dir!=packs:
                        packs.parent.mkdir(parents=True,exist_ok=True)
                        shutil.copytree(cand_dir,packs)
                    selected_obs_seed=int(obs_seed)
                    break
            if selected_obs_seed is not None:break
            obs_cursor=batch_end
        if selected_obs_seed is None:
            raise RuntimeError(f"all presampled observation streams failed discoverability: {name}")
        if "observation_seeds" in entry:
            obs_audit_dir=root/"observation-audits";obs_audit_dir.mkdir(parents=True,exist_ok=True)
            with (obs_audit_dir/f"{name}.json").open("x") as stream:
                json.dump(dict(parent=name,split=entry["split"],selected_seed=selected_obs_seed,
                               stream_index=len(obs_attempts)-1,min_ore_holes=min_ore,
                               holes=list(entry_holes),attempts=obs_attempts,status="passed"),
                          stream,sort_keys=True,indent=2)
        metadata=json.loads((parent/"private-parent.json").read_text())
        lineage=dict(parent_id=metadata["geometry_sha256"],lineage_id=entry["lineage_id"],
            recipe=metadata["recipe"],generation_seed=seed,scope="development")
        records.append(dict(parent=name,parent_directory=str(parent.relative_to(root)),status="accepted",
            family=entry["family"],noisy=entry["noisy"],shift=entry["shift"],coverage=entry["coverage"],
            holes=list(entry_holes),selected_observation_seed=selected_obs_seed,**q))
        return entry["split"],packs,lineage,records
    todo=[e for e in plan["parents"] if e["split"] in target_splits and f"{e['split']}-{e['index']:02}" not in done_parents]
    records=list(existing_progress)
    own_ids={f"{e['split']}-{e['index']:02}" for e in plan["parents"]}
    own_lineage_ids={e["lineage_id"] for e in plan["parents"]}
    lineages=[l for l in existing_lineages if l["lineage_id"] in own_lineage_ids]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for split,packs,lineage,parent_records in pool.map(generate,todo):
            lineages.append(lineage);records.extend(parent_records)
            (root/"progress.json").write_text(json.dumps(dict(records=records,accepted=len(lineages),planned=len(plan["parents"])),indent=2))
            print(json.dumps(records[-1]),flush=True)
    generated_splits=[s for s in all_split_names if any(
        r.get("status")=="accepted" and r["parent"].startswith(s+"-") for r in records)]
    manifests=[]
    for split in generated_splits:
        suite_dir=root/"suites"/split
        if not (suite_dir/"manifest.json").is_file():
            split_packs=[root/"packs"/f"{e['split']}-{e['index']:02}" for e in plan["parents"] if e["split"]==split]
            build(split_packs,suite_dir,split)
        manifests.append(suite_dir/"manifest.json")
    all_lineages=list(lineages)
    if a.ood_for:
        original=a.ood_for.resolve()
        if not (original/"completion.json").is_file():raise ValueError("ID pilot must be completed before OOD audit")
        id_lineages=json.loads((original/"lineage.json").read_text())["parents"]
        all_lineages.extend(id_lineages)
        for p_manifest in sorted((original/"suites").glob("*/manifest.json")):
            manifests.append(p_manifest)
    lineage_file=root/"lineage.json"
    lineage_file.write_text(json.dumps(dict(schema="copper-lineage-v2",parents=all_lineages),indent=2))
    (root/"split-audit.json").write_text(json.dumps(audit(manifests,lineage_file),indent=2))
    missing_splits=[s for s in all_split_names if s not in generated_splits]
    (root/"completion.json").write_text(json.dumps(dict(
        status="ood_pilot_generated" if a.ood_for else "id_pilot_generated",
        parents=len(all_lineages),cases=2*len(all_lineages),
        generated_splits=generated_splits,
        missing=missing_splits+plan["missing"]+["visual geology review","population near-duplicate review"]),indent=2))


if __name__=="__main__":main()
