"""Freeze all development methods before any validation feedback is produced."""
import argparse
import json
from pathlib import Path
from tasks.copper_inversion.core.execution import docker
from tasks.copper_inversion.v2.training_receipts import eligible
from tasks.copper_inversion.v2.selection import freeze,run,ranking


def main():
    p=argparse.ArgumentParser()
    for name in ("root","ldm-run","baseline-run","train","validation","test-id","test-ood"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=False)
    image=docker("image","inspect","ldm-copper-baselines:20261024","--format","{{.Id}}").stdout.strip()
    records=eligible(a.ldm_run,a.train)
    if len(records)!=5:raise ValueError("this pilot requires exactly five completed LDM training evaluations")
    ordered=sorted(records,key=lambda sha:ranking(dict(source_sha256=sha,metrics=records[sha]["metrics"])))
    plans=[("ldm",a.ldm_run,[Path(records[s]["source_path"]) for s in ordered[:3]])]
    baselines=eligible(a.baseline_run,a.train)
    summary=json.loads((a.baseline_run/"results.json").read_text())
    if len(summary["results"])!=5:raise ValueError("all five baseline training methods must finish first")
    for row in summary["results"]:
        sha=row["source_sha256"]
        if sha not in baselines:raise ValueError("baseline missing complete training evidence")
        plans.append((row["method"],a.baseline_run,[Path(baselines[sha]["source_path"])]))
    for name,training,sources in plans:
        freeze(a.root/name,sources,a.validation,[a.test_id,a.test_ood],seconds=10,image=image,
            training_run=training,training_manifest=a.train)
    (a.root/"plan.json").write_text(json.dumps(dict(development_only=True,
        caveat="Fixed baselines have not received equal tuning budgets; no formal superiority claim",
        ldm_rule="top three whole-training-suite scores using frozen ranking",methods=[p[0] for p in plans]),indent=2))
    for phase in ("validation","test"):
        for name,_,_ in plans:
            result=run(a.root/name,phase)
            print(json.dumps(dict(method=name,phase=phase,result=result)),flush=True)


if __name__=="__main__":main()
