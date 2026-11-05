"""Development-only fixed baseline end-to-end check through existing Execution."""
import argparse
import json
import shutil
from pathlib import Path
from tasks.copper_inversion.core.execution import Execution, freeze_model
from tasks.copper_inversion.v2.data import read_problem


def main():
    p=argparse.ArgumentParser();p.add_argument("--packs",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seconds",type=float,default=10);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.packs/"private-manifest.json").read_text())
    if manifest["stage"]!="development-only":raise ValueError("pilot cannot run formal sealed suites")
    execution=Execution(a.output/"jobs","ldm-copper-baselines:20261024")
    results=[]
    for method in ("deterministic","metropolis"):
        source=a.output/"sources"/method;source.mkdir(parents=True)
        call="deterministic(problem, budget, seed)" if method=="deterministic" else "metropolis(problem, budget, seed)[0]"
        (source/"solver.py").write_text(f"from tasks.copper_inversion.v2.baselines import {method}\ndef solve(problem, budget, seed):\n    return {call}\n")
        for case in manifest["cases"]:
            root=a.output/method/case["case_id"];root.mkdir(parents=True)
            inputs=root/"input";shutil.copytree(a.packs/case["case_id"],inputs)
            problem=read_problem(inputs)
            (inputs/"budget.json").write_text(json.dumps(dict(seed=47,wall_time_seconds=a.seconds,cpu_count=2,memory_mb=4096)))
            job,r=execution.run(dict(method=method,case=case["case_id"],seed=47),phase="solver",inputs=inputs,
                timeout=a.seconds,cpus=2,memory_mb=4096,source=source,science_module="v2")
            outcome=dict(status="failed",error="solver failed",metrics={"iou":0.})
            if r["exit_code"]==0 and not r["reason"] and not r["oom"]:
                freeze_model(job/"output/model.npy",root/"model",problem["shape"])
                scoring,s=execution.run(dict(method=method,case=case["case_id"],seed=47),phase="scorer",inputs=inputs,
                    timeout=30,model=root/"model",truth=a.packs/"private",science_module="v2")
                if s["exit_code"] or s["reason"] or s["oom"]:raise RuntimeError("independent scorer infrastructure failure")
                outcome=json.loads((scoring/"output/result.json").read_text())
            results.append(dict(method=method,case=case["case_id"],modality=case["modality"],solver_wall_seconds=r["wall_seconds"],**outcome))
            (a.output/"results.json").write_text(json.dumps(dict(stage="development-only",seconds=a.seconds,image=execution.image,results=results),indent=2))
            print(json.dumps(results[-1]),flush=True)


if __name__=="__main__":main()
