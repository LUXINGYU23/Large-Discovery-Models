"""Remote-only launcher. Provider key arrives on stdin, never in argv or files."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--public-catalog",type=Path,required=True)
    parser.add_argument("--evaluation-socket",type=Path,required=True)
    parser.add_argument("--evaluation-contract",required=True)
    parser.add_argument("--evaluation-service-uid",type=int,required=True)
    parser.add_argument("--evaluations-per-round",type=int,default=2)
    parser.add_argument("--candidates-per-researcher",type=int,default=1)
    parser.add_argument("--harness-seconds",type=int,default=600)
    args=parser.parse_args()
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in args.run_id):
        raise ValueError("run-id must contain only lowercase letters, digits and hyphens")
    base=Path("/Aidata/shared/ldm/copper-v2-development")
    source=base/f"source-{args.run_id}"
    if not (source/"tasks/copper_inversion/v2/workflow.py").is_file():
        raise ValueError("freeze the source snapshot before launching")
    run=base/"runs"/args.run_id
    launch=base/f"{args.run_id}-launch.json"
    if launch.exists() or run.exists():raise ValueError("development run identity already exists")
    key=sys.stdin.read().strip()
    if not key:raise ValueError("protected provider key missing")
    python="/Aidata/shared/ldm/copper-encoding-ablation-v3-20261023/source/tasks/copper_inversion/.venv/bin/python"
    command=[python,"-m","tasks.copper_inversion.v2.workflow",
        "--public-catalog",str(args.public_catalog.resolve()),"--out-dir",str(run),
        "--evaluation-socket",str(args.evaluation_socket.resolve()),
        "--evaluation-contract",args.evaluation_contract,
        "--evaluation-service-uid",str(args.evaluation_service_uid),
        "--embedding-model-path","/Aidata/shared/codebo/models/Qwen3-Embedding-0.6B-97b0c614",
        "--embedding-device","cuda:0","--harness-cache-dir","/Aidata/shared/ldm/cache/gondolin",
        "--sidecar-image","sha256:caffc0f9407f0ac895f8019af3d80f6d2feaa94df38218e800a01af7226a0df4",
        "--iterations","1","--evaluations-per-round",str(args.evaluations_per_round),"--candidates-per-researcher",str(args.candidates_per_researcher),
        "--solver-seconds","10","--harness-seconds",str(args.harness_seconds)]
    env=dict(os.environ,LLM_API_KEY=key,PYTHONPATH=str(source),OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",
        TMPDIR="/Aidata/shared/ldm/tmp")
    with (base/f"{args.run_id}.log").open("xb") as output:
        process=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=process.pid,started_unix=time.time(),source=str(source),run=str(run),command=command,
        proc_start_ticks=Path(f"/proc/{process.pid}/stat").read_text().split()[21])
    launch.write_text(json.dumps(record,indent=2))
    print(json.dumps(dict(pid=process.pid,run=str(run))))


if __name__=="__main__":main()
