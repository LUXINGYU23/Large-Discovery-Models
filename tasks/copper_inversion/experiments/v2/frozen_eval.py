"""Trusted controller CLI; never make this a researcher tool."""
import argparse
import json
from pathlib import Path
from tasks.copper_inversion.core.execution import docker
from tasks.copper_inversion.v2.selection import freeze,run


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="command",required=True)
    f=sub.add_parser("freeze");f.add_argument("--root",type=Path,required=True)
    f.add_argument("--sources",type=Path,nargs="+",required=True)
    f.add_argument("--training-run",type=Path,required=True);f.add_argument("--training-manifest",type=Path,required=True)
    f.add_argument("--validation",type=Path,required=True)
    f.add_argument("--test-id",type=Path,required=True);f.add_argument("--test-ood",type=Path,required=True)
    f.add_argument("--seconds",type=float,default=120);f.add_argument("--image",default="ldm-copper-baselines:20261024")
    for phase in ("validation","test"):
        p=sub.add_parser(phase);p.add_argument("--root",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="freeze":
        image=docker("image","inspect",args.image,"--format","{{.Id}}").stdout.strip()
        result=freeze(args.root,args.sources,args.validation,[args.test_id,args.test_ood],seconds=args.seconds,image=image,
            training_run=args.training_run,training_manifest=args.training_manifest)
        print(json.dumps(dict(stage=result["stage"],candidates=len(result["candidates"]),root=str(args.root))))
    else:print(json.dumps(run(args.root,args.command)))


if __name__=="__main__":main()
