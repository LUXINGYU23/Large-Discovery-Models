"""Trusted private-account generation; qualification required before publication."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--python',type=Path,required=True)
    args=parser.parse_args();root=args.root
    plan=json.loads((root/'generation-plan.json').read_text())
    for mode in ('id','ood'):
        command=[sys.executable,'-m','tasks.copper_inversion.experiments.v2.generate_pilot',
                 '--output',str(root/mode),'--generator-python',str(args.python),
                 '--seed',str(plan[mode+'_seed']),'--full-study']
        if mode=='ood':command+=['--ood-for',str(root/'id')]
        subprocess.run(command,check=True)
    (root/'generation-complete.json').write_text(json.dumps(dict(status='generated_pending_qualification',counts=plan['counts'])))


if __name__=='__main__':main()
