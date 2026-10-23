"""Detach an authorized stage, keeping the provider credential only in pipes/memory."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=('tiny','full'),required=True)
a=p.parse_args();base=a.root.resolve();key=sys.stdin.read().strip()
assert key and not (base/f'{a.stage}-supervisor.json').exists()
python=base/'source/tasks/copper_inversion/.venv/bin/python'
script=base/'source/tasks/copper_inversion/experiments/encoding_ablation.py'
with (base/f'{a.stage}-supervisor.log').open('w') as log:
    proc=subprocess.Popen([str(python),str(script),'--root',str(base),'--stage',a.stage],
                          stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    proc.stdin.write(key.encode());proc.stdin.close()
print(json.dumps({'stage':a.stage,'pid':proc.pid,'log':str(base/f'{a.stage}-supervisor.log')}))
