"""Finish reports for the two explicitly dispatched baseline batches."""
import json
from pathlib import Path
import subprocess
import sys
import time

base=Path(sys.argv[1]);names=('matched-120s-v2','paper-1000-v2')
while True:
    state={}
    for name in names:
        root=base/name
        launch=json.loads((base/f'{name}-launch.json').read_text())
        done=(root/'status.json').exists()
        alive=Path(f'/proc/{launch["pid"]}').exists()
        results=[json.loads(p.read_text()) for p in root.glob('*/*/result-*.json')]
        failures=[str(p.relative_to(root)) for p in root.glob('*/*/failure.json')]
        state[name]=dict(process_alive=alive,finished_results=len(results),
            successful=sum(r['status']=='succeeded' for r in results),failures=failures,
            state='finished' if done else 'running' if alive else 'interrupted')
        subprocess.run([sys.executable,str(Path(__file__).with_name('report.py')),str(root)],
                       check=True,stdout=subprocess.DEVNULL)
    target=base/'benchmark-status.json';temporary=target.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(updated_at=time.time(),batches=state),indent=2));temporary.replace(target)
    if all(s['state']!='running' for s in state.values()):break
    time.sleep(30)
