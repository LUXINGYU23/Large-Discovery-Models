"""One-shot gate. No scientific run is resumed or overwritten."""
import os,sys,time,subprocess
from pathlib import Path
p=Path('/Aidata/shared/ldm/copper-baseline-audit-20261027')
fd=os.open(p/'comparison-gate.started',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
os.write(fd,str(os.getpid()).encode());os.close(fd)
end=time.monotonic()+1200
while not (p/'full-development/results.json').exists():
 if time.monotonic()>end:raise TimeoutError('development qualification incomplete; comparison not started')
 time.sleep(10)
subprocess.run([sys.executable,str(p/'qualify_results.py')],check=True)
os.execv(sys.executable,[sys.executable,str(p/'compare_actual_time.py')])
