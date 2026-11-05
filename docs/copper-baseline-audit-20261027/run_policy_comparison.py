import os,sys,time,subprocess
from pathlib import Path
p=Path('/Aidata/shared/ldm/copper-baseline-audit-20261027')
fd=os.open(p/'policy-comparison-gate.started',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.write(fd,str(os.getpid()).encode());os.close(fd)
end=time.monotonic()+1200
while not all((p/x/'results.json').exists() for x in ['policy-development','projected-policy']):
 if time.monotonic()>end:raise TimeoutError('policy qualification incomplete')
 time.sleep(10)
subprocess.run([sys.executable,str(p/'qualify_policy.py')],check=True)
os.execv(sys.executable,[sys.executable,str(p/'compare_actual_time.py')])
