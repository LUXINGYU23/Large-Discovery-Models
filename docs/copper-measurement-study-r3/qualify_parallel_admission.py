"""Four profile batches of eight candidates against the saturated shared queue."""
from pathlib import Path
import json,hashlib,time
from concurrent.futures import ThreadPoolExecutor
from tasks.copper_inversion.core.execution import Execution,digest
from tasks.copper_inversion.core.candidate import seal
from tasks.copper_inversion.v2.interface_checks import InterfaceChecks
from tasks.copper_inversion.experiments.v2.parallel_runtime import install
OPS=Path('/Aidata/shared/ldm/copper-permanent-repair-20261025')
R=OPS/'parallel-admission-v2';R.mkdir(exist_ok=False)
policy=dict(queue_root=str(OPS/'load-full-v2/queue'),global_jobs=128,interface_slots=8,jobs_per_process=32,startup_slots=4)
policy['policy_sha256']=digest(policy);install(policy)
executor=Execution(R/'jobs','sha256:f6856385a0da9444b1e5edb5048b2f6081593fe568c3473c2fda524d0c4489f5');checks=InterfaceChecks(R,executor)
payloads=[seal(R,('from tasks.copper_inversion.v2.baselines import initialize\ndef solve(problem,budget,seed):\n return initialize(problem)\n# candidate '+str(i)).encode()) for i in range(32)]
start=time.monotonic()
def profile(i):
 for p in payloads[i*8:(i+1)*8]:assert checks(p)
with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(profile,range(4)))
receipts={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'jobs').rglob('receipt.json')}
for p in payloads:assert checks(p)
assert receipts=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'jobs').rglob('receipt.json')}
assert len(receipts)==192
result=dict(status='passed',sources=32,receipts=192,replay_unchanged=True,wall_seconds=time.monotonic()-start)
(R/'completion.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
