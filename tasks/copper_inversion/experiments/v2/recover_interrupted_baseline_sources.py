"""Finish already charged interrupted sources; never reserve a new source."""
import json,concurrent.futures,sys
from pathlib import Path
from tasks.copper_inversion.v2.service_client import ServiceEvaluator
r=Path('/Aidata/shared/ldm/copper-formal-study-20261024-r1');d=json.loads((r/'deployment.json').read_text())
def finish(name):
 p=r/name;rec=json.loads((p/'serialization-recovery.json').read_text());row=json.loads((p/'tuning-receipts/001.json').read_text());extra=set(rec['duplicate_indices']['1'])-{row['source_sha256']}
 ledger=json.loads((p/'budget.json').read_text());out=p/'serialization-interrupted-recovery';out.mkdir(exist_ok=True)
 for sha in extra:
  assert 'copper:v2:scientific-source:'+sha in ledger['metadata']['cumulative_usage']
  cfg=d['configs'][name];client=ServiceEvaluator(p,cfg['socket'],cfg['contract'],service_uid=991)
  result=client.request(dict(operation='evaluate',source=(p/'candidates'/sha/'solver.py').read_text(),contract_digest=cfg['contract']))
  (out/(sha+'.json')).write_text(json.dumps(result,indent=2));print(name,result.get('status'),flush=True)
  if result.get('status') not in ('succeeded','invalid_interface'):raise RuntimeError('recovery incomplete '+name)
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(finish,[n for n in d['configs'] if not n.startswith('ldm-')]))
