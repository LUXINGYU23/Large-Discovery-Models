from pathlib import Path
from collections import Counter
import json
import os
import sys
from tasks.antibody_rfd3.repro.h20.full_mixed_compare import make_cases, audit_absolut_receipts, audit_ldm, SEED_BANK
from tasks.antibody_rfd3.repro.h20.paired_compare import audit, observations, query
from tasks.antibody_rfd3.repro.h20.launch_utils import run_case, save_summary, private_environment_values, ROOT

root=Path(sys.argv[1])
phase=sys.argv[2]
output=root/phase
output.mkdir(parents=True, exist_ok=True)
case=make_cases(output)['ldm_harness']
overrides={}
if phase=='pilot':
    overrides.update({'iterations':32,'research-rounds':2,'scores-per-generation':8,
       'pool-size':32,'research-candidates-per-role':8,'local-proposal-count':16,
       'evaluation-window':8,'concurrency':2})
for key,value in overrides.items():
    case['command'].extend(['--set',f'args.{key}='+json.dumps(value)])
    case['effective_args'][key.replace('-','_')]=value
case['tool_version']='sequence-conditioned-v1'
env=dict(os.environ,ANTBO_HOME=str(ROOT/'env/sources/HEBO-77636146/AntBO'),
         OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',
         TOKENIZERS_PARALLELISM='false',TMPDIR=str(ROOT/'tmp'),LLM_REASONING_EFFORT='low',
         NO_PROXY='localhost,127.0.0.1,172.17.0.1',no_proxy='localhost,127.0.0.1,172.17.0.1')
save=lambda:save_summary(output/'case.json',case)
save()
run_case(case,env,172800,private_environment_values(env,env['LLM_API_KEY']),save)
directory=Path(case['run_dir'])
result=audit({'ldm_harness':case},json.loads(SEED_BANK.read_text()))['ldm_harness']
result['absolut_receipts_verified']=audit_absolut_receipts(directory,'ldm_harness')
rows=observations(directory,'ldm_harness')
jobs=[json.loads(p.read_text()) for p in (directory/'design_jobs').glob('*/receipt.json')]
result['tool_calls']=dict(Counter(j['action'] for j in jobs))
result['measured_actions']=dict(Counter(o['candidate']['payload'].get('generation_action','structure_or_local') for o in rows[16:]))
if phase=='full': result.update(audit_ldm(directory))
result['status']='passed'
save_summary(output/'audit.json',result)
print(json.dumps({'phase':phase,'scores':len(rows),'best':result['best_so_far'][-1],'tool_calls':result['tool_calls']}),flush=True)
