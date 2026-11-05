import json, shutil
from pathlib import Path
import pytest
from tasks.copper_inversion.core import execution
from tasks.copper_inversion.core.data import load_problem,save_problem
from tasks.copper_inversion.core.candidate import seal
from ldm_tts.contracts.evaluation import EvaluationPaused
from ldm_tts.engine.run_store import BudgetLedger
from tasks.copper_inversion.v2.budget import ScientificSlots
R=Path('/Aidata/shared/ldm/copper-measurement-study-20261027-r4-baselines')
@pytest.mark.parametrize('boundary',['missing','created','terminal','terminal137'])
def test_real_docker_recovery(tmp_path,monkeypatch,boundary):
 inputs=tmp_path/'inputs';save_problem(load_problem('drillholes_3d',fixture=True),inputs)
 execution.atomic_json_write(inputs/'budget.json',dict(seed=47,wall_time_seconds=5,cpu_count=1,memory_mb=2048))
 body=b"def solve(problem,budget,seed):\n    return problem['initial_level_set'].copy()\n"
 if boundary=='terminal137':body=b"def solve(problem,budget,seed):\n    import time\n    time.sleep(20)\n    return problem['initial_level_set'].copy()\n"
 p=seal(tmp_path,body)
 runner=execution.Execution(tmp_path/'jobs','ldm-copper-baselines:20261024')
 old=execution.docker;write=execution.atomic_json_write;starts=[];fault=False
 def docker(*a,**k):
  nonlocal fault
  if a[0]=='create' and boundary=='missing' and not fault:
   fault=True;raise EvaluationPaused('injected before create')
  if a[0]=='create' and boundary=='created' and not fault:
   result=old(*a,**k);fault=True;raise EvaluationPaused('injected after create before job persistence')
  if a[0]=='start':starts.append(a[1])
  return old(*a,**k)
 def save(path,data):
  nonlocal fault
  if Path(path).name=='receipt.json' and boundary in ('terminal','terminal137') and not fault:
   fault=True;raise EvaluationPaused('injected after exit before receipt')
  return write(path,data)
 monkeypatch.setattr(execution,'docker',docker);monkeypatch.setattr(execution,'atomic_json_write',save)
 kw=dict(phase='solver',inputs=inputs,timeout=5,source=(tmp_path/p['source_path']).parent)
 with pytest.raises(EvaluationPaused):runner.run(boundary,**kw)
 assert fault
 root,result=runner.run(boundary,**kw)
 assert result['exit_code']==(137 if boundary=='terminal137' else 0) and result['workload_resources']['workload_complete']
 assert len(starts)==1
 before=(root/'receipt.json').read_bytes()
 assert runner.run(boundary,**kw)[1]==result and len(starts)==1
 assert before==(root/'receipt.json').read_bytes()
@pytest.mark.parametrize('name',[f'{m}-{s}' for s in [42,43,3407] for m in ['mcmc','acq']])
def test_original_batch_reservation_replay(tmp_path,name):
 from tasks.copper_inversion.experiments.v2.tune_stochastic import source_for
 root=R/name;before=(root/'budget.json').read_bytes();shutil.copy2(root/'budget.json',tmp_path/'budget.json')
 ledger=BudgetLedger.load(tmp_path/'budget.json');slots=ScientificSlots(ledger.consume_many)
 plan=json.loads((root/'plan.json').read_text());template=(R/'source/tasks/copper_inversion/experiments/v2/stochastic_solver.py').read_text()
 counters=dict(ledger.counters)
 count=counters['algorithm_feedback_slots'];assert count in (113,129)
 import hashlib
 for _ in range(2):
  for p in plan['parameters'][:count]:slots.reserve(hashlib.sha256(source_for(template,p).encode()).hexdigest())
 assert ledger.counters==counters
 assert (root/'budget.json').read_bytes()==before
