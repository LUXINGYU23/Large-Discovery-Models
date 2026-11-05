"""Replay real generated sources through fixed admission, fusion BO and scoring.

Diagnostic only: this does not substitute for a successful live research turn.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path
from ldm_tts.campaign import CampaignRequest,CampaignBudget
from ldm_tts.contracts import RawProposal,ProposalSearchSpec
from ldm_tts.engine import LDMEngine
from ldm_tts.engine.expansion import CallableReservoirExpander,ExpansionResult
from tasks.copper_inversion.core.candidate import seal
from tasks.copper_inversion.core.execution import Execution,digest,source_identity
from tasks.copper_inversion.core.encoding import CodeEncoder
from tasks.copper_inversion.core.surrogate import LossSelector
from tasks.copper_inversion.v2.suite import load_manifest,SuiteEvaluator
from tasks.copper_inversion.v2.interface_checks import InterfaceChecks
from tasks.copper_inversion.v2.domain import CheckedDomain
from tasks.copper_inversion.v2.budget import ScientificSlots
from tasks.copper_inversion.v2.task_spec import describe_task


def main():
    p=argparse.ArgumentParser();p.add_argument("--source-run",type=Path,required=True);p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    manifest=load_manifest(a.manifest)
    if manifest["stage"]!="development":raise ValueError("diagnostic only accepts development cases")
    sources=sorted(a.source_run.glob("harness/branches/round_0001/sessions/*/turns/*/attempts/0001/artifacts/artifact-00.py"))
    if len(sources)!=4:raise ValueError("exactly four first-attempt real candidate snapshots required")
    seed=seal(a.output,b"from tasks.copper_inversion.v2.baselines import initialize\ndef solve(problem,budget,seed):\n    return initialize(problem)\n")
    proposals=[RawProposal(seal(a.output,s.read_bytes()),"frozen_real_research_artifact") for s in sources]
    task=describe_task(digest(manifest))
    task=replace(task,proposal_search=ProposalSearchSpec(name="frozen_real_artifact_replay"),metadata={**task.metadata,"diagnostic_only":True})
    budget=CampaignBudget(rounds=2,reservoir_size=4,batch_size=2,target_observations=3,max_evaluation_attempts=3,
        extra_limits={"algorithm_feedback_slots":3,"interface_jobs":30,"solver_jobs":6,"scorer_jobs":6,"benchmark_jobs":12})
    runtime=CampaignRequest(run_dir=a.output,budget=budget,config=dict(diagnostic_only=True,source_run=str(a.source_run),
        implementation_sha256=source_identity(),source_digests=[p.payload["sha256"] for p in proposals],seconds=10)).open_runtime(task)
    execution=Execution(a.output/"jobs","ldm-copper-baselines:20261024",runtime.consume_many)
    slots=ScientificSlots(runtime.consume_many);checks=InterfaceChecks(a.output,execution,slots)
    def expand(request):
        if request.round_idx==0:return ExpansionResult(proposals=(RawProposal(seed,"fixed_public_seed"),),selection_mode="reservoir_order")
        return ExpansionResult(proposals=tuple(proposals),selection_mode="acquisition")
    engine=LDMEngine(task_spec=task,expander=CallableReservoirExpander(expand),candidate_domain=CheckedDomain(a.output,checks),
        evaluator=SuiteEvaluator(a.output,execution,a.manifest,10,slots),runtime=runtime,
        surrogate_encoder=CodeEncoder(a.output,"fusion",a.model,"cuda:0"),selector=LossSelector("fusion"))
    result=engine.run(budget.engine_config())
    rows=[dict(candidate=o.canonical_key,round=o.round_idx,**o.metrics) for o in result.state.observations]
    (a.output/"replay-result.json").write_text(json.dumps(dict(diagnostic_only=True,observations=rows),indent=2))
    print(json.dumps(rows))


if __name__=="__main__":main()
