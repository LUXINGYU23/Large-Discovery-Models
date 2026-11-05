"""Fixed, untuned development baselines through the same admission and scorer."""
import argparse
import json
from pathlib import Path
from ldm_tts.contracts import RawProposal, CandidateRejection
from ldm_tts.engine.run_store import BudgetLedger,atomic_json_write
from tasks.copper_inversion.core.candidate import seal
from tasks.copper_inversion.core.execution import Execution,source_identity,digest
from tasks.copper_inversion.v2.budget import ScientificSlots
from tasks.copper_inversion.v2.domain import CheckedDomain
from tasks.copper_inversion.v2.interface_checks import InterfaceChecks
from tasks.copper_inversion.v2.suite import load_manifest,SuiteEvaluator


def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--seconds",type=float,default=10)
    args=p.parse_args();manifest=load_manifest(args.manifest)
    if manifest["stage"]!="development" or manifest["split"]!="train":
        raise ValueError("pilot comparisons only use development training cases")
    args.output.mkdir(parents=True,exist_ok=False)
    ledger=BudgetLedger(limits={"algorithm_feedback_slots":5},path=args.output/"budget.json")
    slots=ScientificSlots(ledger.consume_many)
    execution=Execution(args.output/"jobs","ldm-copper-baselines:20261024",ledger.consume_many)
    checks=InterfaceChecks(args.output,execution,slots)
    domain=CheckedDomain(args.output,checks)
    evaluator=SuiteEvaluator(args.output,execution,args.manifest,args.seconds,slots)
    template=(Path(__file__).parent/"stochastic_solver.py").read_text()
    sources={"initialize":"from tasks.copper_inversion.v2.baselines import initialize\ndef solve(problem,budget,seed):\n    return initialize(problem)\n",
             "mcmc_v2":template,"acq_v2":template.replace('"method":"mcmc_v2"','"method":"acq_v2"')}
    deterministic=(Path(__file__).parent/"deterministic_solver.py").read_text()
    sources.update(distance_local=deterministic,contact_rbf=deterministic.replace('METHOD="distance_local"','METHOD="contact_rbf"'))
    atomic_json_write(args.output/"config.json",dict(stage="development",untuned=True,seconds=args.seconds,
        manifest_digest=digest(manifest),implementation_sha256=source_identity(),image=execution.image))
    rows=[]
    for method,source in sources.items():
        payload=seal(args.output,source.encode())
        candidate=domain.admit(RawProposal(payload,method))
        if isinstance(candidate,CandidateRejection):
            rows.append(dict(method=method,status="interface_rejected",detail=str(candidate)))
        else:
            result=evaluator.evaluate(candidate)
            rows.append(dict(method=method,source_sha256=candidate.canonical_key,status=str(result.status),metrics=dict(result.metrics)))
        atomic_json_write(args.output/"results.json",dict(development_only=True,results=rows))
        print(json.dumps(rows[-1]),flush=True)


if __name__=="__main__":main()
