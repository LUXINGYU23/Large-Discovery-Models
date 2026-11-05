"""Prespecified Sobol tuning with the same whole-suite scientific feedback cap.

No test/validation feedback is used. This is low-dimensional parameter search,
not code BO and not a new LDM orchestration layer.
"""
import argparse
import ast
import json
import math
from pathlib import Path
from scipy.stats import qmc
from ldm_tts.contracts import RawProposal,CandidateRejection
from ldm_tts.engine.run_store import BudgetLedger,atomic_json_write
from tasks.copper_inversion.core.candidate import seal
from tasks.copper_inversion.core.execution import Execution,source_identity,digest,docker
from tasks.copper_inversion.v2.budget import ScientificSlots
from tasks.copper_inversion.v2.domain import CheckedDomain
from tasks.copper_inversion.v2.interface_checks import InterfaceChecks
from tasks.copper_inversion.v2.public_catalog import load_catalog,catalog_digest
from tasks.copper_inversion.v2.service_client import ServiceEvaluator


def configurations(method,count,seed):
    if method not in {"mcmc_v2","acq_v2"} or not 1<=count<=321 or seed<0:
        raise ValueError("invalid preregistered tuning plan")
    default=dict(method=method,temperature=.1,step_cells=1.,range_scale=1.,batch=4,warmup=4,beta=.75,initializer="nearest")
    rows=[default]
    samples=qmc.Sobol(d=7,scramble=True,seed=seed).random_base2(math.ceil(math.log2(max(1,count-1))))
    for point in samples[:count-1]:
        rows.append(dict(default,temperature=float(10**(-3+4*point[0])),step_cells=float(2**(-1+3*point[1])),
            range_scale=float(2**(-3+4*point[2])),initializer="nearest" if point[3]<.5 else "random_field",
            batch=(2,4,8)[min(2,int(3*point[4]))] if method=="acq_v2" else 4,
            warmup=(3,4,8)[min(2,int(3*point[5]))] if method=="acq_v2" else 4,
            beta=float(2*point[6]) if method=="acq_v2" else .75))
    return rows


def source_for(template,parameters):
    # Replace exactly the literal CONFIG assignment, preserving all algorithm code.
    tree=ast.parse(template)
    nodes=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="CONFIG" for t in n.targets)]
    if len(nodes)!=1:raise ValueError("template requires one CONFIG assignment")
    node=nodes[0];lines=template.splitlines(keepends=True)
    # Source identity must survive a sorted JSON plan round trip. Parameter
    # insertion order is not an algorithm change and must not spend a new slot.
    ordered = {key: parameters[key] for key in sorted(parameters)}
    return "".join(lines[:node.lineno-1])+"CONFIG="+repr(ordered)+"\n"+"".join(lines[node.end_lineno:])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--public-catalog",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--evaluation-socket",type=Path,required=True)
    parser.add_argument("--evaluation-contract",required=True)
    parser.add_argument("--evaluation-service-uid",type=int,required=True)
    parser.add_argument("--method",choices=["mcmc_v2","acq_v2"],required=True)
    parser.add_argument("--slots",type=int,default=321);parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--seconds",type=float,default=120);parser.add_argument("--resume",action="store_true")
    parser.add_argument("--prepare-only",action="store_true",help="Freeze the plan and verify service before parallel execution")
    args=parser.parse_args();catalog=load_catalog(args.public_catalog)
    if not math.isfinite(args.seconds) or args.seconds<=0:raise ValueError("invalid solver budget")
    if catalog["stage"]=="formal" and (args.slots!=321 or args.seconds!=120 or args.seed not in (42,43,3407)):
        raise ValueError("formal tuning differs from preregistered budget or seeds")
    template=(Path(__file__).parent/"stochastic_solver.py").read_text()
    parameters=configurations(args.method,args.slots,args.seed)
    image=docker("image","inspect","ldm-copper-baselines:20261024","--format","{{.Id}}").stdout.strip()
    plan=dict(method=args.method,discovery_seed=args.seed,slots=args.slots,seconds=args.seconds,image=image,
        suite_sha256=catalog_digest(args.public_catalog),implementation_sha256=source_identity(),template_sha256=digest(template),
        evaluation_contract=args.evaluation_contract,evaluation_socket=str(args.evaluation_socket.resolve()),
        evaluation_service_uid=args.evaluation_service_uid,
        search="scrambled Sobol; default first; no held-out feedback",parameters=parameters)
    root=args.output
    if args.resume:
        if json.loads((root/"plan.json").read_text())!=plan:raise ValueError("tuning identity changed")
        ledger=BudgetLedger.load(root/"budget.json")
    else:
        root.mkdir(parents=True,exist_ok=False);atomic_json_write(root/"plan.json",plan)
        ledger=BudgetLedger(limits={"algorithm_feedback_slots":args.slots},path=root/"budget.json")
    slots=ScientificSlots(ledger.consume_many);execution=Execution(root/"jobs",image,ledger.consume_many)
    checks=InterfaceChecks(root,execution,slots);domain=CheckedDomain(root,checks)
    evaluator=ServiceEvaluator(root,args.evaluation_socket,args.evaluation_contract,
        service_uid=args.evaluation_service_uid,slots=slots)
    evaluator.verify_contract(plan["suite_sha256"],args.seconds,args.slots)
    if args.prepare_only:
        ledger.write()
        return
    rows=[]
    for index,config in enumerate(parameters):
        payload=seal(root,source_for(template,config).encode())
        row_path=root/"tuning-receipts"/f"{index:03}.json"
        if row_path.exists():
            row=json.loads(row_path.read_text())
            if row["source_sha256"]!=payload["sha256"]:raise ValueError("tuning receipt source changed")
        else:
            candidate=domain.admit(RawProposal(payload,"fixed_sobol_parameter_search"))
            if isinstance(candidate,CandidateRejection):
                row=dict(index=index,source_sha256=payload["sha256"],status="interface_rejected",reason=str(candidate))
            else:
                result=evaluator.evaluate(candidate)
                row=dict(index=index,source_sha256=payload["sha256"],status=result.status,metrics=dict(result.metrics))
            atomic_json_write(row_path,row)
        rows.append(row);atomic_json_write(root/"results.json",dict(development_only=catalog["stage"]=="development",results=rows))
        print(json.dumps(row),flush=True)
    atomic_json_write(root/"completion.json",dict(status="completed",plan_sha256=digest(plan),evaluated=len(rows),
        scientific_feedback_slots=ledger.counters.get("algorithm_feedback_slots",0)))


if __name__=="__main__":main()
