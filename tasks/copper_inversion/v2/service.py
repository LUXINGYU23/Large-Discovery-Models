"""Private training evaluator over a local Unix socket, never a guest tool.

Trusted startup configuration owns dataset, image, budget and caller UID. The
wire protocol accepts source text only; it has no paths, shell or test endpoint.
"""
import argparse
import json
import logging
import os
import socket
import socketserver
import struct
from pathlib import Path
from ldm_tts.contracts import RawProposal,CandidateRejection
from ldm_tts.contracts.evaluation import EvaluationPaused
from ldm_tts.engine.run_store import BudgetLedger,BudgetExceededError,atomic_json_write
from ..core.candidate import seal,validate_source
from ..core.execution import Execution,source_identity,digest
from .contract import OBJECTIVE_VERSIONS
from .suite import SuiteEvaluator,load_manifest,release_tiers,manifest_tiers,q1_seconds,OBJECTIVES
from .budget import ScientificSlots
from .domain import CheckedDomain
from .interface_checks import InterfaceChecks
from .public_catalog import observation_digest
from .data import read_problem
from .training_receipts import eligible

MAX_REQUEST=2*1024*1024


def resolve_tier(configuration,manifest):
    """Budget from the release's sealed tier table; old releases keep fixed limits."""
    path=Path(configuration["manifest"]).resolve()
    release,tiers=release_tiers(path,manifest["stage"])
    seconds,slots=configuration["seconds"],configuration["slots"]
    if type(seconds) is not int or type(slots) is not int:raise ValueError("invalid service budget")
    if configuration.get("purpose")=="q1":
        # Q1 measurement is not a tier: budget from the release Q1 record only.
        if "tier" in configuration or slots<1 or seconds!=q1_seconds(path,manifest):
            raise ValueError("Q1 service requires the release Q1 solver budget and no tier")
        return None
    if configuration.get("purpose") is not None:raise ValueError("unknown service purpose")
    if tiers is None:
        if "tier" in configuration:raise ValueError("release has no tier table")
        if manifest["stage"]=="formal" and (slots!=321 or seconds!=120):
            raise ValueError("formal service requires 321 slots and 120 seconds")
        if not 1<=slots<=321 or not 1<=seconds<=120:
            raise ValueError("invalid fixed service budget")
        return None
    names=manifest_tiers(path,manifest,release,tiers,roles=("train",))
    if "tier" in configuration:
        if configuration["tier"] not in names:raise ValueError("manifest is not the training suite of the named tier")
        name=configuration["tier"]
    elif len(names)!=1:raise ValueError("training manifest does not identify one release tier")
    else:name=names[0]
    row=tiers[name]
    if (seconds,slots)!=(row["solver_seconds"],row["slots"]):
        raise ValueError(f"tier {name} requires {row['slots']} slots and {row['solver_seconds']} seconds")
    if len({c["parent_id"] for c in manifest["cases"]})!=row["train_bodies"]:
        raise ValueError("training bodies differ from the release tier table")
    return dict(name=name,**row)


class EvaluatorService:
    configuration_keys=frozenset({"manifest","root","image","seconds","slots","caller_uid","socket"})
    optional_keys=frozenset({"tier","objective","objective_version","purpose","anonymize_cases","canonical_origin"})

    def __init__(self,configuration):
        if not isinstance(configuration,dict) or not (
                self.configuration_keys<=set(configuration)<=self.configuration_keys|self.optional_keys):
            raise ValueError("invalid trusted service configuration")
        self.objective=configuration.get("objective","confirmation")
        if self.objective not in OBJECTIVES:raise ValueError("unknown evaluation objective")
        objective_version=configuration.get("objective_version")
        if objective_version is not None and objective_version not in OBJECTIVE_VERSIONS:
            raise ValueError("unknown evaluation objective version")
        anonymize_cases=configuration.get("anonymize_cases",False)
        canonical_origin=configuration.get("canonical_origin",False)
        if type(anonymize_cases) is not bool or type(canonical_origin) is not bool:
            raise ValueError("invalid case isolation flags")
        manifest=load_manifest(configuration["manifest"])
        if manifest["split"]!="train":raise ValueError("service accepts training only")
        if manifest["stage"]=="formal":
            from .qualification import verify_release
            verify_release(configuration["manifest"])
        self.tier=resolve_tier(configuration,manifest)
        self.root=Path(configuration["root"]);self.root.mkdir(parents=True,exist_ok=True)
        self.manifest_path=Path(configuration["manifest"]).resolve()
        self.caller_uid=configuration["caller_uid"]
        self.public_digest=observation_digest(manifest["stage"],
            (read_problem(Path(configuration["manifest"]).resolve().parent/row["public"]) for row in manifest["cases"]))
        self.seconds=configuration["seconds"]
        self.slot_limit=configuration["slots"]
        self.interface_failure_metrics=dict(mean_iou=0.,loss_total=1.,research_utility=0.,legal_rate=0.,
            parent_count=len({row["parent_id"] for row in manifest["cases"]}),
            solve_count=len(manifest["cases"])*len(manifest["seeds"]),
            failed_solves=len(manifest["cases"])*len(manifest["seeds"]),mean_solver_wall_seconds=0.)
        self.identity=dict(configuration=configuration,suite_digest=digest(manifest),implementation=source_identity(),
            objective=self.objective,tier=self.tier,purpose=configuration.get("purpose"))
        identity_file=self.root/"identity.json"
        if identity_file.exists():
            if json.loads(identity_file.read_text())!=self.identity:raise ValueError("private service identity changed")
            self.ledger=BudgetLedger.load(self.root/"budget.json")
        else:
            self.ledger=BudgetLedger(limits={"algorithm_feedback_slots":configuration["slots"]},path=self.root/"budget.json")
            self.ledger.write()
            atomic_json_write(identity_file,self.identity)
        execution=Execution(self.root/"jobs",configuration["image"],self.ledger.consume_many)
        if execution.image!=configuration["image"]:raise ValueError("service requires immutable image digest")
        slots=ScientificSlots(self.ledger.consume_many)
        self.slots=slots
        self.domain=CheckedDomain(self.root,InterfaceChecks(self.root,execution,slots))
        self.evaluator=SuiteEvaluator(self.root,execution,configuration["manifest"],configuration["seconds"],slots,
            objective=self.objective,purpose=configuration.get("purpose"),
            anonymize_cases=anonymize_cases,canonical_origin=canonical_origin,
            objective_version=objective_version)

    def dispatch(self,request):
        if not isinstance(request,dict):return {"status":"rejected_request"}
        if set(request)=={"operation","contract_digest"} and request["operation"]=="seal_training":
            if request["contract_digest"]!=digest(self.identity):return {"status":"contract_mismatch"}
            try:
                return self.seal_training()
            except Exception:
                logging.exception("private training seal failure")
                return {"status":"infrastructure_error"}
        if request=={"operation":"info"}:
            return dict(status="ready",contract_digest=digest(self.identity),
                public_digest=self.public_digest,seconds=self.seconds,slots=self.slot_limit,
                used_slots=self.ledger.counters.get("algorithm_feedback_slots",0))
        if set(request)!={"operation","source","contract_digest"} or request["operation"]!="evaluate" or not isinstance(request["source"],str):
            return {"status":"rejected_request"}
        if request["contract_digest"]!=digest(self.identity):return {"status":"contract_mismatch"}
        if (self.root/"training-sealed.json").exists():return {"status":"training_sealed"}
        try:
            body=request["source"].encode();validate_source(body)
        except (ValueError,SyntaxError,UnicodeError):return {"status":"invalid_source"}
        try:
            payload=seal(self.root,body)
            # Every accepted wire submission requests a scientific evaluation.
            # Reserve before interface feedback so an exhausted caller cannot
            # keep querying legality as a free oracle.
            self.slots.reserve(payload["sha256"])
            candidate=self.domain.admit(RawProposal(payload,"private_service"))
            if isinstance(candidate,CandidateRejection):
                return dict(status="invalid_interface",source_sha256=payload["sha256"],
                    metrics=self.interface_failure_metrics)
            result=self.evaluator.evaluate(candidate)
            return dict(status="succeeded",source_sha256=candidate.canonical_key,metrics=dict(result.metrics))
        except BudgetExceededError:return {"status":"budget_exhausted"}
        except EvaluationPaused:
            logging.exception("private evaluator paused")
            return {"status":"paused"}
        except Exception:
            logging.exception("private evaluator infrastructure failure")
            return {"status":"infrastructure_error"}

    def seal_training(self):
        """Serialized with evaluate; freeze training before trusted validation.

        No validation/test dataset or path can be supplied over this protocol.
        The trusted operator performs the separate frozen selection afterward.
        """
        path=self.root/"training-sealed.json"
        if path.exists():
            record=json.loads(path.read_text())
            if record["contract_digest"]!=digest(self.identity):raise ValueError("sealed contract differs")
        else:
            records=eligible(self.root,self.manifest_path)
            ordered=sorted(records,key=lambda sha:(records[sha]["metrics"]["loss_total"],
                -records[sha]["metrics"]["legal_rate"],records[sha]["metrics"]["mean_solver_wall_seconds"],sha))[:3]
            record=dict(contract_digest=digest(self.identity),
                used_slots=self.ledger.counters.get("algorithm_feedback_slots",0),
                source_sha256s=ordered,
                selected_receipts={sha:records[sha] for sha in ordered})
            atomic_json_write(path,record)
        return dict(status="training_sealed",contract_digest=record["contract_digest"],
            source_sha256s=record["source_sha256s"],used_slots=record["used_slots"])


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        _,uid,_=struct.unpack("3i",self.request.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
        if uid!=self.server.backend.caller_uid:return
        self.request.settimeout(20)
        line=self.rfile.readline(MAX_REQUEST+1)
        if len(line)>MAX_REQUEST or not line.endswith(b"\n"):
            response={"status":"rejected_request"}
        else:
            try:request=json.loads(line)
            except (ValueError,UnicodeError):response={"status":"rejected_request"}
            else:response=self.server.backend.dispatch(request)
        self.wfile.write(json.dumps(response,allow_nan=False).encode()+b"\n")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,required=True)
    args=parser.parse_args();configuration=json.loads(args.config.read_text())
    backend=EvaluatorService(configuration)
    # A pre-existing socket needs explicit operator reconciliation, not unlink.
    with socketserver.UnixStreamServer(configuration["socket"],Handler) as server:
        os.chmod(configuration["socket"],0o600)
        server.backend=backend
        print(json.dumps(dict(status="listening",pid=os.getpid())),flush=True)
        server.serve_forever()


if __name__=="__main__":main()
