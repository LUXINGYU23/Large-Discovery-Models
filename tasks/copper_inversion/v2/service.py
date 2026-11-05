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
from .suite import SuiteEvaluator,load_manifest
from .budget import ScientificSlots
from .domain import CheckedDomain
from .interface_checks import InterfaceChecks
from .public_catalog import observation_digest
from .data import read_problem
from .training_receipts import eligible

MAX_REQUEST=2*1024*1024


class EvaluatorService:
    def __init__(self,configuration):
        expected={"manifest","root","image","seconds","slots","caller_uid","socket"}
        if set(configuration)!=expected:raise ValueError("invalid trusted service configuration")
        manifest=load_manifest(configuration["manifest"])
        if manifest["split"]!="train":raise ValueError("service accepts training only")
        if manifest["stage"]=="formal":
            from .qualification import verify_release
            verify_release(configuration["manifest"])
            if configuration["slots"]!=321 or configuration["seconds"]!=120:
                raise ValueError("formal service requires 321 slots and 120 seconds")
        if not 1<=configuration["slots"]<=321 or not 1<=configuration["seconds"]<=120:
            raise ValueError("invalid fixed service budget")
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
        self.identity=dict(configuration=configuration,suite_digest=digest(manifest),implementation=source_identity())
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
        self.evaluator=SuiteEvaluator(self.root,execution,configuration["manifest"],configuration["seconds"],slots)

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
