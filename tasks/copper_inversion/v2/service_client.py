"""LDM evaluator adapter with no dataset or truth-path access."""
import json
import math
import socket
import struct
from pathlib import Path
from ldm_tts.contracts import EvaluationResult
from ldm_tts.contracts.evaluation import EvaluationPaused,EVALUATION_ATTEMPT_RECEIPT_KEY
from ldm_tts.engine.run_store import BudgetExceededError,atomic_json_write
from ..core.execution import digest
from ..core.candidate import validate_source

METRICS={"mean_iou","loss_total","research_utility","legal_rate","parent_count","solve_count","failed_solves","mean_solver_wall_seconds"}


def validate_metrics(metrics):
    if not isinstance(metrics,dict) or set(metrics)!=METRICS:raise EvaluationPaused("private evaluator metric schema differs")
    if any(type(v) not in (float,int) or not math.isfinite(v) for v in metrics.values()):
        raise EvaluationPaused("private evaluator returned nonfinite metrics")
    if any(not 0<=metrics[k]<=1 for k in ("mean_iou","loss_total","research_utility","legal_rate")):
        raise EvaluationPaused("private evaluator returned out-of-range metrics")
    if not math.isclose(metrics["loss_total"],1-metrics["research_utility"],abs_tol=1e-12):
        raise EvaluationPaused("private objective identities differ")
    if any(metrics[k]<0 or int(metrics[k])!=metrics[k] for k in ("parent_count","solve_count","failed_solves")) or metrics["mean_solver_wall_seconds"]<0:
        raise EvaluationPaused("invalid private evaluator resource counts")
    if not 0<metrics["parent_count"]<=metrics["solve_count"] or metrics["failed_solves"]>metrics["solve_count"]:
        raise EvaluationPaused("invalid private failure denominator")
    if not math.isclose(metrics["legal_rate"],1-metrics["failed_solves"]/metrics["solve_count"],abs_tol=1e-12):
        raise EvaluationPaused("private legal rate differs from failure counts")
    return metrics


class ServiceEvaluator:
    def __init__(self,root,socket_path,contract_digest,*,service_uid,timeout=7200,slots=None):
        if len(contract_digest)!=64 or any(c not in "0123456789abcdef" for c in contract_digest):
            raise ValueError("pin the trusted evaluator contract digest")
        if type(service_uid) is not int or service_uid<0 or not math.isfinite(timeout) or timeout<=0:
            raise ValueError("invalid evaluator peer or timeout")
        self.root=Path(root).resolve();self.socket=str(socket_path)
        self.contract_digest=contract_digest;self.service_uid=service_uid;self.timeout=timeout
        self.slots=slots

    def verify_contract(self,public_digest,seconds,slots):
        info=self.request({"operation":"info"})
        if (set(info)!={"status","contract_digest","public_digest","seconds","slots","used_slots"}
                or info["status"]!="ready" or info["contract_digest"]!=self.contract_digest
                or info["public_digest"]!=public_digest or info["seconds"]!=seconds
                or info["slots"]!=slots or type(info["used_slots"]) is not int
                or not 0<=info["used_slots"]<=slots):
            raise EvaluationPaused("private evaluator contract or public observations differ")
        return info

    def request(self,payload):
        try:
            with socket.socket(socket.AF_UNIX) as connection:
                connection.settimeout(self.timeout);connection.connect(self.socket)
                _,uid,_=struct.unpack("3i",connection.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
                if uid!=self.service_uid:raise EvaluationPaused("private evaluator peer identity differs")
                connection.sendall(json.dumps(payload).encode()+b"\n")
                with connection.makefile("rb") as stream:line=stream.readline(65537)
            if len(line)>65536 or not line.endswith(b"\n"):raise EvaluationPaused("private evaluator response is incomplete")
            result=json.loads(line)
            if not isinstance(result,dict):raise ValueError("response must be an object")
            return result
        except (OSError,ValueError,UnicodeError) as exc:
            raise EvaluationPaused("private evaluator outcome unavailable; replay same source after reconciliation") from exc

    def evaluation_attempt_usage_key(self,candidate):
        return digest(dict(service=self.contract_digest,source=candidate.canonical_key))

    def evaluate(self,candidate):
        try:
            return self._evaluate(candidate)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            raise EvaluationPaused("private evaluation source or receipt unavailable; reconcile before replay") from exc

    def _evaluate(self,candidate):
        source=(self.root/candidate.payload["source_path"])
        if source.is_symlink() or not source.resolve().is_relative_to(self.root):raise EvaluationPaused("candidate path differs")
        body=source.read_bytes()
        if validate_source(body)!=candidate.canonical_key:raise EvaluationPaused("candidate source differs")
        if self.slots is not None:self.slots.reserve(candidate.canonical_key)
        key=self.evaluation_attempt_usage_key(candidate)
        result_path=self.root/"evaluations"/key/"result.json"
        if result_path.exists():
            saved=json.loads(result_path.read_text());validate_metrics(saved["metrics"])
            if (saved["candidate_id"]!=candidate.candidate_id or saved["status"]!="succeeded"
                    or saved["metadata"].get("service_contract_digest")!=self.contract_digest
                    or saved["metadata"][EVALUATION_ATTEMPT_RECEIPT_KEY]!=key):
                raise EvaluationPaused("private evaluation receipt differs")
            return EvaluationResult(**saved)
        reply=self.request(dict(operation="evaluate",source=body.decode(),contract_digest=self.contract_digest))
        if reply.get("status")=="budget_exhausted":raise BudgetExceededError("private evaluator feedback cap reached")
        if reply.get("status") not in {"succeeded","invalid_interface"}:
            raise EvaluationPaused("private evaluator did not complete: "+str(reply.get("status")))
        if set(reply)!={"status","source_sha256","metrics"} or reply["source_sha256"]!=candidate.canonical_key:
            raise EvaluationPaused("private response source identity differs")
        metrics=validate_metrics(reply["metrics"])
        if reply["status"]=="invalid_interface" and (metrics["mean_iou"]!=0 or metrics["legal_rate"]!=0):
            raise EvaluationPaused("invalid interface must fail every planned solve")
        result=EvaluationResult(candidate.candidate_id,"succeeded",metrics=metrics,
            resource_usage={"private_suite_evaluations":1},
            metadata={EVALUATION_ATTEMPT_RECEIPT_KEY:key,"service_contract_digest":self.contract_digest,
                "service_outcome":reply["status"]})
        atomic_json_write(result_path,result.to_dict())
        return result
