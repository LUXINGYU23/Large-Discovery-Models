"""A thin, hash-checked invocation of the unmodified upstream lightweight judge."""

from __future__ import annotations
import importlib.util
import hashlib
import json
import sys
from pathlib import Path
from ldm_tts.contracts import EvaluationResult
from ldm_tts.contracts.evaluation import EVALUATION_ATTEMPT_RECEIPT_KEY
from tasks.atomworld.core.data import TASK_ROOT, sha256_file, write_json


def load_official_evaluator(upstream: Path):
    path = upstream / "src/atomworld/evaluate.py"
    pin = json.loads((TASK_ROOT / "resources/source_manifest.json").read_text())
    expected = pin["files_sha256"]["src/atomworld/evaluate.py"]
    if sha256_file(path) != expected:
        raise ValueError(
            "AtomWorld evaluator source hash mismatch; review the new source before updating the pin"
        )
    name = "_ldm_atomworld_official_evaluator"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.evaluate


class AtomWorldEvaluator:
    def __init__(
        self, targets: dict[str, str], run_dir: Path, *, official=None, mock=False
    ):
        self._targets = targets
        self.run_dir = run_dir
        self.official = official
        self.mock = mock

    def _identity(self, candidate):
        round_idx = candidate.metadata.get("submission_round", candidate.metadata.get("round_idx"))
        if isinstance(round_idx, bool) or not isinstance(round_idx, int) or round_idx < 0:
            raise ValueError("AtomWorld evaluation requires a scheduled submission round")
        return {
            "sample_id": candidate.payload["sample_id"],
            "round_idx": round_idx,
            "candidate_id": candidate.candidate_id,
            "canonical_key": candidate.canonical_key,
            "payload_sha256": self._digest(candidate.payload),
            "target_sha256": self._digest(self._targets[candidate.payload["sample_id"]]),
            "evaluator": "synthetic_fixture" if self.mock else "upstream_atomworld_evaluate",
        }

    @staticmethod
    def _digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def _receipt(self, candidate):
        identity = self._identity(candidate)
        path = Path("evaluations") / f"{identity['round_idx']:06d}" / f"{candidate.candidate_id}.json"
        receipt = json.loads((self.run_dir / path).read_text()) if (self.run_dir / path).exists() else None
        if receipt is not None:
            if receipt.get("identity") != identity or receipt.get("status") not in {"started", "completed"}:
                raise ValueError("AtomWorld evaluation receipt identity mismatch")
            if receipt["status"] == "completed":
                result = receipt.get("result", {})
                if (receipt.get("result_sha256") != self._digest(result)
                        or result.get("candidate_id") != candidate.candidate_id
                        or result.get("status") != "succeeded"
                        or result.get("metadata", {}).get(EVALUATION_ATTEMPT_RECEIPT_KEY) != self._usage_key(identity)):
                    raise ValueError("AtomWorld evaluation receipt result mismatch")
        return identity, path, receipt

    def _usage_key(self, identity):
        return f"atomworld_evaluation:{self._digest(identity)}"

    def evaluation_attempt_usage_key(self, candidate):
        identity, _, _ = self._receipt(candidate)
        return self._usage_key(identity)

    def evaluate(self, candidate):
        identity, path, receipt = self._receipt(candidate)
        if receipt is not None:
            if receipt["status"] != "completed":
                raise RuntimeError("AtomWorld judge call interrupted; refusing to repeat an ambiguous charged evaluation")
            return EvaluationResult(**receipt["result"])
        write_json(self.run_dir / path, {"identity": identity, "status": "started"})
        payload = candidate.payload
        sample_id = payload["sample_id"]
        if self.mock:
            # Synthetic exact-text fixture is intentionally distinct from official matching.
            from tasks.atomworld.core.proposals import extract_cif

            correct = (
                extract_cif(payload["generated_output"]) == self._targets[sample_id]
            )
            record = {
                "correct": correct,
                "wrong_type": None if correct else "MockTextMismatch",
                "rmsd": None,
                "max_dist": None,
            }
        else:
            result = self.official(
                self._targets[sample_id],
                payload["generated_output"],
                action_name=payload["action_name"],
            )
            record = {
                "correct": bool(result.correct),
                "wrong_type": result.wrong_type,
                "rmsd": None if result.rmsd is None else float(result.rmsd),
                "max_dist": None if result.max_dist is None else float(result.max_dist),
            }
        metrics = {"correct": float(record["correct"])}
        if record["rmsd"] is not None:
            metrics["normalized_rmsd"] = record["rmsd"]
        if record["max_dist"] is not None:
            metrics["normalized_max_dist"] = record["max_dist"]
        evaluation = EvaluationResult(
            candidate.candidate_id,
            "succeeded",
            metrics=metrics,
            artifacts={"evaluation": str(path)},
            resource_usage={"benchmark_jobs": 1},
            metadata={"sample_id": sample_id, "distance_units": "normalized_dimensionless",
                      EVALUATION_ATTEMPT_RECEIPT_KEY: self._usage_key(identity), **record},
        )
        result = evaluation.to_dict()
        write_json(self.run_dir / path, {"identity": identity, "status": "completed",
                                        "result": result, "result_sha256": self._digest(result)})
        return evaluation
