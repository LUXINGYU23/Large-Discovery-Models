"""Formal solver -> frozen geometry -> independent score, behind shared receipts."""

import hashlib
import json
from pathlib import Path
from ldm_tts.contracts import EvaluationResult
from ldm_tts.contracts.evaluation import (
    EVALUATION_ATTEMPT_RECEIPT_KEY,
    EvaluationPaused,
)
from ldm_tts.engine.run_store import atomic_json_write
from .data import save_problem
from .execution import digest, freeze_model, source_identity


class Evaluator:
    def __init__(self, root, execution, problem, seconds):
        self.root = Path(root)
        self.execution = execution
        self.problem = problem
        self.seconds = seconds
        self.context = digest(
            {
                "source": source_identity(),
                "case": problem["case_id"],
                "seconds": seconds,
                "image": execution.image,
                "seed": 47,
            }
        )

    def evaluation_attempt_usage_key(self, candidate):
        return digest({"context": self.context, "source": candidate.canonical_key})

    def evaluate(self, candidate):
        key = self.evaluation_attempt_usage_key(candidate)
        root = self.root / "evaluations" / key
        root.mkdir(parents=True, exist_ok=True)
        receipt = root / "result.json"
        if receipt.exists():
            return EvaluationResult(**json.loads(receipt.read_text()))
        inputs = root / "input"
        save_problem(self.problem, inputs)
        atomic_json_write(
            inputs / "budget.json",
            {
                "seed": 47,
                "wall_time_seconds": self.seconds,
                "cpu_count": 2,
                "memory_mb": 4096,
            },
        )
        source = self.root / candidate.payload["source_path"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != candidate.canonical_key:
            raise EvaluationPaused("source changed")
        job, r = self.execution.run(
            key,
            phase="solver",
            inputs=inputs,
            source=source.parent,
            timeout=self.seconds,
            cpus=2,
            memory_mb=4096,
        )
        status = "failed"
        metrics = {}
        error = ""
        jobs = 1
        artifacts = {}
        if r["reason"] == "timeout":
            status = "timed_out"
            error = "solver timeout"
        elif r["exit_code"] or r["oom"] or r["reason"]:
            error = "solver process failed"
        else:
            try:
                sha = freeze_model(
                    job / "output/model.npy", root / "model", self.problem["shape"]
                )
                score_root, s = self.execution.run(
                    {"evaluation": key, "model": sha},
                    phase="scorer",
                    inputs=inputs,
                    model=root / "model",
                    timeout=30,
                )
                jobs += 1
                if s["exit_code"] or s["oom"] or s["reason"]:
                    raise EvaluationPaused("independent scorer failed")
                outcome = json.loads((score_root / "output/result.json").read_text())
                status = outcome["status"]
                metrics = outcome.get("metrics", {})
                error = outcome.get("error", "")
                artifacts = {
                    "model": str((root / "model/model.npy").relative_to(self.root)),
                    "score": str(
                        (score_root / "output/result.json").relative_to(self.root)
                    ),
                }
            except (ValueError, OSError) as exc:
                status = "invalid"
                error = str(exc)
        result = EvaluationResult(
            candidate.candidate_id,
            status,
            metrics=metrics,
            artifacts=artifacts,
            resource_usage={"benchmark_jobs": jobs},
            error=error,
            metadata={EVALUATION_ATTEMPT_RECEIPT_KEY: key, "seed": 47},
        )
        atomic_json_write(receipt, result.to_dict())
        return result
