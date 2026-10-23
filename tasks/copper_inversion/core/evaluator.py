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
from .feedback import model_slices


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
        resource_usage = {"solver_wall_seconds": r["wall_seconds"]}
        samples = {"solver": r["resource_sampling"]}
        workloads = {"solver": r["workload_resources"]}
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
            except (ValueError, OSError) as exc:
                status = "invalid"
                error = str(exc)
            else:
                score_root, s = self.execution.run(
                    {"evaluation": key, "model": sha},
                    phase="scorer",
                    inputs=inputs,
                    model=root / "model",
                    timeout=30,
                )
                jobs += 1
                resource_usage["scorer_wall_seconds"] = s["wall_seconds"]
                samples["scorer"] = s["resource_sampling"]
                workloads["scorer"] = s["workload_resources"]
                if s["exit_code"] or s["oom"] or s["reason"]:
                    raise EvaluationPaused("independent scorer failed")
                try:
                    outcome = json.loads((score_root / "output/result.json").read_text())
                    if not isinstance(outcome, dict) or outcome.get("status") not in {"succeeded", "invalid"}:
                        raise ValueError("invalid scorer result schema")
                    if outcome["status"] == "succeeded":
                        import math

                        loss = outcome["metrics"]["loss_total"]
                        if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss):
                            raise ValueError("invalid scorer loss")
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    raise EvaluationPaused("independent scorer receipt unavailable or invalid") from exc
                status = outcome["status"]
                metrics = outcome.get("metrics", {})
                error = outcome.get("error", "")
                artifacts = {
                    "model": str((root / "model/model.npy").relative_to(self.root)),
                    "score": str(
                        (score_root / "output/result.json").relative_to(self.root)
                    ),
                }
                if status == "succeeded":
                    preview = root / "model/slices.json"
                    atomic_json_write(preview, model_slices(root / "model/model.npy", self.problem, seed=47))
                    artifacts["model_slices"] = str(preview.relative_to(self.root))
        for phase, sampling in samples.items():
            for name in ("cpu_seconds_observed", "sampled_peak_memory_bytes"):
                if sampling[name] is not None:
                    resource_usage[f"{phase}_{name}"] = sampling[name]
        for phase, workload in workloads.items():
            if workload is not None:
                for name in ("peak_memory_bytes", "cpu_seconds_through_reap"):
                    if workload[name] is not None:
                        resource_usage[f"{phase}_{name}"] = workload[name]
        result = EvaluationResult(
            candidate.candidate_id,
            status,
            metrics=metrics,
            artifacts=artifacts,
            resource_usage={"benchmark_jobs": jobs, **resource_usage},
            error=error,
            metadata={EVALUATION_ATTEMPT_RECEIPT_KEY: key, "seed": 47,
                      "resource_sampling": samples, "workload_resources": workloads},
        )
        atomic_json_write(receipt, result.to_dict())
        return result
