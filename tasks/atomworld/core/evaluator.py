"""A thin, hash-checked invocation of the unmodified upstream lightweight judge."""

from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path
from ldm_tts.contracts import EvaluationResult
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

    def evaluate(self, candidate):
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
        path = Path("evaluations") / f"{candidate.candidate_id}.json"
        write_json(
            self.run_dir / path,
            {
                "sample_id": sample_id,
                "candidate_id": candidate.candidate_id,
                "evaluator": "synthetic_fixture"
                if self.mock
                else "upstream_atomworld_evaluate",
                "distance_units": "normalized_dimensionless",
                **record,
            },
        )
        metrics = {"correct": float(record["correct"])}
        if record["rmsd"] is not None:
            metrics["normalized_rmsd"] = record["rmsd"]
        if record["max_dist"] is not None:
            metrics["normalized_max_dist"] = record["max_dist"]
        return EvaluationResult(
            candidate.candidate_id,
            "succeeded",
            metrics=metrics,
            artifacts={"evaluation": str(path)},
            resource_usage={"benchmark_jobs": 1},
            metadata={"sample_id": sample_id, **record},
        )
