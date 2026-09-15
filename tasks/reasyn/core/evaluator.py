"""Faithful reported metrics behind the shared CandidateEvaluator seam."""

from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from ldm_tts.contracts import EvaluationResult
from ldm_tts.contracts.evaluation import EVALUATION_ATTEMPT_RECEIPT_KEY
from ldm_tts.engine.run_store import atomic_json_write
from .chemistry import canonicalize, similarity


class ReconstructionEvaluator:
    def __init__(self, args, projector, target):
        self.args = args
        self.projector = projector
        self.target = target

    def prepare_evaluations(self, candidates):
        """Checkpoint selected projections before charging scientific trials.

        A worker interruption leaves completed targets reusable and propagates
        through the engine preparation seam so the same round can be resumed.
        """
        for candidate in candidates:
            self.projector.project(
                [candidate.payload["target_smiles"]],
                sampling_seed=candidate.payload["sampling_seed"],
                identity=candidate.candidate_id,
            )

    def evaluate(self, candidate):
        p = candidate.payload
        rows, artifact = self.projector.project(
            [p["target_smiles"]],
            sampling_seed=p["sampling_seed"],
            identity=candidate.candidate_id,
        )
        best = max(
            (similarity(self.target, r["smiles"], mock=self.args.mock) for r in rows),
            default=0.0,
        )
        exact = any(
            canonicalize(self.target, mock=self.args.mock, stereo=False)
            == canonicalize(r["smiles"], mock=self.args.mock, stereo=False)
            for r in rows
        )
        return EvaluationResult(
            candidate.candidate_id,
            "succeeded",
            {
                "similarity": best,
                "reconstructed": float(exact),
                "output_count": len(rows),
            },
            artifacts={"projection": artifact},
            resource_usage={"benchmark_jobs": 1},
            metadata={
                "original_target": self.target,
                "projection_query": p["target_smiles"],
                "mock": self.args.mock,
            },
        )


class TDCOracleEvaluator:
    def __init__(self, args, run_dir, *, scorer=None):
        self.args = args
        self.path = Path(run_dir) / "oracle_cache.json"
        self.scorer = scorer
        self.before_oracle = None
        if self.path.exists():
            data = json.loads(self.path.read_text())
            if data["oracle"] != args.oracle or data["mock"] != args.mock:
                raise ValueError("oracle cache scientific identity mismatch")
            self.entries = data["entries"]
        else:
            self.entries = []

    def _candidate_smiles(self, candidate):
        return canonicalize(candidate.payload["smiles"], mock=self.args.mock)

    def evaluation_attempt_usage_key(self, candidate):
        smi = self._candidate_smiles(candidate)
        return f"reasyn_tdc_oracle:{self.args.oracle}:{int(self.args.mock)}:{smi}"

    def _save(self):
        atomic_json_write(
            self.path,
            {
                "oracle": self.args.oracle,
                "mock": self.args.mock,
                "entries": self.entries,
            },
        )

    def evaluate(self, candidate):
        smi = self._candidate_smiles(candidate)
        found = next((r for r in self.entries if r["smiles"] == smi), None)
        if found:
            if found["status"] != "completed":
                raise RuntimeError(
                    "oracle call previously interrupted; refusing to silently repeat an ambiguous charged call"
                )
            score = found["score"]
        else:
            if len(self.entries) >= self.args.max_oracle_calls:
                return EvaluationResult(
                    candidate.candidate_id,
                    "invalid",
                    error="unique canonical oracle budget exhausted",
                )
            if self.before_oracle:
                self.before_oracle()
            found = {
                "smiles": smi,
                "call_index": len(self.entries) + 1,
                "status": "started",
            }
            self.entries.append(found)
            self._save()
            if self.scorer is None:
                if self.args.mock:
                    self.scorer = lambda s: (
                        int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
                        / (2**32 - 1)
                    )
                else:
                    from tdc import Oracle

                    self.scorer = Oracle(name=self.args.oracle)
            score = float(self.scorer(smi))
            if not math.isfinite(score):
                raise ValueError("TDC oracle returned nonfinite score")
            found.update(status="completed", score=score)
            self._save()
        return EvaluationResult(
            candidate.candidate_id,
            "succeeded",
            {"oracle_score": score},
            artifacts={
                "synthesis": candidate.payload["projection_artifact"],
                "oracle_cache": "oracle_cache.json",
            },
            resource_usage={"benchmark_jobs": 1},
            metadata={
                "oracle_call_index": found["call_index"],
                EVALUATION_ATTEMPT_RECEIPT_KEY: self.evaluation_attempt_usage_key(candidate),
                "mock": self.args.mock,
            },
        )
