"""Campaign adapter for isolated, auditable real Absolut evaluations."""
from pathlib import Path
import hashlib
import subprocess
import time

from ldm_tts.contracts import EvaluationResult
from tasks.antibody.core.absolut_oracle import AbsolutOracle, TARGETS
from tasks.antibody.core.engine_adapters import _sequence_violations


class IsolatedAbsolutEvaluator:
    def __init__(self, args, run_dir):
        self.target = TARGETS.get(args.antigen, args.antigen)
        self.root = Path(run_dir) / "oracle"
        self.oracle = AbsolutOracle(args.absolut_path,
                                    args.absolut_resources or [args.absolut_path],
                                    self.root, args.absolut_timeout)
        self.oracle.preflight(self.target, require_cache=True)

    def evaluate(self, candidate):
        sequence = candidate.payload["sequence"]
        started = time.monotonic()
        metadata = {"sequence": sequence, "oracle": "Absolut", "antigen": self.target}
        candidate_root = self.root / hashlib.sha256(sequence.encode()).hexdigest()[:16]
        artifacts = {"oracle_root": str(candidate_root)}
        try:
            if _sequence_violations(sequence, 11):
                raise ValueError("Candidate must satisfy the 11 aa H3 admission rules")
            oracle = AbsolutOracle(self.oracle.binary.parent.parent.parent, self.oracle.resources,
                                   candidate_root, self.oracle.timeout)
            result = oracle.evaluate([sequence], self.target)
            directory = Path(result["run_dir"])
            artifacts.update(result=str(directory / "result.json"),
                             raw=str(directory / (self.target + "FinalBindings_Process_1_Of_1.txt")),
                             log=str(directory / "evaluation.log"))
            metadata.update(binary_sha256=result["binary_sha256"], run_dir=str(directory))
            return EvaluationResult(candidate.candidate_id, "succeeded",
                                    {"absolut_energy": result["results"][0]["absolut_energy"]},
                                    artifacts=artifacts, metadata=metadata,
                                    resource_usage={"benchmark_jobs": 1, "wall_time_seconds": time.monotonic() - started})
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            return EvaluationResult(candidate.candidate_id, "failed", error=str(exc),
                                    artifacts=artifacts, metadata=metadata)
