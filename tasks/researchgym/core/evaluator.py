"""Official-grader evaluation of one candidate program, with durable receipts."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

from ldm_tts.contracts import Candidate, EvaluationResult
from ldm_tts.contracts.evaluation import EVALUATION_ATTEMPT_RECEIPT_KEY
from ldm_tts.engine.run_store import atomic_json_write

from .cases import CaseSpec, render
from .clock import CampaignClock, WallClockExhausted
from .execution import JobCommand, ProcessTreeRunner, free_devices
from .grading import CoverageError, score_summary
from .source import UpstreamCase, modified_files

SLOT = "ldm"


class DevicesUnavailable(RuntimeError):
    """No configured device currently has the required free memory."""


class _ReceiptMixin:
    run_dir: Path

    def evaluation_attempt_usage_key(self, candidate: Candidate) -> str:
        return f"researchgym:{candidate.candidate_id}"

    def _root(self, candidate: Candidate) -> Path:
        return self.run_dir / "evaluations" / candidate.candidate_id

    def _cached(self, candidate: Candidate) -> EvaluationResult | None:
        root = self._root(candidate)
        if (root / "result.json").exists():
            return EvaluationResult(**json.loads((root / "result.json").read_text()))
        if (root / "started.json").exists():
            # A previous process launched this paid attempt and died before a
            # terminal record. Re-running would repeat an expensive evaluation.
            return self._finish(candidate, EvaluationResult(
                candidate.candidate_id, "failed",
                error="evaluation interrupted before a terminal result; not re-run on resume",
                metadata={"interrupted": True}))
        return None

    def _finish(self, candidate: Candidate, result: EvaluationResult) -> EvaluationResult:
        receipt = self.evaluation_attempt_usage_key(candidate)
        result = EvaluationResult(
            result.candidate_id, result.status, result.metrics, result.artifacts, result.resource_usage,
            result.error, {**result.metadata, EVALUATION_ATTEMPT_RECEIPT_KEY: receipt},
        )
        atomic_json_write(self._root(candidate) / "result.json", result.to_dict())
        return result


class ResearchGymEvaluator(_ReceiptMixin):
    """Pinned workspace -> case jobs on explicit devices -> official grader -> coverage."""

    def __init__(self, *, case: CaseSpec, upstream: UpstreamCase, run_dir: Path, python: str,
                 devices: tuple[str, ...], min_free_mib: int, evaluation_timeout: float,
                 grader_timeout: float, clock: CampaignClock, port_base: int = 29600,
                 grace_seconds: float = 15.0, base_env: dict[str, str] | None = None) -> None:
        self.case, self.upstream, self.run_dir = case, upstream, Path(run_dir)
        self.python, self.devices, self.min_free_mib = python, tuple(devices), min_free_mib
        self.evaluation_timeout, self.grader_timeout = evaluation_timeout, grader_timeout
        self.clock, self.port_base = clock, port_base
        self.grace_seconds, self.base_env = grace_seconds, base_env
        self.active_devices = self.devices

    def prepare_evaluations(self, candidates) -> None:
        if self.clock.remaining() <= 0:
            raise WallClockExhausted("campaign wall-clock allowance is exhausted before evaluation")
        self.active_devices = free_devices(self.devices, self.min_free_mib)
        if self.devices and not self.active_devices:
            raise DevicesUnavailable(
                f"no device in {list(self.devices)} has {self.min_free_mib} MiB free; resume when devices are free")

    def evaluate(self, candidate: Candidate) -> EvaluationResult:
        cached = self._cached(candidate)
        if cached is not None:
            return cached
        root = self._root(candidate)
        root.mkdir(parents=True, exist_ok=True)
        allowance = min(self.evaluation_timeout, self.clock.remaining())
        atomic_json_write(root / "started.json", {
            "candidate_id": candidate.candidate_id, "started_at_unix": time.time(),
            "devices": list(self.active_devices), "allowance_seconds": allowance,
        })
        started = time.monotonic()
        workspace = root / "workspace"
        if workspace.exists():
            shutil.rmtree(workspace)
        build = self.upstream.build_workspace(workspace, slot=SLOT, program=candidate.payload["program"])
        jobs = self.case.jobs(SLOT)
        commands = []
        for job in jobs:
            binding = {**job["binding"], "python": self.python, "port": self.port_base + job["index"]}
            for generated in job["generated_files"]:
                target = workspace / generated["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(generated["json"], indent=2))
                build["file_digests"][generated["path"]] = hashlib.sha256(target.read_bytes()).hexdigest()
            commands.append(JobCommand(job["job_id"], tuple(render(self.case.raw["jobs"]["argv"], binding)),
                                       dict(job["env"])))
        runner = ProcessTreeRunner(devices=self.active_devices, grace_seconds=self.grace_seconds,
                                   base_env=self.base_env)
        outcomes = runner.run(commands, cwd=workspace, log_dir=root / "job_logs",
                              deadline=started + allowance, tick=self.clock.heartbeat)
        usage = {"benchmark_jobs": float(len(commands)), "runtime_seconds": time.monotonic() - started}
        job_records = [outcome.to_dict() for outcome in outcomes]
        for record, job in zip(job_records, jobs, strict=True):
            record["missing_outputs"] = [p for p in job["expected_outputs"] if not (workspace / p).is_file()
                                         or (workspace / p).stat().st_size == 0]
            record["log"] = _relative(record["log"], self.run_dir)
        atomic_json_write(root / "jobs.json", {"build": {k: v for k, v in build.items() if k != "file_digests"},
                                                "jobs": job_records})
        metadata = self._metadata(build)
        artifacts = {"jobs": _relative(root / "jobs.json", self.run_dir)}
        if not all(outcome.group_terminated for outcome in outcomes):
            metadata["orphaned_process_groups"] = [o.job_id for o in outcomes if not o.group_terminated]
        failed = [r for r in job_records if r["status"] != "succeeded" or r["missing_outputs"]]
        if failed:
            timed_out = any(r["status"] in {"timed_out", "not_started"} for r in failed)
            return self._finish(candidate, EvaluationResult(
                candidate.candidate_id, "timed_out" if timed_out else "failed",
                error=f"{len(failed)}/{len(job_records)} case jobs failed or produced no output: "
                      + ", ".join(f"{r['job_id']}={r['status']}" for r in failed[:6]),
                artifacts=artifacts, resource_usage=usage, metadata=metadata))
        # Evaluator, grader, generated configs, and the candidate itself must be
        # byte-identical to what was installed before the official grader reads outputs.
        tampered = modified_files(workspace, build["file_digests"])
        if tampered:
            return self._finish(candidate, EvaluationResult(
                candidate.candidate_id, "invalid",
                error=f"{len(tampered)} evaluator-owned workspace files changed during the run: {tampered[:5]}",
                artifacts=artifacts, resource_usage=usage, metadata={**metadata, "tampered_files": tampered[:50]}))
        summary_path = root / "grade_summary.json"
        grader_binding = {"python": self.python, "slot": SLOT, "summary": str(summary_path),
                          "scratch_markdown": str(root / "grader_description.md")}
        source = self.case.raw["grader"].get("scratch_markdown_source")
        if source:
            shutil.copy(workspace / source, root / "grader_description.md")
        remaining = self.clock.remaining()
        grade = ProcessTreeRunner(max_parallel=1, grace_seconds=self.grace_seconds, base_env=self.base_env).run(
            [JobCommand("grader", tuple(render(self.case.raw["grader"]["argv"], grader_binding)))],
            cwd=workspace, log_dir=root, deadline=time.monotonic() + min(self.grader_timeout, remaining),
            tick=self.clock.heartbeat)[0]
        usage["runtime_seconds"] = time.monotonic() - started
        artifacts["grader_log"] = _relative(grade.log, self.run_dir)
        if grade.status != "succeeded" or not summary_path.is_file():
            return self._finish(candidate, EvaluationResult(
                candidate.candidate_id, "timed_out" if grade.status == "timed_out" else "failed",
                error=f"official grader {grade.status} (exit {grade.returncode})",
                artifacts=artifacts, resource_usage=usage, metadata=metadata))
        artifacts["grade_summary"] = _relative(summary_path, self.run_dir)
        try:
            metrics = score_summary(self.case, json.loads(summary_path.read_text()))
        except (CoverageError, ValueError, TypeError, KeyError) as exc:
            problems = getattr(exc, "problems", [str(exc)])
            return self._finish(candidate, EvaluationResult(
                candidate.candidate_id, "invalid", error=str(exc), artifacts=artifacts, resource_usage=usage,
                metadata={**metadata, "coverage_problems": problems}))
        return self._finish(candidate, EvaluationResult(
            candidate.candidate_id, "succeeded", metrics=metrics, artifacts=artifacts,
            resource_usage=usage, metadata=metadata))

    def _metadata(self, build: dict[str, Any]) -> dict[str, Any]:
        return {"evaluator": "researchgym_official_grader", **self.case.protocol,
                "source_commit": self.upstream.commit, "source_tree_sha256": self.upstream.pinned["tree_sha256"],
                "candidate_sha256": build["candidate_sha256"], "devices": list(self.active_devices)}


class MockProgramEvaluator(_ReceiptMixin):
    """Deterministic analytic score over public program features; not a benchmark result."""

    def __init__(self, *, case: CaseSpec, run_dir: Path, encoder, clock: CampaignClock) -> None:
        self.case, self.run_dir, self.encoder, self.clock = case, Path(run_dir), encoder, clock

    def prepare_evaluations(self, candidates) -> None:
        if self.clock.remaining() <= 0:
            raise WallClockExhausted("campaign wall-clock allowance is exhausted before evaluation")

    def evaluate(self, candidate: Candidate) -> EvaluationResult:
        cached = self._cached(candidate)
        if cached is not None:
            return cached
        features = self.encoder.encode(candidate).values
        target = (0.3, 0.3, 0.2, 0.3, 0.4, 0.2, 0.5, 0.4, 0.3, 0.3, 0.2, 0.3)
        distance = math.sqrt(sum((f - t) ** 2 for f, t in zip(features, target)))
        return self._finish(candidate, EvaluationResult(
            candidate.candidate_id, "succeeded", metrics={"mock_score": 1.0 / (1.0 + distance)},
            resource_usage={"benchmark_jobs": 1.0},
            metadata={"evaluator": "analytic_mock", "protocol_id": "mock_analytic_v1",
                      "official_grader": False, "official_protocol": False, "synthetic_fixture": True}))


def _relative(path, root: Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)
