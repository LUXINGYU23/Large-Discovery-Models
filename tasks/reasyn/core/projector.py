"""Bounded official projector process and deterministic infrastructure fixture."""

from __future__ import annotations
import json
import subprocess
from pathlib import Path
from ldm_tts.engine.run_store import atomic_json_write
from .projection_worker import load_progress, request_digest, write_progress


class ProjectionInterruptedError(RuntimeError):
    """A resumable projection failure; completed targets remain checkpointed."""

    def __init__(self, message, *, completed_targets, target_count, artifact):
        super().__init__(message)
        self.completed_targets = completed_targets
        self.target_count = target_count
        self.artifact = artifact


class Projector:
    def __init__(self, args, run_dir):
        self.args = args
        self.run_dir = Path(run_dir)
        self.before_project = None

    def _request(self, targets, sampling_seed):
        return {
            "asset_digests": getattr(self.args, "asset_digests", {}),
            "targets": targets,
            "sampling_seed": sampling_seed,
            "model_paths": [str(p.resolve()) for p in self.args.model_paths],
            "fpindex": str(self.args.fpindex.resolve()),
            "rxn_matrix": str(self.args.rxn_matrix.resolve()),
            "additional_fpindex": str(self.args.additional_fpindex.resolve())
            if self.args.additional_fpindex
            else None,
            "device": self.args.device,
            "settings": {
                "search_width": self.args.search_width,
                "exhaustiveness": self.args.exhaustiveness,
                "num_cycles": self.args.num_cycles,
                "num_editflow_samples": self.args.num_editflow_samples,
                "max_results": self.args.max_results,
                "exact_break": True,
                "time_limit": self.args.projection_time_limit,
            },
        }

    def completed_receipt(self, targets, *, sampling_seed, identity):
        """Identify verified completed work without starting a projection."""
        request = self._request(targets, sampling_seed)
        folder = self.run_dir / "projections" / identity
        request_path = folder / "request.json"
        if not request_path.exists() or json.loads(request_path.read_text()) != request:
            raise ValueError("projection receipt request mismatch")
        progress = load_progress(folder / "result.json", request)
        if progress is None or not progress["complete"]:
            raise ValueError("projection receipt requires a completed result")
        self._validate(progress["rows"], targets)
        return {"request_sha256": request_digest(request), "result_sha256": request_digest(progress)}

    def project(self, targets, *, sampling_seed, identity):
        folder = self.run_dir / "projections" / identity
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / "result.json"
        request = self._request(targets, sampling_seed)
        request_path = folder / "request.json"
        if request_path.exists():
            prior = json.loads(request_path.read_text())
            if prior != request:
                raise ValueError("projection resume request mismatch")
        elif output.exists():
            raise ValueError("projection result has no matching request")
        progress = load_progress(output, request)
        artifact = str(output.relative_to(self.run_dir))
        if progress is not None:
            self._validate(progress["rows"], targets)
            if progress["complete"]:
                return progress["rows"], artifact
        completed_indices = (
            set(progress["completed_target_indices"]) if progress is not None else set()
        )
        pending_count = len(targets) - len(completed_indices)
        if self.before_project:
            self.before_project(pending_count)
        atomic_json_write(request_path, request)
        if self.args.mock or not targets:
            rows = list(progress["rows"]) if progress is not None else []
            for index, target in enumerate(targets):
                if index in completed_indices:
                    continue
                rows.append(
                    {
                        "target": target,
                        "target_index": index,
                        "sampling_seed": sampling_seed + index,
                        "smiles": target,
                        "synthesis": target,
                        "num_steps": 0,
                        "pathway_verified": True,
                        "mock_fixture": True,
                    }
                )
                completed_indices.add(index)
                write_progress(
                    output, request, rows, completed_indices, mock=self.args.mock
                )
            if not targets:
                write_progress(
                    output, request, rows, completed_indices, mock=self.args.mock
                )
        else:
            command = [
                self.args.evaluator_python,
                str(Path(__file__).with_name("projection_worker.py")),
                "--request",
                str(request_path.resolve()),
                "--output",
                str(output.resolve()),
            ]
            # projection_timeout retains the single-target startup/teardown
            # allowance. Each additional serial target receives its full budget.
            timeout = max(
                self.args.projection_timeout, self.args.projection_time_limit
            ) + (pending_count - 1) * self.args.projection_time_limit
            try:
                with (folder / "worker.log").open("a") as log:
                    completed = subprocess.run(
                        command,
                        cwd=self.args.upstream_root,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=timeout,
                        check=False,
                    )
            except subprocess.TimeoutExpired as exc:
                raise self._interrupted(
                    f"ReaSyn projector timed out after {timeout} seconds",
                    output,
                    request,
                    artifact,
                ) from exc
            if completed.returncode:
                raise self._interrupted(
                    f"ReaSyn projector exited {completed.returncode}",
                    output,
                    request,
                    artifact,
                )
            if not output.exists():
                raise self._interrupted(
                    "ReaSyn projector omitted result.json", output, request, artifact
                )
        data = load_progress(output, request)
        if not data["complete"]:
            raise self._interrupted(
                "ReaSyn projector stopped before completing all targets",
                output,
                request,
                artifact,
            )
        rows = data["rows"]
        self._validate(rows, targets)
        return rows, artifact

    def _interrupted(self, message, output, request, artifact):
        progress = load_progress(output, request)
        if progress is not None:
            self._validate(progress["rows"], request["targets"])
        completed = (
            len(progress["completed_target_indices"])
            if progress is not None
            else 0
        )
        return ProjectionInterruptedError(
            f"{message}; {completed}/{len(request['targets'])} targets checkpointed; "
            f"resume to continue; see {output.parent / 'worker.log'}",
            completed_targets=completed,
            target_count=len(request["targets"]),
            artifact=artifact,
        )

    def _validate(self, rows, targets):
        from .chemistry import canonicalize

        for row in rows:
            if (
                row.get("pathway_verified") is not True
                or row["target"] not in targets
                or not row.get("synthesis")
            ):
                raise ValueError("projector output failed trusted-pathway boundary")
            canonicalize(row["smiles"], mock=self.args.mock)
