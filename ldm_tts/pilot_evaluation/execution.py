"""Durable matrix execution through existing config-driven task procedures."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ldm_tts.cli.runner import apply_override, build_plan, load_config, preflight_plan, run_plan
from ldm_tts.engine.run_store import atomic_json_write
from ldm_tts.pilot_evaluation.config import PilotEvaluationSpec
from ldm_tts.pilot_evaluation.reporting import write_evaluation_reports


_MANIFEST_NAME = "evaluation_manifest.json"


@dataclass(frozen=True)
class _EvaluationRun:
    case_id: str
    method: str
    seed: int
    run_dir: Path

    @property
    def key(self) -> str:
        return f"{self.case_id}/{self.method}/seed_{self.seed}"


def run_evaluation(
    spec: PilotEvaluationSpec,
    *,
    resume: bool,
    dry_run: bool,
    cases: Iterable[str] | None = None,
    methods: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
) -> int:
    """Run a fresh or provenance-checked resumed evaluation matrix."""

    base = load_config(spec.base_config)
    if base.get("task") != spec.task:
        raise ValueError(
            "pilot evaluation task must match base config task: "
            f"{spec.task!r} != {base.get('task')!r}"
        )
    selected = _select_runs(spec, cases=cases, methods=methods, seeds=seeds)
    manifest = _open_manifest(spec, base, resume=resume, dry_run=dry_run)
    plans = [_child_plan(spec, base, item, resume=resume) for item in selected]
    if dry_run:
        print(json.dumps({"manifest": manifest, "plans": plans}, indent=2, sort_keys=True))
        return 0
    for item, plan in zip(selected, plans, strict=True):
        _run_child(manifest, spec, item, plan, resume=resume)
    if _matrix_complete(spec, manifest):
        try:
            write_evaluation_reports(spec, manifest)
        except Exception as error:
            manifest["state"] = "failed"
            manifest["error"] = {
                "stage": "reporting",
                "type": type(error).__name__,
                "message": str(error),
            }
            _write_manifest(spec, manifest)
            raise
    else:
        manifest["state"] = "partial"
        _write_manifest(spec, manifest)
    return 0


def _select_runs(spec, *, cases, methods, seeds) -> tuple[_EvaluationRun, ...]:
    case_ids = _selected_values(cases, (item.case_id for item in spec.cases), "case")
    method_ids = _selected_values(methods, spec.methods, "method")
    seed_ids = _selected_values(seeds, spec.seeds, "seed")
    runs = []
    for case in spec.cases:
        for method in spec.methods:
            for seed in spec.seeds:
                if case.case_id in case_ids and method in method_ids and seed in seed_ids:
                    run_dir = spec.output_root / "campaigns" / case.case_id / method / f"seed_{seed}"
                    runs.append(_EvaluationRun(case.case_id, method, seed, run_dir))
    return tuple(runs)


def _selected_values(requested, available, label: str) -> set:
    available_set = set(available)
    values = available_set if requested is None else set(requested)
    unknown = sorted(values - available_set)
    if unknown:
        raise ValueError(f"unknown pilot evaluation {label}: {', '.join(map(str, unknown))}")
    return values


def _open_manifest(spec, base, *, resume: bool, dry_run: bool) -> dict[str, Any]:
    path = spec.output_root / _MANIFEST_NAME
    repository = _repository_state()
    fingerprint = _sha256_json(
        {"spec": spec.to_dict(), "base": base, "repository": repository}
    )
    if resume:
        if not path.is_file():
            raise FileNotFoundError(f"pilot evaluation manifest does not exist: {path}")
        manifest = _read_json(path)
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("pilot evaluation provenance differs; refuse --resume")
        return manifest
    if spec.output_root.exists():
        raise FileExistsError(f"pilot evaluation output already exists: {spec.output_root}; use --resume")
    if not dry_run and base.get("mode") == "real" and repository["dirty"]:
        raise RuntimeError("real pilot evaluation requires a clean repository")
    manifest = {
        "schema_version": 1,
        "name": spec.name,
        "task": spec.task,
        "state": "planned" if dry_run else "running",
        "created_at_unix": time.time(),
        "fingerprint": fingerprint,
        "spec": spec.to_dict(),
        "repository": repository,
        "base_config_sha256": _sha256_json(base),
        "runs": {},
    }
    if not dry_run:
        spec.output_root.mkdir(parents=True)
        atomic_json_write(path, manifest)
    return manifest


def _child_plan(spec, base, run: _EvaluationRun, *, resume: bool) -> dict[str, Any]:
    config = copy.deepcopy(base)
    case = next(item for item in spec.cases if item.case_id == run.case_id)
    for override in case.overrides:
        apply_override(config, override)
    for override in spec.method_overrides[run.method]:
        apply_override(config, override)
    _set(config, "args.search-method", "ldm" if run.method == "harness" else run.method)
    _set(config, "args.proposal-mode", _proposal_mode(config, run.method))
    _set(config, "args.initialization-mode", spec.initialization_mode)
    _set(config, "args.iterations", spec.iterations)
    _set(config, "args.campaign-index", run.seed)
    _set(config, "args.out-dir", str(run.run_dir.parent))
    _set(config, "args.run-name", run.run_dir.name)
    if resume and run.run_dir.exists() and not _child_complete(run.run_dir):
        _set(config, "args.resume-from", str(run.run_dir))
    return build_plan(config, spec.base_config)


def _run_child(manifest, spec, run, plan, *, resume: bool) -> None:
    entry = manifest["runs"].get(run.key, {})
    if _child_complete(run.run_dir):
        entry.update({"status": "completed", "run_dir": str(run.run_dir.relative_to(spec.output_root))})
        manifest["runs"][run.key] = entry
        _write_manifest(spec, manifest)
        return
    if run.run_dir.exists() and not resume:
        raise FileExistsError(f"child run already exists: {run.run_dir}; use --resume")
    entry.update({
        "status": "running",
        "run_dir": str(run.run_dir.relative_to(spec.output_root)),
        "command": plan["command_display"],
        "contract_sha256": plan["contract_sha256"],
        "contract_profile": plan["contract_profile"],
    })
    manifest["runs"][run.key] = entry
    _write_manifest(spec, manifest)
    preflight_plan(plan)
    return_code = run_plan(plan)
    entry["status"] = "completed" if return_code == 0 and _child_complete(run.run_dir) else "failed"
    entry["return_code"] = return_code
    entry["updated_at_unix"] = time.time()
    _write_manifest(spec, manifest)
    if entry["status"] != "completed":
        manifest["state"] = "failed"
        _write_manifest(spec, manifest)
        raise RuntimeError(f"pilot evaluation child failed: {run.key}")


def _matrix_complete(spec: PilotEvaluationSpec, manifest: dict[str, Any]) -> bool:
    expected = {
        f"{case.case_id}/{method}/seed_{seed}"
        for case in spec.cases
        for method in spec.methods
        for seed in spec.seeds
    }
    return set(manifest["runs"]) == expected and all(
        item.get("status") == "completed" for item in manifest["runs"].values()
    )


def _proposal_mode(config: dict[str, Any], method: str) -> str:
    if method in {"bo", "harness"}:
        return "none"
    return "callable" if config.get("mode") == "mock" else "openai"


def _set(config: dict[str, Any], override: str, value: Any) -> None:
    apply_override(config, f"{override}={json.dumps(value)}")


def _child_complete(path: Path) -> bool:
    if not (path / "result.json").is_file() or not (path / "trajectory.csv").is_file():
        return False
    status = _read_json(path / "status.json")
    return status.get("status") == "completed"


def _repository_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = _git(root, "rev-parse", "HEAD")
        dirty = bool(_git(root, "status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError) as error:
        commit = os.environ.get("LDM_PILOT_EVALUATION_COMMIT", "").strip()
        if not commit:
            raise RuntimeError(
                "pilot evaluation requires Git metadata or an explicit "
                "LDM_PILOT_EVALUATION_COMMIT for an archived release"
            ) from error
        return {"commit": commit, "dirty": False, "source": "environment"}
    return {"commit": commit, "dirty": dirty, "source": "git"}


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_manifest(spec: PilotEvaluationSpec, manifest: dict[str, Any]) -> None:
    atomic_json_write(spec.output_root / _MANIFEST_NAME, manifest)


__all__ = ["run_evaluation"]
