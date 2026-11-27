"""Stage-B loaders and legalisers for offline two-stage baselines (M4, M7, M8-RF).

Stage-A scripts run in dedicated H20 Python environments (env/gempy for M4,
env/embedding for M7 and M8-RF) on public inputs only and save raw fields plus
choice.json metadata. Stage-B runs in the service environment, computes signed
distance (for M4) or loads continuous fields (for M7/RF), applies the selected
or native decision rule, and enforces contract legality via _enforce_legal.
"""

import json
import time
from pathlib import Path
import numpy as np
import skfmm
from . import baseline_harness as harness
from .data import read_problem

OFFLINE_SOLVER_NAMES = ("potential_field", "inr_mlp", "rf_xyz", "wei2024")
OFFLINE_NATIVE_SOLVER_NAMES = ("potential_field_native", "inr_mlp_native", "rf_xyz_native")
OFFLINE_DEFAULT_SOLVER_NAMES = (
    "potential_field_default",
    "inr_mlp_default",
    "rf_xyz_default",
    "wei2024_default",
)


def resolve_stage_a_dir(fields_dir, case_id, *, seed=0, split=None):
    """Locate Stage-A output directory for a (case_id, seed) pair."""
    if fields_dir is None:
        raise ValueError("offline solver requires fields_dir (--fields-dir)")
    base = Path(fields_dir)
    candidates = []
    if split is not None:
        candidates.extend([
            base / split / case_id / f"seed_{seed}",
            base / split / case_id / str(seed),
            base / split / case_id,
        ])
    candidates.extend([
        base / case_id / f"seed_{seed}",
        base / case_id / str(seed),
        base / case_id,
    ])
    for cand in candidates:
        if (cand / "choice.json").is_file():
            return cand
    raise FileNotFoundError(
        f"Stage-A output choice.json not found for case_id={case_id!r}, seed={seed} under {base}"
    )


def _signed_distance_on_feasible(ore_mask, data):
    ore = np.asarray(ore_mask, dtype=bool)
    if ore.shape != data["shape"]:
        raise ValueError(f"precomputed ore.npy shape {ore.shape} != expected {data['shape']}")
    if ore.any() and (~ore).any():
        phi = np.where(ore, 1.0, -1.0)
        dist = np.asarray(skfmm.distance(phi, dx=data["spacing"]), dtype=float)
        return dist[data["feasible"]]
    inward = 1e-4 * np.linalg.norm(data["G"] - data["center"], axis=1)
    if ore.any():
        return np.ones(data["M"], dtype=float) - inward
    return -np.ones(data["M"], dtype=float) - inward


def _load_continuous_on_feasible(field_arr, data):
    arr = np.asarray(field_arr, dtype=float)
    if arr.shape == data["shape"]:
        arr = arr[data["feasible"]]
    else:
        arr = arr.reshape(-1)
    if len(arr) != data["M"] or not np.isfinite(arr).all():
        raise ValueError(f"invalid precomputed field shape {arr.shape} for M={data['M']}")
    return arr


def _solve_offline(method_name, tau0, is_mask, problem, budget=None, seed=0, *,
                   fields_dir=None, split=None, decision_rule=None, diagnostics=None):
    del budget
    t0 = time.monotonic()
    data = harness.extract_public_data(problem)
    if fields_dir is None:
        raise ValueError(
            f"{method_name} is an offline two-stage baseline; supply fields_dir (--fields-dir)"
        )
    case_dir = resolve_stage_a_dir(fields_dir, problem["case_id"], seed=int(seed), split=split)
    choice = json.loads((case_dir / "choice.json").read_text())
    stage_a_wall = float(choice.get("wall_seconds", 0.0))

    if choice.get("no_ore") or not (data["y_b"] > 0.5).any():
        out = harness.no_ore_fallback(problem, data, diagnostics=diagnostics, started=t0)
        if diagnostics is not None:
            diagnostics["wall_seconds"] = stage_a_wall + float(time.monotonic() - t0)
        return out

    force_default = decision_rule == "default"
    force_native = decision_rule in ("native", "default")
    if force_default and "default" in choice:
        block = choice["default"]
    elif force_native and "native" in choice:
        block = choice["native"]
    else:
        block = choice.get("cv", choice)
    if block.get("fallback") or choice.get("fallback"):
        out = harness.stretched_fallback(
            problem, data, diagnostics=diagnostics, started=t0,
            cv_evaluated=int(block.get("cv_evaluated", choice.get("cv_evaluated", 0))),
            deadline_stopped=bool(choice.get("deadline_stopped", False)),
            default_mode=force_default,
        )
        if diagnostics is not None:
            diagnostics["wall_seconds"] = stage_a_wall + float(time.monotonic() - t0)
        return out

    rule = "native" if force_native else str(block.get("selected_rule", "share"))
    if decision_rule == "share":
        rule = "share"
    use_native_file = (decision_rule == "native") and not bool(block.get("same_as_cv", True))
    if is_mask:
        fname = "ore_native.npy" if use_native_file and (case_dir / "ore_native.npy").is_file() else "ore.npy"
        raw = np.load(case_dir / fname, allow_pickle=False)
        f = _signed_distance_on_feasible(raw, data)
    else:
        fname = "field_native.npy" if use_native_file and (case_dir / "field_native.npy").is_file() else "field.npy"
        raw = np.load(case_dir / fname, allow_pickle=False)
        f = _load_continuous_on_feasible(raw, data)

    cfg = block.get("selected_config")
    out = harness.finalize_field(
        problem,
        data,
        f,
        tau0,
        rule,
        selected_config=cfg,
        cv_score=block.get("cv_score"),
        cv_evaluated=int(block.get("cv_evaluated", choice.get("cv_evaluated", 0))),
        deadline_stopped=bool(choice.get("deadline_stopped", False)),
        diagnostics=diagnostics,
        started=t0,
    )
    if diagnostics is not None:
        diagnostics["wall_seconds"] = stage_a_wall + float(time.monotonic() - t0)
        if method_name == "potential_field":
            diagnostics["generator_matched"] = True
    return out


def potential_field(problem, budget=None, seed=0, *, fields_dir=None, split=None,
                    decision_rule=None, diagnostics=None):
    """M4 GemPy potential-field co-kriging Stage-B solver (offline, generator-matched)."""
    return _solve_offline(
        "potential_field", 0.0, True, problem, budget, seed,
        fields_dir=fields_dir, split=split, decision_rule=decision_rule, diagnostics=diagnostics,
    )


def potential_field_native(problem, budget=None, seed=0, *, fields_dir=None, split=None, diagnostics=None):
    """M4 potential_field with decision rule forced to native (generator-matched)."""
    return potential_field(
        problem, budget, seed, fields_dir=fields_dir, split=split,
        decision_rule="native", diagnostics=diagnostics,
    )


def inr_mlp(problem, budget=None, seed=0, *, fields_dir=None, split=None,
            decision_rule=None, diagnostics=None):
    """M7 GeoINR-style Softplus MLP Stage-B solver (offline)."""
    return _solve_offline(
        "inr_mlp", 0.0, False, problem, budget, seed,
        fields_dir=fields_dir, split=split, decision_rule=decision_rule, diagnostics=diagnostics,
    )


def inr_mlp_native(problem, budget=None, seed=0, *, fields_dir=None, split=None, diagnostics=None):
    """M7 inr_mlp with decision rule forced to native."""
    return inr_mlp(
        problem, budget, seed, fields_dir=fields_dir, split=split,
        decision_rule="native", diagnostics=diagnostics,
    )


def rf_xyz(problem, budget=None, seed=0, *, fields_dir=None, split=None,
           decision_rule=None, diagnostics=None):
    """M8 Random-forest coordinate classifier Stage-B solver (offline)."""
    return _solve_offline(
        "rf_xyz", 0.5, False, problem, budget, seed,
        fields_dir=fields_dir, split=split, decision_rule=decision_rule, diagnostics=diagnostics,
    )


def rf_xyz_native(problem, budget=None, seed=0, *, fields_dir=None, split=None, diagnostics=None):
    """M8 rf_xyz with decision rule forced to native."""
    return rf_xyz(
        problem, budget, seed, fields_dir=fields_dir, split=split,
        decision_rule="native", diagnostics=diagnostics,
    )


def wei2024(problem, budget=None, seed=0, *, fields_dir=None, split=None,
            decision_rule=None, diagnostics=None):
    """Wei et al. (2024) 3-D level-set MCMC Stage-B loader (posterior mean >= 0.5)."""
    del decision_rule
    return _solve_offline(
        "wei2024", 0.5 - 1e-9, False, problem, budget, seed,
        fields_dir=fields_dir, split=split, decision_rule="native", diagnostics=diagnostics,
    )


def potential_field_default(problem, budget=None, seed=0, *, fields_dir=None, split=None, diagnostics=None):
    """M4 GemPy potential_field with fixed library defaults (form='L', orient='plane', range=1.7), native rule."""
    return potential_field(
        problem, budget, seed, fields_dir=fields_dir, split=split,
        decision_rule="default", diagnostics=diagnostics,
    )


def inr_mlp_default(problem, budget=None, seed=0, *, fields_dir=None, split=None, diagnostics=None):
    """M7 GeoINR-style MLP with fixed published defaults (3x256 Softplus, lam=0.1, s_z=4), native phi > 0."""
    return inr_mlp(
        problem, budget, seed, fields_dir=fields_dir, split=split,
        decision_rule="default", diagnostics=diagnostics,
    )


def rf_xyz_default(problem, budget=None, seed=0, *, fields_dir=None, split=None, diagnostics=None):
    """M8 Random Forest with sklearn defaults (100 trees, min_samples_leaf=1, max_features='sqrt', s_z=4), p > 0.5."""
    return rf_xyz(
        problem, budget, seed, fields_dir=fields_dir, split=split,
        decision_rule="default", diagnostics=diagnostics,
    )


def wei2024_default(problem, budget=None, seed=0, *, fields_dir=None, split=None, diagnostics=None):
    """Wei et al. (2024) 3-D level-set MCMC with upstream defaults (1,000 steps, mean >= 0.5)."""
    return _solve_offline(
        "wei2024", 0.5 - 1e-9, False, problem, budget, seed,
        fields_dir=fields_dir, split=split, decision_rule="default", diagnostics=diagnostics,
    )


OFFLINE_REFERENCE_SOLVERS = {
    "potential_field": potential_field,
    "potential_field_native": potential_field_native,
    "potential_field_default": potential_field_default,
    "inr_mlp": inr_mlp,
    "inr_mlp_native": inr_mlp_native,
    "inr_mlp_default": inr_mlp_default,
    "rf_xyz": rf_xyz,
    "rf_xyz_native": rf_xyz_native,
    "rf_xyz_default": rf_xyz_default,
    "wei2024": wei2024,
    "wei2024_default": wei2024_default,
}


def materialize_precomputed_reference(solver_name, stage_a_dir, release_dir, output_dir,
                                      *, splits=("train", "validation", "test_id", "test_ood"),
                                      tier=None, source_seed=None):
    """Materialize Stage-B legal model.npy files on public inputs only and return precomputed_manifest sha256."""
    from ldm_tts.engine.run_store import atomic_json_write
    from .selection import precomputed_manifest
    if solver_name not in OFFLINE_REFERENCE_SOLVERS:
        raise ValueError(f"unknown offline solver: {solver_name!r}")
    solver = OFFLINE_REFERENCE_SOLVERS[solver_name]
    stage_a_dir = Path(stage_a_dir).resolve()
    release_dir = Path(release_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if tier is not None:
        rec_path = next((release_dir / n for n in ("qualification.json", "development-receipt.json")
                         if (release_dir / n).is_file()), None)
        if rec_path is None:
            raise ValueError(f"release record missing in {release_dir}")
        row = json.loads(rec_path.read_text())["tiers"][tier]
        budget_s = float(row["solver_seconds"])
        rel_manifests = [row["train_manifest"], row["validation_manifest"], *row["sealed_manifests"]]
        plan = [(release_dir / rel, Path(rel).parent.as_posix(), budget_s) for rel in rel_manifests]
    else:
        plan = [(release_dir / split / "manifest.json", split, 120.0) for split in splits]

    for manifest_path, stage_a_sub, budget_s in plan:
        manifest = json.loads(manifest_path.read_text())
        target_split = manifest["split"]
        use_split = stage_a_sub if (stage_a_dir / stage_a_sub).is_dir() else target_split
        seeds = manifest["seeds"]
        for case in manifest["cases"]:
            # Strictly read ONLY public problem files; never access case["truth"].
            problem = read_problem(manifest_path.parent / case["public"])
            for seed in seeds:
                lookup_seed = int(seed) if source_seed is None else int(source_seed)
                diag = {}
                field = solver(
                    problem, {"wall_time_seconds": budget_s}, lookup_seed,
                    fields_dir=stage_a_dir, split=use_split, diagnostics=diag,
                )
                dst = output_dir / target_split / case["case_id"] / str(seed)
                dst.mkdir(parents=True, exist_ok=True)
                np.save(dst / "model.npy", np.asarray(field, dtype=np.float64), allow_pickle=False)
                atomic_json_write(dst / "meta.json", {"wall_seconds": round(float(diag.get("wall_seconds", 0.0)), 6)})

    _, sha = precomputed_manifest(output_dir)
    return sha
