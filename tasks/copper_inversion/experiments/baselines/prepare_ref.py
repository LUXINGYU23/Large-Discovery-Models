"""Prepare precommitted source files, materialized offline fields, and references.json for REF."""

import argparse
import json
from pathlib import Path

from ldm_tts.engine.run_store import atomic_json_write
from tasks.copper_inversion.core.candidate import validate_source
from tasks.copper_inversion.experiments.rsi import q1
from tasks.copper_inversion.rsi.controller import INITIAL_SOURCE
from tasks.copper_inversion.v2 import offline_baselines

IN_CONTAINER_SOURCES = {
    "initial": INITIAL_SOURCE,
    "stretched_nearest_class": (
        b"from tasks.copper_inversion.v2.baselines import stretched_nearest_class\n"
        b"def solve(problem, budget, seed):\n"
        b"    return stretched_nearest_class(problem, budget, seed)\n"
    ),
    "layered_horizon": (
        b"from tasks.copper_inversion.v2.baselines import layered_horizon\n"
        b"def solve(problem, budget, seed):\n"
        b"    return layered_horizon(problem, budget, seed)\n"
    ),
    "ellipsoid_fit": (
        b"from tasks.copper_inversion.v2.baselines import ellipsoid_fit\n"
        b"def solve(problem, budget, seed):\n"
        b"    return ellipsoid_fit(problem, budget, seed)\n"
    ),
    "rbf_sdf": (
        b"from tasks.copper_inversion.v2.lit_baselines import rbf_sdf\n"
        b"def solve(problem, budget, seed):\n"
        b"    return rbf_sdf(problem, budget, seed)\n"
    ),
    "indicator_kriging": (
        b"from tasks.copper_inversion.v2.lit_baselines import indicator_kriging\n"
        b"def solve(problem, budget, seed):\n"
        b"    return indicator_kriging(problem, budget, seed)\n"
    ),
    "gp_classifier": (
        b"from tasks.copper_inversion.v2.lit_baselines import gp_classifier\n"
        b"def solve(problem, budget, seed):\n"
        b"    return gp_classifier(problem, budget, seed)\n"
    ),
    "svm_rbf": (
        b"from tasks.copper_inversion.v2.lit_baselines import svm_rbf\n"
        b"def solve(problem, budget, seed):\n"
        b"    return svm_rbf(problem, budget, seed)\n"
    ),
    "knn_idw": (
        b"from tasks.copper_inversion.v2.lit_baselines import knn_idw\n"
        b"def solve(problem, budget, seed):\n"
        b"    return knn_idw(problem, budget, seed)\n"
    ),
    "fdi_smooth": (
        b"from tasks.copper_inversion.v2.fdi_baseline import fdi_smooth\n"
        b"def solve(problem, budget, seed):\n"
        b"    return fdi_smooth(problem, budget, seed)\n"
    ),
}

IN_CONTAINER_DEFAULT_SOURCES = {
    "initial": INITIAL_SOURCE,
    "stretched_nearest_class": (
        b"from tasks.copper_inversion.v2.baselines import stretched_nearest_class_default\n"
        b"def solve(problem, budget, seed):\n"
        b"    return stretched_nearest_class_default(problem, budget, seed)\n"
    ),
    "layered_horizon": (
        b"from tasks.copper_inversion.v2.baselines import layered_horizon_default\n"
        b"def solve(problem, budget, seed):\n"
        b"    return layered_horizon_default(problem, budget, seed)\n"
    ),
    "ellipsoid_fit": (
        b"from tasks.copper_inversion.v2.baselines import ellipsoid_fit_default\n"
        b"def solve(problem, budget, seed):\n"
        b"    return ellipsoid_fit_default(problem, budget, seed)\n"
    ),
    "rbf_sdf": (
        b"from tasks.copper_inversion.v2.lit_baselines import rbf_sdf_default\n"
        b"def solve(problem, budget, seed):\n"
        b"    return rbf_sdf_default(problem, budget, seed)\n"
    ),
    "indicator_kriging": (
        b"from tasks.copper_inversion.v2.lit_baselines import indicator_kriging_default\n"
        b"def solve(problem, budget, seed):\n"
        b"    return indicator_kriging_default(problem, budget, seed)\n"
    ),
    "gp_classifier": (
        b"from tasks.copper_inversion.v2.lit_baselines import gp_classifier_default\n"
        b"def solve(problem, budget, seed):\n"
        b"    return gp_classifier_default(problem, budget, seed)\n"
    ),
    "svm_rbf": (
        b"from tasks.copper_inversion.v2.lit_baselines import svm_rbf_default\n"
        b"def solve(problem, budget, seed):\n"
        b"    return svm_rbf_default(problem, budget, seed)\n"
    ),
    "knn_idw": (
        b"from tasks.copper_inversion.v2.lit_baselines import knn_idw_default\n"
        b"def solve(problem, budget, seed):\n"
        b"    return knn_idw_default(problem, budget, seed)\n"
    ),
    "fdi_smooth": (
        b"from tasks.copper_inversion.v2.fdi_baseline import fdi_smooth_default\n"
        b"def solve(problem, budget, seed):\n"
        b"    return fdi_smooth_default(problem, budget, seed)\n"
    ),
}

OFFLINE_STAGE_A_SUBDIRS = {
    "potential_field": "gempy",
    "inr_mlp": "inr",
    "rf_xyz": "rf",
    "wei2024": "wei2024",
}

OFFLINE_DEFAULT_SOLVERS = {
    "potential_field": "potential_field_default",
    "inr_mlp": "inr_mlp_default",
    "rf_xyz": "rf_xyz_default",
    "wei2024": "wei2024_default",
}

REF_OFFLINE_DEFAULT = ("potential_field", "inr_mlp", "rf_xyz")
REF3_OFFLINE_DEFAULT = ("potential_field", "inr_mlp", "rf_xyz", "wei2024")
REF3_SEEDS_DEFAULT = (47, 48, 49, 50)

REF_SOLVER_ORDER = (
    "initial",
    "stretched_nearest_class",
    "layered_horizon",
    "ellipsoid_fit",
    "rbf_sdf",
    "indicator_kriging",
    "gp_classifier",
    "svm_rbf",
    "knn_idw",
    "fdi_smooth",
    "potential_field",
    "inr_mlp",
    "rf_xyz",
)

REF3_SOLVER_ORDER = (*REF_SOLVER_ORDER, "wei2024")


def prepare_ref_inputs(release_dir, stage_a_root, output_dir, *, tier="S",
                       in_container_names=None, offline_names=None,
                       preset="ref", seeds=None):
    """Write wrapper solver.py files, materialize precomputed trees, and verify references.json."""
    if preset not in ("ref", "ref3"):
        raise ValueError(f"unknown prepare_ref preset: {preset!r}")
    release_dir = Path(release_dir).resolve()
    stage_a_root = Path(stage_a_root).resolve() if stage_a_root is not None else None
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_table = IN_CONTAINER_DEFAULT_SOURCES if preset == "ref3" else IN_CONTAINER_SOURCES
    default_off = REF3_OFFLINE_DEFAULT if preset == "ref3" else REF_OFFLINE_DEFAULT
    use_in = tuple(source_table) if in_container_names is None else tuple(in_container_names)
    use_off = default_off if offline_names is None else tuple(offline_names)
    seed_list = None if seeds is None else [int(s) for s in seeds]
    if preset == "ref3" and seed_list is None:
        seed_list = list(REF3_SEEDS_DEFAULT)

    entries = {}
    for name in use_in:
        body = source_table[name]
        sha = validate_source(body)
        src_path = output_dir / "sources" / name / "solver.py"
        src_path.parent.mkdir(parents=True, exist_ok=True)
        src_path.write_bytes(body)
        if seed_list is None:
            entries[name] = {"source": str(src_path), "sha256": sha}
        else:
            for s in seed_list:
                entries[f"{name}-s{s}"] = {"source": str(src_path), "sha256": sha, "seed": int(s)}

    for name in use_off:
        if stage_a_root is None:
            raise ValueError(f"stage_a_root required for offline solver {name!r}")
        subdir = stage_a_root / OFFLINE_STAGE_A_SUBDIRS[name]
        solver_key = OFFLINE_DEFAULT_SOLVERS[name] if preset == "ref3" else name
        if seed_list is None:
            fields_dir = output_dir / "precomputed" / name
            sha = offline_baselines.materialize_precomputed_reference(
                solver_key, subdir, release_dir, fields_dir, tier=tier,
            )
            entries[name] = {"fields": str(fields_dir), "sha256": sha}
        elif name == "potential_field":
            fields_dir = output_dir / "precomputed" / name
            sha = offline_baselines.materialize_precomputed_reference(
                solver_key, subdir, release_dir, fields_dir, tier=tier,
            )
            for s in seed_list:
                entries[f"{name}-s{s}"] = {"fields": str(fields_dir), "sha256": sha, "seed": int(s)}
        else:
            for s in seed_list:
                entry_name = f"{name}-s{s}"
                fields_dir = output_dir / "precomputed" / entry_name
                sha = offline_baselines.materialize_precomputed_reference(
                    solver_key, subdir, release_dir, fields_dir, tier=tier, source_seed=int(s),
                )
                entries[entry_name] = {"fields": str(fields_dir), "sha256": sha, "seed": int(s)}

    refs_path = output_dir / "references.json"
    atomic_json_write(refs_path, entries)
    verified = q1.references(refs_path)
    if set(verified) != set(entries):
        raise ValueError("verified references mismatch")
    return refs_path


def _parse_csv_list(raw):
    if raw is None:
        return None
    return [x.strip() for x in raw.split(",") if x.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--stage-a-root", type=Path, required=True,
                        help="Directory containing Stage-A outputs (gempy/, inr/, rf/, wei2024/).")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tier", default="S")
    parser.add_argument("--preset", default="ref", choices=("ref", "ref3"))
    parser.add_argument("--seeds", default=None,
                        help="Optional comma-separated integer seeds (defaults to 47,48,49,50 for preset=ref3).")
    parser.add_argument("--in-container", default=None,
                        help="Optional comma-separated subset of in-container solver names (empty string for none).")
    parser.add_argument("--offline", default=None,
                        help="Optional comma-separated subset of offline solver names (e.g. wei2024).")
    args = parser.parse_args(argv)
    in_names = _parse_csv_list(args.in_container)
    off_names = _parse_csv_list(args.offline)
    seeds = [int(x) for x in _parse_csv_list(args.seeds)] if args.seeds is not None else None
    refs_path = prepare_ref_inputs(
        args.release, args.stage_a_root, args.output, tier=args.tier,
        in_container_names=in_names, offline_names=off_names,
        preset=args.preset, seeds=seeds,
    )
    data = json.loads(refs_path.read_text())
    print(json.dumps({
        "status": "prepared",
        "references": len(data),
        "path": str(refs_path),
        "names": sorted(data),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
