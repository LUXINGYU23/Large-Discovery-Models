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

OFFLINE_STAGE_A_SUBDIRS = {
    "potential_field": "gempy",
    "inr_mlp": "inr",
    "rf_xyz": "rf",
    "wei2024": "wei2024",
}

REF_OFFLINE_DEFAULT = ("potential_field", "inr_mlp", "rf_xyz")

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


def prepare_ref_inputs(release_dir, stage_a_root, output_dir, *, tier="S",
                       in_container_names=None, offline_names=None):
    """Write wrapper solver.py files, materialize precomputed trees, and verify references.json."""
    release_dir = Path(release_dir).resolve()
    stage_a_root = Path(stage_a_root).resolve() if stage_a_root is not None else None
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    use_in = tuple(IN_CONTAINER_SOURCES) if in_container_names is None else tuple(in_container_names)
    use_off = REF_OFFLINE_DEFAULT if offline_names is None else tuple(offline_names)

    entries = {}
    for name in use_in:
        body = IN_CONTAINER_SOURCES[name]
        sha = validate_source(body)
        src_path = output_dir / "sources" / name / "solver.py"
        src_path.parent.mkdir(parents=True, exist_ok=True)
        src_path.write_bytes(body)
        entries[name] = {"source": str(src_path), "sha256": sha}

    for name in use_off:
        if stage_a_root is None:
            raise ValueError(f"stage_a_root required for offline solver {name!r}")
        subdir = stage_a_root / OFFLINE_STAGE_A_SUBDIRS[name]
        fields_dir = output_dir / "precomputed" / name
        sha = offline_baselines.materialize_precomputed_reference(
            name, subdir, release_dir, fields_dir, tier=tier,
        )
        entries[name] = {"fields": str(fields_dir), "sha256": sha}

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
    parser.add_argument("--in-container", default=None,
                        help="Optional comma-separated subset of in-container solver names (empty string for none).")
    parser.add_argument("--offline", default=None,
                        help="Optional comma-separated subset of offline solver names (e.g. wei2024).")
    args = parser.parse_args(argv)
    in_names = _parse_csv_list(args.in_container)
    off_names = _parse_csv_list(args.offline)
    refs_path = prepare_ref_inputs(
        args.release, args.stage_a_root, args.output, tier=args.tier,
        in_container_names=in_names, offline_names=off_names,
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
