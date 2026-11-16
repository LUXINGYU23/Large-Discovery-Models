"""Export per-run outcome measures from the private side (contracts §9.1).

Run by the evaluator account after the unseal. One row per frozen solver,
fixed columns, aggregates over geology bodies only: no per-body value, case or
parent identity, or dataset file hash leaves the private side.
"""

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np

from ldm_tts.engine.run_store import atomic_json_write
from tasks.copper_inversion.core.execution import digest
from tasks.copper_inversion.experiments.rsi.freeze import layout, parse_prefix, run_manifest
from tasks.copper_inversion.experiments.rsi.unseal import check, load_list
from tasks.copper_inversion.v2 import selection
from tasks.copper_inversion.v2.suite import aggregate, file_hash, load_manifest

SPLITS = ("test_id", "test_ood")
MODALITIES = ("drill", "drill_outcrop")
VALIDATION_ONLY = frozenset({"E1", "E2"})
MIN_PARENTS = 2
COLUMNS = ("experiment", "protocol_id", "run_id", "arm", "tier", "judge", "seed", "prefix", "round",
           "selected_sha256", "finalist_sha256s",
           "validation_risk", "validation_iou", "validation_legal_rate", "validation_seconds_per_solve",
           *[f"{split}_{name}" for split in SPLITS for name in (
               "iou", "risk", "legal_rate", "seconds_per_solve", *[f"iou_{m}" for m in MODALITIES],
               "surface_distance", "volume_error")])


def _rows(work):
    found = sorted(Path(work).glob("evaluations/*/private-case-results.json"))
    if len(found) != 1:
        raise ValueError("expected exactly one private case receipt set")
    return json.loads(found[0].read_text())


def _parent_mean(rows, value):
    grouped = {}
    for row in rows:
        item = value(row)
        if item is not None:
            grouped.setdefault(row["parent_id"], []).append(item)
    if len(grouped) < MIN_PARENTS:
        return None
    return float(np.mean([np.mean(v) for v in grouped.values()]))


def _measures(row, work, manifest_path):
    """Body-level receipts reduced to parent-macro aggregates of one split."""
    metrics = row["metrics"]
    out = {"iou": metrics["mean_iou"], "risk": metrics["loss_total"], "legal_rate": metrics["legal_rate"],
           "seconds_per_solve": metrics["mean_solver_wall_seconds"]}
    if row.get("interface_rejected"):
        return {**out, **{f"iou_{m}": 0.0 for m in MODALITIES}, "surface_distance": None, "volume_error": None}
    cases = _rows(work)
    check_metrics = aggregate(cases)
    if any(not np.isclose(check_metrics[k], metrics[k], rtol=0, atol=1e-12)
           for k in ("mean_iou", "loss_total", "legal_rate")):
        raise ValueError("private case receipts differ from the sealed result")
    modality = {c["case_id"]: c["modality"] for c in load_manifest(manifest_path)["cases"]}
    succeeded = lambda key: lambda r: r["metrics"][key] if r["status"] == "succeeded" else None
    for name in MODALITIES:
        subset = [r for r in cases if modality[r["case_id"]] == name]
        out[f"iou_{name}"] = _parent_mean(subset, lambda r: r["metrics"]["iou"] if r["status"] == "succeeded" else 0.)
    out["surface_distance"] = _parent_mean(cases, succeeded("mean_surface_distance"))
    out["volume_error"] = _parent_mean(cases, succeeded("relative_volume_error"))
    return out


def outcome(target, run, prefix, sealed):
    frozen, state = selection.verify(target)
    report = json.loads((target / "selection.json").read_text())
    if digest(report) != state["selection_sha256"]:
        raise ValueError("frozen selection changed")
    sha = report["selected_sha256"]
    chosen = next(r for r in report["candidates"] if r["source_sha256"] == sha)
    validation = _measures(chosen, target / "validation" / sha / "validation", frozen["manifests"][0]["path"])
    row = {"experiment": run["experiment"], "protocol_id": run["protocol_id"], "run_id": run["run_id"],
           "arm": run.get("arm"), "tier": run.get("tier"), "judge": run.get("judge"), "seed": run.get("seed"),
           "prefix": prefix, "round": parse_prefix(prefix), "selected_sha256": sha,
           "finalist_sha256s": " ".join(c["sha256"] for c in frozen["candidates"]),
           **{f"validation_{k}": validation[k] for k in ("risk", "iou", "legal_rate", "seconds_per_solve")}}
    row.update({column: None for column in COLUMNS if column.startswith(SPLITS)})
    if sealed:
        result = json.loads((target / "test-result.json").read_text())
        if state["phase"] != "tested" or digest(result) != state["test_result_sha256"] or result["selected_sha256"] != sha:
            raise ValueError("sealed result is missing or changed")
        for index, split in enumerate(SPLITS, start=1):
            item = next(r for r in result["results"] if r["split"] == split)
            measures = _measures(item, target / "test" / sha / split, frozen["manifests"][index]["path"])
            row.update({f"{split}_{k}": v for k, v in measures.items()})
    return row


def _forbidden(targets):
    tokens = set()
    for target in targets:
        frozen = json.loads((target / "frozen.json").read_text())
        for path in [frozen["training"]["manifest_path"], *[m["path"] for m in frozen["manifests"]]]:
            for case in load_manifest(path)["cases"]:
                tokens.add(case["parent_id"])
                tokens.update(case["sha256"].values())
            tokens.add(file_hash(path))
    return tokens


def export(root, protocol_id, experiment, *, validation_only=False):
    paths = layout(root, protocol_id, experiment)
    if validation_only:
        if experiment not in VALIDATION_ONLY:
            raise ValueError("validation-only export is limited to experiments without a sealed outcome")
        solvers = [{"run_id": run.name, "prefix": p.name} for run in sorted(paths["selection"].iterdir())
                   for p in sorted(run.iterdir()) if p.is_dir()]
        listing = load_list({"experiment": experiment, "protocol_id": protocol_id, "solvers": solvers})
        list_sha = None
    else:
        completion_path = paths["unseal"] / f"{experiment}.json"
        if not completion_path.is_file():
            raise ValueError("export happens only after the unseal")
        completion = json.loads(completion_path.read_text())
        listing = load_list({k: completion[k] for k in ("experiment", "protocol_id", "solvers")})
        list_sha = digest(listing)
        if completion["frozen_list_sha256"] != list_sha:
            raise ValueError("unseal completion record differs from its list")
    targets = check(root, listing)
    rows = []
    for (run_id, prefix), target in sorted(targets.items()):
        run = run_manifest(layout(root, protocol_id, experiment, run_id)["run"], protocol_id, experiment)
        row = outcome(target, run, prefix, not validation_only)
        if not validation_only and digest(json.loads((target / "test-result.json").read_text())) != \
                completion["results_sha256"][f"{run_id}/{prefix}"]:
            raise ValueError("sealed result differs from the unseal record")
        rows.append({column: row[column] for column in COLUMNS})
    table = {"schema": "rsi-outcomes-v1", "experiment": experiment, "protocol_id": protocol_id,
             "sealed": not validation_only, "frozen_list_sha256": list_sha,
             "unseal_log_sha256": file_hash(paths["unseal_log"]) if paths["unseal_log"].exists() else None,
             "columns": list(COLUMNS), "rows": rows}
    text = json.dumps(table, sort_keys=True, allow_nan=False)
    # Short identities would match by accident; formal parents and hashes are long.
    leaked = sorted(t for t in _forbidden(targets.values()) if len(t) >= 16 and t in text)
    if leaked:
        raise ValueError(f"export would disclose {len(leaked)} private dataset identities")
    return table


def to_csv(table):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=table["columns"], lineterminator="\n")
    writer.writeheader()
    for row in table["rows"]:
        writer.writerow({k: "" if v is None else v for k, v in row.items()})
    return stream.getvalue()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", type=Path, required=True, help="outcomes JSON; a CSV is written beside it")
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args(argv)
    table = export(args.root, args.protocol_id, args.experiment, validation_only=args.validation_only)
    if args.output.exists():
        raise ValueError("outcome table already exists")
    atomic_json_write(args.output, table)
    args.output.with_suffix(".csv").write_text(to_csv(table))
    print(json.dumps({"status": "exported", "rows": len(table["rows"]), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
