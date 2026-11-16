"""Export per-run outcome measures from the private side (contracts §9.1).

Run by the evaluator account after the unseal. One row per frozen solver,
fixed columns, aggregates over geology bodies only: no per-body value, case or
parent identity, or dataset file hash leaves the private side.

Extra blocks read by the analysis package (analysis/outcomes.py):
  programs, validations  aggregates from private receipts (also validation-only)
  body_scores            sealed IoU per body in one anonymous order, sorted by the
                         private parent id, which is never written. A scoped,
                         coordinator-approved exception to AT-31 for the body-paired
                         bootstrap of spec §7; written only after the unseal.
  q1, references         a separate table with experiment "Q1" and no rows.
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
from tasks.copper_inversion.v2.training_receipts import eligible

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


def _bodies(item, work, manifest_path):
    """Parent-macro IoU per body, ordered by private parent id; the ids stay here."""
    parents = sorted({c["parent_id"] for c in load_manifest(manifest_path)["cases"]})
    if item.get("interface_rejected"):
        return [0.0] * len(parents)
    grouped = {}
    for row in _rows(work):
        grouped.setdefault(row["parent_id"], []).append(row["metrics"]["iou"] if row["status"] == "succeeded" else 0.)
    if set(grouped) != set(parents):
        raise ValueError("private case receipts do not cover the sealed bodies")
    values = [float(np.mean(grouped[p])) for p in parents]
    if not np.isclose(np.mean(values), item["metrics"]["mean_iou"], rtol=0, atol=1e-12):
        raise ValueError("per-body IoU differs from the sealed result")
    return values


def outcome(target, run, prefix, sealed):
    """One table row, plus {split: (manifest sha256, per-body IoU)} when sealed."""
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
    bodies = {}
    if sealed:
        result = json.loads((target / "test-result.json").read_text())
        if state["phase"] != "tested" or digest(result) != state["test_result_sha256"] or result["selected_sha256"] != sha:
            raise ValueError("sealed result is missing or changed")
        for index, split in enumerate(SPLITS, start=1):
            item = next(r for r in result["results"] if r["split"] == split)
            measures = _measures(item, target / "test" / sha / split, frozen["manifests"][index]["path"])
            row.update({f"{split}_{k}": v for k, v in measures.items()})
            path = frozen["manifests"][index]["path"]
            bodies[split] = (file_hash(path), _bodies(item, target / "test" / sha / split, path))
    return row, bodies


def validations(targets):
    """Validation risk and IoU of every program validated by any freeze of one run."""
    found = {}
    for target in targets:
        report = json.loads((Path(target) / "selection.json").read_text())
        for row in report["candidates"]:
            if not row.get("interface_rejected"):
                found.setdefault(row["source_sha256"], {"risk": row["metrics"]["loss_total"],
                                                        "iou": row["metrics"]["mean_iou"]})
    return found


def programs(service_root):
    """Hidden training IoU of every program with a complete training receipt."""
    identity = json.loads((Path(service_root) / "identity.json").read_text())
    manifest = identity["configuration"]["manifest"]
    return {sha: r["metrics"]["mean_iou"] for sha, r in eligible(service_root, manifest).items()}, manifest


def _forbidden(targets, manifests=()):
    tokens, paths = set(), set(map(str, manifests))
    for target in targets:
        frozen = json.loads((Path(target) / "frozen.json").read_text())
        paths.update(str(p) for p in [frozen["training"]["manifest_path"], *[m["path"] for m in frozen["manifests"]]])
    for path in sorted(paths):
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
    rows, body_scores, orders, program_rows, validation_rows, manifests = [], [], {}, [], [], set()
    for (run_id, prefix), target in sorted(targets.items()):
        run = run_manifest(layout(root, protocol_id, experiment, run_id)["run"], protocol_id, experiment)
        row, bodies = outcome(target, run, prefix, not validation_only)
        if not validation_only and digest(json.loads((target / "test-result.json").read_text())) != \
                completion["results_sha256"][f"{run_id}/{prefix}"]:
            raise ValueError("sealed result differs from the unseal record")
        rows.append({column: row[column] for column in COLUMNS})
        for split, (manifest_sha, values) in sorted(bodies.items()):
            # One body order per split for the whole table: every run must use the same suite.
            if orders.setdefault(split, manifest_sha) != manifest_sha:
                raise ValueError("runs of one table were sealed on different body suites")
            body_scores.append({"run_id": run_id, "prefix": prefix, "split": split, "iou": values})
    for run_id in sorted({run_id for run_id, _ in targets}):
        trained, manifest = programs(layout(root, protocol_id, experiment, run_id)["service"])
        manifests.add(manifest)
        program_rows += [{"run_id": run_id, "program_id": sha, "train_iou": v} for sha, v in sorted(trained.items())]
        validated = validations(t for (r, _), t in sorted(targets.items()) if r == run_id)
        validation_rows += [{"run_id": run_id, "program_id": sha, **v} for sha, v in sorted(validated.items())]
    table = {"schema": "rsi-outcomes-v1", "experiment": experiment, "protocol_id": protocol_id,
             "sealed": not validation_only, "frozen_list_sha256": list_sha,
             "unseal_log_sha256": file_hash(paths["unseal_log"]) if paths["unseal_log"].exists() else None,
             "columns": list(COLUMNS), "rows": rows, "programs": program_rows, "validations": validation_rows}
    if not validation_only:
        table["body_scores"] = body_scores
    _guard(table, _forbidden(targets.values(), manifests))
    return table


def _guard(table, forbidden):
    text = json.dumps(table, sort_keys=True, allow_nan=False)
    # Short identities would match by accident; formal parents and hashes are long.
    leaked = sorted(t for t in forbidden if len(t) >= 16 and t in text)
    if leaked:
        raise ValueError(f"export would disclose {len(leaked)} private dataset identities")


def _risks(service_root):
    identity = json.loads((Path(service_root) / "identity.json").read_text())
    manifest = identity["configuration"]["manifest"]
    return {sha: r["metrics"]["loss_total"] for sha, r in eligible(service_root, manifest).items()}, manifest


def export_references(root, protocol_id, *, q1=None, references=None):
    """The Q1 table: tier and layout risks of the Q1 programs, and domain references.

    q1: {"short_12x1": service, "full_24x1": service, "short_24x2": service,
         "repeats": [service, ...]} (training service roots, extra execution seeds).
    references: {name: tested v2 selection root holding one reference solver}.
    """
    paths = layout(root, protocol_id, "Q1")
    table = {"schema": "rsi-outcomes-v1", "experiment": "Q1", "protocol_id": protocol_id,
             "sealed": bool(references), "frozen_list_sha256": None,
             "unseal_log_sha256": file_hash(paths["unseal_log"]) if paths["unseal_log"].exists() else None,
             "columns": list(COLUMNS), "rows": []}
    manifests, targets = set(), []
    if q1:
        names = ("short_12x1", "full_24x1", "short_24x2")
        if set(q1) - {*names, "repeats"} or not set(names) <= set(q1):
            raise ValueError("Q1 needs the three layout services")
        risks = {}
        for name in names:
            risks[name], manifest = _risks(q1[name])
            manifests.add(manifest)
        repeats = []
        for service in q1.get("repeats", []):
            values, manifest = _risks(service)
            manifests.add(manifest)
            repeats.append(values)
        common = set.intersection(*(set(v) for v in risks.values()))
        table["q1"] = {"programs": [{"program_id": sha, **{n: risks[n][sha] for n in names}} for sha in sorted(common)],
                       "repeats": [{"program_id": sha, "risks": [risks["short_12x1"][sha], *[r[sha] for r in repeats]]}
                                   for sha in sorted(set(risks["short_12x1"]).intersection(*map(set, repeats)))]
                       if repeats else []}
    if references:
        table["references"] = {}
        for name, target in sorted(references.items()):
            target = Path(target)
            frozen, state = selection.verify(target)
            if state["phase"] != "tested" or len(frozen["candidates"]) != 1:
                raise ValueError("a domain reference needs one tested frozen solver")
            result = json.loads((target / "test-result.json").read_text())
            if digest(result) != state["test_result_sha256"]:
                raise ValueError("domain reference result changed")
            table["references"][name] = {r["split"]: r["metrics"]["mean_iou"] for r in result["results"]
                                         if r["split"] in SPLITS}
            targets.append(target)
    _guard(table, _forbidden(targets, manifests))
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
    parser.add_argument("--q1", type=Path, help="Q1 only: JSON of service roots (see export_references)")
    parser.add_argument("--references", type=Path, help="Q1 only: JSON {name: tested selection root}")
    args = parser.parse_args(argv)
    if args.experiment == "Q1":
        load = lambda path: json.loads(path.read_text()) if path else None
        table = export_references(args.root, args.protocol_id, q1=load(args.q1), references=load(args.references))
    else:
        table = export(args.root, args.protocol_id, args.experiment, validation_only=args.validation_only)
    if args.output.exists():
        raise ValueError("outcome table already exists")
    atomic_json_write(args.output, table)
    args.output.with_suffix(".csv").write_text(to_csv(table))
    print(json.dumps({"status": "exported", "rows": len(table["rows"]), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
