"""Export per-run outcome measures from the private side (contracts §9.1).

Run by the evaluator account after the unseal. One row per frozen solver,
fixed columns, aggregates over geology bodies only: no per-body value, case or
parent identity, or dataset file hash leaves the private side.

Extra blocks read by the analysis package (analysis/outcomes.py):
  programs, validations  aggregates from private receipts (also validation-only)
  diagnostics            per-run aggregate diagnostics (effort per turn, surrogate
                         rank skill, training risk/IoU Spearman overall and best
                         quartile, boundary-round vs other-round risk)
  body_scores            sealed IoU per body in one anonymous order, sorted by the
                         private parent id, which is never written. A scoped,
                         coordinator-approved exception to AT-31 for the body-paired
                         bootstrap of spec §7; written only after the unseal.
  strata_slices          sealed parent-macro IoU sliced by family, modality and
                         noise (written only after the unseal; strata with fewer
                         than MIN_PARENTS parents are omitted as null).
  q1, references         per-program training risks and logged reference outcomes.
"""

import argparse
import csv
import io
import json
import math
import re
from pathlib import Path

import numpy as np

from ldm_tts.engine.run_store import atomic_json_write
from tasks.copper_inversion.core.execution import digest
from tasks.copper_inversion.experiments.rsi.freeze import layout, parse_prefix, run_manifest
from tasks.copper_inversion.experiments.rsi.unseal import (
    MULTI_CANDIDATE_REFERENCE_EXPERIMENTS, REFERENCE_EXPERIMENTS, check, load_list, _log,
)
from tasks.copper_inversion.v2 import selection
from tasks.copper_inversion.v2.suite import aggregate, file_hash, load_manifest, manifest_tiers, release_tiers
from tasks.copper_inversion.v2.training_receipts import eligible

SPLITS = ("test_id", "test_ood")
MODALITIES = ("drill", "drill_outcrop")
VALIDATION_ONLY = frozenset({"E1", "E2"})
MIN_PARENTS = 2
COLUMNS = ("experiment", "protocol_id", "run_id", "arm", "tier", "judge", "seed", "prefix", "round",
           "selected_sha256", "finalist_sha256s", "incumbent_sha256s",
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
    prefix_path = target / "prefix.json"
    incumbents = json.loads(prefix_path.read_text()).get("incumbent_sha256s", []) if prefix_path.is_file() else []
    row = {"experiment": run["experiment"], "protocol_id": run["protocol_id"], "run_id": run["run_id"],
           "arm": run.get("arm"), "tier": run.get("tier"), "judge": run.get("judge"), "seed": run.get("seed"),
           "prefix": prefix, "round": parse_prefix(prefix), "selected_sha256": sha,
           "finalist_sha256s": " ".join(c["sha256"] for c in frozen["candidates"]),
           "incumbent_sha256s": " ".join(i["sha256"] for i in incumbents),
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


STRATA_AXES = ("family", "modality", "noise")
_ROUND_FILE = re.compile(r"round_(\d+)\.json")


def _spearman(x, y):
    from scipy.stats import rankdata
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    value = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    return value if math.isfinite(value) else None


def _normalize_noise(value):
    if type(value) is bool:
        return "noisy" if value else "clean"
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("noisy", "true", "1"):
            return "noisy"
        if low in ("clean", "false", "0"):
            return "clean"
    return None


def _apply_stratum_row(meta, by_parent, row):
    if not isinstance(row, dict):
        return
    cids = []
    cid = row.get("case_id")
    if isinstance(cid, str) and cid in meta:
        cids.append(cid)
    pid = row.get("parent_id")
    if isinstance(pid, str) and pid in by_parent:
        cids.extend(c for c in by_parent[pid] if c not in cids)
    if not cids:
        return
    family = row.get("family")
    noise = _normalize_noise(row["noise"] if "noise" in row else row.get("noisy"))
    for target in cids:
        if isinstance(family, str) and family:
            meta[target].setdefault("family", family)
        if noise is not None:
            meta[target].setdefault("noise", noise)


def _case_strata(manifest_path, manifest):
    """Resolve (family, modality, noise) per case from strata files, release layout evidence, or public inputs."""
    manifest_path = Path(manifest_path).resolve()
    release = manifest_path.parent.parent
    meta, by_parent = {}, {}
    for case in manifest["cases"]:
        cid = case["case_id"]
        meta[cid] = {"modality": case["modality"]}
        by_parent.setdefault(case["parent_id"], []).append(cid)
    for rel in ("strata.json", "evidence/strata.json", "private/strata.json"):
        path = release / rel
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        if isinstance(data, dict) and isinstance(data.get("cases"), list):
            for row in data["cases"]:
                _apply_stratum_row(meta, by_parent, row)
        elif isinstance(data, list):
            for row in data:
                _apply_stratum_row(meta, by_parent, row)
        elif isinstance(data, dict):
            for key, row in data.items():
                if isinstance(row, dict):
                    _apply_stratum_row(meta, by_parent, {"case_id": key, "parent_id": key, **row})
    csv_path = release / "strata.csv"
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                _apply_stratum_row(meta, by_parent, row)
    layout_path = release / "evidence/layout.json"
    if layout_path.is_file():
        assignment = json.loads(layout_path.read_text()).get("assignment", [])
        if isinstance(assignment, list):
            for row in assignment:
                _apply_stratum_row(meta, by_parent, row)
            if any("family" not in m or "noise" not in m for m in meta.values()):
                split_rows = sorted((r for r in assignment if isinstance(r, dict) and r.get("split") == manifest["split"]),
                                    key=lambda r: r.get("index", 0))
                if (len(split_rows) == len(manifest["cases"])
                        and all(r.get("selected_modality") == c["modality"] for r, c in zip(split_rows, manifest["cases"]))):
                    for row, case in zip(split_rows, manifest["cases"]):
                        entry = meta[case["case_id"]]
                        if isinstance(row.get("family"), str) and row["family"]:
                            entry.setdefault("family", row["family"])
                        noise = _normalize_noise(row["noise"] if "noise" in row else row.get("noisy"))
                        if noise is not None:
                            entry.setdefault("noise", noise)
    for case in manifest["cases"]:
        entry = meta[case["case_id"]]
        if "noise" not in entry:
            prob_path = manifest_path.parent / case["public"] / "problem.json"
            if prob_path.is_file():
                prob = json.loads(prob_path.read_text())
                noisy = any((o.get("label_error") or 0) > 0 for o in prob.get("observations", []))
                entry["noise"] = "noisy" if noisy else "clean"
    return meta


def _strata_slices(item, work, manifest_path):
    """Parent-macro sealed IoU sliced by family, modality and noise; strata < MIN_PARENTS become None."""
    manifest = load_manifest(manifest_path)
    strata = _case_strata(manifest_path, manifest)
    out = {axis: {} for axis in STRATA_AXES}
    if item.get("interface_rejected"):
        for axis in STRATA_AXES:
            levels = sorted({strata[c["case_id"]].get(axis) for c in manifest["cases"] if strata[c["case_id"]].get(axis)})
            for level in levels:
                parents = {c["parent_id"] for c in manifest["cases"] if strata[c["case_id"]].get(axis) == level}
                out[axis][level] = 0.0 if len(parents) >= MIN_PARENTS else None
        return out
    cases = _rows(work)
    for axis in STRATA_AXES:
        levels = sorted({strata[r["case_id"]].get(axis) for r in cases if strata.get(r["case_id"], {}).get(axis)})
        for level in levels:
            subset = [r for r in cases if strata.get(r["case_id"], {}).get(axis) == level]
            out[axis][level] = _parent_mean(subset, lambda r: r["metrics"]["iou"] if r["status"] == "succeeded" else 0.)
    return out


def _round_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return []
    found = []
    for path in sorted(directory.iterdir()):
        match = _ROUND_FILE.fullmatch(path.name)
        if match and path.is_file():
            found.append((int(match.group(1)), path))
    return sorted(found)


def run_diagnostics(run_dir, trained):
    """Aggregate-only per-run diagnostics: effort/turn, surrogate skill, best-quartile Spearman, boundary risk."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    config = manifest.get("config") or {}
    reset_period = config.get("reset_period", 5)
    if type(reset_period) is not int or reset_period < 1:
        reset_period = 5
    ledger_path = run_dir / "ledger.jsonl"
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()] if ledger_path.is_file() else []

    out_tokens, requests, wall_seconds, compactions, fallback_tools = [], [], [], [], []
    for _, path in _round_files(run_dir / "usage"):
        by_profile = json.loads(path.read_text()).get("by_profile") or {}
        for usage in by_profile.values():
            if not isinstance(usage, dict):
                continue
            if isinstance(usage.get("output_tokens"), (int, float)):
                out_tokens.append(float(usage["output_tokens"]))
            if isinstance(usage.get("requests"), (int, float)):
                requests.append(float(usage["requests"]))
            if isinstance(usage.get("wall_seconds"), (int, float)):
                wall_seconds.append(float(usage["wall_seconds"]))
            if isinstance(usage.get("compactions"), (int, float)):
                compactions.append(float(usage["compactions"]))
            tc = usage.get("tool_calls")
            if isinstance(tc, dict):
                fallback_tools.append(float(sum(v for v in tc.values() if isinstance(v, (int, float)))))
            elif isinstance(tc, (int, float)):
                fallback_tools.append(float(tc))

    tool_calls, bash_calls = [], []
    for commit in sorted(run_dir.glob("harness/epoch_*/branch_*/turns/*/turn_committed.json")):
        data = json.loads(commit.read_text())
        tc = (data.get("usage") or {}).get("toolCalls")
        if isinstance(tc, dict):
            tool_calls.append(float(sum(v for v in tc.values() if isinstance(v, (int, float)))))
            bash_calls.append(float(tc.get("bash", 0)))
    if not tool_calls and fallback_tools:
        tool_calls = fallback_tools

    valid_by_round = {}
    for row in ledger:
        if row.get("status") == "valid":
            valid_by_round.setdefault(row["round"], {})[row["program_id"]] = float(row["risk"])

    round_skills, pooled_pred, pooled_real = [], [], []
    for round_index, path in _round_files(run_dir / "selection"):
        if round_index < 3:
            continue
        data = json.loads(path.read_text())
        entries = data.get("predictions") or data.get("selected") or []
        preds = {p["program_id"]: float(p["mean"]) for p in entries
                 if isinstance(p, dict) and isinstance(p.get("mean"), (int, float)) and math.isfinite(p["mean"])}
        realized = valid_by_round.get(round_index, {})
        pairs = [(preds[pid], realized[pid]) for pid in sorted(realized) if pid in preds]
        if pairs:
            xs, ys = [a for a, _ in pairs], [b for _, b in pairs]
            pooled_pred.extend(xs)
            pooled_real.extend(ys)
            rho = _spearman(xs, ys)
            if rho is not None:
                round_skills.append(rho)

    train_pairs = [(float(r["risk"]), float(trained[r["program_id"]]))
                   for r in ledger if r.get("status") == "valid" and r["program_id"] in trained]
    if train_pairs:
        t_risks = [x for x, _ in train_pairs]
        t_ious = [y for _, y in train_pairs]
        train_spearman = _spearman(t_risks, t_ious)
        q25 = float(np.quantile(t_risks, 0.25))
        top = [(x, y) for x, y in train_pairs if x <= q25]
        best_q_spearman = _spearman([x for x, _ in top], [y for _, y in top])
    else:
        train_spearman, best_q_spearman, top = None, None, []

    round1_risks = [float(r["risk"]) for r in ledger if r.get("status") == "valid" and r["round"] == 1]
    boundary_risks = [float(r["risk"]) for r in ledger
                      if r.get("status") == "valid" and r["round"] > 1 and (r["round"] - 1) % reset_period == 0]
    other_risks = [float(r["risk"]) for r in ledger
                   if r.get("status") == "valid" and r["round"] > 1 and (r["round"] - 1) % reset_period != 0]

    mean_or_none = lambda xs: float(np.mean(xs)) if xs else None
    median_or_none = lambda xs: float(np.median(xs)) if xs else None
    return {
        "run_id": manifest.get("run_id") or run_dir.name,
        "arm": manifest.get("arm"),
        "seed": manifest.get("seed"),
        "judge": manifest.get("judge"),
        "turns": len(out_tokens),
        "output_tokens_per_turn": mean_or_none(out_tokens),
        "requests_per_turn": mean_or_none(requests),
        "wall_seconds_per_turn": mean_or_none(wall_seconds),
        "compactions_per_turn": mean_or_none(compactions),
        "tool_calls_turns": len(tool_calls),
        "tool_calls_per_turn": mean_or_none(tool_calls),
        "bash_calls_per_turn": mean_or_none(bash_calls),
        "surrogate_rank_skill": mean_or_none(round_skills),
        "surrogate_rank_skill_pooled": _spearman(pooled_pred, pooled_real),
        "surrogate_rounds": len(round_skills),
        "train_risk_iou_spearman": train_spearman,
        "best_quartile_spearman": best_q_spearman,
        "train_valid_programs": len(train_pairs),
        "best_quartile_n": len(top),
        "reset_period": reset_period,
        "boundary_round_risk_mean": mean_or_none(boundary_risks),
        "boundary_round_risk_median": median_or_none(boundary_risks),
        "boundary_round_n": len(boundary_risks),
        "other_round_risk_mean": mean_or_none(other_risks),
        "other_round_risk_median": median_or_none(other_risks),
        "other_round_n": len(other_risks),
        "round1_risk_mean": mean_or_none(round1_risks),
        "round1_risk_median": median_or_none(round1_risks),
        "round1_n": len(round1_risks),
    }


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
    rows, body_scores, strata_rows, orders, program_rows, validation_rows, diagnostic_rows, manifests = (
        [], [], [], {}, [], [], [], set())
    for (run_id, prefix), target in sorted(targets.items()):
        run = run_manifest(layout(root, protocol_id, experiment, run_id)["run"], protocol_id, experiment)
        row, bodies = outcome(target, run, prefix, not validation_only)
        if not validation_only:
            result = json.loads((target / "test-result.json").read_text())
            if digest(result) != completion["results_sha256"][f"{run_id}/{prefix}"]:
                raise ValueError("sealed result differs from the unseal record")
            frozen = json.loads((target / "frozen.json").read_text())
            sha = result["selected_sha256"]
            for index, split in enumerate(SPLITS, start=1):
                item = next(r for r in result["results"] if r["split"] == split)
                slices = _strata_slices(item, target / "test" / sha / split, frozen["manifests"][index]["path"])
                strata_rows.append({"run_id": run_id, "prefix": prefix, "split": split, **slices})
        rows.append({column: row[column] for column in COLUMNS})
        for split, (manifest_sha, values) in sorted(bodies.items()):
            # One body order per split for the whole table: every run must use the same suite.
            if orders.setdefault(split, manifest_sha) != manifest_sha:
                raise ValueError("runs of one table were sealed on different body suites")
            body_scores.append({"run_id": run_id, "prefix": prefix, "split": split, "iou": values})
    for run_id in sorted({run_id for run_id, _ in targets}):
        run_paths = layout(root, protocol_id, experiment, run_id)
        trained, manifest = programs(run_paths["service"])
        manifests.add(manifest)
        program_rows += [{"run_id": run_id, "program_id": sha, "train_iou": v} for sha, v in sorted(trained.items())]
        validated = validations(t for (r, _), t in sorted(targets.items()) if r == run_id)
        validation_rows += [{"run_id": run_id, "program_id": sha, **v} for sha, v in sorted(validated.items())]
        diagnostic_rows.append(run_diagnostics(run_paths["run"], trained))
    table = {"schema": "rsi-outcomes-v1", "experiment": experiment, "protocol_id": protocol_id,
             "sealed": not validation_only, "frozen_list_sha256": list_sha,
             "unseal_log_sha256": file_hash(paths["unseal_log"]) if paths["unseal_log"].exists() else None,
             "columns": list(COLUMNS), "rows": rows, "programs": program_rows, "validations": validation_rows,
             "diagnostics": diagnostic_rows}
    if not validation_only:
        table["body_scores"] = body_scores
        table["strata_slices"] = strata_rows
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
    suite = load_manifest(manifest)
    if suite["split"] != "train" or identity["configuration"].get("objective", "confirmation") != "confirmation":
        raise ValueError("Q1 requires confirmation-risk training suites")
    records = eligible(service_root, manifest)
    for record in records.values():
        rows = json.loads(Path(record["cases_path"]).read_text())
        if any(row.get("objective", "confirmation") != "confirmation" for row in rows):
            raise ValueError("Q1 receipts must use confirmation risk")
    return {sha: r["metrics"]["loss_total"] for sha, r in records.items()}, manifest


def _check_q1_layout(root, name, values, manifest):
    path = Path(root) / "q1-layout.json"
    if not path.exists():
        return  # Compatibility with previously supplied service roots.
    identity = json.loads(path.read_text())
    if (identity.get("layout") != name or identity.get("manifest") != str(Path(manifest).resolve())
            or identity.get("manifest_sha256") != file_hash(manifest)
            or identity.get("program_ids") != sorted(values)):
        raise ValueError("Q1 service layout or program receipts differ")


def _seed_risks(service_root):
    identity = json.loads((Path(service_root) / "identity.json").read_text())
    manifest = identity["configuration"]["manifest"]
    suite = load_manifest(manifest)
    if suite["split"] != "train" or identity["configuration"].get("objective", "confirmation") != "confirmation":
        raise ValueError("Q1 requires confirmation-risk training suites")
    if len(suite["seeds"]) != 3:
        raise ValueError("Q1 repeat suite requires exactly three execution seeds")
    records = eligible(service_root, manifest)
    risks = {}
    for sha, record in records.items():
        rows = json.loads(Path(record["cases_path"]).read_text())
        if any(row.get("objective", "confirmation") != "confirmation" for row in rows):
            raise ValueError("Q1 receipts must use confirmation risk")
        risks[sha] = [aggregate([row for row in rows if row["seed"] == seed])["loss_total"]
                      for seed in suite["seeds"]]
    return risks, manifest


def _reference_tier(frozen):
    train_path = Path(frozen["training"]["manifest_path"])
    training = load_manifest(train_path)
    release, tiers = release_tiers(train_path, training["stage"])
    if tiers is not None:
        names = manifest_tiers(train_path, training, release, tiers, roles=("train",))
        if len(names) == 1:
            return names[0]
    return {30: "S", 60: "M", 120: "F"}.get(frozen.get("seconds"))


def export_references(root, protocol_id, *, q1=None, references=None, experiment="Q1"):
    """The Q1 or REF/REF2 table: tier/layout risks of Q1 programs and/or domain references.

    q1: {"short_12x1": service, "full_24x1": service, "short_24x2": service,
         "repeats": [service, ...]} (training service roots, extra execution seeds).
    references: {name: tested v2 selection root holding one reference solver}.
    """
    allowed = ("Q1", *REFERENCE_EXPERIMENTS)
    if experiment not in allowed:
        raise ValueError(f"export_references supports experiment in {allowed}")
    if experiment in REFERENCE_EXPERIMENTS and (q1 is not None or not references):
        raise ValueError(f"{experiment} export requires --references without --q1")
    paths = layout(root, protocol_id, "Q1")
    table = {"schema": "rsi-outcomes-v1", "experiment": experiment, "protocol_id": protocol_id,
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
            _check_q1_layout(q1[name], name, risks[name], manifest)
            manifests.add(manifest)
        common = set.intersection(*(set(v) for v in risks.values()))
        repeat_input = q1.get("repeats", [])
        if isinstance(repeat_input, str):
            repeat_values, manifest = _seed_risks(repeat_input)
            _check_q1_layout(repeat_input, "repeats", repeat_values, manifest)
            manifests.add(manifest)
            repeats = [{"program_id": sha, "risks": repeat_values[sha]}
                       for sha in sorted(common & set(repeat_values))]
        else:
            values = []
            for service in repeat_input:
                value, manifest = _risks(service)
                manifests.add(manifest)
                values.append(value)
            repeats = [{"program_id": sha, "risks": [risks["short_12x1"][sha], *[r[sha] for r in values]]}
                       for sha in sorted(common.intersection(*map(set, values)))] if values else []
        table["q1"] = {"programs": [{"program_id": sha, **{n: risks[n][sha] for n in names}} for sha in sorted(common)],
                       "repeats": repeats}
    if references:
        unseal_exp = "REF" if experiment == "Q1" else experiment
        entries = [entry for entry in _log(paths["unseal_log"])
                   if entry.get("experiment") == unseal_exp and entry.get("protocol_id") == protocol_id
                   and entry.get("mode") == "completed"]
        if len(entries) != 1 or set(entries[0]["selection_roots"]) != set(references):
            raise ValueError(f"references require a completed {unseal_exp} unseal log")
        logged = entries[0]
        table["references"] = {}
        rows, body_scores, strata_rows, validation_rows, orders = [], [], [], [], {}
        for name, target in sorted(references.items()):
            target = Path(target)
            frozen, state = selection.verify(target)
            min_ok = (len(frozen["candidates"]) >= 1
                      if experiment in MULTI_CANDIDATE_REFERENCE_EXPERIMENTS
                      else len(frozen["candidates"]) == 1)
            if state["phase"] != "tested" or not min_ok:
                raise ValueError("a domain reference needs one tested frozen solver")
            result = json.loads((target / "test-result.json").read_text())
            if (str(target.resolve()) != logged["selection_roots"][name]
                    or digest(frozen) != logged["frozen_sha256"][name]
                    or state["selection_sha256"] != logged["selection_sha256"][name]
                    or digest(result) != state["test_result_sha256"]
                    or digest(result) != logged["results_sha256"][name]):
                raise ValueError(f"domain reference differs from {unseal_exp} unseal log")
            table["references"][name] = {r["split"]: r["metrics"]["mean_iou"] for r in result["results"]
                                         if r["split"] in SPLITS}
            if experiment in REFERENCE_EXPERIMENTS:
                val_seeds = load_manifest(frozen["manifests"][0]["path"])["seeds"]
                seed_val = frozen.get("seed_override")
                if seed_val is None:
                    seed_val = val_seeds[0] if len(val_seeds) == 1 else None
                seed_match = re.fullmatch(r"(.+)-s(\d+)", name)
                if seed_match is not None:
                    arm_name = seed_match.group(1)
                    seed_val = int(seed_match.group(2))
                else:
                    arm_name = ("offline_reference" if frozen["candidates"][0].get("kind") == "precomputed"
                                else "in_container_reference")
                run_info = {
                    "experiment": experiment,
                    "protocol_id": protocol_id,
                    "run_id": name,
                    "arm": arm_name,
                    "tier": _reference_tier(frozen),
                    "judge": frozen.get("selection_objective", "confirmation"),
                    "seed": seed_val,
                }
                row, bodies = outcome(target, run_info, "full", True)
                rows.append({column: row[column] for column in COLUMNS})
                sha = result["selected_sha256"]
                for index, split in enumerate(SPLITS, start=1):
                    item = next(r for r in result["results"] if r["split"] == split)
                    slices = _strata_slices(item, target / "test" / sha / split, frozen["manifests"][index]["path"])
                    strata_rows.append({"run_id": name, "prefix": "full", "split": split, **slices})
                for split, (manifest_sha, values) in sorted(bodies.items()):
                    if orders.setdefault(split, manifest_sha) != manifest_sha:
                        raise ValueError("runs of one table were sealed on different body suites")
                    body_scores.append({"run_id": name, "prefix": "full", "split": split, "iou": values})
                validated = validations([target])
                validation_rows += [{"run_id": name, "program_id": s, **v} for s, v in sorted(validated.items())]
            targets.append(target)
        if experiment in REFERENCE_EXPERIMENTS:
            table["rows"] = rows
            table["validations"] = validation_rows
            table["body_scores"] = body_scores
            table["strata_slices"] = strata_rows
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
    parser.add_argument("--references", type=Path, help="Q1, REF or REF2: JSON {name: tested selection root}")
    args = parser.parse_args(argv)
    if args.experiment in ("Q1", *REFERENCE_EXPERIMENTS):
        load = lambda path: json.loads(path.read_text()) if path else None
        table = export_references(args.root, args.protocol_id, q1=load(args.q1),
                                  references=load(args.references), experiment=args.experiment)
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
