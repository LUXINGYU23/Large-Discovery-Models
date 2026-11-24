"""Private RSI data releases (`rsi-dev`, `rsi-r1`, `rsi-r2`), one observation layout per parent.

`plan` fixes the release plan and records its hash before anything is generated.
Generation reuses `generate_independent`/`generate_pilot` (which pick the plan up
from the dataset root) and `generate_confirmation`. `release` checks the plan
receipts, the layout rules, the whole-hole audit and, for formal releases, the
qualification evidence, then writes every manifest and the qualification record.
The unselected pack of each parent is archived under `unselected/`, outside
every evaluation manifest (except the optional Q1 both-pack training manifest).
"""

import argparse
import csv
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from tasks.copper_inversion.experiments.v2.generate_pilot import vertical_ore_runs
from tasks.copper_inversion.experiments.v2.release_independent import (historical_exclusion, pair_confirmation,
                                                                       qualify_evidence)
from tasks.copper_inversion.experiments.v2.split_audit import shape_fingerprint
from tasks.copper_inversion.rsi.protocol import code_commit
from tasks.copper_inversion.v2.data import read_problem, save_problem
from tasks.copper_inversion.v2.qualification import verify_geological_constraints, verify_release, verify_tiers
from tasks.copper_inversion.v2.suite import file_hash, load_manifest

from .measurements import (ASSIGNMENT_RULE, CONFIRMATION_POLICIES, CONFIRMATION_POLICIES_MULTI, MODALITIES,
                           audit_holes, audit_pack_pair, check_confirmation_attempts, check_discoverability_attempts,
                           check_thinning, confirmation_min_ore_holes, confirmation_mode, confirmation_pack_count,
                           hole_distance_summary, merge_confirmations, modality_balance, ore_hole_count,
                           parse_cli_holes as _parse_cli_holes, public_min_ore_holes, thin_confirmation)
from .plan import (PRESETS, derive_seed, load_generation_plan, make_generation_plan, part_splits, release_splits,
                   verify_part, write_generation_plan)


SPLITS = ("train", "calibration", "calibration_ood", "validation", "test_id", "test_ood")
OPEN_SPLITS = ("train", "calibration", "calibration_ood", "validation")
SEALED_SPLITS = ("test_id", "test_ood")
FILES = ("problem.json", "problem.npz", "truth.npy", "confirmation/problem.json", "confirmation/problem.npz")
OVERLAP = ("a confirmation hole overlaps when at least half of its samples lie within one confirmation "
           "transition_scale of the samples of a single public hole (and, for multi-pack releases, any "
           "hole of another confirmation pack)")


def load_suite(path, packs):
    """load_manifest with the layout given explicitly while the record does not exist yet."""
    if "packs_per_body" in inspect.signature(load_manifest).parameters:
        return load_manifest(path, packs_per_body=packs)
    return load_manifest(path)


def _size_bin(fraction):
    if fraction < .10:
        return "small"
    if fraction < .20:
        return "medium"
    return "large"


def _thickness_bin(cells):
    if cells < 5:
        return "thin"
    if cells < 9:
        return "medium"
    return "thick"


def _density_label(holes):
    pair = tuple(holes)
    for name, bounds in (("lo", (3, 12)), ("mid", (12, 24)), ("hi", (24, 48))):
        if pair == bounds:
            return name
    return f"h{pair[0]}" if pair[0] == pair[1] else f"h{pair[0]}-{pair[1]}"


def collect(dataset, *, open_only=False):
    """Generated parents, checked against the plans fixed before generation."""
    dataset = Path(dataset).resolve()
    rsi, rsi_sha = load_generation_plan(dataset)
    if not (dataset / "confirmation-complete.json").is_file():
        raise ValueError("confirmation generation is incomplete")
    bodies, parts, plans = [], {}, {}
    for part in ("id", "ood"):
        base = dataset / part
        planned_splits = part_splits(rsi, part)
        active_splits = tuple((s, n) for s, n in planned_splits if not (open_only and s in SEALED_SPLITS))
        if not active_splits:
            continue
        if not (base / "completion.json").is_file():
            raise ValueError("parent generation is incomplete")
        if json.loads((base / "split-audit.json").read_text()).get("status") != "passed":
            raise ValueError("generated population failed its split audit")
        plan, _, timing = verify_part(base, rsi, rsi_sha, part, dataset / "generation-plan-receipt.json")
        mode = confirmation_mode(plan)
        packs_required = confirmation_pack_count(plan)
        if mode != rsi["confirmation_policy"]["mode"] or packs_required != rsi["confirmation_policy"].get("packs", 1):
            raise ValueError("parent plan confirmation policy differs from the release plan")
        plans[part], parts[part] = plan, dict(timing, confirmation_mode=mode)
        active_names = {s for s, _ in active_splits}
        entries = {f"{row['split']}-{row['index']:02}": row for row in plan["parents"] if row["split"] in active_names}
        accepted = [row for row in json.loads((base / "progress.json").read_text())["records"]
                    if row["status"] == "accepted" and row["parent"] in entries]
        if sorted(row["parent"] for row in accepted) != sorted(entries):
            raise ValueError("generation does not cover the parent plan")
        for item in accepted:
            name, entry = item["parent"], entries[item["parent"]]
            pack, confirmation = base / "packs" / name, base / "confirmation" / name
            manifest = json.loads((pack / "private-manifest.json").read_text())
            hidden = json.loads((confirmation / "private-manifest.json").read_text())
            if manifest["parent_digest"] != hidden["parent_digest"]:
                raise ValueError("confirmation parent differs")
            min_pub_ore = public_min_ore_holes(plan, entry["split"])
            obs_audit_path = base / "observation-audits" / (name + ".json")
            obs_receipt = None
            if obs_audit_path.is_file():
                obs_receipt = json.loads(obs_audit_path.read_text())
                check_discoverability_attempts(
                    obs_receipt, entry.get("observation_seeds", [entry["observation_seed"]]),
                    min_pub_ore, manifest["observation_seed"])
            elif "observation_seeds" in entry and manifest["observation_seed"] not in entry["observation_seeds"]:
                raise ValueError("public observation seed is not in the preassigned stream")
            min_conf_ore = confirmation_min_ore_holes(plan, entry["split"])
            receipt = json.loads((base / "confirmation-audits" / (name + ".json")).read_text())
            if packs_required == 1:
                hidden_seeds = hidden["observation_seed"]
                check_confirmation_attempts(receipt, entry, mode, hidden_seeds, packs=1, min_ore_holes=min_conf_ore)
                pack_manifests = [hidden]
                pack_dirs = [confirmation]
            else:
                pack_dirs = [confirmation / f"c{k}" for k in range(packs_required)]
                pack_manifests = [json.loads((d / "private-manifest.json").read_text()) for d in pack_dirs]
                for pm in pack_manifests:
                    if pm["parent_digest"] != manifest["parent_digest"]:
                        raise ValueError("confirmation sub-pack parent differs")
                hidden_seeds = [pm["observation_seed"] for pm in pack_manifests]
                check_confirmation_attempts(receipt, entry, mode, hidden_seeds,
                                            packs=packs_required, min_ore_holes=min_conf_ore)
            if entry["selected_modality"] not in MODALITIES or sorted(c["modality"] for c in manifest["cases"]) != sorted(MODALITIES):
                raise ValueError("each parent requires both generated observation packs")
            cases = {}
            for modality in MODALITIES:
                case = next(row for row in manifest["cases"] if row["modality"] == modality)
                hidden_case = next(row for row in hidden["cases"] if row["modality"] == modality)
                hidden_packs = None
                if packs_required > 1:
                    hidden_packs = [
                        d / next(row["case_id"] for row in pm["cases"] if row["modality"] == modality)
                        for d, pm in zip(pack_dirs, pack_manifests)
                    ]
                cases[modality] = dict(case_id=case["case_id"], public=pack / case["case_id"],
                                       hidden=confirmation / hidden_case["case_id"],
                                       hidden_packs=hidden_packs, truth=pack / case["truth"],
                                       public_manifest=manifest, hidden_manifests=pack_manifests)
            stream_info = dict(
                seed=hidden["observation_seed"],
                index=entry["confirmation_seeds"].index(hidden["observation_seed"]),
                attempts=receipt["attempts"])
            if packs_required > 1:
                stream_info["seeds"] = hidden_seeds
                stream_info["indices"] = [entry["confirmation_seeds"].index(s) for s in hidden_seeds]
            bodies.append(dict(
                name=name, part=part, split=entry["split"], family=entry["family"],
                index=entry["index"], noisy=entry["noisy"],
                coverage=entry.get("coverage", "uniform"), shift=entry.get("shift", "id"),
                selected=entry["selected_modality"], parent_id=manifest["parent_digest"],
                holes=manifest.get("holes", item.get("holes", [3, 12])),
                hole_count=manifest.get("hole_count"), ore_holes=manifest.get("ore_holes"),
                volume_fraction=item.get("volume_fraction"),
                median_vertical_thickness=item.get("median_vertical_thickness"),
                observation_receipt=obs_receipt,
                cases=cases, stream=stream_info))
    bodies.sort(key=lambda body: (SPLITS.index(body["split"]), body["index"]))
    return rsi, rsi_sha, parts, plans, bodies


def audit_release(dataset, bodies, mode, *, packs_required=1, plans=None):
    """Whole-hole and discoverability audit of every pack; any violation fails the release."""
    rows, selected, inter_pack_audits, discoverability_failures = [], {}, [], []
    for body in bodies:
        plan = (plans or {}).get(body["part"], {})
        min_pub_ore = public_min_ore_holes(plan, body["split"])
        min_conf_ore = confirmation_min_ore_holes(plan, body["split"])
        for modality, case in body["cases"].items():
            pack_paths = case["hidden_packs"] if case["hidden_packs"] is not None else [case["hidden"]]
            conf_problems = []
            for pack_idx, hidden_path in enumerate(pack_paths):
                public, confirm = pair_confirmation(case["public"], hidden_path, case["truth"])
                conf_problems.append(confirm)
                result = audit_holes(public, confirm)
                conf_man = case["hidden_manifests"][pack_idx] if pack_idx < len(case["hidden_manifests"]) else None
                c_ore = ore_hole_count(confirm, conf_man)
                if modality == body["selected"] and min_conf_ore > 0 and c_ore < min_conf_ore:
                    discoverability_failures.append(f"{body['name']}:c{pack_idx}")
                if modality == body["selected"] and pack_idx == 0:
                    selected[body["name"]] = dict(result, split=body["split"])
                row = dict(parent=body["name"], split=body["split"], modality=modality,
                           selected=modality == body["selected"],
                           **{k: v for k, v in result.items() if k != "nearest_hole_distances"})
                if packs_required > 1:
                    row["pack"] = f"c{pack_idx}"
                    row["ore_holes"] = c_ore
                rows.append(row)
            if modality == body["selected"] and min_pub_ore > 0:
                pub_prob = read_problem(case["public"])
                if ore_hole_count(pub_prob, case["public_manifest"]) < min_pub_ore:
                    discoverability_failures.append(f"{body['name']}:public")
            if modality == body["selected"] and len(conf_problems) > 1:
                for i in range(len(conf_problems)):
                    for j in range(i + 1, len(conf_problems)):
                        pair_res = audit_pack_pair(conf_problems[i], conf_problems[j])
                        inter_pack_audits.append(dict(parent=body["name"], split=body["split"],
                                                      pair=f"c{i}-c{j}", **pair_res))
    overlapping = sorted({row["parent"] for row in rows if row["status"] != "passed"}
                         | {row["parent"] for row in inter_pack_audits if row["status"] != "passed"})
    disc_failed = sorted(set(discoverability_failures))
    audits = list(selected.values())
    streams = []
    for body in bodies:
        stream_row = dict(
            parent=body["name"], split=body["split"], selected_stream=body["stream"]["seed"],
            stream_index=body["stream"]["index"],
            attempts=[{k: v for k, v in attempt.items() if k != "nearest_hole_distances"}
                      for attempt in body["stream"]["attempts"]])
        if "seeds" in body["stream"]:
            stream_row["selected_streams"] = body["stream"]["seeds"]
            stream_row["stream_indices"] = body["stream"]["indices"]
        streams.append(stream_row)
    first = [a for a in (body["stream"]["attempts"][0] for body in bodies)
             if "nearest_hole_distances" in a and "transition_scale" in a]
    attempts = {}
    for row in streams:
        attempts[str(len(row["attempts"]))] = attempts.get(str(len(row["attempts"])), 0) + 1
    redrawn = sorted(row["parent"] for row in streams
                     if (row.get("stream_indices", [row["stream_index"]]) != list(range(packs_required))))
    rule_table = CONFIRMATION_POLICIES_MULTI if packs_required > 1 else CONFIRMATION_POLICIES
    failed = bool(overlapping or disc_failed)
    report = dict(
        status="failed" if failed else "passed", criterion=OVERLAP, policy=mode,
        policy_rule=rule_table[mode], packs=rows, overlapping_parents=overlapping,
        nearest_hole_distance=hole_distance_summary(audits),
        nearest_hole_distance_by_split={split: hole_distance_summary([a for a in audits if a["split"] == split])
                                        for split in sorted({a["split"] for a in audits})},
        distance_definition="closest approach of each selected-pack confirmation hole to any public hole",
        confirmation_streams=streams, attempts_per_parent=attempts,
        redrawn_parents=redrawn,
        first_stream_nearest_hole_distance=hole_distance_summary(first) if len(first) == len(bodies) else None)
    if packs_required > 1:
        report["confirmation_packs_per_body"] = packs_required
        report["inter_pack_overlapping_parents"] = sorted({r["parent"] for r in inter_pack_audits if r["status"] != "passed"})
        report["inter_pack_nearest_hole_distance"] = (
            hole_distance_summary(inter_pack_audits) if inter_pack_audits else None)
        report["discoverability_failures"] = disc_failed
    (Path(dataset) / "hole-audit.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    if overlapping:
        raise ValueError(f"confirmation hole overlap in {len(overlapping)} parents; release not created: "
                         + ", ".join(overlapping))
    if disc_failed:
        raise ValueError("discoverability gate failed; release not created: " + ", ".join(disc_failed))
    return report


def write_case(root, body, modality, confirmation=None):
    case = body["cases"][modality]
    public, truth = root / "public" / case["case_id"], root / "private" / case["case_id"]
    shutil.copytree(case["public"], public)
    truth.mkdir(parents=True)
    shutil.copy2(case["truth"], truth / "truth.npy")
    names = list(FILES)
    if confirmation is None:
        shutil.copytree(case["hidden"], truth / "confirmation")
        if case["hidden_packs"] is not None:
            for k, pack_dir in enumerate(case["hidden_packs"]):
                shutil.copytree(pack_dir, truth / "confirmation" / f"c{k}")
                names.extend((f"confirmation/c{k}/problem.json", f"confirmation/c{k}/problem.npz"))
    else:
        save_problem(confirmation, truth / "confirmation")
    return {"case_id": case["case_id"], "parent_id": body["parent_id"], "modality": modality,
            "public": str(public.relative_to(root)), "truth": str(truth.relative_to(root)),
            "sha256": {name: file_hash((public if name.startswith("problem") else truth) / name) for name in names}}


def write_manifest(out, name, split, stage, seeds, cases):
    path = out / name / "manifest.json"
    path.write_text(json.dumps({"schema": "copper-suite-v2", "stage": stage, "split": split, "seeds": list(seeds),
                                "cases": cases}, sort_keys=True, indent=2) + "\n")
    load_suite(path, 2 if any(sum(c["parent_id"] == r["parent_id"] for c in cases) > 1 for r in cases) else 1)
    verify_geological_constraints(path)
    return str(path.relative_to(out))


def _body_geometry_stats(body):
    case = body["cases"][body["selected"]]
    problem = read_problem(case["public"])
    mask = np.load(case["truth"], allow_pickle=False)
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    frac = body["volume_fraction"]
    if frac is None:
        frac = float(mask.sum() / feasible.sum())
    thick = body["median_vertical_thickness"]
    if thick is None:
        runs = vertical_ore_runs(mask)
        thick = float(np.median(runs)) if len(runs) else 0.
    drill = next(o for o in problem["observations"] if o["name"] == "drill")
    hole_count = body["hole_count"] if body["hole_count"] is not None else int(len(np.unique(drill["groups"])))
    pub_ore = ore_hole_count(problem, case["public_manifest"])
    pack_paths = case["hidden_packs"] if case["hidden_packs"] is not None else [case["hidden"]]
    conf_holes, conf_ore = [], []
    for idx, hp in enumerate(pack_paths):
        cp = read_problem(hp)
        cd = next(o for o in cp["observations"] if o["name"] == "drill")
        conf_holes.append(int(len(np.unique(cd["groups"]))))
        cm = case["hidden_manifests"][idx] if idx < len(case["hidden_manifests"]) else None
        conf_ore.append(ore_hole_count(cp, cm))
    return dict(
        case_id=case["case_id"], shape=list(problem["shape"]), cells=int(np.prod(problem["shape"])),
        holes=list(body["holes"]), hole_count=int(hole_count), ore_holes=int(pub_ore),
        volume_fraction=float(frac), median_vertical_thickness=float(thick),
        conf_holes=conf_holes, conf_ore_holes=conf_ore)


def _write_strata_and_card(output, rsi, rsi_sha, parts, bodies, manifests, hashes, hole, stage, open_only, sub_release):
    open_bodies = [b for b in bodies if b["split"] in OPEN_SPLITS]
    stats_by_name = {b["name"]: _body_geometry_stats(b) for b in open_bodies}
    strata_path = output / "strata.csv"
    fieldnames = [
        "case_id", "split", "family", "shift", "density", "holes_min", "holes_max", "hole_count", "ore_holes",
        "coverage", "noise", "modality", "grid_cells", "grid_shape",
        "volume_fraction", "size_bin", "median_vertical_thickness", "thickness_bin",
    ]
    with strata_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for b in open_bodies:
            st = stats_by_name[b["name"]]
            writer.writerow(dict(
                case_id=st["case_id"], split=b["split"], family=b["family"], shift=b["shift"],
                density=_density_label(st["holes"]), holes_min=st["holes"][0], holes_max=st["holes"][1],
                hole_count=st["hole_count"], ore_holes=st["ore_holes"],
                coverage=b["coverage"], noise="noisy" if b["noisy"] else "clean", modality=b["selected"],
                grid_cells=st["cells"], grid_shape="x".join(str(v) for v in st["shape"]),
                volume_fraction=f"{st['volume_fraction']:.6f}", size_bin=_size_bin(st["volume_fraction"]),
                median_vertical_thickness=f"{st['median_vertical_thickness']:.2f}",
                thickness_bin=_thickness_bin(st["median_vertical_thickness"]),
            ))
    judges_doc = {
        "standard": {"objective": "confirmation", "pack": "c0"},
        "q25": {"objective": "confirmation", "train_manifest": manifests["train_s_q25"],
                "validation_manifest": manifests["validation_q25"]},
        "x4": {"objective": "confirmation", "train_manifest": manifests["train_s_x4"],
               "full_train_manifest": manifests["train_x4"]},
        "noisy25": {"objective": "confirmation", "resample": {"fraction": .25}},
        "oracle": {"objective": "oracle_iou"},
        "legacy_v2": {"objective": "confirmation", "legacy_objective_version": "objective_v2"},
    }
    judges_path = output / "judges.json"
    judges_path.write_text(json.dumps(judges_doc, sort_keys=True, indent=2) + "\n")

    def summarize(items):
        if not items:
            return None
        def dist(vals):
            arr = np.asarray(vals, dtype=float)
            return dict(mean=round(float(arr.mean()), 4), min=round(float(arr.min()), 4),
                        median=round(float(np.median(arr)), 4), max=round(float(arr.max()), 4))
        sts = [stats_by_name[b["name"]] for b in items]
        return dict(
            bodies=len(items),
            volume_fraction=dist([s["volume_fraction"] for s in sts]),
            median_vertical_thickness=dist([s["median_vertical_thickness"] for s in sts]),
            public_holes=dist([s["hole_count"] for s in sts]),
            public_ore_holes=dist([s["ore_holes"] for s in sts]),
            confirmation_holes_per_pack=dist([h for s in sts for h in s["conf_holes"]]),
            confirmation_ore_holes_per_pack=dist([h for s in sts for h in s["conf_ore_holes"]]),
        )

    id_holes = next((stats_by_name[b["name"]]["holes"] for b in open_bodies if b["part"] == "id"), [3, 12])
    generated_counts = {s: sum(b["split"] == s for b in bodies) for s in rsi["counts"]}
    sealed_hashes = {m: hashes[m] for m in ("test_id/manifest.json", "test_ood/manifest.json") if m in hashes}
    obs_redrawn = sum(1 for b in open_bodies if (b.get("observation_receipt") or {}).get("stream_index", 0) > 0)
    card = dict(
        schema="copper-bench-card-v1",
        release=rsi["release"],
        sub_release=sub_release or rsi["release"],
        version="2.0.0",
        stage=stage,
        status="candidate_density_open_only" if open_only else ("qualified" if stage == "formal" else "development_only"),
        generator_commit=code_commit(),
        density=dict(
            label=_density_label(id_holes),
            public_holes=list(id_holes),
            confirmation_holes=[3, 12],
            confirmation_packs=rsi["confirmation_policy"].get("packs", 1)),
        counts=dict(
            planned=dict(rsi["counts"]),
            generated=generated_counts,
            tiers={k: (len({c["parent_id"] for c in json.loads((output / m).read_text())["cases"]}))
                   for k, m in (("S", manifests.get("train_s", manifests["train"])),
                                *([("M", manifests["train_m"])] if "train_m" in manifests else []),
                                ("F", manifests["train"]))}),
        gates=dict(
            size_gates=rsi.get("size_gates"),
            discoverability_policy=rsi.get("discoverability_policy"),
            confirmation_policy={k: v for k, v in rsi["confirmation_policy"].items() if k != "seed_derivation"}),
        strata_levels=dict(
            family=dict(id=list(PRESETS["rsi-r2"]["historical"] and ("folded", "lens", "multiple", "faulted")),
                        ood=["faulted_lens"]),
            density=dict(lo=[3, 12], mid=[12, 24], hi=[24, 48]),
            coverage=dict(id=["uniform"], ood=["clustered"]),
            noise=["clean", "noisy"],
            modality=list(MODALITIES),
            size_bins=dict(small="[0.04, 0.10)", medium="[0.10, 0.20)", large="[0.20, 0.35]"),
            thickness_bins=dict(thin="[3, 5)", medium="[5, 9)", thick="[9, inf)")),
        open_split_aggregates=dict(
            by_split={s: summarize([b for b in open_bodies if b["split"] == s])
                      for s in OPEN_SPLITS if any(b["split"] == s for b in open_bodies)},
            by_id_family={f: summarize([b for b in open_bodies if b["part"] == "id" and b["family"] == f])
                          for f in ("folded", "lens", "multiple", "faulted")
                          if any(b["part"] == "id" and b["family"] == f for b in open_bodies)},
            public_discoverability_redrawn_parents=obs_redrawn,
            hole_audit=dict(
                status=hole["status"],
                overlapping_parents=len(hole["overlapping_parents"]),
                inter_pack_overlapping_parents=len(hole.get("inter_pack_overlapping_parents", [])),
                confirmation_redrawn_parents=len(hole["redrawn_parents"]),
                nearest_public_hole_distance_median_m=round(hole["nearest_hole_distance"]["distance"][3], 4),
                nearest_inter_pack_distance_median_m=(
                    round(hole["inter_pack_nearest_hole_distance"]["distance"][3], 4)
                    if hole.get("inter_pack_nearest_hole_distance") else None))),
        hashes=dict(
            generation_plan_sha256=rsi_sha,
            part_plan_sha256={part: value["plan_sha256"] for part, value in parts.items()},
            manifest_sha256=dict(hashes),
            evidence_sha256={str(p.relative_to(output)): file_hash(p) for p in sorted((output / "evidence").glob("*.json"))},
            strata_csv_sha256=file_hash(strata_path),
            judges_json_sha256=file_hash(judges_path)),
        sealed_commitments=dict(
            generation_plan_sha256=rsi_sha,
            part_plan_sha256={part: value["plan_sha256"] for part, value in parts.items()},
            sealed_splits_generated=not open_only,
            sealed_manifest_sha256=sealed_hashes,
            seed_disclosure="withheld_until_retirement"),
    )
    card_path = output / "bench-card.json"
    card_path.write_text(json.dumps(card, sort_keys=True, indent=2) + "\n")
    return dict(strata_csv=file_hash(strata_path), judges_json=file_hash(judges_path), bench_card=file_hash(card_path))


def build(dataset, output, calibration=None, isolation=None, geological_constraints=None, *,
          open_only=False, sub_release=None):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("release destination already exists")
    rsi, rsi_sha, parts, plans, bodies = collect(dataset, open_only=open_only)
    formal = rsi["stage"] == "formal" and not open_only
    active_splits = tuple(s for part in ("id", "ood") for s, _ in part_splits(rsi, part)
                          if not (open_only and s in SEALED_SPLITS))
    expected_counts = {s: rsi["counts"][s] for s in active_splits}
    counts = {split: sum(body["split"] == split for body in bodies) for split in active_splits}
    if counts != expected_counts:
        raise ValueError("parent counts differ from the release plan")
    balance = modality_balance([dict(split=b["split"], family=b["family"], selected_modality=b["selected"])
                                for b in bodies])
    evidence = {}
    if formal:
        if None in (calibration, isolation, geological_constraints):
            raise ValueError("a formal release requires calibration, isolation and geological-constraint evidence")
        evidence.update(qualify_evidence(dataset, calibration, isolation, geological_constraints))
    packs_required = rsi["confirmation_policy"].get("packs", 1)
    hole = audit_release(dataset, bodies, rsi["confirmation_policy"]["mode"],
                         packs_required=packs_required, plans=plans)
    current = {shape_fingerprint(np.load(body["cases"]["drill"]["truth"], allow_pickle=False)) for body in bodies}
    if formal or rsi["historical"]:
        missing = set(PRESETS[rsi["release"]]["historical"]) - set(rsi["historical"])
        if missing:
            raise ValueError("historical exclusion lacks " + ", ".join(sorted(missing)))
        if "rsi_dev" in rsi["historical"]:
            source = Path(rsi["historical"]["rsi_dev"])
            if not ((source / "development-receipt.json").is_file() or
                    json.loads((source / "generation-plan.json").read_text()).get("release") == "rsi-dev"):
                raise ValueError("rsi_dev historical source is not an rsi-dev release or dataset")
        evidence["historical_disjointness"] = historical_exclusion(rsi["historical"], current)
    by_split = {split: [b for b in bodies if b["split"] == split] for split in active_splits}
    if rsi["short_tier"]["per_family"] is None:
        short = by_split["train"]
    else:
        names = {f"train-{row['index']:02}" for row in plans["id"]["short_train"]}
        short = [b for b in by_split["train"] if b["name"] in names]
        if len(short) != len(names):
            raise ValueError("short-tier parents were not generated")
    medium = None
    if "medium_tier" in rsi and rsi["medium_tier"]["per_family"] is not None:
        med_names = {f"train-{row['index']:02}" for row in plans["id"]["medium_train"]}
        medium = [b for b in by_split["train"] if b["name"] in med_names]
        if len(medium) != len(med_names):
            raise ValueError("medium-tier parents were not generated")
    stage = "formal" if formal else "development"
    seeds = rsi["layout"]["execution_seeds"]
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    manifests, rows = {}, {}
    for split in active_splits:
        rows[split] = [write_case(output / split, b, b["selected"]) for b in by_split[split]]
        manifests[split] = write_manifest(output, split, split, stage, seeds, rows[split])
    short_name = "train_s" if rsi["short_tier"]["per_family"] is not None else "train"
    if short_name == "train_s":
        manifests["train_s"] = write_manifest(output, "train_s", "train", stage, seeds,
                                              [write_case(output / "train_s", b, b["selected"]) for b in short])
    if medium is not None:
        manifests["train_m"] = write_manifest(output, "train_m", "train", stage, seeds,
                                              [write_case(output / "train_m", b, b["selected"]) for b in medium])
    # Weak judge: whole holes and outcrop cells of the selected pack's c0 confirmation.
    q25 = rsi["judges"]["q25"]
    thinning = []
    for name, split, members in (("train_s_q25", "train", short), ("validation_q25", "validation", by_split["validation"])):
        cases = []
        for body in members:
            case = body["cases"][body["selected"]]
            full = read_problem(case["hidden"])
            seed = derive_seed(q25["seed"], case["case_id"])
            thinned, chosen = thin_confirmation(full, seed, q25["fraction"], q25["outcrop_grid"])
            summary = check_thinning(full, thinned, chosen, q25["fraction"], q25["outcrop_grid"])
            thinning.append(dict(manifest=name, parent=body["name"], case_id=case["case_id"], seed=seed,
                                 selection=chosen, counts=summary))
            cases.append(write_case(output / name, body, body["selected"], thinned))
        manifests[name] = write_manifest(output, name, split, stage, seeds, cases)
    if packs_required > 1:
        for name, members in (("train_s_x4", short), ("train_x4", by_split["train"])):
            x4_cases = []
            for body in members:
                case = body["cases"][body["selected"]]
                merged = merge_confirmations([read_problem(p) for p in case["hidden_packs"]])
                x4_cases.append(write_case(output / name, body, body["selected"], merged))
            manifests[name] = write_manifest(output, name, "train", stage, seeds, x4_cases)
    standard = {row["case_id"]: row["sha256"] for split in ("train", "validation") for row in rows[split]}
    derived_names = ["train_s_q25", "validation_q25"] + (["train_s_x4", "train_x4"] if packs_required > 1 else [])
    for name in derived_names:
        for row in json.loads((output / manifests[name]).read_text())["cases"]:
            if any(row["sha256"][key] != standard[row["case_id"]][key] for key in ("problem.json", "problem.npz", "truth.npy")):
                raise ValueError("derived judge public observations differ from the standard manifest")
    # Q1: repeated execution seeds on the short tier, and both packs on all training parents.
    q1 = dict(execution_seeds=rsi["q1"]["execution_seeds"])
    manifests["train_s_q1seeds"] = write_manifest(output, "train_s_q1seeds", "train", stage, q1["execution_seeds"],
                                                  [write_case(output / "train_s_q1seeds", b, b["selected"]) for b in short])
    q1["execution_seeds_manifest"] = manifests["train_s_q1seeds"]
    q1["both_packs_manifest"] = None
    if rsi["q1"]["both_packs_manifest"]:
        manifests["train_q1_both"] = write_manifest(output, "train_q1_both", "train", stage, seeds,
                                                    [write_case(output / "train_q1_both", b, m)
                                                     for b in by_split["train"] for m in MODALITIES])
        q1["both_packs_manifest"] = manifests["train_q1_both"]
    unselected = []
    for body in bodies:
        modality = next(m for m in MODALITIES if m != body["selected"])
        case = body["cases"][modality]
        target = output / "unselected" / body["split"] / case["case_id"]
        shutil.copytree(case["public"], target / "public")
        shutil.copytree(case["hidden"], target / "confirmation")
        if case["hidden_packs"] is not None:
            for k, pack_dir in enumerate(case["hidden_packs"]):
                shutil.copytree(pack_dir, target / "confirmation" / f"c{k}")
        unselected.append(dict(parent_id=body["parent_id"], split=body["split"], modality=modality,
                               case_id=case["case_id"], sha256={str(p.relative_to(target)): file_hash(p)
                                                                for p in sorted(target.rglob("*")) if p.is_file()}))
    (output / "unselected" / "index.json").write_text(json.dumps(dict(
        schema="rsi-unselected-packs-v1", scope="generated and archived; not an evaluation manifest",
        packs=unselected), sort_keys=True, indent=2) + "\n")
    sealed_list = [] if open_only else [manifests["test_id"], manifests["test_ood"]]
    common = dict(validation_manifest=manifests["validation"], sealed_manifests=sealed_list)
    tiers = {"S": dict(solver_seconds=rsi["tiers"]["S"]["solver_seconds"], train_manifest=manifests[short_name],
                       train_bodies=len(short), slots=rsi["tiers"]["S"]["slots"], **common),
             "F": dict(solver_seconds=rsi["tiers"]["F"]["solver_seconds"], train_manifest=manifests["train"],
                       train_bodies=len(by_split["train"]), slots=rsi["tiers"]["F"]["slots"], **common)}
    if medium is not None and "M" in rsi["tiers"]:
        tiers["M"] = dict(solver_seconds=rsi["tiers"]["M"]["solver_seconds"], train_manifest=manifests["train_m"],
                          train_bodies=len(medium), slots=rsi["tiers"]["M"]["slots"], **common)
    layout = {key: rsi["layout"][key] for key in ("packs_per_body", "execution_seeds", "assignment_rule", "sealed_bodies")}
    layout["packs_per_body_by_manifest"] = {q1["both_packs_manifest"]: 2} if q1["both_packs_manifest"] else {}
    q1["solver_seconds"] = rsi["tiers"]["S"]["solver_seconds"]
    judges = {"standard": {"objective": "confirmation"},
              "q25": {"objective": "confirmation", "train_manifest": manifests["train_s_q25"],
                      "validation_manifest": manifests["validation_q25"]},
              "oracle": {"objective": "oracle_iou"}}
    if packs_required > 1:
        judges["x4"] = {"objective": "confirmation", "train_manifest": manifests["train_s_x4"]}
    evidence["confirmation"] = dict(
        status="passed", cases=sum(len(b["cases"]) for b in bodies),
        packs_per_body=packs_required, policy=rsi["confirmation_policy"]["mode"],
        scope="independent whole-hole/location and noise streams; geometry/masks/priors identical; not a new geology parent")
    evidence["hole_audit"] = hole
    layout_ev = dict(
        status="passed", rule=rsi["layout"]["assignment_rule"], allocation_seed=rsi["layout"]["allocation_seed"],
        generation_plan_sha256=rsi_sha, plans=parts, balance=balance,
        short_tier=dict(rule=rsi["short_tier"]["rule"], per_family=rsi["short_tier"]["per_family"],
                        parents=[b["name"] for b in short]),
        assignment=[dict(parent=b["name"], split=b["split"], family=b["family"], index=b["index"],
                         noisy=b["noisy"], selected_modality=b["selected"]) for b in bodies])
    if medium is not None:
        layout_ev["medium_tier"] = dict(rule=rsi["medium_tier"]["rule"], per_family=rsi["medium_tier"]["per_family"],
                                        parents=[b["name"] for b in medium])
    evidence["layout"] = layout_ev
    evidence["weak_judge"] = dict(status="passed", rule=q25["rule"], fraction=q25["fraction"], seed=q25["seed"],
                                  outcrop_grid=q25["outcrop_grid"], cases=thinning)
    (output / "evidence").mkdir()
    for name, value in evidence.items():
        (output / "evidence" / f"{name}.json").write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    hashes = {path: file_hash(output / path) for path in manifests.values()}
    extra_artifacts = None
    if rsi["release"] == "rsi-r2":
        extra_artifacts = _write_strata_and_card(
            output, rsi, rsi_sha, parts, bodies, manifests, hashes, hole, stage, open_only, sub_release)
    sealed = dict(release=rsi["release"], generation_plan_sha256=rsi_sha,
                  plan_sha256={part: value["plan_sha256"] for part, value in parts.items()},
                  manifest_sha256=hashes, counts=counts, tiers=tiers, layout=layout, judges=judges, q1=q1,
                  evidence_sha256={str(p.relative_to(output)): file_hash(p) for p in (output / "evidence").glob("*.json")},
                  unselected_sha256={"unselected/index.json": file_hash(output / "unselected" / "index.json")})
    if extra_artifacts is not None:
        sealed["artifacts_sha256"] = extra_artifacts
    verify_tiers(sealed, output, allow_unsealed=open_only)
    if formal:
        record = dict(sealed, schema="copper-qualified-synthetic-v2", status="qualified", evaluator_uid=os.geteuid(),
                      limitations=["Synthetic benchmark only; no real mine validation",
                                   "Empirical discretization calibration, not an error-bound theorem",
                                   "Not secure against host administrators or trusted Docker operator"])
        (output / "qualification.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    else:
        record = dict(sealed, status="candidate_density_open_only" if open_only else "development_only",
                      scope="synthetic GemPy geology; open splits only (sealed splits deferred)"
                      if open_only else "synthetic GemPy geology; not field validation or formal release",
                      manifest_files=hashes, allocation_plan_sha256=sealed["plan_sha256"],
                      manifest_sha256={str(Path(path).parent): value for path, value in hashes.items()},
                      confirmation_hole_audits=hole["packs"])
        (output / "development-receipt.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    for path in output.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o444)
    output.chmod(0o700)
    if formal:
        for path in manifests.values():
            verify_release(output / path)
    return dict(status="qualified" if formal else ("candidate_density_open_only" if open_only else "development_only"),
                release=rsi["release"], counts=counts,
                manifests=sorted(manifests.values()), hole_audit=hole["status"])


def complete_sealed(dataset, output, generator_python, *, holes, ood_holes=None, workers=8,
                    calibration=None, isolation=None, geological_constraints=None, sub_release=None):
    """Generate deferred sealed splits (`test_id`, `test_ood`), run QA + confirmations, and release."""
    dataset = Path(dataset).resolve()
    rsi, _ = load_generation_plan(dataset)
    h_lo, h_hi = holes
    oh_lo, oh_hi = ood_holes if ood_holes is not None else ((3, 12) if (h_lo, h_hi) != (3, 12) else (24, 48))
    for part, split_name, extra in (
        ("id", "test_id", []),
        ("ood", "test_ood", ["--ood-for", str(dataset / "id"), "--ood-holes", str(oh_lo), str(oh_hi)]),
    ):
        subprocess.run([
            sys.executable, "-m", "tasks.copper_inversion.experiments.v2.generate_pilot",
            "--output", str(dataset / part), "--generator-python", str(generator_python),
            "--seed", str(rsi[part + "_seed"]), "--workers", str(workers),
            "--holes", str(h_lo), str(h_hi), "--splits", split_name, "--resume-planned", *extra
        ], check=True)
    qa_path = dataset / "population-qa.json"
    if qa_path.exists():
        qa_path.rename(dataset / "population-qa-open.json")
    subprocess.run([
        sys.executable, "-m", "tasks.copper_inversion.experiments.v2.qualify_population",
        "--root", str(dataset), "--output", str(qa_path)
    ], check=True)
    subprocess.run([
        sys.executable, "-m", "tasks.copper_inversion.experiments.v2.generate_confirmation",
        "--root", str(dataset), "--python", str(generator_python), "--workers", str(workers)
    ], check=True)
    (dataset / "qa-pipeline-complete.json").write_text(json.dumps(
        {"status": "population_and_confirmation_generated", "formal_release": False}) + "\n")
    return build(dataset, output, calibration, isolation, geological_constraints,
                 open_only=False, sub_release=sub_release)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="fix and hash a release generation plan before generation")
    plan.add_argument("--root", type=Path, required=True)
    plan.add_argument("--release", choices=sorted(PRESETS), required=True)
    plan.add_argument("--master-seed", type=int, required=True)
    plan.add_argument("--confirmation-policy", choices=sorted(CONFIRMATION_POLICIES), default="resample_until_separated")
    plan.add_argument("--historical", nargs="*", default=[], metavar="LABEL=PATH",
                      help="prior releases or generator archives; rsi-r1/r2 require legacy_release and rsi_dev")
    plan.add_argument("--no-q1-both-packs", action="store_true")
    plan.add_argument("--tiers", type=json.loads, metavar="JSON",
                      help='development only, e.g. a one-round smoke: {"S": {"solver_seconds": 10, "slots": 5}, ...}')
    release = commands.add_parser("release", help="check a generated population and write the release")
    release.add_argument("--dataset", type=Path, required=True)
    release.add_argument("--output", type=Path, required=True)
    release.add_argument("--open-only", action="store_true",
                         help="package open splits only while holding off on sealed test splits")
    release.add_argument("--sub-release", help="optional sub-release identifier for bench-card.json")
    for name in ("calibration", "isolation", "geological-constraints"):
        release.add_argument("--" + name, type=Path, help="required for a formal release")
    complete = commands.add_parser("complete-sealed",
                                   help="generate held-off sealed test splits on a chosen density dataset and qualify")
    complete.add_argument("--dataset", type=Path, required=True)
    complete.add_argument("--output", type=Path, required=True)
    complete.add_argument("--python", type=Path, required=True, help="GemPy generator python")
    complete.add_argument("--holes", nargs="+", required=True)
    complete.add_argument("--ood-holes", nargs="+")
    complete.add_argument("--workers", type=int, default=8)
    complete.add_argument("--sub-release")
    for name in ("calibration", "isolation", "geological-constraints"):
        complete.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if os.geteuid() != 991:
        raise PermissionError("releases require the isolated evaluator account")
    if args.command == "plan":
        historical = dict(item.split("=", 1) for item in args.historical)
        value = make_generation_plan(args.release, args.master_seed, args.confirmation_policy, historical,
                                     tiers=args.tiers)
        value["q1"]["both_packs_manifest"] = not args.no_q1_both_packs
        sha = write_generation_plan(args.root, value)
        print(json.dumps({"generation_plan_sha256": sha, "release": args.release,
                          "id_seed": value["id_seed"], "ood_seed": value["ood_seed"]}, sort_keys=True))
    elif args.command == "release":
        print(json.dumps(build(args.dataset, args.output, args.calibration, args.isolation,
                               args.geological_constraints, open_only=args.open_only,
                               sub_release=args.sub_release), sort_keys=True))
    else:
        holes = _parse_cli_holes(args.holes)
        ood_holes = _parse_cli_holes(args.ood_holes) if args.ood_holes else None
        print(json.dumps(complete_sealed(
            args.dataset, args.output, args.python, holes=holes, ood_holes=ood_holes,
            workers=args.workers, calibration=args.calibration, isolation=args.isolation,
            geological_constraints=args.geological_constraints, sub_release=args.sub_release), sort_keys=True))


if __name__ == "__main__":
    main()
