"""Private RSI data releases (`rsi-dev`, `rsi-r1`), one observation layout per parent.

`plan` fixes the release plan and records its hash before anything is generated.
Generation reuses `generate_independent`/`generate_pilot` (which pick the plan up
from the dataset root) and `generate_confirmation`. `release` checks the plan
receipts, the layout rules, the whole-hole audit and, for formal releases, the
qualification evidence, then writes every manifest and the qualification record.
The unselected pack of each parent is archived under `unselected/`, outside
every evaluation manifest (except the optional Q1 both-pack training manifest).
"""

import argparse
import inspect
import json
import os
import shutil
from pathlib import Path

import numpy as np

from tasks.copper_inversion.experiments.v2.release_independent import (historical_exclusion, pair_confirmation,
                                                                       qualify_evidence)
from tasks.copper_inversion.experiments.v2.split_audit import shape_fingerprint
from tasks.copper_inversion.v2.data import read_problem, save_problem
from tasks.copper_inversion.v2.qualification import verify_geological_constraints, verify_release, verify_tiers
from tasks.copper_inversion.v2.suite import file_hash, load_manifest

from .measurements import (ASSIGNMENT_RULE, CONFIRMATION_POLICIES, MODALITIES, audit_holes, check_confirmation_attempts,
                           check_thinning, confirmation_mode, hole_distance_summary, modality_balance,
                           thin_confirmation)
from .plan import PRESETS, derive_seed, load_generation_plan, make_generation_plan, verify_part, write_generation_plan


SPLITS = ("train", "validation", "test_id", "test_ood")
FILES = ("problem.json", "problem.npz", "truth.npy", "confirmation/problem.json", "confirmation/problem.npz")
OVERLAP = ("a confirmation hole overlaps when at least half of its samples lie within one confirmation "
           "transition_scale of the samples of a single public hole")


def load_suite(path, packs):
    """load_manifest with the layout given explicitly while the record does not exist yet."""
    if "packs_per_body" in inspect.signature(load_manifest).parameters:
        return load_manifest(path, packs_per_body=packs)
    return load_manifest(path)


def collect(dataset):
    """Generated parents, checked against the plans fixed before generation."""
    dataset = Path(dataset).resolve()
    rsi, rsi_sha = load_generation_plan(dataset)
    if not (dataset / "confirmation-complete.json").is_file():
        raise ValueError("confirmation generation is incomplete")
    bodies, parts, plans = [], {}, {}
    for part in ("id", "ood"):
        base = dataset / part
        if not (base / "completion.json").is_file():
            raise ValueError("parent generation is incomplete")
        if json.loads((base / "split-audit.json").read_text()).get("status") != "passed":
            raise ValueError("generated population failed its split audit")
        plan, _, timing = verify_part(base, rsi, rsi_sha, part, dataset / "generation-plan-receipt.json")
        mode = confirmation_mode(plan)
        if mode != rsi["confirmation_policy"]["mode"]:
            raise ValueError("parent plan confirmation policy differs from the release plan")
        plans[part], parts[part] = plan, dict(timing, confirmation_mode=mode)
        entries = {f"{row['split']}-{row['index']:02}": row for row in plan["parents"]}
        accepted = [row for row in json.loads((base / "progress.json").read_text())["records"]
                    if row["status"] == "accepted"]
        if sorted(row["parent"] for row in accepted) != sorted(entries):
            raise ValueError("generation does not cover the parent plan")
        for item in accepted:
            name, entry = item["parent"], entries[item["parent"]]
            pack, confirmation = base / "packs" / name, base / "confirmation" / name
            manifest = json.loads((pack / "private-manifest.json").read_text())
            hidden = json.loads((confirmation / "private-manifest.json").read_text())
            if manifest["parent_digest"] != hidden["parent_digest"]:
                raise ValueError("confirmation parent differs")
            receipt = json.loads((base / "confirmation-audits" / (name + ".json")).read_text())
            check_confirmation_attempts(receipt, entry, mode, hidden["observation_seed"])
            if entry["selected_modality"] not in MODALITIES or sorted(c["modality"] for c in manifest["cases"]) != sorted(MODALITIES):
                raise ValueError("each parent requires both generated observation packs")
            cases = {}
            for modality in MODALITIES:
                case = next(row for row in manifest["cases"] if row["modality"] == modality)
                hidden_case = next(row for row in hidden["cases"] if row["modality"] == modality)
                cases[modality] = dict(case_id=case["case_id"], public=pack / case["case_id"],
                                       hidden=confirmation / hidden_case["case_id"], truth=pack / case["truth"])
            bodies.append(dict(name=name, part=part, split=entry["split"], family=entry["family"],
                               index=entry["index"], noisy=entry["noisy"], selected=entry["selected_modality"],
                               parent_id=manifest["parent_digest"], cases=cases,
                               stream=dict(seed=hidden["observation_seed"],
                                           index=entry["confirmation_seeds"].index(hidden["observation_seed"]),
                                           attempts=receipt["attempts"])))
    bodies.sort(key=lambda body: (SPLITS.index(body["split"]), body["index"]))
    return rsi, rsi_sha, parts, plans, bodies


def audit_release(dataset, bodies, mode):
    """Whole-hole audit of every pack; any overlap fails the release."""
    rows, selected = [], {}
    for body in bodies:
        for modality, case in body["cases"].items():
            public, confirm = pair_confirmation(case["public"], case["hidden"], case["truth"])
            result = audit_holes(public, confirm)
            if modality == body["selected"]:
                selected[body["name"]] = dict(result, split=body["split"])
            rows.append(dict(parent=body["name"], split=body["split"], modality=modality,
                             selected=modality == body["selected"],
                             **{k: v for k, v in result.items() if k != "nearest_hole_distances"}))
    overlapping = sorted({row["parent"] for row in rows if row["status"] != "passed"})
    audits = list(selected.values())
    # Which presampled stream each body uses and every generation-time audit
    # attempt, so the appendix can state the confirmation rule and its effect.
    streams = [dict(parent=body["name"], split=body["split"], selected_stream=body["stream"]["seed"],
                    stream_index=body["stream"]["index"], attempts=[
                        {k: v for k, v in attempt.items() if k != "nearest_hole_distances"}
                        for attempt in body["stream"]["attempts"]]) for body in bodies]
    first = [a for a in (body["stream"]["attempts"][0] for body in bodies)
             if "nearest_hole_distances" in a and "transition_scale" in a]
    attempts = {}
    for row in streams:
        attempts[str(len(row["attempts"]))] = attempts.get(str(len(row["attempts"])), 0) + 1
    report = dict(status="failed" if overlapping else "passed", criterion=OVERLAP, policy=mode,
                  policy_rule=CONFIRMATION_POLICIES[mode], packs=rows, overlapping_parents=overlapping,
                  nearest_hole_distance=hole_distance_summary(audits),
                  nearest_hole_distance_by_split={split: hole_distance_summary([a for a in audits if a["split"] == split])
                                                  for split in sorted({a["split"] for a in audits})},
                  distance_definition="closest approach of each selected-pack confirmation hole to any public hole",
                  confirmation_streams=streams, attempts_per_parent=attempts,
                  redrawn_parents=sorted(row["parent"] for row in streams if row["stream_index"]),
                  first_stream_nearest_hole_distance=hole_distance_summary(first) if len(first) == len(bodies) else None)
    (Path(dataset) / "hole-audit.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    if overlapping:
        raise ValueError(f"confirmation hole overlap in {len(overlapping)} parents; release not created: "
                         + ", ".join(overlapping))
    return report


def write_case(root, body, modality, confirmation=None):
    case = body["cases"][modality]
    public, truth = root / "public" / case["case_id"], root / "private" / case["case_id"]
    shutil.copytree(case["public"], public)
    truth.mkdir(parents=True)
    shutil.copy2(case["truth"], truth / "truth.npy")
    if confirmation is None:
        shutil.copytree(case["hidden"], truth / "confirmation")
    else:
        save_problem(confirmation, truth / "confirmation")
    return {"case_id": case["case_id"], "parent_id": body["parent_id"], "modality": modality,
            "public": str(public.relative_to(root)), "truth": str(truth.relative_to(root)),
            "sha256": {name: file_hash((public if name.startswith("problem") else truth) / name) for name in FILES}}


def write_manifest(out, name, split, stage, seeds, cases):
    path = out / name / "manifest.json"
    path.write_text(json.dumps({"schema": "copper-suite-v2", "stage": stage, "split": split, "seeds": list(seeds),
                                "cases": cases}, sort_keys=True, indent=2) + "\n")
    load_suite(path, 2 if any(sum(c["parent_id"] == r["parent_id"] for c in cases) > 1 for r in cases) else 1)
    verify_geological_constraints(path)
    return str(path.relative_to(out))


def build(dataset, output, calibration=None, isolation=None, geological_constraints=None):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("release destination already exists")
    rsi, rsi_sha, parts, plans, bodies = collect(dataset)
    formal = rsi["stage"] == "formal"
    counts = {split: sum(body["split"] == split for body in bodies) for split in SPLITS}
    if counts != rsi["counts"]:
        raise ValueError("parent counts differ from the release plan")
    balance = modality_balance([dict(split=b["split"], family=b["family"], selected_modality=b["selected"])
                                for b in bodies])
    evidence = {}
    if formal:
        if None in (calibration, isolation, geological_constraints):
            raise ValueError("a formal release requires calibration, isolation and geological-constraint evidence")
        evidence.update(qualify_evidence(dataset, calibration, isolation, geological_constraints))
    hole = audit_release(dataset, bodies, rsi["confirmation_policy"]["mode"])
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
    by_split = {split: [b for b in bodies if b["split"] == split] for split in SPLITS}
    if rsi["short_tier"]["per_family"] is None:
        short = by_split["train"]
    else:
        names = {f"train-{row['index']:02}" for row in plans["id"]["short_train"]}
        short = [b for b in by_split["train"] if b["name"] in names]
        if len(short) != len(names):
            raise ValueError("short-tier parents were not generated")
    stage = "formal" if formal else "development"
    seeds = rsi["layout"]["execution_seeds"]
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    manifests, rows = {}, {}
    for split in SPLITS:
        rows[split] = [write_case(output / split, b, b["selected"]) for b in by_split[split]]
        manifests[split] = write_manifest(output, split, split, stage, seeds, rows[split])
    short_name = "train_s" if rsi["short_tier"]["per_family"] is not None else "train"
    if short_name == "train_s":
        manifests["train_s"] = write_manifest(output, "train_s", "train", stage, seeds,
                                              [write_case(output / "train_s", b, b["selected"]) for b in short])
    # Weak judge: whole holes and outcrop cells of the selected pack's confirmation.
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
    standard = {row["case_id"]: row["sha256"] for split in ("train", "validation") for row in rows[split]}
    for name in ("train_s_q25", "validation_q25"):
        for row in json.loads((output / manifests[name]).read_text())["cases"]:
            if any(row["sha256"][key] != standard[row["case_id"]][key] for key in ("problem.json", "problem.npz", "truth.npy")):
                raise ValueError("weak-judge public observations differ from the standard manifest")
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
    # The unselected pack is archived for provenance, outside every manifest above
    # except the Q1 both-pack training manifest.
    unselected = []
    for body in bodies:
        modality = next(m for m in MODALITIES if m != body["selected"])
        case = body["cases"][modality]
        target = output / "unselected" / body["split"] / case["case_id"]
        shutil.copytree(case["public"], target / "public")
        shutil.copytree(case["hidden"], target / "confirmation")
        unselected.append(dict(parent_id=body["parent_id"], split=body["split"], modality=modality,
                               case_id=case["case_id"], sha256={str(p.relative_to(target)): file_hash(p)
                                                                for p in sorted(target.rglob("*")) if p.is_file()}))
    (output / "unselected" / "index.json").write_text(json.dumps(dict(
        schema="rsi-unselected-packs-v1", scope="generated and archived; not an evaluation manifest",
        packs=unselected), sort_keys=True, indent=2) + "\n")
    common = dict(validation_manifest=manifests["validation"],
                  sealed_manifests=[manifests["test_id"], manifests["test_ood"]])
    tiers = {"S": dict(solver_seconds=rsi["tiers"]["S"]["solver_seconds"], train_manifest=manifests[short_name],
                       train_bodies=len(short), slots=rsi["tiers"]["S"]["slots"], **common),
             "F": dict(solver_seconds=rsi["tiers"]["F"]["solver_seconds"], train_manifest=manifests["train"],
                       train_bodies=len(by_split["train"]), slots=rsi["tiers"]["F"]["slots"], **common)}
    layout = {key: rsi["layout"][key] for key in ("packs_per_body", "execution_seeds", "assignment_rule", "sealed_bodies")}
    # Exceptions to packs_per_body, read per manifest by the evaluator (Q1 only).
    layout["packs_per_body_by_manifest"] = {q1["both_packs_manifest"]: 2} if q1["both_packs_manifest"] else {}
    q1["solver_seconds"] = rsi["tiers"]["S"]["solver_seconds"]
    judges = {"standard": {"objective": "confirmation"},
              "q25": {"objective": "confirmation", "train_manifest": manifests["train_s_q25"],
                      "validation_manifest": manifests["validation_q25"]},
              "oracle": {"objective": "oracle_iou"}}
    evidence["confirmation"] = dict(status="passed", cases=sum(len(b["cases"]) for b in bodies),
                                    policy=rsi["confirmation_policy"]["mode"],
                                    scope="independent whole-hole/location and noise streams; geometry/masks/priors identical; not a new geology parent")
    evidence["hole_audit"] = hole
    evidence["layout"] = dict(status="passed", rule=ASSIGNMENT_RULE, allocation_seed=rsi["layout"]["allocation_seed"],
                              generation_plan_sha256=rsi_sha, plans=parts, balance=balance,
                              short_tier=dict(rule=rsi["short_tier"]["rule"], per_family=rsi["short_tier"]["per_family"],
                                              parents=[b["name"] for b in short]),
                              assignment=[dict(parent=b["name"], split=b["split"], family=b["family"], index=b["index"],
                                               noisy=b["noisy"], selected_modality=b["selected"]) for b in bodies])
    evidence["weak_judge"] = dict(status="passed", rule=q25["rule"], fraction=q25["fraction"], seed=q25["seed"],
                                  outcrop_grid=q25["outcrop_grid"], cases=thinning)
    (output / "evidence").mkdir()
    for name, value in evidence.items():
        (output / "evidence" / f"{name}.json").write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    hashes = {path: file_hash(output / path) for path in manifests.values()}
    sealed = dict(release=rsi["release"], generation_plan_sha256=rsi_sha,
                  plan_sha256={part: value["plan_sha256"] for part, value in parts.items()},
                  manifest_sha256=hashes, counts=counts, tiers=tiers, layout=layout, judges=judges, q1=q1,
                  evidence_sha256={str(p.relative_to(output)): file_hash(p) for p in (output / "evidence").glob("*.json")},
                  unselected_sha256={"unselected/index.json": file_hash(output / "unselected" / "index.json")})
    verify_tiers(sealed, output)
    if formal:
        record = dict(sealed, schema="copper-qualified-synthetic-v2", status="qualified", evaluator_uid=os.geteuid(),
                      limitations=["Synthetic benchmark only; no real mine validation",
                                   "Empirical discretization calibration, not an error-bound theorem",
                                   "Not secure against host administrators or trusted Docker operator"])
        (output / "qualification.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    else:
        # Development receipts key the four split manifests by name, as read by
        # rsi/development_selection.py; `manifest_files` holds the path keys.
        record = dict(sealed, status="development_only",
                      scope="synthetic GemPy geology; not field validation or formal release",
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
    return dict(status="qualified" if formal else "development_only", release=rsi["release"], counts=counts,
                manifests=sorted(manifests.values()), hole_audit=hole["status"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="fix and hash a release generation plan before generation")
    plan.add_argument("--root", type=Path, required=True)
    plan.add_argument("--release", choices=sorted(PRESETS), required=True)
    plan.add_argument("--master-seed", type=int, required=True)
    plan.add_argument("--confirmation-policy", choices=sorted(CONFIRMATION_POLICIES), default="resample_until_separated")
    plan.add_argument("--historical", nargs="*", default=[], metavar="LABEL=PATH",
                      help="prior releases or generator archives; rsi-r1 requires legacy_release and rsi_dev")
    plan.add_argument("--no-q1-both-packs", action="store_true")
    plan.add_argument("--tiers", type=json.loads, metavar="JSON",
                      help='development only, e.g. a one-round smoke: {"S": {"solver_seconds": 10, "slots": 5}, ...}')
    release = commands.add_parser("release", help="check a generated population and write the release")
    release.add_argument("--dataset", type=Path, required=True)
    release.add_argument("--output", type=Path, required=True)
    for name in ("calibration", "isolation", "geological-constraints"):
        release.add_argument("--" + name, type=Path, help="required for a formal release")
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
    else:
        print(json.dumps(build(args.dataset, args.output, args.calibration, args.isolation,
                               args.geological_constraints), sort_keys=True))


if __name__ == "__main__":
    main()
