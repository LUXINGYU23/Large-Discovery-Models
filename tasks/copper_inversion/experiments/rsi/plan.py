"""Release generation plans, fixed and hashed before any geology is generated.

`generation-plan.json` fixes a release (seeds, counts, layout and policies). Each
population part (`id`, `ood`) then writes `plan.json` with one row per parent.
Both files get a write-once receipt holding their hash before generation starts.
"""

import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .measurements import (ASSIGNMENT_RULE, ASSIGNMENT_RULE_V2, CONFIRMATION_POLICIES, CONFIRMATION_POLICIES_MULTI,
                           FAMILIES, ID_SPLITS, SHORT_TIER_RULE, THINNING_RULE, assign_modalities, preselect_short_train)


SCHEMA = "rsi-generation-plan-v1"
SPLITS = {"id": ("train", "validation", "test_id"), "ood": ("test_ood",)}
SPLITS_R2 = {"id": ("train", "calibration", "validation", "test_id"), "ood": ("calibration_ood", "test_ood")}
PRESETS = {
    "rsi-dev": dict(stage="development", counts=dict(train=8, validation=4, test_id=4, test_ood=4),
                    short_per_family=None, historical=()),
    "rsi-r1": dict(stage="formal", counts=dict(train=24, validation=8, test_id=32, test_ood=16),
                   short_per_family=3, historical=("legacy_release", "rsi_dev")),
    "rsi-r2": dict(stage="formal",
                   counts=dict(train=48, calibration=24, calibration_ood=8, validation=16, test_id=48, test_ood=24),
                   splits=SPLITS_R2, short_per_family=3, medium_per_family=6,
                   assignment_rule=ASSIGNMENT_RULE_V2, confirmation_packs=4, confirmation_streams=64,
                   generation_attempts=16, observation_streams=16,
                   id_size_gates=dict(volume_bounds=[.04, .35], min_vertical_thickness=3),
                   discoverability=dict(min_public_ore_holes=2, min_confirmation_ore_holes=1),
                   density_shift_splits=("test_ood",),
                   historical=("legacy_release", "rsi_dev")),
}
DERIVED = ("selected_modality", "confirmation_seeds", "observation_seeds")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive_seed(master, *labels):
    """A 32-bit seed derived from the release master seed and fixed labels."""
    return int.from_bytes(hashlib.sha256(json.dumps([master, *labels]).encode()).digest()[:4], "big")


def write_once(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _receipt(path, **extra):
    return dict(sha256=file_hash(path), recorded_at=datetime.now(timezone.utc).isoformat(), **extra)


def release_splits(release_or_plan):
    name = release_or_plan["release"] if isinstance(release_or_plan, dict) else release_or_plan
    return PRESETS.get(name, {}).get("splits", SPLITS)


def make_generation_plan(release, master_seed, confirmation_policy="resample_until_separated", historical=None,
                         counts=None, short_per_family="preset", medium_per_family="preset", tiers=None):
    if type(master_seed) is not int or master_seed < 0:
        raise ValueError("a nonnegative master seed is required")
    if confirmation_policy not in CONFIRMATION_POLICIES:
        raise ValueError("unknown confirmation policy")
    preset = PRESETS[release]
    counts = dict(counts or preset["counts"])
    expected_splits = set(preset["counts"])
    if set(counts) != expected_splits or any(type(v) is not int or v < 1 for v in counts.values()):
        raise ValueError("invalid split counts")
    if tiers is not None and preset["stage"] != "development":
        raise ValueError("only a development release may seal its own tier table")
    per_family = preset["short_per_family"] if short_per_family == "preset" else short_per_family
    historical = dict(historical or {})
    if set(preset["historical"]) - set(historical):
        raise ValueError("historical exclusion requires " + ", ".join(preset["historical"]))
    packs = preset.get("confirmation_packs", 1)
    streams = preset.get("confirmation_streams", 16)
    rule_table = CONFIRMATION_POLICIES_MULTI if packs > 1 else CONFIRMATION_POLICIES
    assignment_rule = preset.get("assignment_rule", ASSIGNMENT_RULE)
    plan = dict(
        schema=SCHEMA, release=release, stage=preset["stage"], master_seed=master_seed,
        id_seed=derive_seed(master_seed, "id"), ood_seed=derive_seed(master_seed, "ood"), counts=counts,
        layout=dict(packs_per_body=1, execution_seeds=[47], assignment_rule=assignment_rule,
                    allocation_seed=derive_seed(master_seed, "allocation"),
                    sealed_bodies=dict(test_id=counts["test_id"], test_ood=counts["test_ood"])),
        short_tier=dict(rule=SHORT_TIER_RULE, per_family=per_family, families=list(FAMILIES)),
        tiers=tiers or (
            dict(S=dict(solver_seconds=30, slots=321), M=dict(solver_seconds=60, slots=321),
                 F=dict(solver_seconds=120, slots=321))
            if "medium_per_family" in preset else
            dict(S=dict(solver_seconds=30, slots=321), F=dict(solver_seconds=120, slots=321))),
        judges=dict(q25=dict(rule=THINNING_RULE, fraction=.25, min_groups=1, outcrop_grid=4,
                             seed=derive_seed(master_seed, "q25"))),
        q1=dict(execution_seeds=[47, 48, 49], both_packs_manifest=True),
        confirmation_policy=dict(mode=confirmation_policy, rule=rule_table[confirmation_policy],
                                 streams=streams, seed_derivation="SeedSequence([observation_seed, 920381])"),
        historical={label: str(Path(path).resolve()) for label, path in sorted(historical.items())})
    if packs > 1:
        plan["confirmation_policy"]["packs"] = packs
        plan["judges"]["x4"] = dict(packs=packs, rule="merge-four-separated-confirmation-packs-v1")
    if "medium_per_family" in preset:
        med = preset["medium_per_family"] if medium_per_family == "preset" else medium_per_family
        plan["medium_tier"] = dict(rule=SHORT_TIER_RULE, per_family=med, families=list(FAMILIES))
    if "discoverability" in preset:
        disc = preset["discoverability"]
        plan["discoverability_policy"] = dict(
            id_splits=list(ID_SPLITS),
            min_public_ore_holes=disc["min_public_ore_holes"],
            min_confirmation_ore_holes=disc["min_confirmation_ore_holes"],
            observation_streams=preset["observation_streams"],
            seed_derivation="[observation_seed, *SeedSequence([observation_seed, 510437]).generate_state(15)]")
    if "id_size_gates" in preset:
        gates = preset["id_size_gates"]
        plan["size_gates"] = dict(
            id_splits=list(ID_SPLITS),
            volume_bounds=list(gates["volume_bounds"]),
            min_vertical_thickness=gates["min_vertical_thickness"],
            generation_attempts=preset["generation_attempts"])
    if "density_shift_splits" in preset:
        plan["density_shift_splits"] = list(preset["density_shift_splits"])
        plan["densities"] = dict(lo=[3, 12], mid=[12, 24], hi=[24, 48])
    return plan


def write_generation_plan(root, plan):
    """Record the release plan and its hash before any population exists."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if any((root / name).exists() for name in ("id", "ood", "generation-plan.json")):
        raise ValueError("generation plan must precede every generated population")
    write_once(root / "generation-plan.json", plan)
    receipt = _receipt(root / "generation-plan.json")
    write_once(root / "generation-plan-receipt.json", receipt)
    return receipt["sha256"]


def load_generation_plan(root):
    root = Path(root)
    plan_path, receipt_path = root / "generation-plan.json", root / "generation-plan-receipt.json"
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != SCHEMA:
        raise ValueError("not an RSI generation plan")
    receipt = json.loads(receipt_path.read_text())
    if receipt["sha256"] != file_hash(plan_path) or plan_path.stat().st_mtime_ns > receipt_path.stat().st_mtime_ns:
        raise ValueError("generation plan changed after its hash was recorded")
    return plan, receipt["sha256"]


def part_splits(rsi, part):
    splits = release_splits(rsi)[part]
    return tuple((split, rsi["counts"][split]) for split in splits if split in rsi["counts"])


def part_plan(plan, rsi, rsi_sha, part):
    """Fix layout, confirmation streams and short/medium tiers in a parent plan."""
    rule = rsi["layout"].get("assignment_rule", ASSIGNMENT_RULE)
    assign_modalities(plan, rsi["layout"]["allocation_seed"], rule=rule)
    policy = rsi["confirmation_policy"]
    plan["confirmation_policy"] = dict(policy)
    for entry in plan["parents"]:
        entry["confirmation_seeds"] = [int(s) for s in np.random.SeedSequence(
            [entry["observation_seed"], 920381]).generate_state(policy["streams"])]
    if "discoverability_policy" in rsi:
        disc = rsi["discoverability_policy"]
        plan["discoverability_policy"] = dict(disc)
        n_obs = disc["observation_streams"]
        for entry in plan["parents"]:
            extra = [int(s) for s in np.random.SeedSequence(
                [entry["observation_seed"], 510437]).generate_state(n_obs - 1)]
            entry["observation_seeds"] = [entry["observation_seed"], *extra]
    if "size_gates" in rsi:
        plan["size_gates"] = dict(rsi["size_gates"])
    per_family = rsi["short_tier"]["per_family"]
    if part == "id" and per_family is not None:
        plan["short_train"] = preselect_short_train(plan, per_family, tuple(rsi["short_tier"]["families"]))
    if part == "id" and "medium_tier" in rsi and rsi["medium_tier"]["per_family"] is not None:
        plan["medium_train"] = preselect_short_train(
            plan, rsi["medium_tier"]["per_family"], tuple(rsi["medium_tier"]["families"]))
    plan["rsi"] = dict(release=rsi["release"], generation_plan_sha256=rsi_sha, part=part)
    return plan


def write_part_plan(root, plan):
    """Write a population plan and record its hash before any parent is generated."""
    root = Path(root)
    if (root / "parents").exists():
        raise ValueError("parent plan must precede parent generation")
    write_once(root / "plan.json", plan)
    receipt = _receipt(root / "plan.json", generation_plan_sha256=plan.get("rsi", {}).get("generation_plan_sha256"))
    write_once(root / "plan-receipt.json", receipt)
    return receipt["sha256"]


def replay_part_plan(plan, rsi, rsi_sha, part):
    """Recompute every rule-derived field from the plan's own parent rows."""
    base = copy.deepcopy(plan)
    for key in ("modality_allocation", "confirmation_policy", "discoverability_policy",
                "size_gates", "short_train", "medium_train", "rsi"):
        base.pop(key, None)
    for entry in base["parents"]:
        for key in DERIVED:
            entry.pop(key, None)
    return part_plan(base, rsi, rsi_sha, part)


def verify_part(root, rsi, rsi_sha, part, generation_receipt):
    """The parent plan was hashed after the release plan and before any geology."""
    root = Path(root)
    plan_path, receipt_path = root / "plan.json", root / "plan-receipt.json"
    plan = json.loads(plan_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    sha = file_hash(plan_path)
    if receipt["sha256"] != sha or receipt.get("generation_plan_sha256") != rsi_sha:
        raise ValueError("parent plan changed after its hash was recorded")
    if replay_part_plan(plan, rsi, rsi_sha, part) != plan:
        raise ValueError("parent plan does not reproduce from the fixed rules")
    planned = {split: sum(row["split"] == split for row in plan["parents"]) for split, _ in part_splits(rsi, part)}
    if planned != dict(part_splits(rsi, part)):
        raise ValueError("parent plan counts differ from the release plan")
    geology = sorted((root / "parents").glob("*/geology.npz"))
    if not geology:
        raise ValueError("no generated geology")
    recorded = receipt_path.stat().st_mtime_ns
    first = min(path.stat().st_mtime_ns for path in geology)
    if not (Path(generation_receipt).stat().st_mtime_ns <= plan_path.stat().st_mtime_ns <= recorded <= first):
        raise ValueError("plan hash was not recorded before geology generation")
    return plan, sha, dict(plan_sha256=sha, plan_recorded_at=receipt["recorded_at"],
                           plan_receipt_mtime_ns=recorded, first_geology_mtime_ns=first)
