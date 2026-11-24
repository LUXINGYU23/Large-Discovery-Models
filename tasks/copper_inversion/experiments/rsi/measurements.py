"""Whole-group thinning, public/confirmation borehole overlap checks and layout rules."""

import copy
import hashlib
import math

import numpy as np
from scipy.spatial import cKDTree


from tasks.copper_inversion.experiments.v2.measurement_policy import DENSITIES, parse_cli_holes, parse_holes

FAMILIES = ("folded", "lens", "multiple", "faulted")
MODALITIES = ("drill", "drill_outcrop")
ASSIGNMENT_RULE = "alternate-within-family-v1"
ASSIGNMENT_RULE_V2 = "alternate-within-family-noise-balanced-v2"
SHORT_TIER_RULE = "first-n-per-family-by-plan-index-noise-mixed-v1"
THINNING_RULE = "whole-drill-holes-and-outcrop-spatial-cells-quarter-v1"
ID_SPLITS = ("train", "calibration", "validation", "test_id")
# Author decision (2026-11-15): redraw each body's confirmation from its
# presampled streams until the whole-hole audit passes. Failing the release on
# any overlap (the original plan) remains available as the alternative.
CONFIRMATION_POLICIES = {
    "fail_release": "first presampled stream only; any overlapping whole hole fails the release",
    "resample_until_separated": "first of 16 presampled streams with no overlapping whole holes",
}
CONFIRMATION_POLICIES_MULTI = {
    "fail_release": "first k presampled streams only; any overlapping or undiscovered pack fails the release",
    "resample_until_separated": "first k presampled streams separated from public holes and prior packs",
}


def confirmation_mode(plan):
    """The confirmation policy fixed in a generation plan."""
    policy = plan.get("confirmation_policy")
    if not isinstance(policy, dict):
        raise ValueError("confirmation policy was not fixed in the generation plan")
    # Prototype plans carried only the resampling rule string.
    mode = policy.get("mode", "resample_until_separated")
    packs = policy.get("packs", 1)
    expected = CONFIRMATION_POLICIES_MULTI if packs > 1 else CONFIRMATION_POLICIES
    if expected.get(mode) != policy.get("rule") and CONFIRMATION_POLICIES.get(mode) != policy.get("rule"):
        raise ValueError("unknown confirmation policy")
    return mode


def confirmation_pack_count(plan):
    policy = plan.get("confirmation_policy")
    if not isinstance(policy, dict):
        return 1
    packs = policy.get("packs", 1)
    if type(packs) is not int or packs < 1:
        raise ValueError("invalid confirmation pack count")
    return packs


def public_min_ore_holes(plan, split):
    disc = plan.get("discoverability_policy")
    if not isinstance(disc, dict) or split not in disc.get("id_splits", ID_SPLITS):
        return 0
    return int(disc.get("min_public_ore_holes", 0))


def confirmation_min_ore_holes(plan, split):
    disc = plan.get("discoverability_policy")
    if not isinstance(disc, dict) or split not in disc.get("id_splits", ID_SPLITS):
        return 0
    return int(disc.get("min_confirmation_ore_holes", 0))


def confirmation_streams(entry, mode, packs=1):
    """Presampled confirmation seeds that a policy may try, in order."""
    seeds = entry["confirmation_seeds"]
    if len(seeds) < max(16, packs) or len(set(seeds)) != len(seeds):
        raise ValueError("invalid presampled confirmation streams")
    return list(seeds[:packs]) if mode == "fail_release" else list(seeds)


def ore_hole_count(problem, manifest=None):
    """Number of drill boreholes intersecting ore (uses true-ore manifest count when available)."""
    if isinstance(manifest, dict) and type(manifest.get("ore_holes")) is int:
        return manifest["ore_holes"]
    found = [o for o in problem["observations"] if o["name"] == "drill"]
    if len(found) != 1:
        raise ValueError("each problem requires one drill observation")
    drill = found[0]
    labels = np.asarray(drill["labels"])
    groups = np.asarray(drill["groups"])
    binary = labels != .5
    return int(sum(bool((labels[binary & (groups == g)] == 1).any()) for g in np.unique(groups)))


def check_discoverability_attempts(receipt, seeds, min_ore_holes, selected_seed):
    """A public discoverability receipt must try presampled observation seeds in order."""
    attempts = receipt.get("attempts", [])
    tried = [row["seed"] for row in attempts]
    if not attempts or tried != list(seeds[:len(attempts)]) or selected_seed != tried[-1]:
        raise ValueError("public observations do not match the preassigned seed stream")
    if attempts[-1]["status"] != "passed" or attempts[-1].get("ore_holes", 0) < min_ore_holes:
        raise ValueError("selected public observation pack failed discoverability")
    if any(row["status"] != "failed" or row.get("ore_holes", 0) >= min_ore_holes for row in attempts[:-1]):
        raise ValueError("public observation redraw violated discoverability policy")


def check_confirmation_attempts(receipt, entry, mode, hidden_seeds, packs=1, min_ore_holes=0):
    """A confirmation receipt must follow the plan's separation and discoverability policy."""
    expected = [hidden_seeds] if isinstance(hidden_seeds, int) else list(hidden_seeds)
    if len(expected) != packs:
        raise ValueError("confirmation pack count differs from the plan")
    attempts = receipt["attempts"]
    seeds = [row["seed"] for row in attempts]
    if mode == "fail_release" and len(attempts) > packs:
        raise ValueError("confirmation was redrawn under the fail-release policy")
    allowed = confirmation_streams(entry, mode, packs)
    if not attempts or seeds != allowed[:len(attempts)]:
        raise ValueError("confirmation does not match the preassigned stream policy")
    if mode == "fail_release":
        if seeds != expected:
            raise ValueError("confirmation does not match the preassigned stream policy")
        return
    passed = [row["seed"] for row in attempts if row["status"] == "passed"]
    if (passed != expected or attempts[-1]["status"] != "passed"
            or any(row["status"] not in ("passed", "failed") for row in attempts)):
        raise ValueError("confirmation does not match the preassigned separation policy")
    if min_ore_holes > 0 and any(
            row.get("ore_holes", min_ore_holes) < min_ore_holes for row in attempts if row["status"] == "passed"):
        raise ValueError("confirmation pack failed discoverability policy")


def spatial_groups(problem, points, grid):
    """Outcrop cells of a grid x grid partition of the horizontal domain."""
    origin = np.asarray(problem["origin"], float)
    extent = np.asarray(problem["shape"], float) * np.asarray(problem["spacing"], float)
    normalized = np.clip((np.asarray(points)[:, :2] - origin[:2]) / extent[:2], 0., 1. - 1e-12)
    cells = (normalized * grid).astype(int)
    return cells[:, 0] * grid + cells[:, 1]


def _grouping(problem, observation, grid):
    if observation["name"] == "drill":
        return np.asarray(observation["groups"]), "groups"
    if observation["name"] == "outcrop":
        return spatial_groups(problem, observation["points"], grid), f"spatial-{grid}x{grid}"
    raise ValueError("no thinning rule for observation " + observation["name"])


def thin_confirmation(problem, seed, fraction=.25, grid=4):
    """Keep a deterministic quarter (at least one) of each observation's whole groups."""
    if type(seed) is not int or seed < 0:
        raise ValueError("a nonnegative thinning seed is required")
    result = copy.deepcopy(problem)
    selected = {}
    for index, observation in enumerate(result["observations"]):
        groups, grouping = _grouping(result, observation, grid)
        unique = np.unique(groups)
        if not len(unique):
            raise ValueError("cannot thin an empty observation")
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        count = max(1, int(math.floor(len(unique) * fraction + .5)))
        keep = np.sort(rng.choice(unique, size=count, replace=False))
        mask = np.isin(groups, keep)
        for key in ("points", "labels", "groups"):
            observation[key] = np.asarray(observation[key])[mask].copy()
        selected[observation["name"]] = {"grouping": grouping, "groups_total": int(len(unique)),
                                         "kept": [int(value) for value in keep]}
    return result, selected


def check_thinning(full, thinned, selected, fraction=.25, grid=4):
    """Whole groups only, at least one per observation, within one group of the fraction."""
    if [o["name"] for o in full["observations"]] != [o["name"] for o in thinned["observations"]]:
        raise ValueError("thinning changed the observation modalities")
    summary = {}
    for whole, part in zip(full["observations"], thinned["observations"]):
        groups, _ = _grouping(full, whole, grid)
        kept_groups, _ = _grouping(thinned, part, grid)
        unique, kept = np.unique(groups), np.unique(kept_groups)
        if [int(v) for v in kept] != selected[whole["name"]]["kept"] or not len(kept):
            raise ValueError("thinned groups differ from the recorded selection")
        if abs(len(kept) - fraction * len(unique)) > 1:
            raise ValueError("thinned group count is outside one group of the fraction")
        for group in kept:
            full_rows = np.flatnonzero(groups == group)
            part_rows = np.flatnonzero(kept_groups == group)
            if len(full_rows) != len(part_rows) or not np.array_equal(
                    np.asarray(whole["points"])[full_rows], np.asarray(part["points"])[part_rows]):
                raise ValueError("thinning split a whole group")
        summary[whole["name"]] = {"groups_total": int(len(unique)), "groups_kept": int(len(kept))}
    return summary


def audit_holes(public, confirmation):
    """Reject a confirmation borehole that substantially overlaps any public one."""
    def drill(problem):
        found = [o for o in problem["observations"] if o["name"] == "drill"]
        if len(found) != 1:
            raise ValueError("each problem requires one drill observation")
        return found[0]
    a, b = drill(public), drill(confirmation)
    public_groups = np.unique(a["groups"])
    hidden_groups = np.unique(b["groups"])
    if not len(public_groups) or not len(hidden_groups):
        raise ValueError("hole audit requires nonempty public and confirmation boreholes")
    trees = [cKDTree(np.asarray(a["points"])[np.asarray(a["groups"]) == group])
             for group in public_groups]
    threshold = float(b["transition_scale"])
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("invalid confirmation transition scale")
    minima, overlapping = [], 0
    for group in hidden_groups:
        points = np.asarray(b["points"])[np.asarray(b["groups"]) == group]
        distances = np.vstack([tree.query(points)[0] for tree in trees])
        minima.append(float(np.min(distances)))
        # Overlap: at least half of the hole's samples lie within one transition
        # scale of the samples of a single public hole.
        if np.any(np.mean(distances <= threshold, axis=1) >= .5):
            overlapping += 1
    return {"status": "failed" if overlapping else "passed", "confirmation_holes": len(hidden_groups),
            "overlapping_holes": overlapping, "transition_scale": threshold,
            "nearest_hole_distances": minima,
            "nearest_hole_distance_min": min(minima),
            "nearest_hole_distance_median": float(np.median(minima))}


def audit_pack_pair(first, second):
    """Symmetric whole-hole separation between two confirmation packs of the same body."""
    ab, ba = audit_holes(first, second), audit_holes(second, first)
    overlapping = ab["overlapping_holes"] + ba["overlapping_holes"]
    distances = ab["nearest_hole_distances"] + ba["nearest_hole_distances"]
    return {"status": "failed" if overlapping else "passed",
            "overlapping_holes": overlapping,
            "forward_overlapping_holes": ab["overlapping_holes"],
            "reverse_overlapping_holes": ba["overlapping_holes"],
            "transition_scale": ab["transition_scale"],
            "nearest_hole_distances": distances,
            "nearest_hole_distance_min": min(distances),
            "nearest_hole_distance_median": float(np.median(distances))}


def merge_confirmations(problems):
    """Combine disjoint confirmation packs into one validated problem (for judge X4)."""
    if not problems:
        raise ValueError("cannot merge empty confirmation pack list")
    base = copy.deepcopy(problems[0])
    for other in problems[1:]:
        if [o["name"] for o in other["observations"]] != [o["name"] for o in base["observations"]]:
            raise ValueError("confirmation packs have different observation modalities")
        for key in ("shape", "origin", "spacing"):
            if other[key] != base[key]:
                raise ValueError("confirmation pack geometry differs")
    for idx, obs in enumerate(base["observations"]):
        pts, lbls, grps = [], [], []
        offset = 0
        for pack in problems:
            item = pack["observations"][idx]
            p = np.asarray(item["points"], dtype=float)
            l = np.asarray(item["labels"], dtype=float)
            g = np.asarray(item["groups"], dtype=int)
            pts.append(p)
            lbls.append(l)
            if obs["name"] == "drill":
                unique = np.unique(g)
                remap = {int(old): offset + k for k, old in enumerate(unique)}
                grps.append(np.asarray([remap[int(v)] for v in g], dtype=int))
                offset += len(unique)
            else:
                grps.append(g.copy())
        obs["points"] = np.concatenate(pts, axis=0)
        obs["labels"] = np.concatenate(lbls, axis=0)
        obs["groups"] = np.concatenate(grps, axis=0)
    from tasks.copper_inversion.v2.contract import validate
    return validate(base)


def hole_distance_summary(audits):
    """Distribution of each confirmation hole's closest approach to any public hole."""
    distances = np.asarray([d for row in audits for d in row["nearest_hole_distances"]], float)
    scaled = np.asarray([d / row["transition_scale"] for row in audits for d in row["nearest_hole_distances"]])
    if not len(distances):
        raise ValueError("no audited confirmation holes")
    quantiles = (0., .05, .25, .5, .75, .95, 1.)
    edges = [0., 1., 2., 4., 8., 16., math.inf]
    counts = np.histogram(scaled, bins=[*edges[:-1], max(32., float(scaled.max()) + 1.)])[0]
    return {"holes": int(len(distances)), "quantiles": [*quantiles],
            "distance": [float(v) for v in np.quantile(distances, quantiles)],
            "distance_in_transition_scales": [float(v) for v in np.quantile(scaled, quantiles)],
            "histogram_transition_scales": {"edges": [str(v) for v in edges], "counts": [int(v) for v in counts]}}


def preselect_short_train(plan, per_family=3, families=FAMILIES):
    """Choose the first planned training parents in each family, noise-mixed."""
    parents = [row for row in plan["parents"] if row["split"] == "train"]
    result = []
    for family in families:
        options = sorted((row for row in parents if row["family"] == family), key=lambda row: row["index"])
        selected = options[:per_family]
        if len(selected) != per_family or {row["noisy"] for row in selected} != {False, True}:
            raise ValueError("generation plan lacks noise-mixed short-tier parents per family")
        result.extend(selected)
    if len({(row["split"], row["index"]) for row in result}) != per_family * len(families):
        raise ValueError("short-tier parent selection is not unique")
    return [{"split": row["split"], "index": row["index"], "family": row["family"],
             "noisy": row["noisy"], "lineage_id": row["lineage_id"]} for row in result]


def _assign_modalities_v2(plan, seed):
    by_split_family = {}
    for row in plan["parents"]:
        by_split_family.setdefault((row["split"], row["family"]), []).append(row)
    for split in sorted({k[0] for k in by_split_family}):
        families = sorted(fam for sp, fam in by_split_family if sp == split)
        both_odd = []
        single_odd = {False: [], True: []}
        for family in families:
            rows = by_split_family[split, family]
            clean = sorted((r for r in rows if not r["noisy"]), key=lambda r: r["index"])
            noisy = sorted((r for r in rows if r["noisy"]), key=lambda r: r["index"])
            if len(clean) % 2 == 0 and len(noisy) % 2 == 0:
                for flag, sub in ((0, clean), (1, noisy)):
                    start = hashlib.sha256(f"{seed}:{split}:{family}:{flag}".encode()).digest()[0] % 2
                    for offset, row in enumerate(sub):
                        row["selected_modality"] = MODALITIES[(start + offset) % 2]
            elif len(clean) % 2 == 1 and len(noisy) % 2 == 1:
                both_odd.append((family, clean, noisy))
            else:
                for flag, sub in ((False, clean), (True, noisy)):
                    if len(sub) % 2 == 0:
                        start = hashlib.sha256(f"{seed}:{split}:{family}:{int(flag)}".encode()).digest()[0] % 2
                        for offset, row in enumerate(sub):
                            row["selected_modality"] = MODALITIES[(start + offset) % 2]
                    else:
                        single_odd[flag].append((family, sub))
        both_odd.sort(key=lambda item: hashlib.sha256(f"{seed}:{split}:{item[0]}".encode()).digest())
        base_start = hashlib.sha256(f"{seed}:{split}:both_odd".encode()).digest()[0] % 2
        for idx, (family, clean, noisy) in enumerate(both_odd):
            m0 = (base_start + idx) % 2
            for offset, row in enumerate(clean):
                row["selected_modality"] = MODALITIES[(m0 + offset) % 2]
            for offset, row in enumerate(noisy):
                row["selected_modality"] = MODALITIES[(1 - m0 + offset) % 2]
        odd_start = hashlib.sha256(f"{seed}:{split}:odd".encode()).digest()[0] % 2
        ordered_single = []
        for flag in (False, True):
            ordered_single.extend(sorted(
                single_odd[flag],
                key=lambda item: hashlib.sha256(f"{seed}:{split}:{item[0]}:{int(flag)}:odd".encode()).digest()))
        for idx, (_, sub) in enumerate(ordered_single):
            m0 = (odd_start + idx) % 2
            for offset, row in enumerate(sub):
                row["selected_modality"] = MODALITIES[(m0 + offset) % 2]
    plan["modality_allocation"] = {"seed": seed, "rule": ASSIGNMENT_RULE_V2}
    return plan


def assign_modalities(plan, seed, rule=ASSIGNMENT_RULE):
    """Fix one observation layout per parent, before generating the parents."""
    if type(seed) is not int or seed < 0:
        raise ValueError("a nonnegative allocation seed is required")
    if rule == ASSIGNMENT_RULE_V2:
        return _assign_modalities_v2(plan, seed)
    if rule != ASSIGNMENT_RULE:
        raise ValueError("unknown modality assignment rule: " + str(rule))
    groups = {}
    for row in plan["parents"]:
        groups.setdefault((row["split"], row["family"]), []).append(row)
    for split in sorted({key[0] for key in groups}):
        families = sorted(family for group_split, family in groups if group_split == split)
        odd = sorted((family for family in families if len(groups[split, family]) % 2),
                     key=lambda family: hashlib.sha256(f"{seed}:{split}:{family}".encode()).digest())
        split_start = hashlib.sha256(f"{seed}:{split}".encode()).digest()[0] % 2
        odd_starts = {family: (split_start + index) % 2 for index, family in enumerate(odd)}
        for family in families:
            rows = groups[split, family]
            start = (odd_starts[family] if family in odd_starts else
                     hashlib.sha256(f"{seed}:{split}:{family}".encode()).digest()[0] % 2)
            for offset, row in enumerate(sorted(rows, key=lambda item: item["index"])):
                row["selected_modality"] = MODALITIES[(start + offset) % 2]
    plan["modality_allocation"] = {"seed": seed, "rule": ASSIGNMENT_RULE}
    return plan


def modality_balance(rows):
    """Per split and per split/family layout counts; differ by at most one."""
    result = {}
    for row in rows:
        for key in (row["split"], f"{row['split']}/{row['family']}"):
            counts = result.setdefault(key, {name: 0 for name in MODALITIES})
            counts[row["selected_modality"]] += 1
    for key, counts in result.items():
        if abs(counts["drill"] - counts["drill_outcrop"]) > 1:
            raise ValueError("unbalanced observation layouts in " + key)
    return result
