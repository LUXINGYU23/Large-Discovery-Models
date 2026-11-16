"""Whole-group thinning, public/confirmation borehole overlap checks and layout rules."""

import copy
import hashlib
import math

import numpy as np
from scipy.spatial import cKDTree


FAMILIES = ("folded", "lens", "multiple", "faulted")
MODALITIES = ("drill", "drill_outcrop")
ASSIGNMENT_RULE = "alternate-within-family-v1"
SHORT_TIER_RULE = "first-n-per-family-by-plan-index-noise-mixed-v1"
THINNING_RULE = "whole-drill-holes-and-outcrop-spatial-cells-quarter-v1"
# Author decision (2026-11-15): redraw each body's confirmation from its 16
# presampled streams until the whole-hole audit passes. Failing the release on
# any overlap (the original plan) remains available as the alternative.
CONFIRMATION_POLICIES = {
    "fail_release": "first presampled stream only; any overlapping whole hole fails the release",
    "resample_until_separated": "first of 16 presampled streams with no overlapping whole holes",
}


def confirmation_mode(plan):
    """The confirmation policy fixed in a generation plan."""
    policy = plan.get("confirmation_policy")
    if not isinstance(policy, dict):
        raise ValueError("confirmation policy was not fixed in the generation plan")
    # Prototype plans carried only the resampling rule string.
    mode = policy.get("mode", "resample_until_separated")
    if CONFIRMATION_POLICIES.get(mode) != policy.get("rule"):
        raise ValueError("unknown confirmation policy")
    return mode


def confirmation_streams(entry, mode):
    """Presampled confirmation seeds that a policy may try, in order."""
    seeds = entry["confirmation_seeds"]
    if len(seeds) != 16 or len(set(seeds)) != 16:
        raise ValueError("invalid presampled confirmation streams")
    return seeds[:1] if mode == "fail_release" else list(seeds)


def check_confirmation_attempts(receipt, entry, mode, hidden_seed):
    """A confirmation receipt must follow the plan's policy exactly."""
    attempts = receipt["attempts"]
    seeds = [row["seed"] for row in attempts]
    if mode == "fail_release" and len(attempts) > 1:
        raise ValueError("confirmation was redrawn under the fail-release policy")
    if not attempts or seeds != confirmation_streams(entry, mode)[:len(attempts)] or hidden_seed != seeds[-1]:
        raise ValueError("confirmation does not match the preassigned stream policy")
    if mode == "resample_until_separated" and (attempts[-1]["status"] != "passed"
                                              or any(row["status"] != "failed" for row in attempts[:-1])):
        raise ValueError("confirmation does not match the preassigned separation policy")


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


def assign_modalities(plan, seed):
    """Fix one observation layout per parent, before generating the parents."""
    if type(seed) is not int or seed < 0:
        raise ValueError("a nonnegative allocation seed is required")
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
