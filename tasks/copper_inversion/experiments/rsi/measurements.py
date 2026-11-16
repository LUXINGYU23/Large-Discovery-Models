"""Whole-group thinning and public/confirmation borehole overlap checks."""

import copy
import hashlib
import math

import numpy as np
from scipy.spatial import cKDTree


def thin_confirmation(problem, seed):
    """Select a deterministic quarter of each observation's complete groups."""
    if type(seed) is not int or seed < 0:
        raise ValueError("a nonnegative thinning seed is required")
    result = copy.deepcopy(problem)
    selected = {}
    for index, observation in enumerate(result["observations"]):
        groups = np.asarray(observation["groups"])
        unique = np.unique(groups)
        if not len(unique):
            selected[observation["name"]] = []
            continue
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        count = max(1, int(math.floor(len(unique) / 4.0 + .5)))
        keep = np.sort(rng.choice(unique, size=count, replace=False))
        mask = np.isin(groups, keep)
        for key in ("points", "labels", "groups"):
            observation[key] = np.asarray(observation[key])[mask].copy()
        selected[observation["name"]] = [int(value) for value in keep]
    return result, selected


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
        if np.any(np.mean(distances <= threshold, axis=1) >= .5):
            overlapping += 1
    return {"status": "failed" if overlapping else "passed", "confirmation_holes": len(hidden_groups),
            "overlapping_holes": overlapping,
            "nearest_hole_distance_min": min(minima),
            "nearest_hole_distance_median": float(np.median(minima))}


def preselect_short_train(plan):
    """Choose the first three planned training parents in each family."""
    parents = [row for row in plan["parents"] if row["split"] == "train"]
    families = ("folded", "lens", "multiple", "faulted")
    result = []
    for family in families:
        options = sorted((row for row in parents if row["family"] == family), key=lambda row: row["index"])
        selected = options[:3]
        if len(selected) != 3 or {row["noisy"] for row in selected} != {False, True}:
            raise ValueError("generation plan lacks three noise-balanced parents per family")
        result.extend(selected)
    if len({(row["split"], row["index"]) for row in result}) != 12:
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
                row["selected_modality"] = ("drill", "drill_outcrop")[(start + offset) % 2]
    plan["modality_allocation"] = {"seed": seed, "rule": "seeded split-balanced alternation by index within family"}
    return plan
