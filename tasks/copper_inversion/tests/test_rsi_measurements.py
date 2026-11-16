"""Whole-borehole selection and overlap policy on explicit synthetic data."""

import copy
import numpy as np

from tasks.copper_inversion.experiments.rsi.measurements import assign_modalities, audit_holes, preselect_short_train, thin_confirmation


def problem():
    groups = np.repeat(np.arange(8), 4)
    points = np.column_stack((groups * 10., np.zeros(len(groups)), np.tile(np.arange(4), 8)))
    return {"observations": [{"name": "drill", "groups": groups, "points": points,
                              "labels": np.arange(len(groups)), "transition_scale": 1.}]}


def test_thinning_never_splits_a_borehole_or_mutates_input():
    original = problem()
    saved = copy.deepcopy(original)
    thinned, chosen = thin_confirmation(original, 11)
    again, same = thin_confirmation(original, 11)
    assert same == chosen and len(chosen["drill"]) == 2
    assert np.array_equal(thinned["observations"][0]["points"], again["observations"][0]["points"])
    assert np.array_equal(original["observations"][0]["points"], saved["observations"][0]["points"])
    group_values = thinned["observations"][0]["groups"]
    for group in chosen["drill"]:
        assert np.count_nonzero(group_values == group) == 4


def test_hole_audit_fails_on_reused_hole_and_passes_separated_holes():
    original = problem()
    same = copy.deepcopy(original)
    assert audit_holes(original, same)["status"] == "failed"
    same["observations"][0]["points"][:, 1] += 100.
    assert audit_holes(original, same)["status"] == "passed"


def test_short_tier_is_chosen_from_plan_only():
    families = ("folded", "lens", "multiple", "faulted")
    parents = [{"split": "train", "index": i, "family": families[i % 4],
                "noisy": (i // 4 + i % 2) % 2 == 1, "lineage_id": str(i)} for i in range(24)]
    selection = preselect_short_train({"parents": parents})
    assert len(selection) == 12
    assert {(row["family"], row["noisy"]) for row in selection} == {(f, n) for f in families for n in (False, True)}
    assert {row["index"] for row in selection} == set(range(12))


def test_modality_assignment_is_reproducible_and_balanced_within_family():
    parents = [{"split": split, "family": family, "index": i}
               for split in ("train", "validation")
               for family in ("folded", "lens") for i in range(5)]
    plan = {"parents": parents}
    assigned = assign_modalities(copy.deepcopy(plan), 45)
    assert assigned == assign_modalities(copy.deepcopy(plan), 45)
    assert all("selected_modality" not in row for row in plan["parents"])
    for split in ("train", "validation"):
        split_values = [r["selected_modality"] for r in assigned["parents"] if r["split"] == split]
        assert split_values.count("drill") == split_values.count("drill_outcrop")
        for family in ("folded", "lens"):
            values = [r["selected_modality"] for r in assigned["parents"]
                      if (r["split"], r["family"]) == (split, family)]
            assert abs(values.count("drill") - values.count("drill_outcrop")) == 1
