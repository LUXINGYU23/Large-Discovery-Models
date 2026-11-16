"""Whole-borehole selection and overlap policy on explicit synthetic data."""

import copy
import numpy as np
import pytest

from tasks.copper_inversion.experiments.rsi.measurements import (
    assign_modalities, audit_holes, check_confirmation_attempts, check_thinning, confirmation_mode,
    confirmation_streams, hole_distance_summary, modality_balance, preselect_short_train, spatial_groups,
    thin_confirmation, CONFIRMATION_POLICIES)


def problem(outcrop=False):
    groups = np.repeat(np.arange(8), 4)
    points = np.column_stack((groups * 10., np.zeros(len(groups)), np.tile(np.arange(4), 8)))
    observations = [{"name": "drill", "groups": groups, "points": points,
                     "labels": np.arange(len(groups)), "transition_scale": 1.}]
    if outcrop:
        xy = np.stack(np.meshgrid(np.arange(1., 80., 2.5), np.arange(1., 80., 2.5)), -1).reshape(-1, 2)
        observations.append({"name": "outcrop", "groups": np.zeros(len(xy), int),
                             "points": np.column_stack((xy, np.full(len(xy), 9.))),
                             "labels": np.zeros(len(xy)), "transition_scale": 1.})
    return {"origin": [0., 0., 0.], "shape": [8, 8, 10], "spacing": [10., 10., 1.], "observations": observations}


def test_thinning_never_splits_a_borehole_or_mutates_input():
    original = problem()
    saved = copy.deepcopy(original)
    thinned, chosen = thin_confirmation(original, 11)
    again, same = thin_confirmation(original, 11)
    assert same == chosen and len(chosen["drill"]["kept"]) == 2 and chosen["drill"]["groups_total"] == 8
    assert np.array_equal(thinned["observations"][0]["points"], again["observations"][0]["points"])
    assert np.array_equal(original["observations"][0]["points"], saved["observations"][0]["points"])
    group_values = thinned["observations"][0]["groups"]
    for group in chosen["drill"]["kept"]:
        assert np.count_nonzero(group_values == group) == 4
    assert check_thinning(original, thinned, chosen)["drill"] == {"groups_total": 8, "groups_kept": 2}


def test_outcrop_is_thinned_by_spatial_cells_not_its_group_field():
    original = problem(outcrop=True)
    thinned, chosen = thin_confirmation(original, 5)
    assert chosen["outcrop"]["grouping"] == "spatial-4x4" and chosen["outcrop"]["groups_total"] == 16
    assert len(chosen["outcrop"]["kept"]) == 4
    cells = spatial_groups(original, thinned["observations"][1]["points"], 4)
    assert set(cells.tolist()) == set(chosen["outcrop"]["kept"])
    check_thinning(original, thinned, chosen)
    broken = copy.deepcopy(thinned)
    broken["observations"][1]["points"] = broken["observations"][1]["points"][1:]
    with pytest.raises(ValueError, match="split a whole group"):
        check_thinning(original, broken, chosen)


def test_thinning_keeps_at_least_one_hole():
    single = problem()
    single["observations"][0]["groups"] = np.zeros(32, int)
    _, chosen = thin_confirmation(single, 3)
    assert chosen["drill"]["kept"] == [0]


def test_hole_audit_fails_on_reused_hole_and_passes_separated_holes():
    original = problem()
    same = copy.deepcopy(original)
    assert audit_holes(original, same)["status"] == "failed"
    same["observations"][0]["points"][:, 1] += 100.
    result = audit_holes(original, same)
    assert result["status"] == "passed" and len(result["nearest_hole_distances"]) == 8
    summary = hole_distance_summary([result])
    assert summary["holes"] == 8 and summary["distance"][0] == pytest.approx(100.)
    assert summary["histogram_transition_scales"]["counts"][-1] == 8


def test_hole_audit_detects_one_copied_public_hole():
    public = problem()
    hidden = copy.deepcopy(public)
    hidden["observations"][0]["points"][:, 1] += 100.
    drill = hidden["observations"][0]
    copied = public["observations"][0]["groups"] == 3
    drill["points"] = np.vstack((drill["points"], public["observations"][0]["points"][copied]))
    drill["groups"] = np.concatenate((drill["groups"], np.full(4, 99)))
    result = audit_holes(public, hidden)
    assert result["status"] == "failed" and result["overlapping_holes"] == 1


def test_short_tier_is_chosen_from_plan_only():
    families = ("folded", "lens", "multiple", "faulted")
    parents = [{"split": "train", "index": i, "family": families[i % 4],
                "noisy": (i // 4 + i % 2) % 2 == 1, "lineage_id": str(i)} for i in range(24)]
    selection = preselect_short_train({"parents": parents})
    assert len(selection) == 12
    assert {(row["family"], row["noisy"]) for row in selection} == {(f, n) for f in families for n in (False, True)}
    assert {row["index"] for row in selection} == set(range(12))
    assert len(preselect_short_train({"parents": parents}, per_family=2)) == 8


def test_modality_assignment_is_reproducible_and_balanced_within_family():
    parents = [{"split": split, "family": family, "index": i}
               for split in ("train", "validation")
               for family in ("folded", "lens") for i in range(5)]
    plan = {"parents": parents}
    assigned = assign_modalities(copy.deepcopy(plan), 45)
    assert assigned == assign_modalities(copy.deepcopy(plan), 45)
    assert assigned["modality_allocation"]["rule"] == "alternate-within-family-v1"
    assert all("selected_modality" not in row for row in plan["parents"])
    for split in ("train", "validation"):
        split_values = [r["selected_modality"] for r in assigned["parents"] if r["split"] == split]
        assert split_values.count("drill") == split_values.count("drill_outcrop")
        for family in ("folded", "lens"):
            values = [r["selected_modality"] for r in assigned["parents"]
                      if (r["split"], r["family"]) == (split, family)]
            assert abs(values.count("drill") - values.count("drill_outcrop")) == 1
    modality_balance(assigned["parents"])
    for row in assigned["parents"][:5]:
        row["selected_modality"] = "drill"
    with pytest.raises(ValueError, match="unbalanced"):
        modality_balance(assigned["parents"])


def test_confirmation_policies_differ_only_in_redrawing():
    entry = {"confirmation_seeds": list(range(100, 116))}
    assert confirmation_streams(entry, "fail_release") == [100]
    assert len(confirmation_streams(entry, "resample_until_separated")) == 16
    legacy = {"confirmation_policy": {"rule": CONFIRMATION_POLICIES["resample_until_separated"]}}
    assert confirmation_mode(legacy) == "resample_until_separated"
    with pytest.raises(ValueError):
        confirmation_mode({"confirmation_policy": {"mode": "fail_release", "rule": "other"}})
    # Fail-release keeps the single, possibly overlapping, stream; the release audit decides.
    check_confirmation_attempts({"attempts": [{"seed": 100, "status": "failed"}]}, entry, "fail_release", 100)
    with pytest.raises(ValueError, match="redrawn"):
        check_confirmation_attempts({"attempts": [{"seed": 100, "status": "failed"}, {"seed": 101, "status": "passed"}]},
                                    entry, "fail_release", 101)
    check_confirmation_attempts({"attempts": [{"seed": 100, "status": "failed"}, {"seed": 101, "status": "passed"}]},
                                entry, "resample_until_separated", 101)
    with pytest.raises(ValueError):
        check_confirmation_attempts({"attempts": [{"seed": 100, "status": "failed"}]}, entry,
                                    "resample_until_separated", 100)
