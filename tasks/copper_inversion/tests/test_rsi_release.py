"""RSI release layout, weak judge, hole audit and qualification on small synthetic fixtures.

The fixture mimics the files of `generate_pilot` and `generate_confirmation`
without GemPy: tiny boxes as geology and hand-placed boreholes.
"""

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from tasks.copper_inversion.experiments.rsi import plan as rsiplan
from tasks.copper_inversion.experiments.rsi.measurements import (FAMILIES, MODALITIES,
                                                                 modality_balance, spatial_groups)
from tasks.copper_inversion.experiments.rsi.release import build, load_suite
from tasks.copper_inversion.experiments.rsi.release_dev import release as release_dev
from tasks.copper_inversion.experiments.v2.generate_pilot import make_plan
from tasks.copper_inversion.v2.data import read_problem, save_problem
from tasks.copper_inversion.v2.geological_policy import POLICY_VERSION, public_priors
from tasks.copper_inversion.v2.qualification import TIER_KEYS, load_tiers, verify_release, verify_tiers

SOURCE = Path(__file__).parents[1] / "experiments" / "v2"
SHAPE = (8, 8, 6)
PUBLIC_HOLES = [(1.5, 1.5), (5.5, 1.5), (1.5, 5.5), (5.5, 5.5)]
HIDDEN_HOLES = [(3.5, 3.5), (3.5, .5), (.5, 3.5), (6.5, 6.5)]
SMALL = dict(train=12, validation=4, test_id=4, test_ood=4)


def forbidden():
    mask = np.zeros(SHAPE, bool)
    mask[:, :, -1] = True
    return mask


def truth(k):
    mask = np.zeros(SHAPE, bool)
    mask[1:3 + k % 4, 1:3 + (k // 4) % 4, :2 + k % 3] = True
    return mask


def problem(case_id, holes, outcrop, shift=0.):
    groups = np.repeat(np.arange(len(holes)), 5)
    points = np.array([(x, y, z + .5) for x, y in holes for z in range(5)])
    drill = dict(name="drill", points=points, labels=(points[:, 2] < 2).astype(float), groups=groups,
                 weight=1., transition_scale=.5, contact_sigma=.5, label_error=0.)
    observations = [drill]
    if outcrop:
        axis = np.arange(.5 + shift, 7.6 - shift, 1.)
        xy = np.stack(np.meshgrid(axis, axis), -1).reshape(-1, 2)
        xyz = np.column_stack((xy, np.full(len(xy), 4.9)))
        observations.append(dict(name="outcrop", points=xyz, labels=np.zeros(len(xyz)),
                                 groups=spatial_groups({"origin": [0.] * 3, "shape": SHAPE, "spacing": [1.] * 3}, xyz, 4),
                                 weight=1., transition_scale=.5, contact_sigma=.5, label_error=0.))
    return dict(schema_version="problem_schema_v2", geometry_version="occupancy_v2", objective_version="objective_v2",
                case_id=case_id, shape=list(SHAPE), origin=[0., 0., 0.], spacing=[1., 1., 1.],
                domain_mask=np.ones(SHAPE, bool), forbidden_mask=forbidden(),
                open_boundary_faces=["x-", "x+", "y-", "y+", "z-", "z+"], observations=observations,
                priors=public_priors(~forbidden(), [1., 1., 1.]))


def pack(directory, digest, seed, holes, shift, mask):
    manifest = dict(stage="development-only", parent_digest=digest, observation_seed=seed, cases=[])
    for modality in MODALITIES:
        case_id = hashlib.sha256(f"{digest}{modality}{seed}".encode()).hexdigest()[:20]
        save_problem(problem(case_id, holes, modality == "drill_outcrop", shift), directory / case_id)
        manifest["cases"].append(dict(case_id=case_id, modality=modality, truth="private/truth.npy"))
    (directory / "private").mkdir(parents=True)
    np.save(directory / "private" / "truth.npy", mask)
    (directory / "private-manifest.json").write_text(json.dumps(manifest))


def attempt(seed, holes, status):
    """A generation-time audit receipt row, as generate_confirmation writes it."""
    distances = [0. if hole in PUBLIC_HOLES else 2. for hole in holes]
    return dict(seed=seed, status=status, confirmation_holes=len(holes), overlapping_holes=distances.count(0.),
                transition_scale=.5, nearest_hole_distances=distances, nearest_hole_distance_min=min(distances),
                nearest_hole_distance_median=float(np.median(distances)))


def dataset(root, release="rsi-r1", counts=SMALL, per_family=2, policy="resample_until_separated", historical=None,
            overlap=(), redraw=(), formal_inputs=True):
    """A generated population, written in the order the real pipeline writes it."""
    rsi = rsiplan.make_generation_plan(release, 7, policy, historical, counts=counts, short_per_family=per_family)
    sha = rsiplan.write_generation_plan(root, rsi)
    k = 0
    for part in ("id", "ood"):
        base = root / part
        base.mkdir()
        plan = make_plan(rsi[part + "_seed"], rsiplan.part_splits(rsi, part), part == "ood", SOURCE)
        rsiplan.part_plan(plan, rsi, sha, part)
        rsiplan.write_part_plan(base, plan)
        records = []
        for entry in plan["parents"]:
            name = f"{entry['split']}-{entry['index']:02}"
            mask = truth(k)
            k += 1
            parent = base / "parents" / f"{name}-0"
            parent.mkdir(parents=True)
            np.savez(parent / "geology.npz", truth=mask, forbidden_mask=forbidden())
            digest = hashlib.sha256(f"{part}:{name}".encode()).hexdigest()
            pack(base / "packs" / name, digest, entry["observation_seed"], PUBLIC_HOLES, 0., mask)
            streams = entry["confirmation_seeds"]
            hidden = HIDDEN_HOLES + ([PUBLIC_HOLES[0]] if name in overlap else [])
            # A redrawn body's first stream overlapped; the second one is used.
            attempts = ([attempt(streams[0], HIDDEN_HOLES + [PUBLIC_HOLES[1]], "failed")] if name in redraw else [])
            attempts.append(attempt(streams[len(attempts)], hidden, "passed"))
            pack(base / "confirmation" / name, digest, attempts[-1]["seed"], hidden, .25, mask)
            audits = base / "confirmation-audits"
            audits.mkdir(exist_ok=True)
            (audits / f"{name}.json").write_text(json.dumps(dict(attempts=attempts)))
            records.append(dict(parent=name, status="accepted", parent_directory=f"parents/{name}-0"))
        (base / "progress.json").write_text(json.dumps(dict(records=records)))
        (base / "completion.json").write_text("{}")
        (base / "split-audit.json").write_text(json.dumps(dict(status="passed")))
    (root / "confirmation-complete.json").write_text("{}")
    inputs = {}
    if formal_inputs:
        (root / "qa-pipeline-complete.json").write_text("{}")
        (root / "population-qa.json").write_text(json.dumps(dict(status="passed", parents=[])))
        for name, value in (("calibration", dict(status="passed")), ("isolation", dict(status="passed")),
                            ("geological_constraints", dict(status="passed", policy=POLICY_VERSION,
                                                            calibration_scope="independent_development",
                                                            used_evaluation_truth=False))):
            inputs[name] = root / f"{name}-input.json"
            inputs[name].write_text(json.dumps(value))
    return rsi, inputs


def historical(tmp_path):
    legacy, dev = tmp_path / "legacy", tmp_path / "rsi-dev"
    for path, shape in ((legacy, (slice(0, 6), slice(0, 1), slice(0, 5))), (dev, (slice(0, 1), slice(0, 7), slice(0, 4)))):
        (path / "x").mkdir(parents=True)
        mask = np.zeros(SHAPE, bool)
        mask[shape] = True
        mask[0, 0, 0] = False
        np.save(path / "x" / "truth.npy", mask)
    (dev / "development-receipt.json").write_text("{}")
    return {"legacy_release": legacy, "rsi_dev": dev}


def formal(tmp_path, **kwargs):
    root = tmp_path / "dataset"
    rsi, inputs = dataset(root, historical=historical(tmp_path), **kwargs)
    out = tmp_path / "release"
    result = build(root, out, inputs["calibration"], inputs["isolation"], inputs["geological_constraints"])
    return rsi, root, out, result


def manifests(out):
    return {str(p.relative_to(out)): json.loads(p.read_text()) for p in out.glob("*/manifest.json")}


def test_full_plan_fixes_layout_and_short_tier_before_generation():
    """AT-20: the plan file alone determines the short tier and each parent's layout."""
    rsi = rsiplan.make_generation_plan("rsi-r1", 2026, historical={"legacy_release": "/a", "rsi_dev": "/b"})
    assert rsi["counts"] == dict(train=24, validation=8, test_id=32, test_ood=16)
    assert rsi["layout"]["sealed_bodies"] == dict(test_id=32, test_ood=16) and rsi["layout"]["execution_seeds"] == [47]
    plans = {}
    for part in ("id", "ood"):
        plan = make_plan(rsi[part + "_seed"], rsiplan.part_splits(rsi, part), part == "ood", SOURCE)
        plans[part] = rsiplan.part_plan(plan, rsi, "a" * 64, part)
        replay = rsiplan.replay_part_plan(json.loads(json.dumps(plans[part])), rsi, "a" * 64, part)
        assert replay == json.loads(json.dumps(plans[part]))
    short = plans["id"]["short_train"]
    assert len(short) == 12 and all(sum(r["family"] == f for r in short) == 3 for f in FAMILIES)
    assert all({r["noisy"] for r in short if r["family"] == f} == {False, True} for f in FAMILIES)
    balance = modality_balance(plans["id"]["parents"] + plans["ood"]["parents"])
    assert {k: sum(v.values()) for k, v in balance.items() if "/" not in k} == rsi["counts"]
    assert all(v["drill"] == v["drill_outcrop"] for k, v in balance.items() if "/" not in k)
    ood = plans["ood"]["parents"]
    assert sum(r["shift"] == "structure" for r in ood) == 8 and sum(r["coverage"] == "clustered" for r in ood) == 8
    again = rsiplan.make_generation_plan("rsi-r1", 2026, historical={"legacy_release": "/a", "rsi_dev": "/b"})
    assert again == rsi
    with pytest.raises(ValueError, match="rsi_dev"):
        rsiplan.make_generation_plan("rsi-r1", 2026, historical={"legacy_release": "/a"})


def test_development_release_has_one_pack_per_parent_and_all_extra_manifests(tmp_path):
    root = tmp_path / "dataset"
    rsi, _ = dataset(root, release="rsi-dev", counts=None, per_family="preset", formal_inputs=False)
    out = tmp_path / "release"
    result = release_dev(root, out)
    assert result["status"] == "development_only" and result["counts"] == dict(train=8, validation=4, test_id=4, test_ood=4)
    receipt = json.loads((out / "development-receipt.json").read_text())
    assert set(receipt["manifest_sha256"]) == {"train", "validation", "test_id", "test_ood", "train_s_q25",
                                               "validation_q25", "train_s_q1seeds", "train_q1_both"}
    found = manifests(out)
    for split in ("train", "validation", "test_id", "test_ood"):
        cases = found[f"{split}/manifest.json"]["cases"]
        assert len({c["parent_id"] for c in cases}) == len(cases) == rsi["counts"][split]
        assert found[f"{split}/manifest.json"]["seeds"] == [47]
    # rsi-dev uses all 8 training parents for the short tier.
    assert receipt["tiers"]["S"]["train_manifest"] == "train/manifest.json" and receipt["tiers"]["S"]["train_bodies"] == 8
    assert set(receipt["tiers"]["S"]) == TIER_KEYS
    assert json.loads((root / "hole-audit.json").read_text())["status"] == "passed"


def test_formal_release_qualification_covers_every_manifest(tmp_path):
    """AT-24 (fixture scale): every manifest is sealed and passes verify_release."""
    rsi, root, out, result = formal(tmp_path)
    record = json.loads((out / "qualification.json").read_text())
    found = manifests(out)
    assert set(found) == set(record["manifest_sha256"]) == set(result["manifests"])
    assert set(found) == {"train/manifest.json", "train_s/manifest.json", "validation/manifest.json",
                          "test_id/manifest.json", "test_ood/manifest.json", "train_s_q25/manifest.json",
                          "validation_q25/manifest.json", "train_s_q1seeds/manifest.json",
                          "train_q1_both/manifest.json"}
    for path in found:
        assert verify_release(out / path) == record
        packs = record["layout"]["packs_per_body_by_manifest"].get(path, record["layout"]["packs_per_body"])
        assert load_suite(out / path, packs)["stage"] == "formal"
    # The tier table has exactly the contracts §2.1 shape.
    assert record["tiers"] == {
        "S": {"solver_seconds": 30, "train_manifest": "train_s/manifest.json", "train_bodies": 8,
              "validation_manifest": "validation/manifest.json",
              "sealed_manifests": ["test_id/manifest.json", "test_ood/manifest.json"], "slots": 321},
        "F": {"solver_seconds": 120, "train_manifest": "train/manifest.json", "train_bodies": 12,
              "validation_manifest": "validation/manifest.json",
              "sealed_manifests": ["test_id/manifest.json", "test_ood/manifest.json"], "slots": 321}}
    assert load_tiers(out) == record["tiers"]
    assert record["layout"] == {"packs_per_body": 1, "execution_seeds": [47], "assignment_rule": "alternate-within-family-v1",
                                "sealed_bodies": {"test_id": 4, "test_ood": 4},
                                "packs_per_body_by_manifest": {"train_q1_both/manifest.json": 2}}
    assert record["q1"] == {"execution_seeds": [47, 48, 49], "execution_seeds_manifest": "train_s_q1seeds/manifest.json",
                            "both_packs_manifest": "train_q1_both/manifest.json", "solver_seconds": 30}
    assert record["judges"]["q25"] == {"objective": "confirmation", "train_manifest": "train_s_q25/manifest.json",
                                       "validation_manifest": "validation_q25/manifest.json"}
    assert found["train_s_q1seeds/manifest.json"]["seeds"] == [47, 48, 49]
    assert [c["case_id"] for c in found["train_s_q1seeds/manifest.json"]["cases"]] == \
        [c["case_id"] for c in found["train_s/manifest.json"]["cases"]]
    # The short tier is the planned subset, and the unselected packs are in no manifest but the Q1 both-pack one.
    layout = json.loads((out / "evidence" / "layout.json").read_text())
    names = {f"train-{i:02}" for i in (0, 1, 2, 3, 4, 5, 6, 7)}
    assert set(layout["short_tier"]["parents"]) == names
    unselected = json.loads((out / "unselected" / "index.json").read_text())["packs"]
    assert len(unselected) == sum(SMALL.values())
    for path, manifest in found.items():
        ids = {c["case_id"] for c in manifest["cases"]}
        extra = ids & {row["case_id"] for row in unselected}
        if path == "train_q1_both/manifest.json":
            assert extra == {row["case_id"] for row in unselected if row["split"] == "train"}
            assert len(manifest["cases"]) == 2 * SMALL["train"]
        else:
            assert not extra
    for key, counts in layout["balance"].items():
        assert abs(counts["drill"] - counts["drill_outcrop"]) <= 1
    hole = json.loads((out / "evidence" / "hole_audit.json").read_text())
    assert hole["status"] == "passed" and hole["nearest_hole_distance"]["holes"] == 4 * sum(SMALL.values())
    assert set(hole["nearest_hole_distance_by_split"]) == set(SMALL)
    assert json.loads((out / "evidence" / "historical_disjointness.json").read_text())["historical_sources"].keys() == \
        {"legacy_release", "rsi_dev"}


def test_weak_judge_keeps_whole_holes_and_public_files(tmp_path):
    """AT-21: whole holes, at least one per case, a quarter within one hole, same public hashes."""
    _, _, out, _ = formal(tmp_path)
    found = manifests(out)
    for name, standard in (("train_s_q25", "train_s"), ("validation_q25", "validation")):
        weak, full = found[f"{name}/manifest.json"]["cases"], found[f"{standard}/manifest.json"]["cases"]
        assert [c["case_id"] for c in weak] == [c["case_id"] for c in full]
        for thin, whole in zip(weak, full):
            assert {k: thin["sha256"][k] for k in ("problem.json", "problem.npz", "truth.npy")} == \
                {k: whole["sha256"][k] for k in ("problem.json", "problem.npz", "truth.npy")}
            a = read_problem(out / name / thin["truth"] / "confirmation")
            b = read_problem(out / standard / whole["truth"] / "confirmation")
            for kept, complete in zip(a["observations"], b["observations"]):
                if kept["name"] == "drill":
                    groups, all_groups = kept["groups"], complete["groups"]
                else:
                    groups = spatial_groups(a, kept["points"], 4)
                    all_groups = spatial_groups(b, complete["points"], 4)
                total, chosen = len(np.unique(all_groups)), np.unique(groups)
                assert len(chosen) >= 1 and abs(len(chosen) - total / 4) <= 1
                for group in chosen:
                    assert np.count_nonzero(groups == group) == np.count_nonzero(all_groups == group)


def test_resampled_confirmation_records_streams_and_attempts(tmp_path):
    """Default policy: per-body redraws are allowed and fully recorded."""
    rsi, root, out, _ = formal(tmp_path, redraw=("train-03", "test_ood-02"))
    assert rsi["confirmation_policy"]["mode"] == "resample_until_separated"
    hole = json.loads((out / "evidence" / "hole_audit.json").read_text())
    assert hole["policy"] == "resample_until_separated" and hole["redrawn_parents"] == ["test_ood-02", "train-03"]
    assert hole["attempts_per_parent"] == {"1": sum(SMALL.values()) - 2, "2": 2}
    rows = {row["parent"]: row for row in hole["confirmation_streams"]}
    assert len(rows) == sum(SMALL.values()) and rows["train-03"]["stream_index"] == 1
    assert [a["status"] for a in rows["train-03"]["attempts"]] == ["failed", "passed"]
    assert rows["train-03"]["selected_stream"] == rows["train-03"]["attempts"][-1]["seed"]
    # The unconditioned first-stream distances include the rejected overlaps.
    assert hole["first_stream_nearest_hole_distance"]["distance"][0] == 0.
    assert hole["nearest_hole_distance"]["distance"][0] > 0.


def test_redraw_is_refused_under_the_fail_release_policy(tmp_path):
    root = tmp_path / "dataset"
    _, inputs = dataset(root, historical=historical(tmp_path), policy="fail_release", redraw=("train-03",))
    with pytest.raises(ValueError, match="redrawn"):
        build(root, tmp_path / "release", inputs["calibration"], inputs["isolation"], inputs["geological_constraints"])


@pytest.mark.parametrize("policy", ["fail_release", "resample_until_separated"])
def test_copied_public_hole_fails_the_release(tmp_path, policy):
    """AT-22 negative control: one public hole copied into one confirmation.

    Under resampling the generator would have redrawn this body; a receipt that
    claims a pass is still caught by the release's own audit."""
    root = tmp_path / "dataset"
    _, inputs = dataset(root, historical=historical(tmp_path), overlap=("validation-01",), policy=policy)
    with pytest.raises(ValueError, match="hole overlap in 1 parents.*validation-01"):
        build(root, tmp_path / "release", inputs["calibration"], inputs["isolation"], inputs["geological_constraints"])
    assert not (tmp_path / "release").exists()
    report = json.loads((root / "hole-audit.json").read_text())
    assert report["status"] == "failed" and report["overlapping_parents"] == ["validation-01"]
    assert report["policy"] == policy and report["nearest_hole_distance"]["distance"][0] == 0.


def test_plan_hash_must_precede_geology_and_plans_cannot_change(tmp_path):
    root = tmp_path / "dataset"
    _, inputs = dataset(root, historical=historical(tmp_path))
    args = (inputs["calibration"], inputs["isolation"], inputs["geological_constraints"])
    geology = next((root / "id" / "parents").glob("*/geology.npz"))
    receipt = (root / "id" / "plan-receipt.json").stat().st_mtime_ns
    os.utime(geology, ns=(receipt - 10**9, receipt - 10**9))
    with pytest.raises(ValueError, match="before geology"):
        build(root, tmp_path / "r1", *args)
    os.utime(geology, ns=(receipt + 10**9, receipt + 10**9))
    plan = json.loads((root / "id" / "plan.json").read_text())
    changed = copy.deepcopy(plan)
    first = changed["parents"][0]
    first["selected_modality"] = "drill" if first["selected_modality"] == "drill_outcrop" else "drill_outcrop"
    (root / "id" / "plan.json").write_text(json.dumps(changed, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError, match="changed after its hash"):
        build(root, tmp_path / "r2", *args)
    (root / "id" / "plan.json").write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError, match="before geology"):
        build(root, tmp_path / "r3", *args)


def test_historical_collision_with_rsi_dev_is_refused(tmp_path):
    sources = historical(tmp_path)
    np.save(sources["rsi_dev"] / "x" / "truth.npy", truth(0))
    root = tmp_path / "dataset"
    _, inputs = dataset(root, historical=sources)
    with pytest.raises(ValueError, match="collides"):
        build(root, tmp_path / "release", inputs["calibration"], inputs["isolation"], inputs["geological_constraints"])


def test_formal_release_requires_evidence_and_tier_shape(tmp_path):
    root = tmp_path / "dataset"
    dataset(root, historical=historical(tmp_path))
    with pytest.raises(ValueError, match="formal release requires"):
        build(root, tmp_path / "release")
    record = {"manifest_sha256": {"a/manifest.json": "x", "b/manifest.json": "y"},
              "tiers": {t: {"solver_seconds": 30, "train_manifest": "a/manifest.json", "train_bodies": 1,
                            "validation_manifest": "a/manifest.json", "sealed_manifests": ["b/manifest.json"],
                            "slots": 321} for t in ("S", "F")}}
    verify_tiers(record)
    for broken in ({**record["tiers"]["S"], "extra": 1}, {**record["tiers"]["S"], "train_manifest": "c/manifest.json"},
                   {**record["tiers"]["S"], "solver_seconds": 0}):
        with pytest.raises(ValueError):
            verify_tiers({**record, "tiers": {**record["tiers"], "S": broken}})


def test_legacy_release_counts_come_from_the_generation_plan(tmp_path):
    from tasks.copper_inversion.experiments.v2.release_independent import expected_cases
    assert expected_cases(tmp_path) == 112
    (tmp_path / "generation-plan.json").write_text(json.dumps(dict(counts=dict(train=24, validation=8, test_id=32, test_ood=16))))
    assert expected_cases(tmp_path) == 160
    (tmp_path / "generation-plan.json").write_text(json.dumps(dict(schema="rsi-generation-plan-v1")))
    with pytest.raises(ValueError, match="experiments.rsi.release"):
        expected_cases(tmp_path)
