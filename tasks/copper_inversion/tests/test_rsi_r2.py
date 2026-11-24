"""Unit tests for rsi-r2 generator gates, 4-pack confirmations, calibration split, and determinism."""

import csv
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tasks.copper_inversion.experiments.rsi import plan as rsiplan
from tasks.copper_inversion.experiments.rsi.measurements import (
    ASSIGNMENT_RULE_V2, FAMILIES, MODALITIES, audit_pack_pair, check_confirmation_attempts,
    check_discoverability_attempts, merge_confirmations, modality_balance, ore_hole_count, spatial_groups)
from tasks.copper_inversion.experiments.rsi.release import OPEN_SPLITS, SEALED_SPLITS, build, load_suite
from tasks.copper_inversion.experiments.v2.generate_pilot import make_plan, quality, vertical_ore_runs
from tasks.copper_inversion.experiments.v2.observations import DENSITIES, _parse_cli_holes, parse_holes
from tasks.copper_inversion.v2.data import read_problem, save_problem
from tasks.copper_inversion.v2.geological_policy import POLICY_VERSION, public_priors
from tasks.copper_inversion.v2.qualification import load_tiers, verify_release
from tasks.copper_inversion.v2.suite import load_manifest, manifest_tiers

SOURCE = Path(__file__).parents[1] / "experiments" / "v2"
SHAPE = (10, 10, 10)
PUBLIC_HOLES = [(1.5, 1.5), (8.5, 1.5), (1.5, 8.5), (8.5, 8.5)]
CONF_PACK_HOLES = [
    [(3.5, 1.5), (3.5, 8.5)],
    [(6.5, 1.5), (6.5, 8.5)],
    [(1.5, 4.5), (8.5, 4.5)],
    [(4.5, 4.5), (6.5, 5.5)],
]
SMALL_R2 = dict(train=16, calibration=8, calibration_ood=4, validation=8, test_id=8, test_ood=6)


def _forbidden(shape=SHAPE):
    mask = np.zeros(shape, bool)
    mask[:, :, -1] = True
    return mask


def _truth(k, shape=SHAPE):
    mask = np.zeros(shape, bool)
    x0 = 1 + (k % 3)
    y0 = 1 + ((k // 3) % 3)
    z0 = 1 + ((k // 9) % 2)
    w = 3 + (k % 2)
    h = 3 + ((k // 2) % 2)
    mask[x0:x0 + w, y0:y0 + h, z0:z0 + 4] = True
    if k >= 18:
        mask[x0, y0, z0 + 4] = True
    if k >= 36:
        mask[x0 + 1, y0, z0 + 4] = True
    return mask


def _problem(case_id, holes, outcrop, *, ore_holes=2):
    groups = np.repeat(np.arange(len(holes)), 5)
    points = np.array([(x, y, z + .5) for x, y in holes for z in range(5)])
    labels = np.zeros(len(points), float)
    for g in range(min(ore_holes, len(holes))):
        labels[(groups == g) & (points[:, 2] < 3)] = 1.
    drill = dict(name="drill", points=points, labels=labels, groups=groups,
                 weight=1., transition_scale=.4, contact_sigma=.4, label_error=0.)
    observations = [drill]
    if outcrop:
        axis = np.arange(1.5, 8.6, 2.)
        xy = np.stack(np.meshgrid(axis, axis), -1).reshape(-1, 2)
        xyz = np.column_stack((xy, np.full(len(xy), 8.9)))
        observations.append(dict(
            name="outcrop", points=xyz, labels=np.zeros(len(xyz)),
            groups=spatial_groups({"origin": [0.] * 3, "shape": SHAPE, "spacing": [1.] * 3}, xyz, 4),
            weight=.25, transition_scale=.4, contact_sigma=.4, label_error=0.))
    return dict(schema_version="problem_schema_v2", geometry_version="occupancy_v2", objective_version="objective_v2",
                case_id=case_id, shape=list(SHAPE), origin=[0., 0., 0.], spacing=[1., 1., 1.],
                domain_mask=np.ones(SHAPE, bool), forbidden_mask=_forbidden(),
                open_boundary_faces=["x-", "x+", "y-", "y+", "z-", "z+"], observations=observations,
                priors=public_priors(~_forbidden(), [1., 1., 1.]))


def _write_pack(directory, digest, seed, holes, mask, *, ore_holes=2, holes_bounds=(8, 8)):
    manifest = dict(stage="development-only", parent_digest=digest, observation_seed=seed,
                    holes=list(holes_bounds), hole_count=len(holes), ore_holes=ore_holes,
                    observed_ore_holes=ore_holes, cases=[])
    for modality in MODALITIES:
        case_id = hashlib.sha256(f"{digest}:{modality}:{seed}:{holes_bounds}".encode()).hexdigest()[:20]
        save_problem(_problem(case_id, holes, modality == "drill_outcrop", ore_holes=ore_holes), directory / case_id)
        manifest["cases"].append(dict(case_id=case_id, modality=modality, truth="private/truth.npy"))
    (directory / "private").mkdir(parents=True)
    np.save(directory / "private" / "truth.npy", mask)
    (directory / "private-manifest.json").write_text(json.dumps(manifest))


def _conf_attempt(seed, holes, passed, *, ore_holes=1, pack_overlap=0, pub_overlap=0):
    distances = [0. if i < pub_overlap else 2.5 for i in range(len(holes))]
    return dict(seed=seed, status="passed" if passed else "failed",
                reason="passed" if passed else "failed",
                confirmation_holes=len(holes), overlapping_holes=pub_overlap,
                pack_overlapping_holes=pack_overlap, ore_holes=ore_holes,
                transition_scale=.4, nearest_hole_distances=distances,
                nearest_hole_distance_min=min(distances),
                nearest_hole_distance_median=float(np.median(distances)))


def _historical(base):
    legacy, dev = base / "legacy", base / "rsi-dev"
    for path, slc in ((legacy, (slice(0, 8), slice(0, 1), slice(0, 5))),
                      (dev, (slice(0, 1), slice(0, 8), slice(0, 5)))):
        (path / "x").mkdir(parents=True, exist_ok=True)
        mask = np.zeros(SHAPE, bool)
        mask[slc] = True
        np.save(path / "x" / "truth.npy", mask)
    (dev / "development-receipt.json").write_text("{}")
    return {"legacy_release": legacy, "rsi_dev": dev}


def _populate_part(base, plan, part, k_start, *, only_splits=None, holes_bounds=(8, 8),
                   public_ore_override=None, conf_ore_override=None, inter_pack_overlap=()):
    public_ore_override = public_ore_override or {}
    conf_ore_override = conf_ore_override or {}
    existing_records = []
    if (base / "progress.json").is_file():
        existing_records = json.loads((base / "progress.json").read_text())["records"]
    done = {r["parent"] for r in existing_records if r.get("status") == "accepted"}
    records = list(existing_records)
    for idx, entry in enumerate(plan["parents"]):
        if only_splits is not None and entry["split"] not in only_splits:
            continue
        name = f"{entry['split']}-{entry['index']:02}"
        if name in done:
            continue
        mask = _truth(k_start + idx)
        parent = base / "parents" / f"{name}-0"
        parent.mkdir(parents=True)
        np.savez(parent / "geology.npz", truth=mask, forbidden_mask=_forbidden())
        digest = hashlib.sha256(f"{part}:{entry['lineage_id']}".encode()).hexdigest()
        pub_ore = public_ore_override.get(name, 2)
        obs_seeds = entry["observation_seeds"]
        obs_attempts = [
            dict(seed=obs_seeds[0], attempt=0, hole_count=len(PUBLIC_HOLES), ore_holes=0, status="failed"),
            dict(seed=obs_seeds[1], attempt=1, hole_count=len(PUBLIC_HOLES), ore_holes=max(2, pub_ore), status="passed"),
        ] if name == "train-01" else [
            dict(seed=obs_seeds[0], attempt=0, hole_count=len(PUBLIC_HOLES), ore_holes=max(2, pub_ore), status="passed"),
        ]
        sel_obs = obs_attempts[-1]["seed"]
        _write_pack(base / "packs" / name, digest, sel_obs, PUBLIC_HOLES, mask,
                    ore_holes=pub_ore, holes_bounds=holes_bounds)
        obs_audits = base / "observation-audits"
        obs_audits.mkdir(exist_ok=True)
        (obs_audits / f"{name}.json").write_text(json.dumps(dict(
            parent=name, split=entry["split"], selected_seed=sel_obs,
            stream_index=len(obs_attempts) - 1, min_ore_holes=2 if part == "id" else 0,
            holes=list(holes_bounds), attempts=obs_attempts, status="passed")))

        c_seeds = entry["confirmation_seeds"]
        c_attempts = [_conf_attempt(c_seeds[0], CONF_PACK_HOLES[0], False, ore_holes=0)] if name == "train-02" else []
        start = len(c_attempts)
        chosen_seeds = c_seeds[start:start + 4]
        conf_dir = base / "confirmation" / name
        pack_rows = []
        for k, c_seed in enumerate(chosen_seeds):
            holes = list(CONF_PACK_HOLES[k])
            if name in inter_pack_overlap and k == 3:
                holes = list(CONF_PACK_HOLES[1])
            c_ore = conf_ore_override.get((name, k), 1)
            c_attempts.append(_conf_attempt(c_seed, holes, True, ore_holes=max(1, c_ore)))
            sub_dir = conf_dir / f"c{k}"
            _write_pack(sub_dir, digest, c_seed, holes, mask, ore_holes=c_ore, holes_bounds=(3, 12))
            sub_man = json.loads((sub_dir / "private-manifest.json").read_text())
            pack_rows.append(dict(pack=f"c{k}", observation_seed=c_seed,
                                  hole_count=len(holes), ore_holes=c_ore, cases=sub_man["cases"]))
            if k == 0:
                for item in sub_dir.iterdir():
                    if item.is_dir():
                        save_problem_or_copy = item.name
                        if item.name != "private":
                            save_problem(read_problem(item), conf_dir / item.name)
                (conf_dir / "private").mkdir(exist_ok=True)
                np.save(conf_dir / "private" / "truth.npy", mask)
                top_man = dict(sub_man, packs=pack_rows)
                (conf_dir / "private-manifest.json").write_text(json.dumps(top_man))
        top_man = json.loads((conf_dir / "private-manifest.json").read_text())
        top_man["packs"] = pack_rows
        (conf_dir / "private-manifest.json").write_text(json.dumps(top_man))
        c_audits = base / "confirmation-audits"
        c_audits.mkdir(exist_ok=True)
        (c_audits / f"{name}.json").write_text(json.dumps(dict(
            parent=name, split=entry["split"], selected_stream=chosen_seeds[0],
            selected_streams=chosen_seeds, packs=4, min_ore_holes=1 if part == "id" else 0,
            attempts=c_attempts, status="passed", policy="resample_until_separated")))
        records.append(dict(parent=name, status="accepted", parent_directory=f"parents/{name}-0",
                            holes=list(holes_bounds), volume_fraction=float(mask.sum() / (~_forbidden()).sum()),
                            median_vertical_thickness=4.))
    (base / "progress.json").write_text(json.dumps(dict(records=records)))
    (base / "completion.json").write_text("{}")
    (base / "split-audit.json").write_text(json.dumps(dict(status="passed")))


def _build_r2_dataset(root, *, master_seed=19, counts=SMALL_R2, hist=None, open_only=False,
                      holes_bounds=(8, 8), public_ore_override=None, conf_ore_override=None,
                      inter_pack_overlap=()):
    rsi = rsiplan.make_generation_plan(
        "rsi-r2", master_seed, "resample_until_separated", hist,
        counts=counts, short_per_family=2, medium_per_family=3)
    sha = rsiplan.write_generation_plan(root, rsi)
    k_offsets = {"id": 0, "ood": 32}
    for part in ("id", "ood"):
        base = root / part
        base.mkdir()
        splits = rsiplan.part_splits(rsi, part)
        plan = make_plan(rsi[part + "_seed"], splits, part == "ood", SOURCE,
                         generation_attempts=rsi["size_gates"]["generation_attempts"],
                         density_shift_splits=rsi["density_shift_splits"])
        rsiplan.part_plan(plan, rsi, sha, part)
        rsiplan.write_part_plan(base, plan)
        target = ({s for s, _ in splits if s not in SEALED_SPLITS} if open_only else None)
        _populate_part(base, plan, part, k_offsets[part], only_splits=target, holes_bounds=holes_bounds,
                       public_ore_override=public_ore_override, conf_ore_override=conf_ore_override,
                       inter_pack_overlap=inter_pack_overlap)
    (root / "confirmation-complete.json").write_text("{}")
    (root / "qa-pipeline-complete.json").write_text("{}")
    (root / "population-qa.json").write_text(json.dumps(dict(status="passed", parents=[])))
    inputs = {}
    for name, value in (("calibration", dict(status="passed")), ("isolation", dict(status="passed")),
                        ("geological_constraints", dict(status="passed", policy=POLICY_VERSION,
                                                        calibration_scope="independent_development",
                                                        used_evaluation_truth=False))):
        inputs[name] = root / f"{name}-input.json"
        inputs[name].write_text(json.dumps(value))
    return rsi, inputs


def test_parse_holes_and_density_presets():
    assert parse_holes(None) == (3, 12)
    assert parse_holes(8) == (8, 8)
    assert parse_holes([16]) == (16, 16)
    assert parse_holes((12, 24)) == (12, 24)
    assert parse_holes("lo") == (3, 12) and parse_holes("mid") == (12, 24) and parse_holes("hi") == (24, 48)
    assert _parse_cli_holes(["12"]) == (12, 12)
    assert _parse_cli_holes(["12", "24"]) == (12, 24)
    assert _parse_cli_holes(["mid"]) == DENSITIES["mid"]
    for invalid in (0, -3, True, (12, 8), (0, 5), [1, 2, 3], "ultra"):
        with pytest.raises(ValueError):
            parse_holes(invalid)


def test_vertical_ore_runs_and_id_size_thickness_gates(tmp_path):
    mask = np.zeros((4, 4, 8), bool)
    mask[1, 1, 1:4] = True
    mask[1, 1, 5:7] = True
    mask[2, 2, 2:6] = True
    runs = vertical_ore_runs(mask)
    assert sorted(runs.tolist()) == [2, 3, 4]

    forb = np.zeros((10, 10, 10), bool)
    forb[:, :, -1] = True  # 900 feasible voxels
    valid = np.zeros((10, 10, 10), bool)
    valid[2:6, 2:6, 2:5] = True  # 48 voxels -> 48/900 = 0.0533, closed, thickness 3
    p_ok = tmp_path / "ok"
    p_ok.mkdir()
    np.savez(p_ok / "geology.npz", truth=valid, forbidden_mask=forb)
    q = quality(p_ok, "lens", volume_bounds=(.04, .35), min_vertical_thickness=3)
    assert q["median_vertical_thickness"] == 3.0 and .04 <= q["volume_fraction"] <= .35

    thin = np.zeros((10, 10, 10), bool)
    thin[1:7, 1:7, 2:4] = True  # 72 voxels -> 0.08 volume, but thickness 2 < 3
    p_thin = tmp_path / "thin"
    p_thin.mkdir()
    np.savez(p_thin / "geology.npz", truth=thin, forbidden_mask=forb)
    assert quality(p_thin, "lens")["median_vertical_thickness"] == 2.0
    with pytest.raises(ValueError, match="vertical thickness quality gate"):
        quality(p_thin, "lens", volume_bounds=(.04, .35), min_vertical_thickness=3)

    tiny = np.zeros((10, 10, 10), bool)
    tiny[2:4, 2:4, 2:7] = True  # 20 voxels -> 20/900 = 0.0222 (< 0.04, though >= 0.02)
    p_tiny = tmp_path / "tiny"
    p_tiny.mkdir()
    np.savez(p_tiny / "geology.npz", truth=tiny, forbidden_mask=forb)
    assert quality(p_tiny, "lens")["volume_fraction"] == pytest.approx(20 / 900)
    with pytest.raises(ValueError, match="volume fraction quality gate"):
        quality(p_tiny, "lens", volume_bounds=(.04, .35), min_vertical_thickness=3)

    huge = np.zeros((10, 10, 10), bool)
    huge[1:9, 1:9, 1:7] = True  # 384/900 = 0.4267 > 0.35
    p_huge = tmp_path / "huge"
    p_huge.mkdir()
    np.savez(p_huge / "geology.npz", truth=huge, forbidden_mask=forb)
    with pytest.raises(ValueError, match="volume fraction quality gate"):
        quality(p_huge, "lens", volume_bounds=(.04, .35), min_vertical_thickness=3)


def test_discoverability_gate_and_negative_controls(tmp_path):
    p = _problem("c", PUBLIC_HOLES, False, ore_holes=2)
    assert ore_hole_count(p) == 2
    # True-ore count recorded in private-manifest overrides spurious noisy label flips.
    assert ore_hole_count(p, {"ore_holes": 1}) == 1
    seeds = [10, 20, 30, 40]
    check_discoverability_attempts(
        {"attempts": [{"seed": 10, "ore_holes": 1, "status": "failed"},
                      {"seed": 20, "ore_holes": 2, "status": "passed"}]},
        seeds, 2, 20)
    with pytest.raises(ValueError):
        check_discoverability_attempts(
            {"attempts": [{"seed": 10, "ore_holes": 2, "status": "failed"},
                          {"seed": 20, "ore_holes": 2, "status": "passed"}]},
            seeds, 2, 20)

    hist = _historical(tmp_path / "hist")
    root_pub = tmp_path / "ds_pub_fail"
    _, inputs = _build_r2_dataset(root_pub, hist=hist, public_ore_override={"train-00": 1})
    with pytest.raises(ValueError, match="discoverability gate failed.*train-00:public"):
        build(root_pub, tmp_path / "rel_pub", inputs["calibration"], inputs["isolation"], inputs["geological_constraints"])
    assert not (tmp_path / "rel_pub").exists()
    assert json.loads((root_pub / "hole-audit.json").read_text())["discoverability_failures"] == ["train-00:public"]

    root_conf = tmp_path / "ds_conf_fail"
    _, inputs2 = _build_r2_dataset(root_conf, hist=hist, conf_ore_override={("validation-00", 2): 0})
    with pytest.raises(ValueError, match="discoverability gate failed.*validation-00:c2"):
        build(root_conf, tmp_path / "rel_conf", inputs2["calibration"], inputs2["isolation"], inputs2["geological_constraints"])
    assert not (tmp_path / "rel_conf").exists()


def test_inter_pack_confirmation_overlap_fails_release(tmp_path):
    hist = _historical(tmp_path / "hist")
    root = tmp_path / "ds_pack_overlap"
    _, inputs = _build_r2_dataset(root, hist=hist, inter_pack_overlap=("calibration-01",))
    with pytest.raises(ValueError, match="hole overlap in 1 parents.*calibration-01"):
        build(root, tmp_path / "rel", inputs["calibration"], inputs["isolation"], inputs["geological_constraints"])
    audit = json.loads((root / "hole-audit.json").read_text())
    assert audit["status"] == "failed" and audit["inter_pack_overlapping_parents"] == ["calibration-01"]


def test_rsi_r2_preset_counts_calibration_and_factorial_balance():
    rsi = rsiplan.make_generation_plan("rsi-r2", 20261124, historical={"legacy_release": "/a", "rsi_dev": "/b"})
    assert rsi["counts"] == dict(train=48, calibration=24, calibration_ood=8, validation=16, test_id=48, test_ood=24)
    assert sum(rsi["counts"].values()) == 168
    assert rsi["confirmation_policy"]["packs"] == 4 and rsi["confirmation_policy"]["streams"] == 64
    assert rsi["layout"]["assignment_rule"] == ASSIGNMENT_RULE_V2
    assert set(rsi["tiers"]) == {"S", "M", "F"}
    plans = {}
    for part in ("id", "ood"):
        splits = rsiplan.part_splits(rsi, part)
        plan = make_plan(rsi[part + "_seed"], splits, part == "ood", SOURCE,
                         generation_attempts=rsi["size_gates"]["generation_attempts"],
                         density_shift_splits=rsi["density_shift_splits"])
        plans[part] = rsiplan.part_plan(plan, rsi, "b" * 64, part)
        assert rsiplan.replay_part_plan(json.loads(json.dumps(plans[part])), rsi, "b" * 64, part) == plans[part]
    assert len(plans["id"]["short_train"]) == 12 and len(plans["id"]["medium_train"]) == 24
    for entry in plans["id"]["parents"] + plans["ood"]["parents"]:
        assert len(entry["observation_seeds"]) == 16 and entry["observation_seeds"][0] == entry["observation_seed"]
        assert len(entry["confirmation_seeds"]) == 64 and len(set(entry["confirmation_seeds"])) == 64
    modality_balance(plans["id"]["parents"] + plans["ood"]["parents"])
    # Exact full-factorial family x noise x modality balance in train, validation, and test_id.
    for split, expected_per_cell in (("train", 3), ("validation", 1), ("test_id", 3)):
        rows = [r for r in plans["id"]["parents"] if r["split"] == split]
        for fam in FAMILIES:
            for noisy in (False, True):
                for mod in MODALITIES:
                    cell = [r for r in rows if (r["family"], r["noisy"], r["selected_modality"]) == (fam, noisy, mod)]
                    assert len(cell) == expected_per_cell
    cal_ood = [r for r in plans["ood"]["parents"] if r["split"] == "calibration_ood"]
    assert sum(r["shift"] == "structure" and r["family"] == "faulted_lens" for r in cal_ood) == 4
    assert sum(r["shift"] == "coverage" and r["coverage"] == "clustered" for r in cal_ood) == 4
    test_ood = [r for r in plans["ood"]["parents"] if r["split"] == "test_ood"]
    for shift in ("structure", "coverage", "density"):
        sub = [r for r in test_ood if r["shift"] == shift]
        assert len(sub) == 8 and sum(r["noisy"] for r in sub) == 4
    # Modality balance holds across arbitrary master seeds for both full and small counts.
    for s in range(1, 25):
        for counts in (None, SMALL_R2):
            r = rsiplan.make_generation_plan("rsi-r2", s, historical={"legacy_release": "/a", "rsi_dev": "/b"},
                                             counts=counts, short_per_family=2 if counts else "preset",
                                             medium_per_family=3 if counts else "preset")
            for part in ("id", "ood"):
                pl = make_plan(r[part + "_seed"], rsiplan.part_splits(r, part), part == "ood", SOURCE,
                               generation_attempts=r["size_gates"]["generation_attempts"],
                               density_shift_splits=r["density_shift_splits"])
                rsiplan.part_plan(pl, r, "c" * 64, part)
                modality_balance(pl["parents"])


def test_r2_determinism_open_only_and_deferred_sealed_equivalence(tmp_path):
    hist = _historical(tmp_path / "hist")
    # 1. Open-only sub-release + deferred completion of sealed splits.
    ds_inc = tmp_path / "ds_inc"
    rsi_inc, inputs_inc = _build_r2_dataset(ds_inc, master_seed=77, hist=hist, open_only=True, holes_bounds=(12, 12))
    rel_open = tmp_path / "rel_open"
    res_open = build(ds_inc, rel_open, open_only=True, sub_release="rsi-r2-h12")
    assert res_open["status"] == "candidate_density_open_only"
    assert set(res_open["counts"]) == set(OPEN_SPLITS)
    assert not (rel_open / "test_id").exists() and not (rel_open / "test_ood").exists()
    card_open = json.loads((rel_open / "bench-card.json").read_text())
    assert card_open["sub_release"] == "rsi-r2-h12" and card_open["density"]["public_holes"] == [12, 12]
    assert card_open["sealed_commitments"]["sealed_splits_generated"] is False
    assert "master_seed" not in json.dumps(card_open) and "id_seed" not in json.dumps(card_open)
    assert card_open["sealed_commitments"]["seed_disclosure"] == "withheld_until_retirement"
    with (rel_open / "strata.csv").open() as f:
        strata_rows = list(csv.DictReader(f))
    assert len(strata_rows) == sum(SMALL_R2[s] for s in OPEN_SPLITS)
    assert all(r["split"] in OPEN_SPLITS for r in strata_rows)

    # Complete sealed splits on ds_inc and build formal release.
    for part, k0 in (("id", 0), ("ood", 32)):
        plan = json.loads((ds_inc / part / "plan.json").read_text())
        _populate_part(ds_inc / part, plan, part, k0, only_splits=SEALED_SPLITS, holes_bounds=(12, 12))
    rel_inc_full = tmp_path / "rel_inc_full"
    build(ds_inc, rel_inc_full, inputs_inc["calibration"], inputs_inc["isolation"], inputs_inc["geological_constraints"])

    # 2. Full one-shot dataset with same master_seed=77 -> byte-identical manifest & plan SHA-256s.
    ds_one = tmp_path / "ds_one"
    _, inputs_one = _build_r2_dataset(ds_one, master_seed=77, hist=hist, open_only=False, holes_bounds=(12, 12))
    rel_one = tmp_path / "rel_one"
    build(ds_one, rel_one, inputs_one["calibration"], inputs_one["isolation"], inputs_one["geological_constraints"])

    q_inc = json.loads((rel_inc_full / "qualification.json").read_text())
    q_one = json.loads((rel_one / "qualification.json").read_text())
    assert q_inc["generation_plan_sha256"] == q_one["generation_plan_sha256"]
    assert q_inc["plan_sha256"] == q_one["plan_sha256"]
    assert q_inc["manifest_sha256"] == q_one["manifest_sha256"]
    assert q_inc["artifacts_sha256"]["strata_csv"] == q_one["artifacts_sha256"]["strata_csv"]

    # Verify all 14 formal manifests, tiers S/M/F, X4 merged confirmation, and strata exclusion of sealed splits.
    tiers = load_tiers(rel_one)
    assert set(tiers) == {"S", "M", "F"}
    for m_rel in q_one["manifest_sha256"]:
        assert verify_release(rel_one / m_rel) == q_one
    x4_man = load_manifest(rel_one / "train_s_x4/manifest.json")
    assert manifest_tiers(rel_one / "train_s_x4/manifest.json", x4_man, rel_one, tiers, roles=("train",)) == ["S"]
    sample_x4 = read_problem(rel_one / "train_s_x4" / x4_man["cases"][0]["truth"] / "confirmation")
    assert len(np.unique(sample_x4["observations"][0]["groups"])) == 8  # 4 packs x 2 holes each
    with (rel_one / "strata.csv").open() as f:
        formal_strata = list(csv.DictReader(f))
    assert len(formal_strata) == sum(SMALL_R2[s] for s in OPEN_SPLITS)
    assert not any(r["split"] in SEALED_SPLITS for r in formal_strata)

    # Different master seed produces distinct hashes.
    ds_other = tmp_path / "ds_other"
    _build_r2_dataset(ds_other, master_seed=78, hist=hist, open_only=True, holes_bounds=(12, 12))
    rel_other = tmp_path / "rel_other"
    build(ds_other, rel_other, open_only=True)
    assert json.loads((rel_other / "bench-card.json").read_text())["hashes"]["generation_plan_sha256"] != \
        card_open["hashes"]["generation_plan_sha256"]


def test_multiple_recipe_uses_enlarged_radii():
    """Verify multiple recipe draws radii[0] in [0.16, 0.22] and radii[2] in [0.08, 0.12]."""
    import ast
    src = (SOURCE / "geology.py").read_text()
    tree = ast.parse(src)
    uniform_bounds = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "uniform" and len(node.args) == 2
                and all(isinstance(a, ast.Constant) and isinstance(a.value, float) for a in node.args)):
            uniform_bounds.append((node.args[0].value, node.args[1].value))
    assert (.16, .22) in uniform_bounds and (.08, .12) in uniform_bounds
    assert (.13, .18) not in uniform_bounds and (.045, .075) not in uniform_bounds


def test_gempy_multiple_recipe_and_observations_holes_determinism(tmp_path):
    gempy_py = Path("/Aidata/shared/xingyulu/rsi4geo/env/gempy/bin/python")
    if not gempy_py.is_file():
        pytest.skip("H20 GemPy environment not present")
    import os
    import subprocess
    repo_root = Path(__file__).resolve().parents[3]
    env = {**os.environ, "PYTHONPATH": str(repo_root), "TMPDIR": str(tmp_path)}
    parent = None
    for seed in range(10, 26):
        cand = tmp_path / f"parent-{seed}"
        res = subprocess.run(
            [str(gempy_py), "-m", "tasks.copper_inversion.experiments.v2.geology",
             "--family", "multiple", "--seed", str(seed), "--output", str(cand)],
            env=env, capture_output=True, text=True, timeout=180)
        if res.returncode == 0:
            try:
                q = quality(cand, "multiple", volume_bounds=(.04, .35), min_vertical_thickness=3)
                assert q["component_count"] == 2 and q["median_vertical_thickness"] >= 3
                parent = cand
                break
            except ValueError:
                continue
    assert parent is not None, "enlarged multiple recipe should clear ID size/thickness gates quickly"

    obs_a, obs_b = tmp_path / "obs_a", tmp_path / "obs_b"
    batch_spec = tmp_path / "batch.json"
    batch_spec.write_text(json.dumps([{"seed": 42, "output": str(obs_b)}]))
    subprocess.run(
        [str(gempy_py), "-m", "tasks.copper_inversion.experiments.v2.observations",
         "--parent", str(parent), "--output", str(obs_a), "--seed", "42", "--holes", "16"],
        env=env, check=True, capture_output=True, text=True, timeout=180)
    subprocess.run(
        [str(gempy_py), "-m", "tasks.copper_inversion.experiments.v2.observations",
         "--parent", str(parent), "--batch", str(batch_spec), "--holes", "16"],
        env=env, check=True, capture_output=True, text=True, timeout=180)
    man_a = json.loads((obs_a / "private-manifest.json").read_text())
    man_b = json.loads((obs_b / "private-manifest.json").read_text())
    assert man_a == man_b
    assert man_a["holes"] == [16, 16] and man_a["hole_count"] == 16
    case_outcrop = next(c["case_id"] for c in man_a["cases"] if c["modality"] == "drill_outcrop")
    prob_a = read_problem(obs_a / case_outcrop)
    prob_b = read_problem(obs_b / case_outcrop)
    assert prob_a["observations"][1]["weight"] == .25
    for obs1, obs2 in zip(prob_a["observations"], prob_b["observations"]):
        assert np.array_equal(obs1["points"], obs2["points"])
        assert np.array_equal(obs1["labels"], obs2["labels"])
        assert np.array_equal(obs1["groups"], obs2["groups"])


def test_generate_confirmation_four_packs_cli_with_prepopulated_candidates(tmp_path, monkeypatch):
    """Exercise generate_confirmation.main end-to-end on a 4-pack rsi-r2 dataset."""
    import sys
    from tasks.copper_inversion.experiments.v2 import generate_confirmation
    hist = _historical(tmp_path / "hist")
    ds = tmp_path / "ds"
    _build_r2_dataset(ds, master_seed=31, hist=hist, open_only=True, holes_bounds=(8, 8))
    # Move one parent's confirmation into confirmation-candidates and re-run generate_confirmation.main.
    import shutil
    part_dir = ds / "id"
    plan = json.loads((part_dir / "plan.json").read_text())
    entry = next(e for e in plan["parents"] if e["split"] == "train" and e["index"] == 0)
    name = "train-00"
    shutil.rmtree(part_dir / "confirmation" / name)
    (part_dir / "confirmation-audits" / f"{name}.json").unlink()
    (ds / "confirmation-complete.json").unlink()
    digest = json.loads((part_dir / "packs" / name / "private-manifest.json").read_text())["parent_digest"]
    mask = np.load(part_dir / "packs" / name / "private" / "truth.npy", allow_pickle=False)
    # Seed 0 fails discoverability (0 ore holes); seeds 1..4 pass and are separated.
    _write_pack(part_dir / "confirmation-candidates" / f"{name}-00", digest, entry["confirmation_seeds"][0],
                CONF_PACK_HOLES[0], mask, ore_holes=0, holes_bounds=(3, 12))
    for k in range(4):
        _write_pack(part_dir / "confirmation-candidates" / f"{name}-{k + 1:02}", digest,
                    entry["confirmation_seeds"][k + 1], CONF_PACK_HOLES[k], mask, ore_holes=1, holes_bounds=(3, 12))
    # Seed 5 is also pre-written so batching of 6 finds all candidates without invoking gempy.
    _write_pack(part_dir / "confirmation-candidates" / f"{name}-05", digest,
                entry["confirmation_seeds"][5], [(5.5, 2.5), (5.5, 7.5)], mask, ore_holes=1, holes_bounds=(3, 12))
    monkeypatch.setattr(sys, "argv", ["generate_confirmation", "--root", str(ds), "--python", sys.executable, "--workers", "2"])
    generate_confirmation.main()
    receipt = json.loads((part_dir / "confirmation-audits" / f"{name}.json").read_text())
    assert receipt["status"] == "passed" and receipt["packs"] == 4
    assert [a["status"] for a in receipt["attempts"]] == ["failed", "passed", "passed", "passed", "passed"]
    assert receipt["attempts"][0]["reason"] == "undiscovered"
    build(ds, tmp_path / "rel", open_only=True)

