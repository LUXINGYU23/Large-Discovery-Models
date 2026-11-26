"""Tests for C1: judge cells (standard, x4, noisy25, w25, oracle), r2 loop preset, finalists=5, standard validation."""

import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tasks.copper_inversion.core.execution import digest
from tasks.copper_inversion.experiments.rsi import freeze, launch
from tasks.copper_inversion.experiments.rsi.measurements import thin_confirmation
from tasks.copper_inversion.rsi import protocol as protocols, workflow
from tasks.copper_inversion.rsi.controller import Config
from tasks.copper_inversion.rsi.exposure import FIELDS
from tasks.copper_inversion.rsi.ledger import append
from tasks.copper_inversion.rsi.service import RiskOnlyService
from tasks.copper_inversion.tests.test_rsi_launch import FakeHost, VALUES
from tasks.copper_inversion.tests.test_rsi_service import IMAGE, evaluate, slab, tier, truth_for, write_record
from tasks.copper_inversion.tests.test_v2_science import fixture
from tasks.copper_inversion.v2.data import read_problem, save_problem
from tasks.copper_inversion.v2.objective import score
from tasks.copper_inversion.v2.scorer import score_case
from tasks.copper_inversion.v2.suite import (SuiteEvaluator, confirmation_resample_seed, file_hash,
                                             load_manifest, merge_confirmations)

R1_EXAMPLE = Path(protocols.RESOURCES) / "rsi/protocol.example.json"
R2_EXAMPLE = Path(protocols.RESOURCES) / "rsi/protocol.r2.example.json"


class RealScorerExecution:
    """In-process execution delegating scoring to the real v2.scorer.score_case."""

    def __init__(self, root, image, account=None):
        self.root, self.image = Path(root), image
        self.root.mkdir(parents=True, exist_ok=True)
        self.seen_truths = []

    def run(self, identity, *, phase, inputs, timeout, cpus=1, memory_mb=2048, source=None,
            model=None, science_module="core", truth=None):
        job = self.root / digest(identity)
        (job / "output").mkdir(parents=True, exist_ok=True)
        if phase == "solver":
            problem = read_problem(inputs)
            namespace = {}
            try:
                exec(compile((Path(source) / "solver.py").read_text(), "solver.py", "exec"), namespace)
                field = np.asarray(namespace["solve"](problem, {}, 47), dtype=float)
            except Exception:
                return job, dict(exit_code=1, reason=None, oom=False, wall_seconds=0.5)
            np.save(job / "output/model.npy", field)
            return job, dict(exit_code=0, reason=None, oom=False, wall_seconds=1.0)
        self.seen_truths.append((identity, Path(truth)))
        outcome = score_case(inputs, model, truth)
        (job / "output/result.json").write_text(json.dumps(outcome, allow_nan=False))
        return job, dict(exit_code=0, reason=None, oom=False)


def make_multipack_confirmation(case_index, pack_index, holes=8):
    """Build one confirmation pack with `holes` distinct boreholes whose labels vary by x-coordinate and pack."""
    base = fixture()
    pts, lbls, grps = [], [], []
    # Spread holes across x = 0.5..8.5, distinct per pack_index so c0..c3 give distinct scores.
    for h in range(holes):
        x = 0.5 + ((h + pack_index * 1.7 + case_index * 0.3) % 8.0)
        y = 1.5 + ((h * 1.9 + pack_index * 2.1) % 12.0)
        for z in (1.5, 4.5, 7.5, 10.5):
            pts.append([x, y, z])
            # Pack 0 favors shallow x < 3.5; packs 1..3 extend deeper to x < 6.5 so x4 differs from c0.
            cutoff = 3.5 if pack_index == 0 else 6.5
            lbls.append(1.0 if x < cutoff else 0.0)
            grps.append(h)
    base["observations"][0].update(
        points=np.asarray(pts, dtype=float),
        labels=np.asarray(lbls, dtype=float),
        groups=np.asarray(grps, dtype=int),
    )
    return base


def write_r2_suite(release, name, *, bodies=3, split="train", offset=0, packs=4, holes=8):
    """Write a suite with c0..c{packs-1} confirmation packs and valid copper-suite-v2 manifest."""
    root = release / name
    rows = []
    for index in range(offset, offset + bodies):
        parent = hashlib.sha256(f"r2-parent-{split}-{index}".encode()).hexdigest()
        cid = hashlib.sha256(f"r2-case-{split}-{index}".encode()).hexdigest()[:20]
        truth = root / "private" / f"body{index}"
        truth.mkdir(parents=True)
        np.save(truth / "truth.npy", truth_for(index))
        sha = {"truth.npy": file_hash(truth / "truth.npy")}
        for p_idx in range(packs):
            conf = make_multipack_confirmation(index, p_idx, holes=holes)
            conf["case_id"] = f"confirm-{index}-c{p_idx}"
            folder = truth / "confirmation" / f"c{p_idx}"
            save_problem(conf, folder)
            for fname in ("problem.json", "problem.npz"):
                sha[f"confirmation/c{p_idx}/{fname}"] = file_hash(folder / fname)
        prob = fixture()
        prob["case_id"] = cid
        pub = root / "public" / cid
        save_problem(prob, pub)
        for fname in ("problem.json", "problem.npz"):
            sha[fname] = file_hash(pub / fname)
        rows.append({
            "case_id": cid, "parent_id": parent, "modality": "drill",
            "public": str(pub.relative_to(root)), "truth": str(truth.relative_to(root)),
            "sha256": sha,
        })
    path = root / "manifest.json"
    path.write_text(json.dumps({"schema": "copper-suite-v2", "stage": "development", "split": split,
                                "seeds": [47], "cases": rows}, sort_keys=True))
    return path


def make_r2_service(tmp_path, manifest, name, *, objective="confirmation", resample=None,
                    confirmation_packs=None, finalists=5, slots=20):
    cfg = {"manifest": str(manifest), "root": str(tmp_path / name), "image": IMAGE,
           "seconds": 30, "slots": slots, "caller_uid": 0, "socket": str(tmp_path / (name + ".sock")),
           "objective": objective, "finalists": finalists}
    if resample is not None:
        cfg["resample"] = resample
    if confirmation_packs is not None:
        cfg["confirmation_packs"] = confirmation_packs
    with patch("tasks.copper_inversion.v2.service.Execution", RealScorerExecution):
        svc = RiskOnlyService(cfg)
    svc.domain.checks = lambda payload: True
    return svc


# 1. Old protocols remain byte-identical; new r2 protocol validates cleanly ----------------

def test_old_protocols_byte_identical_and_r2_example_validates():
    r1 = protocols.load(R1_EXAMPLE)
    assert protocols.validation_judge(r1) == "match_training"
    assert protocols.objective_version(r1) == "objective_v2"
    assert protocols.case_isolation(r1)["randomize_orientation"] is False
    # Exact canonical JSON of protocol.example.json has no new fields injected.
    assert "validation_judge" not in r1 and "objective_version" not in r1 and "loop_preset" not in r1

    r2 = protocols.load(R2_EXAMPLE)
    assert r2["release"]["name"] == "rsi-r2"
    assert protocols.objective_version(r2) == "objective_v3"
    assert protocols.validation_judge(r2) == "standard"
    assert protocols.case_isolation(r2)["randomize_orientation"] is True
    assert r2["loop_preset"] == "r2"
    assert r2["loop"] == protocols.loop_preset("r2")
    assert (r2["loop"]["proposals_per_branch"], r2["loop"]["batch"],
            r2["loop"]["reserved_per_branch"], r2["loop"]["finalists"],
            r2["loop"]["rounds"], r2["loop"]["reset_period"]) == (4, 16, 4, 5, 20, 5)
    assert set(r2["judges"]) == {"standard", "x4", "noisy25", "w25", "oracle"}
    assert r2["judges"]["noisy25"]["resample"] == {"fraction": 0.25}
    assert r2["judges"]["x4"]["confirmation_packs"] == ["c0", "c1", "c2", "c3"]
    assert r2["tiers"]["S"]["slots"] == 321

    cfg = Config.from_protocol(r2, "ldm4geo", "S", seed=101)
    assert (cfg.proposals_per_branch, cfg.batch, cfg.reserved_per_branch, cfg.finalists, cfg.slots) == (4, 16, 4, 5, 321)


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(objective_version="objective_v9"),
    lambda p: p.update(validation_judge="oracle"),
    lambda p: p.update(loop_preset="r99"),
    lambda p: p["loop"].update(proposals_per_branch=8),  # mismatches loop_preset="r2"
    lambda p: p["judges"]["noisy25"]["resample"].update(fraction=1.0),
    lambda p: p["judges"]["noisy25"]["resample"].update(fraction=0.0),
    lambda p: p["judges"]["noisy25"]["resample"].update(extra=1),
    lambda p: p["judges"]["oracle"].update(resample={"fraction": 0.25}),
    lambda p: p["judges"]["oracle"].update(confirmation_packs=["c0"]),
    lambda p: p["judges"]["x4"].update(confirmation_packs=[]),
    lambda p: p["judges"]["x4"].update(confirmation_packs=["c0", "c0"]),
    lambda p: p["judges"]["x4"].update(confirmation_packs=["../escape"]),
    lambda p: p["judges"]["w25"].update(validation_manifest="validation_w25/manifest.json"),
    lambda p: p["judges"]["standard"].update(unknown_field=True),
])
def test_protocol_validation_rejects_invalid_r2_judge_and_preset_fields(mutate):
    data = json.loads(R2_EXAMPLE.read_text())
    mutate(data)
    with pytest.raises(ValueError, match="invalid protocol"):
        protocols.validate(data)


# 2. noisy25: deterministic per evaluation key, fresh across keys, full-judge finalist ranking ---

def test_noisy25_deterministic_per_key_fresh_across_keys_and_ranks_by_full_judge(tmp_path):
    release = tmp_path / "release"
    train = write_r2_suite(release, "train_s", bodies=3, split="train", packs=1, holes=12)

    svc = make_r2_service(tmp_path, train, "svc-noisy25", resample={"fraction": 0.25}, finalists=5)
    # Two distinct sources producing the exact same model (only differing by a comment) get different evaluation keys
    # and therefore fresh independent random quarters of the confirmation holes!
    src_a = slab(3) + "# candidate A\n"
    src_b = slab(3) + "# candidate B\n"
    src_c = slab(3) + "# candidate C\n"

    fb_a = evaluate(svc, src_a)
    fb_b = evaluate(svc, src_b)
    fb_c = evaluate(svc, src_c)

    # Whitelist still strips private full_loss_total from public feedback!
    assert set(fb_a) == FIELDS and "full_loss_total" not in fb_a

    # Replay of src_a hits the exact same evaluation key and returns identical feedback.
    svc_replay = make_r2_service(tmp_path, train, "svc-noisy25-replay", resample={"fraction": 0.25}, finalists=5)
    fb_a_replay = evaluate(svc_replay, src_a)
    assert fb_a_replay == fb_a

    # Check the actual thinned confirmation holes on disk for A, A_replay, B, C.
    key_a = svc.evaluator.evaluation_attempt_usage_key(type("C", (), {"canonical_key": fb_a["source_sha256"]})())
    tkey_a = svc.evaluator.case_transformation_key(type("C", (), {"canonical_key": fb_a["source_sha256"]})())
    key_b = svc.evaluator.evaluation_attempt_usage_key(type("C", (), {"canonical_key": fb_b["source_sha256"]})())
    key_c = svc.evaluator.evaluation_attempt_usage_key(type("C", (), {"canonical_key": fb_c["source_sha256"]})())
    assert key_a != key_b != key_c
    cases = json.loads(train.read_text())["cases"]

    for case in cases:
        cid = case["case_id"]
        thinned_a = read_problem(svc.root / "evaluations" / key_a / cid / "47/score-truth/confirmation")
        thinned_a_rep = read_problem(svc_replay.root / "evaluations" / key_a / cid / "47/score-truth/confirmation")
        full_a = read_problem(svc.root / "evaluations" / key_a / cid / "47/score-truth/full_confirmation")
        raw_c0 = read_problem(train.parent / case["truth"] / "confirmation/c0")
        # Exact replay of thinned holes and matches manual thin_confirmation(raw_c0, seed, 0.25).
        np.testing.assert_array_equal(thinned_a["observations"][0]["points"], thinned_a_rep["observations"][0]["points"])
        expected_thin, _ = thin_confirmation(raw_c0, confirmation_resample_seed(tkey_a, cid), 0.25, 4)
        np.testing.assert_array_equal(thinned_a["observations"][0]["points"], expected_thin["observations"][0]["points"])
        # Thinned has ~1/4 of 12 holes = 3 holes; full_confirmation retains all 12 holes.
        assert len(np.unique(thinned_a["observations"][0]["groups"])) == 3
        assert len(np.unique(full_a["observations"][0]["groups"])) == 12

    # Across 3 cases x 3 candidates, the sampled hole subsets differ between evaluation keys.
    groups_by_key = {
        k: [tuple(np.unique(read_problem(svc.root / "evaluations" / k / c["case_id"] / "47/score-truth/confirmation")[
            "observations"][0]["groups"]).tolist()) for c in cases]
        for k in (key_a, key_b, key_c)
    }
    assert len({tuple(v) for v in groups_by_key.values()}) >= 2

    # Private receipt stores BOTH noisy loss_total and exact full_loss_total (identical across A, B, C since same model!).
    res_a = json.loads((svc.root / "evaluations" / key_a / "result.json").read_text())["metrics"]
    res_b = json.loads((svc.root / "evaluations" / key_b / "result.json").read_text())["metrics"]
    res_c = json.loads((svc.root / "evaluations" / key_c / "result.json").read_text())["metrics"]
    assert res_a["full_loss_total"] == pytest.approx(res_b["full_loss_total"], abs=1e-12)
    assert res_a["full_loss_total"] == pytest.approx(res_c["full_loss_total"], abs=1e-12)
    assert len({round(r["loss_total"], 10) for r in (res_a, res_b, res_c)}) >= 2

    # Evaluate 6 distinct programs (slab 2..7) and verify seal_training keeps top 5 ranked by private full_loss_total!
    for rows_n in (2, 4, 5, 6, 7):
        evaluate(svc, slab(rows_n))
    sealed = svc.seal_training()
    assert len(sealed["source_sha256s"]) == 5
    sealed_disk = json.loads((svc.root / "training-sealed.json").read_text())
    full_risks = [sealed_disk["selected_receipts"][sha]["metrics"]["full_loss_total"] for sha in sealed["source_sha256s"]]
    assert full_risks == sorted(full_risks)


# 3. x4 merges all four confirmation packs c0..c3 without group collisions ----------------

def test_x4_merges_all_four_confirmation_packs(tmp_path):
    release = tmp_path / "release"
    train = write_r2_suite(release, "train_s", bodies=2, split="train", packs=4, holes=6)
    manifest = load_manifest(train)
    assert len(manifest["cases"]) == 2

    svc_std = make_r2_service(tmp_path, train, "svc-std")
    svc_x4 = make_r2_service(tmp_path, train, "svc-x4", confirmation_packs=["c0", "c1", "c2", "c3"])
    svc_all = make_r2_service(tmp_path, train, "svc-all", confirmation_packs="all")

    fb_std = evaluate(svc_std, slab(5))
    fb_x4 = evaluate(svc_x4, slab(5))
    fb_all = evaluate(svc_all, slab(5))

    # x4 with explicit ["c0".."c3"] and "all" produce the exact same risk, and differ from standard (c0 only).
    assert fb_x4["risk"] == pytest.approx(fb_all["risk"], abs=1e-12)
    assert fb_x4["risk"] != pytest.approx(fb_std["risk"], abs=1e-6)

    # Verify x4 score matches manual score(merge_confirmations([c0, c1, c2, c3]), field) on every body!
    ns = {}
    exec(slab(5), ns)
    expected_utils = []
    for case in manifest["cases"]:
        pub = read_problem(train.parent / case["public"])
        field = ns["solve"](pub, {}, 47)
        packs = [read_problem(train.parent / case["truth"] / f"confirmation/c{i}") for i in range(4)]
        merged = merge_confirmations(packs)
        assert len(np.unique(merged["observations"][0]["groups"])) == 24  # 4 packs x 6 distinct holes
        loss = score(merged, field)["loss_total"]
        expected_utils.append(1.0 / (1.0 + loss))
    assert fb_x4["risk"] == pytest.approx(1.0 - float(np.mean(expected_utils)), abs=1e-12)


# 4. Validation judge is always fixed to standard across all 5 cells ----------------------

def test_validation_judge_always_standard_across_all_cells_and_finalists_5(tmp_path):
    release = tmp_path / "release"
    train_s = write_r2_suite(release, "train_s", bodies=2, split="train", offset=0, packs=4, holes=8)
    train_w25 = write_r2_suite(release, "train_s_w25", bodies=2, split="train", offset=0, packs=1, holes=2)
    val = write_r2_suite(release, "validation", bodies=2, split="validation", offset=10, packs=4, holes=6)
    tid = write_r2_suite(release, "test_id", bodies=2, split="test_id", offset=20, packs=1, holes=6)
    tood = write_r2_suite(release, "test_ood", bodies=2, split="test_ood", offset=30, packs=1, holes=6)
    write_record(release, "development", layout={"packs_per_body": 1},
                 tiers={"S": tier(30, "train_s/manifest.json", 2, slots=321)})

    r2_proto = json.loads(R2_EXAMPLE.read_text())
    r2_proto["tiers"] = {"S": tier(30, "train_s/manifest.json", 2, slots=321)}

    active = tmp_path / "active"
    pid = "rsi-r2-test000000000000"
    val_picks = {}
    for judge_name, train_path, extra_cfg in (
        ("standard", train_s, {}),
        ("x4", train_s, {"confirmation_packs": ["c0", "c1", "c2", "c3"]}),
        ("noisy25", train_s, {"resample": {"fraction": 0.25}}),
        ("w25", train_w25, {}),
        ("oracle", train_s, {"objective": "oracle_iou"}),
    ):
        run_id = f"ldm4geo-{judge_name}-s201"
        paths = freeze.layout(active, pid, "E3", run_id)
        svc = make_r2_service(paths["service"].parent, train_path, paths["service"].name,
                              finalists=5, slots=321, **extra_cfg)
        for seq, rows_n in enumerate((2, 3, 4, 5, 6, 7)):
            fb = evaluate(svc, slab(rows_n))
            append(paths["run"] / "ledger.jsonl", {
                "seq": seq, "program_id": fb["source_sha256"], "epoch": 0, "round": 1,
                "branch": seq % 4, "selection": "reserved",
                **{k: v for k, v in fb.items() if k != "source_sha256"},
            })
        svc.seal_training()
        (paths["run"] / "manifest.json").write_text(json.dumps({
            "run_id": run_id, "experiment": "E3", "protocol_id": pid,
            "arm": "ldm4geo", "tier": "S", "judge": judge_name, "seed": 201,
            "blind": judge_name != "oracle",
        }))
        (paths["run"] / "result.json").write_text(json.dumps({"status": "completed", "stop_reason": "completed"}))

        with patch("tasks.copper_inversion.v2.selection.Execution", RealScorerExecution), \
                patch("tasks.copper_inversion.v2.selection.InterfaceChecks", lambda root, ex: lambda p: True):
            rec = freeze.freeze_run(active, pid, "E3", run_id, release=release, protocol=r2_proto)
            assert len(rec["source_sha256s"]) == 5
            assert rec["selection_objective"] == "confirmation"
            target = paths["run_selection"] / "full"
            frozen = json.loads((target / "frozen.json").read_text())
            assert frozen["manifests"][0]["path"] == str(val.resolve())
            assert frozen["selection_objective"] == "confirmation"
            assert "resample" not in frozen and "confirmation_packs" not in frozen
            report = freeze.validate_run(active, pid, "E3", run_id)
            assert report["objective"] == "confirmation"
            val_picks[judge_name] = report["selected_sha256"]

    # Every cell's validation used the exact same standard judge (c0 confirmation on validation/manifest.json).
    assert len(val_picks) == 5


# 5. Launcher forwards resample, confirmation_packs, and finalists=5 ----------------------

def test_launcher_forwards_r2_judge_and_finalist_settings_to_service(tmp_path):
    r2_proto = json.loads(R2_EXAMPLE.read_text())
    r2_proto["release"]["name"] = "rsi-dev"  # development so no G3 freeze needed in test
    r2_proto["token_cap_input"] = 0
    proto_path = tmp_path / "r2.json"
    proto_path.write_text(json.dumps(r2_proto))

    root = tmp_path / "active"
    policy = launch.make_policy(VALUES, tmp_path / "queue", tmp_path / "policy.json")
    site = {"root": str(root), "release": str(tmp_path / "release"), "source": str(tmp_path),
            "service_python": "py-e", "client_python": "py-c", "service_user": "evaluator",
            "client_user": "client", "image": IMAGE, "sidecar_image": "sidecar@sha256",
            "embedding_model_path": "/models/e", "harness_cache_dir": "/cache",
            "api_key_file": "/keys/k", "policy": str(tmp_path / "policy.json"),
            "public_catalogs": {"S": "/catalogs/S.json"}}
    batch = tmp_path / "batch.json"
    batch.write_text(json.dumps({
        "schema": launch.SCHEMA, "name": "b-r2", "protocol": str(proto_path),
        "experiment": "smoke", "tier": "S",
        "runs": [
            {"arm": "ldm4geo", "judge": "noisy25", "seed": 1},
            {"arm": "ldm4geo", "judge": "x4", "seed": 1},
            {"arm": "ldm4geo", "judge": "standard", "seed": 1},
        ],
    }))
    host = FakeHost(root, {})
    host.queue = policy["queue_root"]
    launcher = launch.Launcher(site, batch, host=host)
    for name, run in launcher.runs():
        launcher.start_service(name, run)
        cfg = json.loads((launcher.paths(name)["service"] / "service-config.json").read_text())
        assert cfg["finalists"] == 5
        assert cfg["case_isolation"]["randomize_orientation"] is True
        if run["judge"] == "noisy25":
            assert cfg["resample"] == {"fraction": 0.25} and "confirmation_packs" not in cfg
        elif run["judge"] == "x4":
            assert cfg["confirmation_packs"] == ["c0", "c1", "c2", "c3"] and "resample" not in cfg
        else:
            assert "resample" not in cfg and "confirmation_packs" not in cfg
