"""Tests for precomputed offline references in q1.py, selection.py, unseal.py, and export_outcomes.py."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest

from tasks.copper_inversion.core.execution import digest
from tasks.copper_inversion.experiments.rsi import export_outcomes, q1
from tasks.copper_inversion.experiments.rsi.freeze import layout
from tasks.copper_inversion.experiments.rsi.unseal import unseal_references
from tasks.copper_inversion.tests.test_rsi_unseal import IMAGE, PID, setup
from tasks.copper_inversion.tests.test_v2_science import fixture, model
from tasks.copper_inversion.v2 import offline_baselines, scorer, selection
from tasks.copper_inversion.v2.data import save_problem
from tasks.copper_inversion.v2.suite import file_hash
from tasks.copper_inversion.v2.training_receipts import eligible


def _make_release_with_confirmations(root):
    manifests, index = {}, 0
    for split in ("train", "validation", "test_id", "test_ood"):
        base, rows = root / split, []
        for parent in range(2):
            modality = ("drill", "drill_outcrop")[parent % 2]
            truth_dir = base / f"truth-{parent}"
            truth_dir.mkdir(parents=True)
            mask = model(fixture()) > 0
            iy, iz = index % 6 + 1, (index // 6) % 4 + 1
            mask[3, iy, iz] = not bool(mask[3, iy, iz])
            index += 1
            np.save(truth_dir / "truth.npy", mask, allow_pickle=False)
            problem = fixture()
            problem["case_id"] = f"{split}-{parent}-{modality}"
            save_problem(problem, base / f"case-{parent}")
            conf = fixture()
            conf["case_id"] = problem["case_id"]
            save_problem(conf, truth_dir / "confirmation")
            rows.append({
                "case_id": problem["case_id"],
                "parent_id": hashlib.sha256(f"{split}-{parent}".encode()).hexdigest(),
                "modality": modality,
                "public": f"case-{parent}",
                "truth": f"truth-{parent}",
                "sha256": {
                    "problem.json": file_hash(base / f"case-{parent}/problem.json"),
                    "problem.npz": file_hash(base / f"case-{parent}/problem.npz"),
                    "truth.npy": file_hash(truth_dir / "truth.npy"),
                    "confirmation/problem.json": file_hash(truth_dir / "confirmation/problem.json"),
                    "confirmation/problem.npz": file_hash(truth_dir / "confirmation/problem.npz"),
                },
            })
        manifests[split] = base / "manifest.json"
        manifests[split].write_text(json.dumps({
            "schema": "copper-suite-v2",
            "stage": "development",
            "split": split,
            "seeds": [47],
            "cases": rows,
        }))
    tier = {
        "train_manifest": "train/manifest.json",
        "solver_seconds": 10,
        "slots": 321,
        "train_bodies": 2,
        "validation_manifest": "validation/manifest.json",
        "sealed_manifests": ["test_id/manifest.json", "test_ood/manifest.json"],
    }
    (root / "development-receipt.json").write_text(json.dumps({"tiers": {"S": tier}}))
    return manifests


class LocalScorerExecution:
    """In-process Execution substitute calling v2.scorer.score_case directly without Docker."""
    merged = True

    def __init__(self, root, image, account=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.image = image

    def run(self, identity, *, phase, cases=None, inputs=None, model=None, truth=None, **kwargs):
        assert phase == "scorer"
        job = self.root / digest(identity)
        out = job / "output"
        out.mkdir(parents=True, exist_ok=True)
        if cases is not None:
            results = [scorer.score_case(c["inputs"], c["model"], c["truth"]) for c in cases]
            (out / "results.json").write_text(json.dumps(results))
        else:
            res = scorer.score_case(inputs, model, truth)
            (out / "result.json").write_text(json.dumps(res))
        return job, {"exit_code": 0, "oom": False, "reason": None, "wall_seconds": 0.05}


def test_precomputed_references_real_scorer_unseal_tamper_and_export(tmp_path, monkeypatch):
    release = tmp_path / "release_conf"
    manifests = _make_release_with_confirmations(release)

    # Create Stage-A outputs for potential_field and rf_xyz across all splits using ONLY public inputs.
    stage_a_m4 = tmp_path / "stage_a_m4"
    stage_a_rf = tmp_path / "stage_a_rf"
    for split, m_path in manifests.items():
        m = json.loads(m_path.read_text())
        for case in m["cases"]:
            prob = fixture()
            cdir_m4 = stage_a_m4 / split / case["case_id"]
            cdir_m4.mkdir(parents=True)
            ore_pred = model(prob) > 0
            np.save(cdir_m4 / "ore.npy", ore_pred, allow_pickle=False)
            (cdir_m4 / "choice.json").write_text(json.dumps({
                "method": "potential_field",
                "generator_matched": True,
                "case_id": case["case_id"],
                "no_ore": False,
                "fallback": False,
                "cv": {"selected_rule": "native", "selected_config": {"form": "L", "orient": "plane", "range": 1.7}, "cv_score": 0.8, "cv_evaluated": 12},
                "wall_seconds": 2.0,
            }))
            for seed in m["seeds"]:
                cdir_rf = stage_a_rf / split / case["case_id"] / str(seed)
                cdir_rf.mkdir(parents=True)
                feasible = prob["domain_mask"] & ~prob["forbidden_mask"]
                f_rf = np.where((model(prob) > 0)[feasible], 0.85, 0.15).astype(np.float32)
                np.save(cdir_rf / "field.npy", f_rf, allow_pickle=False)
                (cdir_rf / "choice.json").write_text(json.dumps({
                    "method": "rf_xyz",
                    "case_id": case["case_id"],
                    "seed": seed,
                    "no_ore": False,
                    "fallback": False,
                    "cv": {"selected_rule": "native", "selected_config": {"s_z": 4}, "cv_score": 0.75, "cv_evaluated": 12},
                    "wall_seconds": 1.0,
                }))

    frozen_m4_dir = tmp_path / "frozen_m4"
    frozen_rf_dir = tmp_path / "frozen_rf"
    sha_m4 = offline_baselines.materialize_precomputed_reference("potential_field", stage_a_m4, release, frozen_m4_dir)
    sha_rf = offline_baselines.materialize_precomputed_reference("rf_xyz", stage_a_rf, release, frozen_rf_dir)

    refs_file = tmp_path / "references.json"
    refs_file.write_text(json.dumps({
        "potential_field": {"fields": str(frozen_m4_dir), "sha256": sha_m4},
        "rf_xyz": {"fields": str(frozen_rf_dir), "sha256": sha_rf},
    }))

    # Hash mismatch in references.json is rejected.
    bad_refs = tmp_path / "bad_references.json"
    bad_refs.write_text(json.dumps({"potential_field": {"fields": str(frozen_m4_dir), "sha256": "0" * 64}}))
    with pytest.raises(ValueError, match="reference fields hash differs"):
        q1.references(bad_refs)

    # Unexpected file in precomputed directory is rejected.
    (frozen_m4_dir / "stray.txt").write_text("bad")
    with pytest.raises(ValueError, match="unexpected file"):
        selection.precomputed_manifest(frozen_m4_dir)
    (frozen_m4_dir / "stray.txt").unlink()

    active_root = tmp_path / "active"
    with patch("tasks.copper_inversion.v2.selection.Execution", LocalScorerExecution):
        mapping = q1.build_references(refs_file, release, tmp_path / "refs_out", IMAGE)
        assert set(mapping) == {"potential_field", "rf_xyz"}

        # Before unseal: validation ran, test has NOT run!
        for target in mapping.values():
            _, st = selection.verify(target)
            assert st["phase"] == "selected"
            assert not (Path(target) / "test-result.json").exists()

        # Unseal once under REF log.
        res = unseal_references(active_root, PID, refs_file, tmp_path / "refs_out/references.json", operator="tester")
        assert res == {"status": "unsealed", "experiment": "REF", "references": 2}

        # Second unseal is refused (state is now 'tested', not 'selected').
        with pytest.raises(ValueError, match="untested|already been unsealed"):
            unseal_references(active_root, PID, refs_file, tmp_path / "refs_out/references.json")

        table = export_outcomes.export_references(active_root, PID, references=mapping)
        assert set(table["references"]) == {"potential_field", "rf_xyz"}
        for name in ("potential_field", "rf_xyz"):
            assert set(table["references"][name]) == {"test_id", "test_ood"}
            assert 0.5 < table["references"][name]["test_id"] <= 1.0

    # Check unseal log records precomputed reference kinds.
    log_entries = [json.loads(l) for l in layout(active_root, PID, "Q1")["unseal_log"].read_text().splitlines()]
    assert log_entries[-1]["reference_kinds"] == {"potential_field": "precomputed", "rf_xyz": "precomputed"}

    # Tampering with any frozen model.npy inside selection root is caught by verify() and export_references().
    m4_target = Path(mapping["potential_field"])
    one_model = next((m4_target / "models").rglob("model.npy"))
    one_model.chmod(0o644)
    one_model.write_bytes(one_model.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="frozen source changed"):
        selection.verify(m4_target)
    with pytest.raises(ValueError, match="frozen source changed"):
        export_outcomes.export_references(active_root, PID, references=mapping)


def test_mixed_source_and_precomputed_references_together(tmp_path, monkeypatch):
    root, manifests, _ = setup(tmp_path, ["ldm4geo-s101"], prefixes=(None,))
    service = layout(root, PID, "M1", "ldm4geo-s101")["service"]
    trained = eligible(service, manifests["train"])
    sha_src = next(iter(trained))
    release = manifests["train"].parent.parent
    tier = {
        "train_manifest": "train/manifest.json",
        "solver_seconds": 10,
        "slots": 321,
        "train_bodies": 4,
        "validation_manifest": "validation/manifest.json",
        "sealed_manifests": ["test_id/manifest.json", "test_ood/manifest.json"],
    }
    (release / "development-receipt.json").write_text(json.dumps({"tiers": {"S": tier}}))

    # Build a precomputed tree for all 4 splits.
    pre_dir = tmp_path / "pre_fields"
    for split, m_path in manifests.items():
        m = json.loads(m_path.read_text())
        for c in m["cases"]:
            d = pre_dir / split / c["case_id"]
            d.mkdir(parents=True)
            np.save(d / "model.npy", model(fixture()).astype(np.float64), allow_pickle=False)
    _, sha_pre = selection.precomputed_manifest(pre_dir)

    refs = tmp_path / "mixed_refs.json"
    refs.write_text(json.dumps({
        "A0_in_container": {"source": trained[sha_src]["source_path"], "sha256": sha_src},
        "M4_offline": {"fields": str(pre_dir), "sha256": sha_pre},
    }))
    monkeypatch.setattr(q1, "evaluate_sources", lambda *args, **kwargs: service)
    from tasks.copper_inversion.tests.test_rsi_unseal import fake_evaluator
    with fake_evaluator():
        mapping = q1.build_references(refs, release, tmp_path / "mixed_out", IMAGE)
        unseal_references(root, PID, refs, tmp_path / "mixed_out/references.json")
        table = export_outcomes.export_references(root, PID, references=mapping)
        assert set(table["references"]) == {"A0_in_container", "M4_offline"}


def test_ref_export_outcomes_rows_csv_objective_v3_and_outcomes_loader(tmp_path, monkeypatch):
    from tasks.copper_inversion.experiments.baselines import prepare_ref
    from tasks.copper_inversion.experiments.rsi.analysis.outcomes import Outcomes
    from tasks.copper_inversion.tests.test_rsi_unseal import fake_evaluator

    root, manifests, _ = setup(tmp_path, ["ldm4geo-s101"], prefixes=(None,))
    service = layout(root, PID, "M1", "ldm4geo-s101")["service"]
    release = manifests["train"].parent.parent

    # Emulate h12-formal tier S where train_manifest lives under train_s/manifest.json with split=="train".
    shutil_src = release / "train"
    train_s_dir = release / "train_s"
    import shutil
    shutil.copytree(shutil_src, train_s_dir)
    tier = {
        "train_manifest": "train_s/manifest.json",
        "solver_seconds": 10,
        "slots": 321,
        "train_bodies": 4,
        "validation_manifest": "validation/manifest.json",
        "sealed_manifests": ["test_id/manifest.json", "test_ood/manifest.json"],
    }
    (release / "development-receipt.json").write_text(json.dumps({"tiers": {"S": tier}}))

    # Populate Stage-A outputs for potential_field under stage_a/gempy/{train_s,validation,test_id,test_ood}.
    stage_a_root = tmp_path / "stage_a"
    prob = fixture()
    for split_sub in ("train_s", "validation", "test_id", "test_ood"):
        m = json.loads((release / split_sub / "manifest.json").read_text())
        for c in m["cases"]:
            cdir = stage_a_root / "gempy" / split_sub / c["case_id"]
            cdir.mkdir(parents=True)
            np.save(cdir / "ore.npy", model(prob) > 0, allow_pickle=False)
            (cdir / "choice.json").write_text(json.dumps({
                "method": "potential_field",
                "generator_matched": True,
                "case_id": c["case_id"],
                "no_ore": False,
                "fallback": False,
                "cv": {"selected_rule": "native", "selected_config": {"form": "L", "orient": "plane", "range": 1.7},
                       "cv_score": 0.8, "cv_evaluated": 12},
                "wall_seconds": 1.5,
            }))

    refs_path = prepare_ref.prepare_ref_inputs(
        release, stage_a_root, tmp_path / "ref_inputs", tier="S",
        in_container_names=("initial",), offline_names=("potential_field",),
    )
    assert set(json.loads(refs_path.read_text())) == {"initial", "potential_field"}

    # Ensure initial source has a training receipt in the service for train_s/manifest.json.
    init_sha = json.loads(refs_path.read_text())["initial"]["sha256"]
    def fake_eval_sources(svc_root, manifest, seconds, slots, image, sources, **kwargs):
        assert kwargs.get("anonymize_cases") is True
        assert kwargs.get("canonical_origin") is True
        assert kwargs.get("randomize_orientation") is True
        assert kwargs.get("objective_version") == "objective_v3"
        from tasks.copper_inversion.core.candidate import seal
        from tasks.copper_inversion.v2.suite import aggregate
        svc_root = Path(svc_root)
        svc_root.mkdir(parents=True, exist_ok=True)
        train_m = json.loads(Path(manifest).read_text())
        (svc_root / "identity.json").write_text(json.dumps({
            "configuration": {"manifest": str(manifest), "seconds": seconds, "image": image},
            "suite_digest": digest(train_m),
        }))
        for src_p, sha in sources:
            seal(svc_root, Path(src_p).read_bytes())
            ev = svc_root / "evaluations" / sha
            rows = [
                {"case_id": c["case_id"], "seed": 47, "parent_id": c["parent_id"], "solver_wall_seconds": 1.0,
                 "status": "succeeded", "metrics": {"iou": 0.4, "confirmation_loss_total": 0.25}}
                for c in train_m["cases"]
            ]
            for r in rows:
                rp = ev / r["case_id"] / "47/receipt.json"
                rp.parent.mkdir(parents=True, exist_ok=True)
                rp.write_text(json.dumps(r))
            (ev / "private-case-results.json").write_text(json.dumps(rows))
            (ev / "result.json").write_text(json.dumps({
                "candidate_id": sha, "status": "succeeded", "metrics": aggregate(rows),
                "metadata": {"suite_digest": digest(train_m)},
            }))
        return svc_root

    from tasks.copper_inversion.tests.test_rsi_unseal import FakeSuite
    class _FakeSuiteKw(FakeSuite):
        def __init__(self, root, execution, manifest, seconds, **kwargs):
            super().__init__(root, execution, manifest, seconds)

    monkeypatch.setattr(q1, "evaluate_sources", fake_eval_sources)
    out_sel = tmp_path / "ref_selections"
    with fake_evaluator(), patch("tasks.copper_inversion.v2.selection.SuiteEvaluator", _FakeSuiteKw):
        mapping = q1.build_references(
            refs_path, release, out_sel, IMAGE,
            anonymize_cases=True, canonical_origin=True, randomize_orientation=True,
            objective_version="objective_v3",
        )
        assert set(mapping) == {"initial", "potential_field"}
        for name, target in mapping.items():
            frozen, st = selection.verify(target)
            assert st["phase"] == "selected"
            assert frozen["objective_version"] == "objective_v3"
        unseal_references(root, PID, refs_path, out_sel / "references.json", operator="eval")

    # Export M1 loop outcomes AND REF outcomes into the same directory and verify Outcomes.load().
    unseal_m1 = root / "private" / PID / "unsealed"
    unseal_m1.mkdir(parents=True, exist_ok=True)
    with fake_evaluator(), patch("tasks.copper_inversion.v2.selection.SuiteEvaluator", _FakeSuiteKw):
        from tasks.copper_inversion.experiments.rsi.unseal import unseal
        unseal(root, {"experiment": "M1", "protocol_id": PID,
                      "solvers": [{"run_id": "ldm4geo-s101", "prefix": "full"}]}, operator="eval")

    outcomes_dir = tmp_path / "private_exports"
    outcomes_dir.mkdir()
    assert export_outcomes.main([
        "--root", str(root), "--protocol-id", PID, "--experiment", "M1",
        "--output", str(outcomes_dir / "outcomes-M1.json"),
    ]) == 0
    ref_json = outcomes_dir / "outcomes-REF.json"
    rc = export_outcomes.main([
        "--root", str(root),
        "--protocol-id", PID,
        "--experiment", "REF",
        "--references", str(out_sel / "references.json"),
        "--output", str(ref_json),
    ])
    assert rc == 0
    assert ref_json.is_file() and (outcomes_dir / "outcomes-REF.csv").is_file()

    table = json.loads(ref_json.read_text())
    assert table["experiment"] == "REF"
    assert len(table["rows"]) == 2
    by_id = {r["run_id"]: r for r in table["rows"]}
    assert by_id["initial"]["selected_sha256"] == init_sha
    assert by_id["initial"]["arm"] == "in_container_reference"
    assert by_id["potential_field"]["arm"] == "offline_reference"
    assert by_id["potential_field"]["tier"] == "S"
    for name in ("initial", "potential_field"):
        for col in ("validation_iou", "test_id_iou", "test_ood_iou"):
            assert isinstance(by_id[name][col], float)

    loaded = Outcomes.load(outcomes_dir)
    assert set(loaded.references) == {"initial", "potential_field"}
    assert loaded.sealed("REF", "initial", "full", "test_id") == by_id["initial"]["test_id_iou"]
    assert loaded.validation_value("REF", "potential_field", "full", "iou") == by_id["potential_field"]["validation_iou"]


def test_wei2024_helpers_and_stage_b_loader(tmp_path):
    from tasks.copper_inversion.experiments.baselines import prepare_ref, wei2024_reference
    from tasks.copper_inversion.v2.geometry import canonicalize

    rng = np.random.default_rng(47)
    labels = rng.integers(0, 2, size=24).astype(float)
    distances = rng.normal(0.0, 2.5, size=24)
    contacts = rng.normal(0.1, 0.7, size=6)
    weight = 0.35
    upstream_formula = weight * (
        np.sum(np.log(1.0 + np.exp(distances[labels == 0])) / np.log2(2.0))
        + np.sum(np.log(1.0 + np.exp(-distances[labels == 1])) / np.log2(2.0))
        + (np.mean(contacts) ** 2 + np.mean(contacts ** 2))
    )
    assert np.isclose(
        wei2024_reference.finite_binary_loss(labels, distances, contacts, weight),
        upstream_formula,
        rtol=0,
        atol=1e-12,
    )
    # Empty contacts (e.g. outcrop observations) and extreme distances stay finite without overflow or NaN.
    val_empty_c = wei2024_reference.finite_binary_loss(
        np.array([0.0, 1.0]), np.array([950.0, -950.0]), np.zeros(0), 0.1
    )
    assert np.isfinite(val_empty_c) and val_empty_c > 0.0

    prob = fixture()
    origin = np.asarray(prob["origin"], dtype=float)
    spacing = np.asarray(prob["spacing"], dtype=float)
    c000 = origin + 0.5 * spacing
    c100 = origin + np.array([1.5, 0.5, 0.5]) * spacing
    c200 = origin + np.array([2.5, 0.5, 0.5]) * spacing
    obs = {
        "name": "drill",
        "points": np.array([c000, c000, c100, c100, c200, c200]),
        "labels": np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.5]),
    }
    grid = wei2024_reference.obs_to_grid(prob, obs)
    assert grid.shape == tuple(prob["shape"])
    assert grid[0, 0, 0] == 1.0
    assert grid[1, 0, 0] == 0.5  # tie -> 0.5
    assert grid[2, 0, 0] == 0.5  # explicit contact overrides binary sample in same cell
    assert np.isnan(grid[3, 3, 3])

    # Stage-B loader and prepare_ref offline-only mode for wei2024.
    release = tmp_path / "rel_wei"
    manifests = _make_release_with_confirmations(release)
    stage_a_root = tmp_path / "stage_a"
    for split, m_path in manifests.items():
        m = json.loads(m_path.read_text())
        for c in m["cases"]:
            p = fixture()
            p["case_id"] = c["case_id"]
            feasible = p["domain_mask"] & ~p["forbidden_mask"]
            f_prob = np.where((model(p) > 0)[feasible], 0.8, 0.1).astype(np.float32)
            cdir = stage_a_root / "wei2024" / split / c["case_id"] / "47"
            cdir.mkdir(parents=True)
            np.save(cdir / "field.npy", f_prob, allow_pickle=False)
            (cdir / "choice.json").write_text(json.dumps({
                "method": "wei2024",
                "case_id": c["case_id"],
                "seed": 47,
                "no_ore": False,
                "fallback": False,
                "cv": {"selected_rule": "native", "selected_config": {"iter_num": 1000}, "cv_score": None, "cv_evaluated": 0},
                "wall_seconds": 542.5,
            }))

    diag = {}
    sample_prob = fixture()
    sample_prob["case_id"] = json.loads(manifests["train"].read_text())["cases"][0]["case_id"]
    out = offline_baselines.wei2024(
        sample_prob, {"wall_time_seconds": 30}, seed=47,
        fields_dir=stage_a_root / "wei2024", split="train", diagnostics=diag,
    )
    canonicalize(sample_prob, out)
    assert diag["wall_seconds"] >= 542.5

    refs_path = prepare_ref.prepare_ref_inputs(
        release, stage_a_root, tmp_path / "ref2_inputs", tier="S",
        in_container_names=(), offline_names=("wei2024",),
    )
    refs_data = json.loads(refs_path.read_text())
    assert set(refs_data) == {"wei2024"}
    assert "fields" in refs_data["wei2024"]


def test_ref_and_ref2_independent_one_shot_unseal_and_merged_outcomes(tmp_path, monkeypatch):
    from tasks.copper_inversion.experiments.baselines import prepare_ref
    from tasks.copper_inversion.experiments.rsi.analysis.outcomes import Outcomes
    from tasks.copper_inversion.experiments.rsi.unseal import unseal
    from tasks.copper_inversion.tests.test_rsi_unseal import FakeSuite, fake_evaluator

    root, manifests, _ = setup(tmp_path, ["ldm4geo-s101"], prefixes=(None,))
    release = manifests["train"].parent.parent
    import shutil
    shutil.copytree(release / "train", release / "train_s")
    tier = {
        "train_manifest": "train_s/manifest.json",
        "solver_seconds": 10,
        "slots": 321,
        "train_bodies": 4,
        "validation_manifest": "validation/manifest.json",
        "sealed_manifests": ["test_id/manifest.json", "test_ood/manifest.json"],
    }
    (release / "development-receipt.json").write_text(json.dumps({"tiers": {"S": tier}}))

    stage_a_root = tmp_path / "stage_a"
    prob = fixture()
    for split_sub in ("train_s", "validation", "test_id", "test_ood"):
        m = json.loads((release / split_sub / "manifest.json").read_text())
        for c in m["cases"]:
            feasible = prob["domain_mask"] & ~prob["forbidden_mask"]
            f_prob = np.where((model(prob) > 0)[feasible], 0.85, 0.10).astype(np.float32)
            cdir = stage_a_root / "wei2024" / split_sub / c["case_id"] / "47"
            cdir.mkdir(parents=True)
            np.save(cdir / "field.npy", f_prob, allow_pickle=False)
            (cdir / "choice.json").write_text(json.dumps({
                "method": "wei2024",
                "case_id": c["case_id"],
                "seed": 47,
                "no_ore": False,
                "fallback": False,
                "cv": {"selected_rule": "native", "selected_config": {"iter_num": 1000}, "cv_score": None, "cv_evaluated": 0},
                "wall_seconds": 550.0,
            }))

    ref1_path = prepare_ref.prepare_ref_inputs(
        release, stage_a_root, tmp_path / "ref1_in", tier="S",
        in_container_names=("initial",), offline_names=(),
    )
    ref2_path = prepare_ref.prepare_ref_inputs(
        release, stage_a_root, tmp_path / "ref2_in", tier="S",
        in_container_names=(), offline_names=("wei2024",),
    )

    def fake_eval_sources(svc_root, manifest, seconds, slots, image, sources, **kwargs):
        from tasks.copper_inversion.core.candidate import seal
        from tasks.copper_inversion.v2.suite import aggregate
        svc_root = Path(svc_root)
        svc_root.mkdir(parents=True, exist_ok=True)
        train_m = json.loads(Path(manifest).read_text())
        (svc_root / "identity.json").write_text(json.dumps({
            "configuration": {"manifest": str(manifest), "seconds": seconds, "image": image},
            "suite_digest": digest(train_m),
        }))
        for src_p, sha in sources:
            seal(svc_root, Path(src_p).read_bytes())
            ev = svc_root / "evaluations" / sha
            rows = [
                {"case_id": c["case_id"], "seed": 47, "parent_id": c["parent_id"], "solver_wall_seconds": 1.0,
                 "status": "succeeded", "metrics": {"iou": 0.4, "confirmation_loss_total": 0.25}}
                for c in train_m["cases"]
            ]
            for r in rows:
                rp = ev / r["case_id"] / "47/receipt.json"
                rp.parent.mkdir(parents=True, exist_ok=True)
                rp.write_text(json.dumps(r))
            (ev / "private-case-results.json").write_text(json.dumps(rows))
            (ev / "result.json").write_text(json.dumps({
                "candidate_id": sha, "status": "succeeded", "metrics": aggregate(rows),
                "metadata": {"suite_digest": digest(train_m)},
            }))
        return svc_root

    class _FakeSuiteKw(FakeSuite):
        def __init__(self, root, execution, manifest, seconds, **kwargs):
            super().__init__(root, execution, manifest, seconds)

    monkeypatch.setattr(q1, "evaluate_sources", fake_eval_sources)
    sel_ref1 = tmp_path / "sel_ref1"
    sel_ref2 = tmp_path / "sel_ref2"
    with fake_evaluator(), patch("tasks.copper_inversion.v2.selection.SuiteEvaluator", _FakeSuiteKw):
        q1.build_references(ref1_path, release, sel_ref1, IMAGE,
                            anonymize_cases=True, canonical_origin=True, randomize_orientation=True,
                            objective_version="objective_v3")
        q1.build_references(ref2_path, release, sel_ref2, IMAGE,
                            anonymize_cases=True, canonical_origin=True, randomize_orientation=True,
                            objective_version="objective_v3")

        res1 = unseal_references(root, PID, ref1_path, sel_ref1 / "references.json",
                                 operator="eval", experiment="REF")
        assert res1 == {"status": "unsealed", "experiment": "REF", "references": 1}
        with pytest.raises(ValueError, match="untested|already been unsealed"):
            unseal_references(root, PID, ref1_path, sel_ref1 / "references.json", experiment="REF")

        # Exporting REF2 before REF2 is unsealed is rejected even though REF is already unsealed!
        with pytest.raises(ValueError, match="completed REF2 unseal log|tested frozen solver"):
            export_outcomes.export_references(
                root, PID, references=json.loads((sel_ref2 / "references.json").read_text()),
                experiment="REF2",
            )

        # REF2 unseals cleanly once on the same (root, PID), then locks independently.
        res2 = unseal_references(root, PID, ref2_path, sel_ref2 / "references.json",
                                 operator="eval", experiment="REF2")
        assert res2 == {"status": "unsealed", "experiment": "REF2", "references": 1}
        with pytest.raises(ValueError, match="untested|already been unsealed"):
            unseal_references(root, PID, ref2_path, sel_ref2 / "references.json", experiment="REF2")

        unseal(root, {"experiment": "M1", "protocol_id": PID,
                      "solvers": [{"run_id": "ldm4geo-s101", "prefix": "full"}]}, operator="eval")

    outcomes_dir = tmp_path / "merged_exports"
    outcomes_dir.mkdir()
    assert export_outcomes.main([
        "--root", str(root), "--protocol-id", PID, "--experiment", "M1",
        "--output", str(outcomes_dir / "outcomes-M1.json"),
    ]) == 0
    assert export_outcomes.main([
        "--root", str(root), "--protocol-id", PID, "--experiment", "REF",
        "--references", str(sel_ref1 / "references.json"),
        "--output", str(outcomes_dir / "outcomes-REF.json"),
    ]) == 0
    assert export_outcomes.main([
        "--root", str(root), "--protocol-id", PID, "--experiment", "REF2",
        "--references", str(sel_ref2 / "references.json"),
        "--output", str(outcomes_dir / "outcomes-REF2.json"),
    ]) == 0

    loaded = Outcomes.load(outcomes_dir)
    assert set(loaded.references) == {"initial", "wei2024"}
    assert isinstance(loaded.sealed("REF", "initial", "full", "test_id"), float)
    assert isinstance(loaded.sealed("REF2", "wei2024", "full", "test_id"), float)
    assert isinstance(loaded.sealed("M1", "ldm4geo-s101", "full", "test_id"), float)
