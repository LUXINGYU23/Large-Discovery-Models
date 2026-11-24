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
