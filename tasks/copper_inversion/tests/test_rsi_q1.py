"""Reproducible risk-decile selection for Q1."""
import json

import pytest

from tasks.copper_inversion.experiments.rsi import q1


def test_selection_strata_repeat_determinism_and_overwrite(tmp_path, monkeypatch):
    runs = [tmp_path / "a", tmp_path / "b"]
    for run in runs:
        run.mkdir()
        (run / "ledger.jsonl").write_text(run.name)
    rows = [{"program_id": f"{i:064x}", "risk": i / 100, "status": "valid"} for i in range(100)]
    monkeypatch.setattr(q1, "read", lambda path: rows[:50] if path.parent.name == "a" else rows[50:])
    a, b = tmp_path / "selection.json", tmp_path / "same.json"
    sha = q1.select(runs, a, 42)
    assert sha == q1.select(reversed(runs), b, 42)
    selection = json.loads(a.read_text())
    assert len(selection["programs"]) == 60
    assert len(selection["repeat_program_ids"]) == 12
    assert set(selection["repeat_program_ids"]) <= {r["program_id"] for r in selection["programs"]}
    for decile in range(1, 11):
        group = [r for r in selection["programs"] if r["decile"] == decile]
        assert len(group) == 6
        assert all((decile - 1) * 10 <= int(r["program_id"], 16) < decile * 10 for r in group)
    with pytest.raises(FileExistsError):
        q1.select(runs, a, 42)
    assert q1.select(runs, tmp_path / "other.json", 43) != sha


def test_selection_requires_sixty_distinct_evaluated_programs(tmp_path, monkeypatch):
    (tmp_path / "ledger.jsonl").write_text("ledger")
    monkeypatch.setattr(q1, "read", lambda _: [{"program_id": "a" * 64, "risk": .2, "status": "valid"}])
    with pytest.raises(ValueError, match="60 distinct"):
        q1.select([tmp_path], tmp_path / "selection.json", 0)


def test_evaluate_and_export_training_roots(tmp_path, monkeypatch):
    from tasks.copper_inversion.core.candidate import seal
    from tasks.copper_inversion.core.execution import digest
    from tasks.copper_inversion.experiments.rsi import export_outcomes
    from tasks.copper_inversion.tests.test_rsi_unseal import make_release, IMAGE, PID
    from tasks.copper_inversion.v2.suite import aggregate

    manifests = make_release(tmp_path / "release")
    train = manifests["train"]
    data = json.loads(train.read_text())
    data["seeds"] = [47, 48, 49]
    train.write_text(json.dumps(data))
    tier = {"train_manifest": "train/manifest.json", "solver_seconds": 10, "slots": 321,
            "train_bodies": 4, "validation_manifest": "validation/manifest.json",
            "sealed_manifests": ["test_id/manifest.json", "test_ood/manifest.json"]}
    release = train.parent.parent
    (release / "development-receipt.json").write_text(json.dumps({"tiers": {"S": tier, "F": tier},
        "q1": {"both_packs_manifest": "train/manifest.json", "execution_seeds_manifest": "train/manifest.json",
               "solver_seconds": 10}}))
    run = tmp_path / "run"
    run.mkdir()
    (run / "ledger.jsonl").write_text("ledger")
    rows = []
    for i in range(60):
        body = f"def solve(problem,budget,seed):\n    return {i}\n".encode()
        sha = seal(run, body)["sha256"]
        rows.append({"program_id": sha, "source_run": str(run)})
    from tasks.copper_inversion.v2.suite import file_hash
    chosen = {"schema": "rsi-q1-selection-v1", "programs": rows,
              "repeat_program_ids": [r["program_id"] for r in rows[:12]],
              "input_run_hashes": {str(run): file_hash(run / "ledger.jsonl")}}
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps(chosen))

    class FakeService:
        def __init__(self, config):
            self.identity = {"configuration": config}
            self.root = Path(config["root"])
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "identity.json").write_text(json.dumps(self.identity))
        def dispatch(self, request):
            sha = seal(self.root, request["source"].encode())["sha256"]
            suite = json.loads(Path(self.identity["configuration"]["manifest"]).read_text())
            evaluation = self.root / "evaluations" / sha
            cases = []
            for case in suite["cases"]:
                for seed in suite["seeds"]:
                    row = {"parent_id": case["parent_id"], "case_id": case["case_id"], "seed": seed,
                           "status": "succeeded", "metrics": {"iou": .5, "confirmation_loss_total": seed / 100},
                           "solver_wall_seconds": 1.}
                    receipt = evaluation / case["case_id"] / str(seed) / "receipt.json"
                    receipt.parent.mkdir(parents=True, exist_ok=True)
                    receipt.write_text(json.dumps(row))
                    cases.append(row)
            (evaluation / "private-case-results.json").write_text(json.dumps(cases))
            (evaluation / "result.json").write_text(json.dumps({"status": "succeeded", "candidate_id": sha,
                "metrics": aggregate(cases), "metadata": {"suite_digest": digest(suite)}}))
            return {"status": "succeeded", "source_sha256": sha}

    from pathlib import Path
    monkeypatch.setattr(q1, "EvaluatorService", FakeService)
    mapping = q1.evaluate(sample, release, tmp_path / "output", IMAGE)
    table = export_outcomes.export_references(tmp_path / "active", PID, q1=mapping)
    assert len(table["q1"]["programs"]) == 60
    assert len(table["q1"]["repeats"]) == 12
    assert table["q1"]["repeats"][0]["risks"] == pytest.approx([1 - 1 / (1 + s / 100) for s in [47, 48, 49]])
    assert "iou" not in json.dumps(table["q1"])
    assert json.loads((tmp_path / "output/short_24x2/q1-layout.json").read_text())["layout"] == "short_24x2"
    missing = rows[0]["program_id"]
    (tmp_path / "output/full_24x1/evaluations" / missing / "result.json").unlink()
    with pytest.raises(ValueError, match="program receipts differ"):
        export_outcomes.export_references(tmp_path / "active", PID, q1=mapping)
    mapping["short_12x1"] = str(tmp_path / "bad")
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad/identity.json").write_text(json.dumps({"configuration": {"manifest": str(manifests["validation"])}}))
    with pytest.raises(ValueError, match="training"):
        export_outcomes.export_references(tmp_path / "active", PID, q1=mapping)


def test_reference_build_unseal_and_log_guard(tmp_path, monkeypatch):
    from tasks.copper_inversion.experiments.rsi import export_outcomes
    from tasks.copper_inversion.experiments.rsi.freeze import layout
    from tasks.copper_inversion.experiments.rsi.unseal import unseal_references
    from tasks.copper_inversion.tests.test_rsi_unseal import setup, fake_evaluator, IMAGE, PID
    from tasks.copper_inversion.v2.training_receipts import eligible

    root, manifests, _ = setup(tmp_path, ["ldm4geo-s101"], prefixes=(None,))
    service = layout(root, PID, "M1", "ldm4geo-s101")["service"]
    trained = eligible(service, manifests["train"])
    sha = next(iter(trained))
    release = manifests["train"].parent.parent
    tier = {"train_manifest": "train/manifest.json", "solver_seconds": 10, "slots": 321, "train_bodies": 4,
            "validation_manifest": "validation/manifest.json",
            "sealed_manifests": ["test_id/manifest.json", "test_ood/manifest.json"]}
    (release / "development-receipt.json").write_text(json.dumps({"tiers": {"S": tier}}))
    refs = tmp_path / "sources.json"
    refs.write_text(json.dumps({"A0": {"source": trained[sha]["source_path"], "sha256": sha}}))
    monkeypatch.setattr(q1, "evaluate_sources", lambda *args, **kwargs: service)
    with fake_evaluator():
        mapping = q1.build_references(refs, release, tmp_path / "refs", IMAGE)
        with pytest.raises(ValueError, match="REF unseal"):
            export_outcomes.export_references(root, PID, references=mapping)
        unseal_references(root, PID, refs, tmp_path / "refs/references.json")
        table = export_outcomes.export_references(root, PID, references=mapping)
        assert set(table["references"]["A0"]) == {"test_id", "test_ood"}
        with pytest.raises(ValueError):
            unseal_references(root, PID, refs, tmp_path / "refs/references.json")
    log = layout(root, PID, "Q1")["unseal_log"]
    entries = [json.loads(line) for line in log.read_text().splitlines()]
    entry = entries[-1]
    assert entry["experiment"] == "REF" and entry["mode"] == "completed"
    assert entry["references_sha256"] == __import__("hashlib").sha256(refs.read_bytes()).hexdigest()
    entry["results_sha256"]["A0"] = "0" * 64
    log.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    with pytest.raises(ValueError, match="REF unseal log"):
        export_outcomes.export_references(root, PID, references=mapping)
    with pytest.raises(ValueError, match="disclose"):
        export_outcomes._guard({"q1": {"programs": [{"risk": .1, "leak": "f" * 64}]}}, {"f" * 64})
