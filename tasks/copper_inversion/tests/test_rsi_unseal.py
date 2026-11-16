"""Prefix freeze, unseal and outcome export with a fake evaluator (no Docker): AT-29, AT-30, AT-31."""
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from tasks.copper_inversion.core.candidate import seal
from tasks.copper_inversion.core.execution import digest
from tasks.copper_inversion.experiments.rsi import export_outcomes, unseal as unseal_module
from tasks.copper_inversion.experiments.rsi.freeze import freeze_run, layout, validate_run
from tasks.copper_inversion.experiments.rsi.unseal import unseal
from tasks.copper_inversion.rsi.exposure import risk_by_case
from tasks.copper_inversion.rsi.ledger import append, read
from tasks.copper_inversion.rsi.service import RiskOnlyService
from tasks.copper_inversion.tests.test_v2_science import fixture, model
from tasks.copper_inversion.v2 import selection
from tasks.copper_inversion.v2.data import save_problem
from tasks.copper_inversion.v2.suite import aggregate, file_hash

PID = "rsi-test-" + "c" * 12
IMAGE = "sha256:" + "a" * 64
MODALITIES = ("drill", "drill_outcrop")
# (round, branch, selection, status, training risk). Rounds 0-2 favour other programs than 3-4.
SCHEDULE = [(0, -1, "initial", "valid", .60), (1, 0, "reserved", "valid", .50), (1, 1, "reserved", "rejected", 1.),
            (2, 0, "reserved", "valid", .40), (2, 1, "reserved", "valid", .45), (3, 0, "reserved", "failed", 1.),
            (3, 1, "reserved", "valid", .30), (4, 0, "reserved", "valid", .20), (4, 1, "reserved", "valid", .35)]


def make_release(root):
    manifests, index = {}, 0
    for split in ("train", "validation", "test_id", "test_ood"):
        base, rows = root / split, []
        for parent in range(4):
            modality = MODALITIES[parent % 2]
            truth = base / f"truth-{parent}"
            truth.mkdir(parents=True)
            mask = model(fixture()) > 0
            mask[0, index % 8, index // 8] = False
            index += 1
            np.save(truth / "truth.npy", mask)
            problem = fixture()
            problem["case_id"] = f"{split}-{parent}-{modality}"
            save_problem(problem, base / f"case-{parent}")
            rows.append(dict(case_id=problem["case_id"], parent_id=hashlib.sha256(f"{split}{parent}".encode()).hexdigest(),
                             modality=modality, public=f"case-{parent}", truth=f"truth-{parent}",
                             sha256={name: file_hash(base / (f"truth-{parent}" if name == "truth.npy" else f"case-{parent}") / name)
                                     for name in ("problem.json", "problem.npz", "truth.npy")}))
        manifests[split] = base / "manifest.json"
        manifests[split].write_text(json.dumps(dict(schema="copper-suite-v2", stage="development", split=split,
                                                    seeds=[47], cases=rows)))
    return manifests


def make_run(root, run_id, offset, train_manifest, experiment="M1"):
    paths = layout(root, PID, experiment, run_id)
    service, run = paths["service"], paths["run"]
    train = json.loads(train_manifest.read_text())
    identity = {"configuration": {"manifest": str(train_manifest), "seconds": 10, "image": IMAGE},
                "suite_digest": digest(train)}
    service.mkdir(parents=True)
    (service / "identity.json").write_text(json.dumps(identity))
    case_ids = sorted(c["case_id"] for c in train["cases"])
    for seq, (round_index, branch, reason, status, risk) in enumerate(SCHEDULE):
        body = f"def solve(problem,budget,seed):\n    return {offset * 100 + seq}\n".encode()
        sha = seal(service, body)["sha256"]
        seal(run, body)
        feedback = dict(risk=1., risk_by_case={c: 1. for c in case_ids}, legal_rate=0.,
                        failed_solves=len(case_ids), seconds_per_solve=0., message="interface rejected")
        if status != "rejected":
            loss = (risk - offset / 100) / (1 - risk + offset / 100) if status == "valid" else None
            rows = [dict(case_id=c["case_id"], seed=47, parent_id=c["parent_id"], solver_wall_seconds=2.,
                         **(dict(status="succeeded", metrics={"iou": .3, "confirmation_loss_total": loss})
                            if status == "valid" else dict(status="failed", metrics={"iou": 0.})))
                    for c in train["cases"]]
            evaluation = service / "evaluations" / sha
            for row in rows:
                receipt = evaluation / row["case_id"] / "47/receipt.json"
                receipt.parent.mkdir(parents=True)
                receipt.write_text(json.dumps(row))
            (evaluation / "private-case-results.json").write_text(json.dumps(rows))
            metrics = aggregate(rows)
            (evaluation / "result.json").write_text(json.dumps(dict(
                candidate_id=sha, status="succeeded", metrics=metrics, metadata={"suite_digest": digest(train)})))
            feedback = dict(risk=metrics["loss_total"], risk_by_case=risk_by_case(rows, case_ids),
                            legal_rate=metrics["legal_rate"], failed_solves=metrics["failed_solves"],
                            seconds_per_solve=2., message="" if status == "valid" else "all planned solves failed")
        append(run / "ledger.jsonl", dict(seq=seq, program_id=sha, epoch=0, round=round_index, branch=branch,
                                          status=status, selection=reason, **feedback))
    # The real service rule writes the full-run training seal.
    RiskOnlyService.seal_training(SimpleNamespace(root=service, identity=identity, manifest_path=train_manifest,
                                                  ledger=SimpleNamespace(counters={"algorithm_feedback_slots": len(SCHEDULE)})))
    arm, seed = run_id.rsplit("-s", 1)
    (run / "manifest.json").write_text(json.dumps(dict(run_id=run_id, experiment=experiment, protocol_id=PID, arm=arm,
                                                       tier="S", judge="standard", seed=int(seed))))
    (run / "result.json").write_text(json.dumps(dict(status="completed", stop_reason="completed")))
    return paths


class FakeSuite:
    """Writes per-body receipts like SuiteEvaluator; values vary by program and body."""
    calls = []
    bodies = {}

    def __init__(self, root, execution, manifest, seconds):
        self.root, self.path = Path(root), Path(manifest)
        self.manifest = json.loads(self.path.read_text())

    def evaluate(self, candidate):
        sha = candidate.canonical_key
        FakeSuite.calls.append((sha, self.manifest["split"]))
        q = int(sha[:8], 16) / 2 ** 32 / 10
        key = digest(dict(source=sha, manifest=str(self.path)))
        rows = []
        for i, case in enumerate(self.manifest["cases"]):
            row = dict(parent_id=case["parent_id"], case_id=case["case_id"], seed=47, status="succeeded",
                       metrics=dict(iou=.3 + q + .01 * i * i, confirmation_loss_total=.2 + q + .01 * i,
                                    mean_surface_distance=1.5 + q + .1 * i * i, relative_volume_error=q + .05 * i * i),
                       solver_wall_seconds=1. + i / 10, jobs=2, error="")
            receipt = self.root / "evaluations" / key / case["case_id"] / "47/receipt.json"
            receipt.parent.mkdir(parents=True)
            receipt.write_text(json.dumps(row))
            FakeSuite.bodies.setdefault(sha, set()).update(row["metrics"].values())
            rows.append(row)
        (self.root / "evaluations" / key / "private-case-results.json").write_text(json.dumps(rows))
        (self.root / "evaluations" / key / "result.json").write_text(json.dumps({"candidate_id": sha}))
        return SimpleNamespace(metrics=aggregate(rows))


@contextmanager
def fake_evaluator():
    with (patch("tasks.copper_inversion.v2.selection.Execution", lambda root, image: SimpleNamespace(image=image)),
          patch("tasks.copper_inversion.v2.selection.SuiteEvaluator", FakeSuite),
          patch("tasks.copper_inversion.v2.selection.InterfaceChecks", lambda root, execution: lambda payload: True)):
        yield


def setup(tmp_path, runs, prefixes=(None, 2), validate=True, experiment="M1"):
    FakeSuite.calls.clear()
    manifests = make_release(tmp_path / "release")
    root = tmp_path / "active"
    for index, run_id in enumerate(runs):
        make_run(root, run_id, index, manifests["train"], experiment)
        with fake_evaluator():
            for round_index in prefixes:
                freeze_run(root, PID, experiment, run_id, round_index, release=tmp_path / "release")
                if validate:
                    validate_run(root, PID, experiment, run_id, round_index)
    listing = {"experiment": experiment, "protocol_id": PID,
               "solvers": [{"run_id": r, "prefix": "full" if p is None else f"round_{p:02d}"} for r in runs for p in prefixes]}
    return root, manifests, listing


def snapshot(folder):
    return {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in Path(folder).rglob("*") if p.is_file()}


def test_prefix_freeze_uses_only_its_rounds_and_leaves_full_freeze_alone(tmp_path):  # AT-29
    root, manifests, _ = setup(tmp_path, ["ldm4geo-s101"], prefixes=(None,))
    paths = layout(root, PID, "M1", "ldm4geo-s101")
    full_dir = paths["run_selection"] / "full"
    before = snapshot(full_dir)
    sealed = json.loads((paths["service"] / "training-sealed.json").read_text())
    assert json.loads((full_dir / "prefix.json").read_text())["source_sha256s"] == sealed["source_sha256s"]
    rounds = {row["program_id"]: row["round"] for row in read(paths["run"] / "ledger.jsonl")}
    with fake_evaluator():
        record = freeze_run(root, PID, "M1", "ldm4geo-s101", 2, release=tmp_path / "release")
        validate_run(root, PID, "M1", "ldm4geo-s101", 2)
        assert freeze_run(root, PID, "M1", "ldm4geo-s101", 2, release=tmp_path / "release") == record
    assert record["prefix"] == "round_02" and record["last_round"] == 2 and record["records"] == 5
    assert len(record["source_sha256s"]) == 3 and all(rounds[s] <= 2 for s in record["source_sha256s"])
    assert [rounds[s] for s in sealed["source_sha256s"]] == [4, 3, 4]
    frozen = json.loads((paths["run_selection"] / "round_02/frozen.json").read_text())
    assert [c["sha256"] for c in frozen["candidates"]] == record["source_sha256s"]
    assert snapshot(full_dir) == before
    assert {p.name for p in paths["run_selection"].iterdir()} == {"full", "round_02"}
    # The public ledger is research-writable; it must agree with the private receipts.
    rows = read(paths["run"] / "ledger.jsonl")
    rows[3].update(risk=rows[3]["risk"] - .1, risk_by_case={k: v - .1 for k, v in rows[3]["risk_by_case"].items()})
    (paths["run"] / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="private training receipt"):
        freeze_run(root, PID, "M1", "ldm4geo-s101", 3, release=tmp_path / "release")


def test_unseal_refuses_incomplete_lists_logs_and_replays(tmp_path):  # AT-30
    root, _, listing = setup(tmp_path, ["ldm4geo-s101", "accumulated-s101"])
    log = layout(root, PID, "M1")["unseal_log"]
    validation_calls = len(FakeSuite.calls)
    with fake_evaluator():
        partial = {**listing, "solvers": [s for s in listing["solvers"] if s["run_id"] != "accumulated-s101"]}
        with pytest.raises(ValueError, match="missing runs: accumulated-s101"):
            unseal(root, partial)
        with pytest.raises(ValueError, match="same freeze prefixes"):
            unseal(root, {**listing, "solvers": listing["solvers"][:-1]})
        with pytest.raises(ValueError, match="missing runs: ldm4geo-s102"):
            unseal(root, listing, expected_runs=["ldm4geo-s101", "accumulated-s101", "ldm4geo-s102"])
        manifest = layout(root, PID, "M1", "accumulated-s101")["run"] / "manifest.json"
        original = manifest.read_text()
        manifest.write_text(json.dumps({**json.loads(original), "protocol_id": "rsi-test-other"}))
        with pytest.raises(ValueError, match="protocol differs"):
            unseal(root, listing)
        manifest.write_text(original)
        assert not log.exists() and len(FakeSuite.calls) == validation_calls
        first = unseal(root, listing, operator="tester")
        assert first["mode"] == "unseal" and len(FakeSuite.calls) == validation_calls + 4 * 2
        entries = [json.loads(line) for line in log.read_text().splitlines()]
        assert len(entries) == 1 and entries[0]["frozen_list_sha256"] == first["frozen_list_sha256"]
        assert {"time", "experiment", "frozen_list_sha256", "protocol_id", "operator"} <= set(entries[0])
        assert entries[0]["protocol_id"] == PID and entries[0]["operator"] == "tester"
        receipts = unseal_module._receipts(p for p in (root / "private").rglob("test") if p.is_dir())
        assert len([k for k in receipts if k.endswith("receipt.json")]) == 4 * 2 * 4
        again = unseal(root, dict(reversed(list(listing.items()))))
        assert again["mode"] == "replay" and len(FakeSuite.calls) == validation_calls + 8
        assert unseal_module._receipts(p for p in (root / "private").rglob("test") if p.is_dir()) == receipts
        assert len(log.read_text().splitlines()) == 2


def test_unseal_refuses_unvalidated_or_unlogged_sealed_evaluation(tmp_path):  # AT-30
    root, _, listing = setup(tmp_path, ["ldm4geo-s101"], prefixes=(None,), validate=False)
    with fake_evaluator():
        with pytest.raises(ValueError, match="validate before unsealing"):
            unseal(root, listing)
        target = layout(root, PID, "M1", "ldm4geo-s101")["run_selection"] / "full"
        selection.run(target, "validation")
        selection.run(target, "test")
        with pytest.raises(ValueError, match="without an unseal log"):
            unseal(root, listing)


def test_unseal_needs_a_signed_off_blind_review(tmp_path):
    root, _, listing = setup(tmp_path, ["ldm4geo-s101", "accumulated-s101"])
    service = layout(root, PID, "M1", "accumulated-s101")["service"]
    (service / unseal_module.REVIEW_FILE).write_text(json.dumps(
        {"run_id": "accumulated-s101", "review_sha256": "a" * 64, "hits": 2}))
    review = tmp_path / "review.json"
    entry = {"review_sha256": "a" * 64, "verdict": "explained", "reviewer": "author", "note": "agent notes"}
    with fake_evaluator():
        with pytest.raises(ValueError, match="blind review required for: accumulated-s101"):
            unseal(root, listing)
        for bad in ({**entry, "review_sha256": "b" * 64}, {**entry, "verdict": "leak"}, {**entry, "reviewer": ""}):
            review.write_text(json.dumps({"schema": "rsi-blind-review-v1", "experiment": "M1",
                                          "runs": {"accumulated-s101": bad}}))
            with pytest.raises(ValueError, match="blind review"):
                unseal(root, listing, review=review)
        assert not layout(root, PID, "M1")["unseal_log"].exists()
        body = {"schema": "rsi-blind-review-v1", "experiment": "M1", "runs": {"accumulated-s101": entry}}
        review.write_text(json.dumps(body))
        assert unseal(root, listing, review=review)["mode"] == "unseal"
    logged = json.loads(layout(root, PID, "M1")["unseal_log"].read_text().splitlines()[0])
    assert logged["blind_review_sha256"] == digest(body)


def test_exported_outcomes_hold_no_per_body_truth(tmp_path):  # AT-31
    root, manifests, listing = setup(tmp_path, ["ldm4geo-s101", "accumulated-s101"])
    with pytest.raises(ValueError, match="after the unseal"):
        export_outcomes.export(root, PID, "M1")
    with pytest.raises(ValueError, match="validation-only"):
        export_outcomes.export(root, PID, "M1", validation_only=True)
    with fake_evaluator():
        unseal(root, listing)
    table = export_outcomes.export(root, PID, "M1")
    assert table["sealed"] and len(table["rows"]) == 4 and table["columns"] == list(export_outcomes.COLUMNS)
    assert all(list(row) == table["columns"] for row in table["rows"])
    assert not any(word in c for c in table["columns"] for word in ("case", "parent", "body", "truth"))
    text = json.dumps(table) + export_outcomes.to_csv(table)
    for path in manifests.values():
        for case in json.loads(path.read_text())["cases"]:
            assert case["parent_id"] not in text and case["case_id"] not in text
            assert not any(value in text for value in case["sha256"].values())
        assert file_hash(path) not in text
    for row in table["rows"]:
        per_body = FakeSuite.bodies[row["selected_sha256"]]
        numbers = [v for v in row.values() if isinstance(v, float)]
        assert numbers and not any(v in per_body for v in numbers)
        assert row["test_id_iou_drill"] is not None and row["test_ood_surface_distance"] is not None
        assert 0 < row["test_id_iou"] < 1 and row["round"] in (None, 2)
