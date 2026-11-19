"""Extra export blocks read by the analysis package, loaded end to end by its reader."""
import json

import pytest

from tasks.copper_inversion.experiments.rsi import export_outcomes
from tasks.copper_inversion.experiments.rsi.analysis.outcomes import Outcomes
from tasks.copper_inversion.experiments.rsi.freeze import layout
from tasks.copper_inversion.experiments.rsi.unseal import unseal, unseal_references
from tasks.copper_inversion.tests.test_rsi_unseal import IMAGE, PID, FakeSuite, fake_evaluator, setup
from tasks.copper_inversion.v2 import selection
from tasks.copper_inversion.v2.training_receipts import eligible


def private_identities(manifests):
    tokens = set()
    for path in manifests.values():
        for case in json.loads(path.read_text())["cases"]:
            tokens.update([case["parent_id"], case["case_id"], *case["sha256"].values()])
    return tokens


def test_sealed_export_blocks_load_in_the_analysis_reader(tmp_path):
    root, manifests, listing = setup(tmp_path, ["ldm4geo-s101", "accumulated-s101"])
    with fake_evaluator():
        unseal(root, listing)
    out = tmp_path / "outcomes"
    out.mkdir()
    assert export_outcomes.main(["--root", str(root), "--protocol-id", PID, "--experiment", "M1",
                                 "--output", str(out / "M1.json")]) == 0
    text = (out / "M1.json").read_text() + (out / "M1.csv").read_text()
    assert not any(token in text for token in private_identities(manifests))
    table = json.loads((out / "M1.json").read_text())
    for row in table["rows"]:  # AT-31 still holds outside body_scores
        per_body = FakeSuite.bodies[row["selected_sha256"]]
        assert not any(v in per_body for v in row.values() if isinstance(v, float))
    assert not any(v["risk"] in FakeSuite.bodies[v["program_id"]] or v["iou"] in FakeSuite.bodies[v["program_id"]]
                   for v in table["validations"])
    outcomes = Outcomes.load(out)
    assert outcomes.protocol_id == PID and len(outcomes.freezes) == 4
    test_cases = {split: json.loads(manifests[split].read_text())["cases"] for split in ("test_id", "test_ood")}
    for run_id in ("ldm4geo-s101", "accumulated-s101"):
        paths = layout(root, PID, "M1", run_id)
        for prefix, key in (("full", "full"), ("round_02", 2)):
            sha = json.loads((paths["run_selection"] / prefix / "selection.json").read_text())["selected_sha256"]
            q = int(sha[:8], 16) / 2 ** 32 / 10
            for split, cases in test_cases.items():
                # The fake gives body i the IoU .3 + q + .01 i^2; order is by private parent id.
                expected = [.3 + q + .01 * i * i for i, _ in sorted(enumerate(cases), key=lambda c: c[1]["parent_id"])]
                assert outcomes.body_row("M1", run_id, key, split) == pytest.approx(expected, abs=1e-12)
                assert outcomes.sealed("M1", run_id, key, split) == pytest.approx(sum(expected) / 4, abs=1e-12)
        trained = eligible(paths["service"], manifests["train"])
        assert {sha: outcomes.train_iou[("M1", run_id, sha)] for sha in trained} == \
            {sha: r["metrics"]["mean_iou"] for sha, r in trained.items()}
        finalists = set()
        for prefix in ("full", "round_02"):
            report = json.loads((paths["run_selection"] / prefix / "selection.json").read_text())
            for row in report["candidates"]:
                finalists.add(row["source_sha256"])
                assert outcomes.validation[("M1", run_id, row["source_sha256"])]["risk"] == row["metrics"]["loss_total"]
        assert {k[2] for k in outcomes.validation if k[1] == run_id} == finalists


def test_validation_only_export_has_no_body_scores(tmp_path):
    root, _, _ = setup(tmp_path, ["ldm4geo-s101"], prefixes=(None,), experiment="E1")
    table = export_outcomes.export(root, PID, "E1", validation_only=True)
    assert "body_scores" not in table and not table["sealed"]
    assert table["programs"] and table["validations"] and table["rows"][0]["test_id_iou"] is None
    outcomes = Outcomes([table])
    assert outcomes.bodies == {} and outcomes.validation_value("E1", "ldm4geo-s101", "full") is not None


def test_q1_and_reference_table(tmp_path):
    root, manifests, _ = setup(tmp_path, ["ldm4geo-s101"], prefixes=(None,))
    service = layout(root, PID, "M1", "ldm4geo-s101")["service"]
    trained = eligible(service, manifests["train"])
    best = min(trained, key=lambda sha: trained[sha]["metrics"]["loss_total"])
    reference = tmp_path / "references" / "initial"
    selection.freeze(reference, [trained[best]["source_path"]], manifests["validation"],
                     [manifests["test_id"], manifests["test_ood"]], seconds=10, image=IMAGE,
                     training_run=service, training_manifest=manifests["train"])
    source = trained[best]["source_path"]
    refs = tmp_path / "source-refs.json"
    refs.write_text(json.dumps({"Initial solver": {"source": source, "sha256": best}}))
    refmap = tmp_path / "refmap.json"
    refmap.write_text(json.dumps({"Initial solver": str(reference)}))
    with fake_evaluator():
        with pytest.raises(ValueError, match="REF unseal"):
            export_outcomes.export_references(root, PID, references={"Initial solver": reference})
        selection.run(reference, "validation")
        unseal_references(root, PID, refs, refmap)
        result = json.loads((reference / "test-result.json").read_text())
    q1 = {"short_12x1": str(service), "full_24x1": str(service), "short_24x2": str(service),
          "repeats": [str(service), str(service)]}
    (tmp_path / "q1.json").write_text(json.dumps(q1))
    (tmp_path / "refs.json").write_text(json.dumps({"Initial solver": str(reference)}))
    output = tmp_path / "Q1.json"
    export_outcomes.main(["--root", str(root), "--protocol-id", PID, "--experiment", "Q1", "--output", str(output),
                          "--q1", str(tmp_path / "q1.json"), "--references", str(tmp_path / "refs.json")])
    assert not any(token in output.read_text() for token in private_identities(manifests))
    outcomes = Outcomes.load(output)
    assert outcomes.freezes == {}
    assert outcomes.references["Initial solver"] == {r["split"]: r["metrics"]["mean_iou"] for r in result["results"]}
    assert len(outcomes.q1["programs"]) == len(trained) and len(outcomes.q1["repeats"]) == len(trained)
    row = next(p for p in outcomes.q1["programs"] if p["program_id"] == best)
    assert row["short_12x1"] == row["full_24x1"] == trained[best]["metrics"]["loss_total"]
