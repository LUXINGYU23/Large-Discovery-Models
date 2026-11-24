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
    assert "body_scores" not in table and "strata_slices" not in table and not table["sealed"]
    assert table["programs"] and table["validations"] and table["diagnostics"]
    assert table["rows"][0]["test_id_iou"] is None
    outcomes = Outcomes([table])
    assert outcomes.bodies == {} and outcomes.strata_slices == {}
    assert ("E1", "ldm4geo-s101") in outcomes.diagnostics
    assert outcomes.validation_value("E1", "ldm4geo-s101", "full") is not None


def test_c4_diagnostics_export_effort_surrogate_quartile_boundary_and_strata(tmp_path):
    """C4 diagnostics block and sealed strata slices (family, modality, noise), aggregates only."""
    import hashlib
    from types import SimpleNamespace
    from tasks.copper_inversion.core.candidate import seal
    from tasks.copper_inversion.core.execution import digest
    from tasks.copper_inversion.experiments.rsi.freeze import freeze_run, validate_run
    from tasks.copper_inversion.rsi.exposure import risk_by_case
    from tasks.copper_inversion.rsi.ledger import append
    from tasks.copper_inversion.rsi.service import RiskOnlyService
    from tasks.copper_inversion.tests.test_rsi_unseal import make_release
    from tasks.copper_inversion.v2.suite import aggregate

    release = tmp_path / "release"
    manifests = make_release(release)
    # Add release evidence/layout.json with family + noisy metadata aligned with manifest cases.
    # test_id: 2 folded (idx 0,1) + 2 lens (idx 2,3), 2 clean (idx 0,2) + 2 noisy (idx 1,3), 2 drill + 2 drill_outcrop.
    # test_ood: 3 faulted_lens (idx 0,1,2) + 1 folded (idx 3) -> single-parent 'folded' must be suppressed to None.
    assignment = []
    for split, families, noises in (
        ("test_id", ["folded", "folded", "lens", "lens"], [False, True, False, True]),
        ("test_ood", ["faulted_lens", "faulted_lens", "faulted_lens", "folded"], [False, False, True, True]),
    ):
        cases = json.loads(manifests[split].read_text())["cases"]
        for idx, (case, fam, noisy) in enumerate(zip(cases, families, noises)):
            assignment.append({"parent": f"{split}-{idx:02d}", "split": split, "index": idx,
                               "family": fam, "noisy": noisy, "selected_modality": case["modality"]})
    (release / "evidence").mkdir()
    (release / "evidence/layout.json").write_text(json.dumps({"assignment": assignment}))

    root = tmp_path / "active"
    run_id = "ldm4geo-s101"
    paths = layout(root, PID, "M1", run_id)
    service, run = paths["service"], paths["run"]
    train = json.loads(manifests["train"].read_text())
    identity = {"configuration": {"manifest": str(manifests["train"]), "seconds": 10, "image": IMAGE},
                "suite_digest": digest(train)}
    service.mkdir(parents=True)
    (service / "identity.json").write_text(json.dumps(identity))
    case_ids = sorted(c["case_id"] for c in train["cases"])

    # reset_period=3: round 1 = round1; rounds 4, 7 = boundary; rounds 2, 3, 5, 6 = other.
    # 16 valid evaluated programs (4 in best quartile: risk in [0.10..0.16] with INCREASING iou -> best_q spearman = +1.0;
    # overall 16 programs have risk descending with iou ascending outside top quartile -> negative overall spearman).
    # Format: (round, branch, risk, train_iou)
    schedule = [
        (0, -1, 0.80, 0.20),
        (1, 0, 0.70, 0.22), (1, 1, 0.72, 0.21), (1, 2, 0.74, 0.19),
        (2, 0, 0.50, 0.35), (2, 1, 0.52, 0.34), (2, 2, 0.54, 0.33),
        (3, 0, 0.42, 0.40), (3, 1, 0.44, 0.39), (3, 2, 0.46, 0.38),
        (4, 0, 0.60, 0.30), (4, 1, 0.62, 0.29), (4, 2, 0.64, 0.28),  # boundary round 4
        (5, 0, 0.24, 0.55), (5, 1, 0.26, 0.53), (5, 2, 0.28, 0.51),
        (6, 0, 0.10, 0.60), (6, 1, 0.12, 0.65), (6, 2, 0.14, 0.70),  # best quartile (top 5: 0.10..0.18)
        (7, 0, 0.16, 0.75), (7, 1, 0.18, 0.80), (7, 2, 0.58, 0.31),  # boundary round 7
    ]
    by_round = {}
    for seq, (r_idx, branch, risk, iou) in enumerate(schedule):
        body = f"def solve(problem,budget,seed):\n    return {seq}\n".encode()
        sha = seal(service, body)["sha256"]
        seal(run, body)
        loss = risk / (1.0 - risk)
        rows = [dict(case_id=c["case_id"], seed=47, parent_id=c["parent_id"], solver_wall_seconds=1.5,
                     status="succeeded", metrics={"iou": iou, "confirmation_loss_total": loss})
                for c in train["cases"]]
        ev = service / "evaluations" / sha
        for row in rows:
            rec = ev / row["case_id"] / "47/receipt.json"
            rec.parent.mkdir(parents=True)
            rec.write_text(json.dumps(row))
        (ev / "private-case-results.json").write_text(json.dumps(rows))
        metrics = aggregate(rows)
        (ev / "result.json").write_text(json.dumps(dict(
            candidate_id=sha, status="succeeded", metrics=metrics, metadata={"suite_digest": digest(train)})))
        fb = dict(risk=metrics["loss_total"], risk_by_case=risk_by_case(rows, case_ids),
                  legal_rate=1.0, failed_solves=0, seconds_per_solve=1.5, message="")
        append(run / "ledger.jsonl", dict(seq=seq, program_id=sha, epoch=(r_idx - 1) // 3 if r_idx else 0,
                                          round=r_idx, branch=branch, status="valid",
                                          selection="initial" if r_idx == 0 else "reserved", **fb))
        by_round.setdefault(r_idx, []).append((sha, risk))

    RiskOnlyService.seal_training(SimpleNamespace(root=service, identity=identity, manifest_path=manifests["train"],
                                                  ledger=SimpleNamespace(counters={"algorithm_feedback_slots": len(schedule)})))
    (run / "manifest.json").write_text(json.dumps(dict(
        run_id=run_id, experiment="M1", protocol_id=PID, arm="ldm4geo", tier="S", judge="standard", seed=101,
        config={"reset_period": 3, "rounds": 7})))
    (run / "result.json").write_text(json.dumps(dict(status="completed", stop_reason="completed")))

    # Write usage/round_01..07.json and harness turn_committed.json with known effort numbers.
    (run / "usage").mkdir()
    for r_idx in range(1, 8):
        (run / "usage" / f"round_{r_idx:02d}.json").write_text(json.dumps({
            "round": r_idx,
            "by_profile": {
                "researcher_0": {"output_tokens": 1000 * r_idx, "requests": 4, "wall_seconds": 10.0 * r_idx, "compactions": 0},
                "researcher_1": {"output_tokens": 2000 * r_idx, "requests": 6, "wall_seconds": 20.0 * r_idx, "compactions": 1},
            },
        }))
        for b, (bash_n, write_n) in ((0, (2, 3)), (1, (4, 5))):
            ep = (r_idx - 1) // 3
            commit = run / f"harness/epoch_{ep:02d}/branch_{b}/turns/round_{r_idx:03d}/turn_committed.json"
            commit.parent.mkdir(parents=True)
            commit.write_text(json.dumps({"usage": {"providerCalls": 4 + 2 * b,
                                                    "toolCalls": {"bash": bash_n, "write": write_n, "submit_solver": 1}}}))

    # Write selection/round_01..07.json: rounds 1-2 anti-correlated (must be ignored), rounds 3-7 exact +1.0 rank skill!
    (run / "selection").mkdir()
    for r_idx in range(1, 8):
        sign = -1.0 if r_idx < 3 else 1.0
        preds = [{"program_id": sha, "branch": b, "mean": sign * risk, "std": 0.05, "acquisition": -risk}
                 for b, (sha, risk) in enumerate(by_round[r_idx])]
        (run / "selection" / f"round_{r_idx:02d}.json").write_text(json.dumps({"predictions": preds, "selected": preds}))

    with fake_evaluator():
        freeze_run(root, PID, "M1", run_id, None, release=release)
        validate_run(root, PID, "M1", run_id, None)
        listing = {"experiment": "M1", "protocol_id": PID, "solvers": [{"run_id": run_id, "prefix": "full"}]}
        unseal(root, listing)

    table = export_outcomes.export(root, PID, "M1")
    text = json.dumps(table)
    assert not any(token in text for token in private_identities(manifests))

    assert len(table["diagnostics"]) == 1
    diag = table["diagnostics"][0]
    assert diag["run_id"] == run_id and diag["reset_period"] == 3
    # Effort per turn: 14 turns, mean output_tokens = mean([1000r, 2000r for r=1..7]) = 1500 * 4 = 6000
    assert diag["turns"] == 14
    assert diag["output_tokens_per_turn"] == pytest.approx(6000.0)
    assert diag["requests_per_turn"] == pytest.approx(5.0)
    assert diag["wall_seconds_per_turn"] == pytest.approx(60.0)
    assert diag["compactions_per_turn"] == pytest.approx(0.5)
    # Tool calls: branch 0 has 2+3+1=6 (bash=2), branch 1 has 4+5+1=10 (bash=4) -> mean 8.0, bash 3.0
    assert diag["tool_calls_turns"] == 14
    assert diag["tool_calls_per_turn"] == pytest.approx(8.0)
    assert diag["bash_calls_per_turn"] == pytest.approx(3.0)
    # Surrogate rank skill: rounds 3..7 all have +1.0 spearman (rounds 1..2 excluded)
    assert diag["surrogate_rounds"] == 5
    assert diag["surrogate_rank_skill"] == pytest.approx(1.0)
    assert diag["surrogate_rank_skill_pooled"] == pytest.approx(1.0)
    # Overall Spearman(risk, train_iou) is strongly negative; best-quartile (risk <= q25, 6 lowest risks: 0.10..0.24)
    # has 5 strictly increasing (0.10..0.18 -> iou 0.60..0.80) and 0.24 -> 0.55; let's verify exact quantile slice!
    assert diag["train_valid_programs"] == len(schedule)
    assert diag["train_risk_iou_spearman"] < -0.7
    # q25 of 22 values (0.10, 0.12, 0.14, 0.16, 0.18, 0.24, ...) is 0.25*21 = index 5.25 -> 0.245, so 6 points:
    # (0.10,0.60),(0.12,0.65),(0.14,0.70),(0.16,0.75),(0.18,0.80),(0.24,0.55) -> distinct from overall!
    assert diag["best_quartile_n"] == 6
    assert diag["best_quartile_spearman"] == pytest.approx(1 / 7, abs=1e-12)
    # Boundary rounds (4 and 7): risks [0.60, 0.62, 0.64, 0.16, 0.18, 0.58]
    assert diag["boundary_round_n"] == 6
    assert diag["boundary_round_risk_mean"] == pytest.approx(sum([0.60, 0.62, 0.64, 0.16, 0.18, 0.58]) / 6)
    assert diag["boundary_round_risk_median"] == pytest.approx(0.59)
    # Other rounds (2, 3, 5, 6): 12 programs
    assert diag["other_round_n"] == 12
    other_expected = [0.50, 0.52, 0.54, 0.42, 0.44, 0.46, 0.24, 0.26, 0.28, 0.10, 0.12, 0.14]
    assert diag["other_round_risk_mean"] == pytest.approx(sum(other_expected) / 12)
    assert diag["other_round_risk_median"] == pytest.approx(0.35)
    assert diag["round1_n"] == 3 and diag["round1_risk_mean"] == pytest.approx(0.72)

    # Sealed strata slices (family, modality, noise) and Outcomes reader compatibility.
    outcomes = Outcomes([table])
    assert outcomes.diagnostics[("M1", run_id)] == diag
    id_slice = outcomes.strata_slices[("M1", run_id, "full", "test_id")]
    ood_slice = outcomes.strata_slices[("M1", run_id, "full", "test_ood")]
    sha = table["rows"][0]["selected_sha256"]
    q = int(sha[:8], 16) / 2 ** 32 / 10
    # FakeSuite gives body i IoU = 0.3 + q + 0.01 * i^2 for i in 0, 1, 2, 3 -> [0.3+q, 0.31+q, 0.34+q, 0.39+q]
    assert id_slice["family"]["folded"] == pytest.approx((0.30 + 0.31) / 2 + q, abs=1e-12)
    assert id_slice["family"]["lens"] == pytest.approx((0.34 + 0.39) / 2 + q, abs=1e-12)
    assert id_slice["modality"]["drill"] == pytest.approx((0.30 + 0.34) / 2 + q, abs=1e-12)
    assert id_slice["modality"]["drill_outcrop"] == pytest.approx((0.31 + 0.39) / 2 + q, abs=1e-12)
    assert id_slice["noise"]["clean"] == pytest.approx((0.30 + 0.34) / 2 + q, abs=1e-12)
    assert id_slice["noise"]["noisy"] == pytest.approx((0.31 + 0.39) / 2 + q, abs=1e-12)
    # test_ood has 3 faulted_lens (indices 0,1,2) and 1 folded (index 3 -> suppressed to None by MIN_PARENTS=2).
    assert ood_slice["family"]["faulted_lens"] == pytest.approx((0.30 + 0.31 + 0.34) / 3 + q, abs=1e-12)
    assert ood_slice["family"]["folded"] is None
    assert ood_slice["noise"]["clean"] == pytest.approx((0.30 + 0.31) / 2 + q, abs=1e-12)
    assert ood_slice["noise"]["noisy"] == pytest.approx((0.34 + 0.39) / 2 + q, abs=1e-12)


def test_c4_strata_csv_source_for_rsi_r2(tmp_path):
    """Verify strata.csv (rsi-r2 layout) populates family/noise slices without layout.json."""
    root, manifests, listing = setup(tmp_path, ["ldm4geo-s101"], prefixes=(None,))
    release = tmp_path / "release"
    rows = ["case_id,parent_id,split,family,modality,noise"]
    for split in ("test_id", "test_ood"):
        for idx, case in enumerate(json.loads(manifests[split].read_text())["cases"]):
            fam = "folded" if idx < 2 else "faulted"
            noise = "clean" if idx % 2 == 0 else "noisy"
            rows.append(f"{case['case_id']},{case['parent_id']},{split},{fam},{case['modality']},{noise}")
    (release / "strata.csv").write_text("\n".join(rows) + "\n")
    with fake_evaluator():
        unseal(root, listing)
    table = export_outcomes.export(root, PID, "M1")
    text = json.dumps(table)
    assert not any(token in text for token in private_identities(manifests))
    outcomes = Outcomes([table])
    sl = outcomes.strata_slices[("M1", "ldm4geo-s101", "full", "test_id")]
    assert set(sl["family"]) == {"faulted", "folded"} and all(v is not None for v in sl["family"].values())
    assert set(sl["noise"]) == {"clean", "noisy"} and all(v is not None for v in sl["noise"].values())


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
