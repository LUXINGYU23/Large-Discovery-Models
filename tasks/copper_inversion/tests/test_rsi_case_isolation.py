"""Tests for C1 training-case memorisation countermeasures across solver inputs, workspace, feedback, arms and splits."""

import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from ldm_tts.contracts.evaluation import EvaluationPaused
from tasks.copper_inversion.core.candidate import validate_source
from tasks.copper_inversion.experiments.rsi import export_outcomes, q1
from tasks.copper_inversion.experiments.rsi.freeze import freeze_run, layout, validate_run
from tasks.copper_inversion.experiments.rsi.unseal import unseal
from tasks.copper_inversion.experiments.v2.publish_training import publish
from tasks.copper_inversion.rsi import continuation as continuations, protocol as protocols
from tasks.copper_inversion.rsi.agent import parse_submission
from tasks.copper_inversion.rsi.audit import reconcile_budget, scan_blind, verify_briefs
from tasks.copper_inversion.rsi.case_isolation import (DEFAULT_CASE_ISOLATION, LEGACY_CASE_ISOLATION,
                                                       case_orientation, check_source_case_ids,
                                                       forbidden_case_tokens, opaque_case_id,
                                                       validate_case_isolation)
from tasks.copper_inversion.rsi.controller import Config, Controller
from tasks.copper_inversion.rsi.ledger import read
from tasks.copper_inversion.rsi.render import render
from tasks.copper_inversion.rsi.scripted import ScriptedAgent
from tasks.copper_inversion.tests.test_rsi_controller import briefs, tree
from tasks.copper_inversion.tests.test_rsi_service import (FAILING, FIXED, IMAGE, InProcessExecution,
                                                           LoopbackClient, REJECTED, RecordingAgent, VALID,
                                                           evaluate, make_service, scripted_select, tier,
                                                           truth_for, write_record)
from tasks.copper_inversion.tests.test_v2_science import fixture
from tasks.copper_inversion.v2.baselines import initialize
from tasks.copper_inversion.v2.contract import validate
from tasks.copper_inversion.v2.data import (ORIENTATION_COUNT, apply_orientation, canonical_problem,
                                            invert_orientation, read_problem, save_problem)
from tasks.copper_inversion.v2.metrics import compare
from tasks.copper_inversion.v2.objective import score
from tasks.copper_inversion.v2.research import PublicTrainingExport
from tasks.copper_inversion.v2.suite import file_hash
from ldm_tts.harness import HarnessSubmittedArtifact


def write_hex_suite(release, name, *, stage="development", bodies=4, split="train", offset=0, seeds=(47,)):
    """Suite with realistic 20-hex case_ids and distinct non-zero physical origins per body."""
    root = release / name
    rows = []
    for index in range(offset, offset + bodies):
        parent = hashlib.sha256(f"private-parent-{split}-{index}".encode()).hexdigest()
        cid = hashlib.sha256(f"public-case-{split}-{index}".encode()).hexdigest()[:20]
        origin = np.array([-140.25 + 37.5 * index, 85.125 - 19.25 * index, -115.5 + 4.75 * index])
        spacing = np.array([1.75 + 0.2 * (index % 3), 2.5 + 0.3 * (index % 2), 1.25])
        base = fixture()
        points = np.asarray(base["observations"][0]["points"], float) * spacing / np.array([1., 2., 3.]) + origin
        truth = root / "private" / f"body{index}"
        truth.mkdir(parents=True)
        np.save(truth / "truth.npy", truth_for(index))
        confirm = copy.deepcopy(base)
        confirm.update(case_id=f"confirm-{index}", origin=origin.tolist(), spacing=spacing.tolist())
        confirm["observations"][0].update(points=points + np.array([0.05, 0.0, 0.0]))
        save_problem(confirm, truth / "confirmation")
        problem = copy.deepcopy(base)
        problem.update(case_id=cid, origin=origin.tolist(), spacing=spacing.tolist())
        problem["observations"][0].update(points=points)
        public = root / "public" / cid
        save_problem(problem, public)
        files = ("problem.json", "problem.npz", "truth.npy", "confirmation/problem.json", "confirmation/problem.npz")
        rows.append({
            "case_id": cid, "parent_id": parent, "modality": "drill",
            "public": str(public.relative_to(root)), "truth": str(truth.relative_to(root)),
            "sha256": {n: file_hash((truth if n == "truth.npy" or n.startswith("confirmation/") else public) / n)
                       for n in files},
        })
    path = root / "manifest.json"
    path.write_text(json.dumps({"schema": "copper-suite-v2", "stage": stage, "split": split,
                                "seeds": list(seeds), "cases": rows}, sort_keys=True))
    return path


def test_canonical_problem_is_translation_invariant_and_bit_exact():
    raw = fixture()
    origin = np.array([-142.375, 88.125, -112.5])
    spacing = np.array([2.25, 3.5, 1.75])
    raw["origin"] = origin.tolist()
    raw["spacing"] = spacing.tolist()
    raw["case_id"] = "de5c023df76d130023ae"
    raw["observations"][0]["points"] = raw["observations"][0]["points"] * spacing / np.array([1., 2., 3.]) + origin
    validate(raw)

    canon = canonical_problem(raw, case_id="case-0123456789abcdef", canonical_origin=True)
    assert canon["case_id"] == "case-0123456789abcdef"
    assert canon["origin"] == [0.0, 0.0, 0.0]
    assert not canon["observations"][0]["points"].flags.writeable

    field_raw = initialize(raw)
    field_canon = initialize(canon)
    np.testing.assert_array_equal(field_raw, field_canon)
    assert score(raw, field_raw) == score(canon, field_canon)
    truth = truth_for(1)
    assert compare(raw, field_raw, truth)["iou"] == compare(canon, field_canon, truth)["iou"]


def test_static_refusal_blocks_release_ids_prefixes_and_workspace_ids(tmp_path):
    train = write_hex_suite(tmp_path / "release", "train", bodies=3)
    catalog = publish(train, tmp_path / "catalog")
    export = PublicTrainingExport(catalog, anonymize_cases=True, canonical_origin=True)
    export(tmp_path / "turn", None)

    exported_cases = json.loads((tmp_path / "turn/public/cases.json").read_text())
    assert [c["case_id"] for c in exported_cases] == ["case-0000", "case-0001", "case-0002"]
    for idx in range(3):
        prob = read_problem(tmp_path / f"turn/public/cases/{idx:04d}")
        assert prob["case_id"] == f"case-{idx:04d}" and prob["origin"] == [0.0, 0.0, 0.0]

    raw_ids = [row["case_id"] for row in json.loads(train.read_text())["cases"]]
    tokens = export.forbidden_tokens
    assert raw_ids[0] in tokens and raw_ids[0][:8] in tokens and "case-0000" in tokens

    for bad_literal in (raw_ids[0], raw_ids[1][:8], "case-0001"):
        bad_src = (f"CASES = {{'{bad_literal}': 0.65}}\n" + VALID).encode()
        with pytest.raises(ValueError, match="training case identifier"):
            check_source_case_ids(bad_src, tokens)
        snap = tmp_path / "snap/candidates/0/solver.py"
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_bytes(bad_src)
        art = HarnessSubmittedArtifact("/candidates/0/artifact_path", "candidates/0/solver.py",
                                       "snap/candidates/0/solver.py", hashlib.sha256(bad_src).hexdigest(), len(bad_src))
        sub = {"candidates": [{"artifact_path": "candidates/0/solver.py", "parent_sha256": None,
                               "change_summary": "c", "rationale": "r", "predicted_risk": .5,
                               "interval80_low": .4, "interval80_high": .6}]}
        with pytest.raises(ValueError, match="training case identifier"):
            parse_submission(sub, (art,), tmp_path, 1, set(), set(), forbidden_case_tokens=tokens)

    # Innocent solver using 'case' as a variable or comment must never be refused.
    innocent = ("# edge case check\n" + VALID).encode()
    check_source_case_ids(innocent, tokens)
    check_source_case_ids(innocent, forbidden_case_tokens(["case"], exported_count=2))


def test_opaque_solver_inputs_aggregate_feedback_and_service_refusal(tmp_path):
    train = write_hex_suite(tmp_path / "release", "train", bodies=3)
    raw_ids = [row["case_id"] for row in json.loads(train.read_text())["cases"]]
    seen_inputs = []

    class CapturingExecution(InProcessExecution):
        def run(self, identity, *, phase, inputs, **kw):
            if phase == "solver":
                prob = read_problem(inputs)
                seen_inputs.append((prob["case_id"], list(prob["origin"])))
            return super().run(identity, phase=phase, inputs=inputs, **kw)

    legacy = make_service(tmp_path, train, name="legacy")
    with patch("tasks.copper_inversion.v2.service.Execution", CapturingExecution):
        isolated = RiskOnlyService_with_checks(tmp_path, train, "isolated", DEFAULT_CASE_ISOLATION)

    legacy_reply = evaluate(legacy, VALID)
    seen_inputs.clear()
    first_reply = evaluate(isolated, VALID)
    first_seen = list(seen_inputs)
    seen_inputs.clear()
    second_reply = evaluate(isolated, VALID + "# second candidate\n")
    second_seen = list(seen_inputs)

    # General translation-invariant solver gets exact same aggregate risk.
    assert first_reply["risk"] == pytest.approx(legacy_reply["risk"], abs=1e-12)
    assert first_reply["risk_by_case"] == {} and second_reply["risk_by_case"] == {}

    # Solver inputs have canonical zero origin and disjoint per-evaluation opaque IDs across candidates.
    first_ids = {cid for cid, _ in first_seen}
    second_ids = {cid for cid, _ in second_seen}
    assert len(first_ids) == 3 and len(second_ids) == 3
    assert first_ids.isdisjoint(second_ids)
    assert first_ids.isdisjoint(raw_ids) and second_ids.isdisjoint(raw_ids)
    assert all(cid.startswith("case-") and len(cid) == 21 for cid in first_ids | second_ids)
    assert all(origin == [0.0, 0.0, 0.0] for _, origin in first_seen + second_seen)

    # Private receipt still keeps per-case risks keyed by release case_ids.
    private = json.loads((isolated.root / "rsi-feedback" / (first_reply["source_sha256"] + ".json")).read_text())
    assert private["feedback"]["risk_by_case"] == {}
    assert set(private["private_record"]["risk_by_case"]) == set(raw_ids)

    # Direct RPC submission containing a training case_id is statically rejected without running solver containers.
    seen_inputs.clear()
    memoriser = f"TABLE = {{'{raw_ids[0]}': 1}}\n" + VALID
    rejected = evaluate(isolated, memoriser)
    assert rejected["status"] == "rejected" and rejected["risk"] == 1.0 and rejected["risk_by_case"] == {}
    assert seen_inputs == []
    assert evaluate(isolated, FAILING)["risk_by_case"] == {}
    assert evaluate(isolated, REJECTED)["risk_by_case"] == {}


def RiskOnlyService_with_checks(tmp_path, manifest, name, isolation, **extra):
    from tasks.copper_inversion.rsi.service import RiskOnlyService
    cfg = {"manifest": str(manifest), "root": str(tmp_path / name), "image": IMAGE,
           "seconds": extra.pop("seconds", 10), "slots": extra.pop("slots", 25),
           "caller_uid": 0, "socket": str(tmp_path / (name + ".sock")),
           "case_isolation": isolation, **extra}
    service = RiskOnlyService(cfg)

    def checks(payload):
        if b"interface_reject" in (service.root / payload["source_path"]).read_bytes():
            service.slots.reserve(payload["sha256"], interface_rejected=True)
            raise ValueError("fixture interface failure")
        return True
    service.domain.checks = checks
    return service


def test_opaque_case_feedback_mode_renders_positional_columns_without_sparse_blowup(tmp_path):
    train = write_hex_suite(tmp_path / "release", "train", bodies=3)
    opaque_cfg = {**DEFAULT_CASE_ISOLATION, "case_feedback": "opaque"}
    service = make_service(tmp_path, train, name="opaque", case_isolation=opaque_cfg, slots=20)
    run = tmp_path / "run"
    ctrl = Controller(run, Config("ldm4geo", 7, rounds=2, proposals_per_branch=1, batch=4, reset_period=1,
                                  token_cap_input=0),
                      RecordingAgent(), LoopbackClient(run, service), scripted_select,
                      "Fixed task brief.", {"synthetic_test": True})
    ctrl.run()
    rows = read(run / "ledger.jsonl")
    valid = [r for r in rows if r["status"] == "valid"]
    key_sets = [frozenset(r["risk_by_case"]) for r in valid]
    assert len(set(key_sets)) == len(valid)  # every evaluated candidate got distinct opaque case keys
    text = render(rows)
    assert "# id            case_01  case_02  case_03" in text
    for cid in service.audit["case_ids"]:
        assert cid[:8] not in text
    assert scan_blind(run, service.root)["marker_hits"]["release_case_id"] == 0


@pytest.mark.parametrize("arm", ["ldm4geo", "accumulated", "restart_best", "independent_sampling"])
def test_all_core_arms_run_identically_with_blind_scan_and_no_per_case_briefs(tmp_path, arm):
    train = write_hex_suite(tmp_path / "release", "train", bodies=3)
    catalog = publish(train, tmp_path / f"cat-{arm}")
    export = PublicTrainingExport(catalog, anonymize_cases=True, canonical_origin=True)
    service = make_service(tmp_path, train, name=f"svc-{arm}", case_isolation=DEFAULT_CASE_ISOLATION, slots=20)
    run = tmp_path / arm

    class ExportingAgent(RecordingAgent):
        def propose(self, *, turn_root, **kw):
            export(turn_root, None)
            return super().propose(turn_root=turn_root, **kw)

    ctrl = Controller(run, Config(arm, 7, rounds=3, proposals_per_branch=2, batch=4, reset_period=2,
                                  token_cap_input=0),
                      ExportingAgent(), LoopbackClient(run, service), scripted_select,
                      "Fixed task brief.", {"synthetic_test": True},
                      forbidden_case_tokens=export.forbidden_tokens)
    result = ctrl.run()
    assert result["charged_records"] == 13
    assert verify_briefs(run)["briefs_checked"] == 12
    for row in read(run / "ledger.jsonl"):
        assert row["risk_by_case"] == {}
    for msg_bytes in briefs(run).values():
        assert b"per-case risk" not in msg_bytes
    report = scan_blind(run, service.root)
    assert report["leaks"] == 0 and report["marker_hits"]["release_case_id"] == 0


def test_ldm4geo_vs_accumulated_brief_identity_replay_and_continuation_under_isolation(tmp_path):
    train = write_hex_suite(tmp_path / "release", "train", bodies=3)
    for arm in ("ldm4geo", "accumulated"):
        svc = make_service(tmp_path, train, name=f"svc-{arm}", case_isolation=DEFAULT_CASE_ISOLATION, slots=25)
        run = tmp_path / arm
        agent = continuations.CheckpointingAgent(RecordingAgent(), run, 2) if arm == "accumulated" else RecordingAgent()
        Controller(run, Config(arm, 7, rounds=4, proposals_per_branch=2, batch=4, reset_period=2, token_cap_input=0),
                   agent, LoopbackClient(run, svc), scripted_select, "Fixed task brief.", {"synthetic_test": True}).run()

    # AT-11 holds under aggregate-only + anonymised isolation.
    assert briefs(tmp_path / "ldm4geo") == briefs(tmp_path / "accumulated")

    # Interrupted evaluation resumes deterministically with identical opaque inputs and ledger.
    svc_int = make_service(tmp_path, train, name="svc-int", case_isolation=DEFAULT_CASE_ISOLATION, slots=25)
    interrupted = tmp_path / "interrupted"
    client = LoopbackClient(interrupted, svc_int)
    calls = 0
    orig_eval = client.evaluate_source

    def flaky(path):
        nonlocal calls
        calls += 1
        if calls == 6:
            raise RuntimeError("mid-batch interruption")
        return orig_eval(path)

    client.evaluate_source = flaky
    with pytest.raises(RuntimeError, match="mid-batch"):
        Controller(interrupted, Config("ldm4geo", 7, rounds=4, proposals_per_branch=2, batch=4, reset_period=2,
                                       token_cap_input=0),
                   RecordingAgent(), client, scripted_select, "Fixed task brief.", {"synthetic_test": True}).run()
    Controller(interrupted, Config("ldm4geo", 7, rounds=4, proposals_per_branch=2, batch=4, reset_period=2,
                                   token_cap_input=0),
               RecordingAgent(), LoopbackClient(interrupted, svc_int), scripted_select,
               "Fixed task brief.", {"synthetic_test": True}).run(resume=True)
    assert (interrupted / "ledger.jsonl").read_bytes() == (tmp_path / "ldm4geo/ledger.jsonl").read_bytes()

    # Continuation service import_prefix succeeds under DEFAULT_CASE_ISOLATION.
    checkpoint = continuations.open_checkpoint(tmp_path / "accumulated/checkpoints/round_02")
    prefix = checkpoint.path / "ledger-prefix.jsonl"
    spec = {"source_root": str(tmp_path / "svc-accumulated"), "ledger_prefix": str(prefix),
            "ledger_prefix_sha256": checkpoint.snapshot["ledger_prefix_sha256"]}
    continued_svc = make_service(tmp_path, train, name="svc-cont", case_isolation=DEFAULT_CASE_ISOLATION,
                                 slots=25, **{"import": spec})
    cont_run = tmp_path / "ck7-history-r1"
    cont_client = LoopbackClient(cont_run, continued_svc)
    assert cont_client.verify_contract(continued_svc.public_digest, 10, 25,
                                       import_ledger_prefix_sha256=checkpoint.snapshot["ledger_prefix_sha256"],
                                       case_isolation=DEFAULT_CASE_ISOLATION)
    with pytest.raises(EvaluationPaused):
        cont_client.verify_contract(continued_svc.public_digest, 10, 25,
                                    import_ledger_prefix_sha256=checkpoint.snapshot["ledger_prefix_sha256"],
                                    case_isolation=LEGACY_CASE_ISOLATION)
    cfg = Config("ldm4geo", 7, rounds=4, proposals_per_branch=2, batch=4, reset_period=2, token_cap_input=0)
    ctrl = Controller(cont_run, cfg, ScriptedAgent(), cont_client, scripted_select, "Fixed task brief.",
                      {"synthetic_test": True, "run_id": "ck7-history-r1"},
                      start=continuations.continuation_start(checkpoint, "history"))
    res = continuations.run_continuation(ctrl, checkpoint, "history")
    assert res["charged_records"] == 17 and reconcile_budget(cont_run, continued_svc.root)["charged_sources"] == 17

    # Planting a release case_id in the public directory is caught by scan_blind.
    leaked = tmp_path / "ldm4geo/harness/epoch_00/branch_0/leaked.txt"
    leaked.write_text(continued_svc.audit["case_ids"][0])
    with pytest.raises(ValueError, match="release_case_id"):
        scan_blind(tmp_path / "ldm4geo", (tmp_path / "svc-ldm4geo"))


def test_validation_sealed_unseal_and_q1_apply_same_anonymisation_with_unchanged_scores(tmp_path):
    release = tmp_path / "release"
    train = write_hex_suite(release, "train_s", bodies=3, split="train", offset=0)
    write_hex_suite(release, "train", bodies=4, split="train", offset=0)
    val = write_hex_suite(release, "validation", bodies=2, split="validation", offset=10)
    tid = write_hex_suite(release, "test_id", bodies=2, split="test_id", offset=20)
    tood = write_hex_suite(release, "test_ood", bodies=2, split="test_ood", offset=30)
    write_hex_suite(release, "train_s_q1seeds", bodies=3, split="train", offset=0, seeds=(47, 48, 49))
    write_record(release, "development", layout={"packs_per_body": 1},
                 tiers={"S": tier(30, "train_s/manifest.json", 3, slots=9),
                        "F": tier(120, "train/manifest.json", 4, slots=9)},
                 q1={"execution_seeds": [47, 48, 49], "execution_seeds_manifest": "train_s_q1seeds/manifest.json",
                     "both_packs_manifest": "train/manifest.json", "solver_seconds": 30})

    proto = {
        "tiers": {"S": tier(30, "train_s/manifest.json", 3, slots=9),
                  "F": tier(120, "train/manifest.json", 4, slots=9)},
        "judges": {"standard": {"objective": "confirmation"}},
        "case_isolation": DEFAULT_CASE_ISOLATION,
    }
    pid, exp, run_id = "rsi-dev-c1test0000000000", "M1", "ldm4geo-s101"
    active = tmp_path / "active"

    solver_seen = []

    class RecordingInProcess(InProcessExecution):
        def run(self, identity, *, phase, inputs, **kw):
            if phase == "solver":
                p = read_problem(inputs)
                solver_seen.append((p["case_id"], list(p["origin"])))
            return super().run(identity, phase=phase, inputs=inputs, **kw)

    outcomes_by_mode = {}
    for label, iso in (("legacy", LEGACY_CASE_ISOLATION), ("isolated", DEFAULT_CASE_ISOLATION)):
        base = active / label
        paths = layout(base, pid, exp, run_id)
        with patch("tasks.copper_inversion.v2.service.Execution", RecordingInProcess):
            svc = RiskOnlyService_with_checks(base, train, f"private/{pid}/services/{exp}/{run_id}", iso,
                                              seconds=30, slots=9)
        Controller(paths["run"], Config("ldm4geo", 101, rounds=2, proposals_per_branch=1, batch=4, reset_period=1,
                                        token_cap_input=0),
                   ScriptedAgent(), LoopbackClient(paths["run"], svc), scripted_select,
                   "Fixed task brief.",
                   {"run_id": run_id, "experiment": exp, "protocol_id": pid, "arm": "ldm4geo",
                    "tier": "S", "judge": "standard", "seed": 101, "blind": True}).run()
        svc.seal_training()
        solver_seen.clear()
        run_proto = {**proto, "case_isolation": iso}
        with patch("tasks.copper_inversion.v2.selection.Execution", RecordingInProcess), \
                patch("tasks.copper_inversion.v2.selection.InterfaceChecks", lambda root, ex: lambda payload: True):
            freeze_run(base, pid, exp, run_id, release=release, protocol=run_proto)
            val_report = validate_run(base, pid, exp, run_id)
            unseal(base, {"experiment": exp, "protocol_id": pid, "solvers": [{"run_id": run_id, "prefix": "full"}]},
                   operator="tester", workers=1)
        if label == "isolated":
            # Validation (3 finalists x 2 cases) + sealed test_id (2) + test_ood (2): all opaque + zero origin!
            assert solver_seen
            held_out_raw = {r["case_id"] for m in (val, tid, tood) for r in json.loads(m.read_text())["cases"]}
            assert all(cid.startswith("case-") and cid not in held_out_raw for cid, _ in solver_seen)
            assert all(origin == [0.0, 0.0, 0.0] for _, origin in solver_seen)
        outcomes_by_mode[label] = (val_report, export_outcomes.export(base, pid, exp))

    # Numerical validation and sealed outcomes are identical (program hashes differ only because
    # ScriptedAgent embeds sha256(message), and isolated briefs omit the per-case risk table).
    iso_row = outcomes_by_mode["isolated"][1]["rows"][0]
    leg_row = outcomes_by_mode["legacy"][1]["rows"][0]
    metric_keys = [k for k in iso_row if k.startswith(("validation_", "test_id_", "test_ood_"))]
    assert metric_keys and {k: iso_row[k] for k in metric_keys} == {k: leg_row[k] for k in metric_keys}
    assert ([b["iou"] for b in outcomes_by_mode["isolated"][1]["body_scores"]]
            == [b["iou"] for b in outcomes_by_mode["legacy"][1]["body_scores"]])

    # Freezing the exact same three candidate solvers under legacy vs isolated selects the exact same sha256.
    from tasks.copper_inversion.tests.test_rsi_service import slab
    from tasks.copper_inversion.v2 import selection
    candidates = []
    for rows_n in (3, 4, 7):
        p = tmp_path / f"cand_{rows_n}.py"
        p.write_text(slab(rows_n))
        candidates.append(p)
    picks = {}
    for label, flag in (("legacy", False), ("isolated", True)):
        with patch("tasks.copper_inversion.v2.service.Execution", RecordingInProcess):
            svc_c = RiskOnlyService_with_checks(tmp_path, train, f"same-svc-{label}",
                                                DEFAULT_CASE_ISOLATION if flag else LEGACY_CASE_ISOLATION,
                                                seconds=30, slots=9)
            for p in candidates:
                evaluate(svc_c, p.read_text())
            svc_c.seal_training()
        with patch("tasks.copper_inversion.v2.selection.Execution", RecordingInProcess), \
                patch("tasks.copper_inversion.v2.selection.InterfaceChecks", lambda root, ex: lambda payload: True):
            sel_dir = tmp_path / f"same-sel-{label}"
            selection.freeze(sel_dir, candidates, val, [tid, tood], seconds=30, image=IMAGE,
                             training_run=svc_c.root, training_manifest=train,
                             anonymize_cases=flag, canonical_origin=flag)
            v_rep = selection.run(sel_dir, "validation")
            t_rep = selection.run(sel_dir, "test")
            picks[label] = (v_rep["selected_sha256"], v_rep["candidates"], t_rep["results"])
    assert picks["isolated"] == picks["legacy"]

    # Q1 domain reference evaluation with anonymize_cases=True, canonical_origin=True.
    ref_src = tmp_path / "ref_solver.py"
    ref_src.write_text(VALID)
    ref_sha = validate_source(ref_src.read_bytes())
    refs_json = tmp_path / "refs.json"
    refs_json.write_text(json.dumps({"A0": {"source": str(ref_src), "sha256": ref_sha}}))
    solver_seen.clear()
    with patch("tasks.copper_inversion.v2.service.Execution", RecordingInProcess), \
            patch("tasks.copper_inversion.v2.service.InterfaceChecks", lambda root, ex, slots=None: lambda p: True), \
            patch("tasks.copper_inversion.v2.selection.Execution", RecordingInProcess), \
            patch("tasks.copper_inversion.v2.selection.InterfaceChecks", lambda root, ex: lambda p: True):
        mapping = q1.build_references(refs_json, release, tmp_path / "q1-refs", IMAGE,
                                      anonymize_cases=True, canonical_origin=True)
    assert "A0" in mapping and solver_seen
    assert all(cid.startswith("case-") and origin == [0.0, 0.0, 0.0] for cid, origin in solver_seen)


def test_protocol_schema_and_example_enable_recommended_case_isolation():
    example = protocols.load(Path(protocols.RESOURCES) / "rsi/protocol.example.json")
    assert protocols.case_isolation(example) == DEFAULT_CASE_ISOLATION

    legacy_proto = {k: v for k, v in example.items() if k != "case_isolation"}
    assert protocols.case_isolation(protocols.validate(legacy_proto)) == LEGACY_CASE_ISOLATION

    for bad in (
        {"anonymize_cases": False, "canonical_origin": True, "case_feedback": "opaque", "refuse_case_ids": True},
        {"anonymize_cases": True, "canonical_origin": True, "case_feedback": "unknown", "refuse_case_ids": True},
        {"anonymize_cases": "yes", "canonical_origin": True, "case_feedback": "none", "refuse_case_ids": True},
        {"anonymize_cases": True, "canonical_origin": True, "case_feedback": "none", "refuse_case_ids": True,
         "randomize_orientation": "yes"},
        {"anonymize_cases": True, "canonical_origin": True, "case_feedback": "none", "refuse_case_ids": True,
         "unknown": False},
        {"anonymize_cases": True},
    ):
        with pytest.raises(ValueError):
            validate_case_isolation(bad)
        with pytest.raises(ValueError, match="invalid protocol"):
            protocols.validate({**example, "case_isolation": bad})

    # Old 4-key case_isolation defaults randomize_orientation to False; new protocols may enable it.
    four_key = {"anonymize_cases": True, "canonical_origin": True, "case_feedback": "none", "refuse_case_ids": True}
    assert validate_case_isolation(four_key) == {**four_key, "randomize_orientation": False}
    oriented = {**four_key, "randomize_orientation": True}
    assert protocols.case_isolation(protocols.validate({**example, "case_isolation": oriented})) == oriented


def test_canonical_problem_horizontal_orientations_and_exact_inverse():
    raw = fixture()
    origin = np.array([-142.375, 88.125, -112.5])
    spacing = np.array([1.5, 2.25, 1.25])
    shape = tuple(raw["shape"])  # (9, 8, 7) — asymmetric nx != ny, dx != dy
    forbidden = raw["forbidden_mask"].copy()
    forbidden[0, :, :] = True   # close x- face
    forbidden[:, -1, :] = True  # close y+ face
    raw["origin"] = origin.tolist()
    raw["spacing"] = spacing.tolist()
    raw["forbidden_mask"] = forbidden
    raw["open_boundary_faces"] = ["x+", "y-", "z-"]
    pts = np.array([[2.5, 2.5, 2.5], [6.5, 2.5, 2.5], [4.5, 2.5, 2.5]]) * spacing + origin
    raw["observations"][0]["points"] = pts
    outcrop_pts = np.array([[3.5, 4.5, 5.5], [5.5, 1.5, 5.5]]) * spacing + origin
    raw["observations"].append({
        "name": "outcrop", "points": outcrop_pts, "labels": np.array([1., 0.]), "groups": np.array([0, 1]),
        "weight": 0.25, "transition_scale": 1., "contact_sigma": 1., "label_error": 0.02,
    })
    validate(raw)
    field_raw = initialize(raw)
    score_raw = score(raw, field_raw)
    truth = np.zeros(shape, bool)
    truth[1:5, 1:5, :4] = True
    compare_raw = compare(raw, field_raw, truth)

    for canonical_origin in (True, False):
        for o in range(ORIENTATION_COUNT):
            canon = canonical_problem(raw, case_id=f"case-{o:016x}", canonical_origin=canonical_origin, orientation=o)
            expected_shape = [shape[1], shape[0], shape[2]] if o % 4 in (1, 3) else list(shape)
            expected_spacing = [spacing[1], spacing[0], spacing[2]] if o % 4 in (1, 3) else spacing.tolist()
            assert canon["shape"] == expected_shape
            assert canon["spacing"] == expected_spacing
            np.testing.assert_array_equal(invert_orientation(apply_orientation(field_raw, o), o), field_raw)

            field_canon = initialize(canon)
            restored = invert_orientation(field_canon, o)
            np.testing.assert_array_equal(restored, field_raw)
            assert score(raw, restored) == score_raw
            assert compare(raw, restored, truth) == compare_raw
            # Pure geometric diagnostics match closely; skfmm 2nd-order upwind has slight axis-reversal asymmetry,
            # which is why SuiteEvaluator applies invert_orientation before scoring on raw_problem.
            sc_canon = score(canon, field_canon)
            assert sc_canon["positive_volume"] == pytest.approx(score_raw["positive_volume"], abs=1e-12)
            assert sc_canon["internal_area"] == pytest.approx(score_raw["internal_area"], abs=1e-12)
            assert sc_canon["component_count"] == score_raw["component_count"]
            assert sc_canon["loss_total"] == pytest.approx(score_raw["loss_total"], abs=1e-3)

    for bad in (-1, 8, True, 1.5, "0"):
        with pytest.raises(ValueError, match="orientation"):
            canonical_problem(raw, orientation=bad)
        with pytest.raises(ValueError, match="orientation"):
            invert_orientation(field_raw, bad)


def test_orientation_replay_invariance_and_fingerprinting_defeat(tmp_path):
    """C2 replay test: equivariant solver scores are unchanged while case-fingerprinting solvers no longer match."""
    release = tmp_path / "release"
    # Give every body a unique spacing so un-rotated fingerprints never collide across the 8 bodies.
    train_root = release / "train"
    train_rows = []
    for index in range(8):
        parent = hashlib.sha256(f"private-parent-c2-{index}".encode()).hexdigest()
        cid = hashlib.sha256(f"public-case-c2-{index}".encode()).hexdigest()[:20]
        origin = np.array([-140.25 + 37.5 * index, 85.125 - 19.25 * index, -115.5 + 4.75 * index])
        spacing = np.array([1.5 + 0.13 * index, 2.1 + 0.17 * index, 1.25])
        base = fixture()
        points = np.asarray(base["observations"][0]["points"], float) * spacing / np.array([1., 2., 3.]) + origin
        truth_dir = train_root / "private" / f"body{index}"
        truth_dir.mkdir(parents=True)
        # Corner-asymmetric truth (strictly in one x<3, y<3 quadrant) so no non-identity D4 symmetry overlaps it.
        asym_truth = np.zeros(base["shape"], bool)
        asym_truth[:3, :3, :3 + (index % 3)] = True
        np.save(truth_dir / "truth.npy", asym_truth)
        confirm = copy.deepcopy(base)
        confirm.update(case_id=f"confirm-{index}", origin=origin.tolist(), spacing=spacing.tolist())
        confirm["observations"][0].update(points=points + np.array([0.05, 0.0, 0.0]))
        save_problem(confirm, truth_dir / "confirmation")
        prob = copy.deepcopy(base)
        prob.update(case_id=cid, origin=origin.tolist(), spacing=spacing.tolist())
        prob["observations"][0].update(points=points)
        pub_dir = train_root / "public" / cid
        save_problem(prob, pub_dir)
        files = ("problem.json", "problem.npz", "truth.npy", "confirmation/problem.json", "confirmation/problem.npz")
        train_rows.append({
            "case_id": cid, "parent_id": parent, "modality": "drill",
            "public": str(pub_dir.relative_to(train_root)), "truth": str(truth_dir.relative_to(train_root)),
            "sha256": {n: file_hash((truth_dir if n == "truth.npy" or n.startswith("confirmation/") else pub_dir) / n)
                       for n in files},
        })
    train = train_root / "manifest.json"
    train.write_text(json.dumps({"schema": "copper-suite-v2", "stage": "development", "split": "train",
                                 "seeds": [47], "cases": train_rows}, sort_keys=True))
    val = write_hex_suite(release, "validation", bodies=4, split="validation", offset=10)
    tid = write_hex_suite(release, "test_id", bodies=4, split="test_id", offset=20)
    tood = write_hex_suite(release, "test_ood", bodies=4, split="test_ood", offset=30)

    train_cases = json.loads(train.read_text())["cases"]
    # Record exact un-rotated fingerprints and truths (representing a solver that memorized public cases).
    fp_table = {}
    invariant_table = {}
    for case in train_cases:
        prob = canonical_problem(read_problem(train.parent / case["public"]), canonical_origin=True)
        truth = np.load(train.parent / case["truth"] / "truth.npy")
        p0 = tuple(np.round(prob["observations"][0]["points"][0], 4).tolist())
        fp_table[(tuple(prob["shape"]), p0)] = truth.tolist()
        # Orientation-invariant key: first observation z-coordinate and spacing_z/min-xy signature.
        inv_key = (round(float(prob["spacing"][0] * prob["spacing"][1]), 6),
                   round(float(prob["observations"][0]["points"][0, 2]), 4))
        invariant_table[inv_key] = truth.tolist()

    seen_solver_inputs = []

    class CapturingExecution(InProcessExecution):
        def run(self, identity, *, phase, inputs, **kw):
            if phase == "solver":
                p = read_problem(inputs)
                seen_solver_inputs.append((identity["case"], tuple(p["shape"]),
                                           tuple(np.round(p["observations"][0]["points"][0], 6).tolist())))
            return super().run(identity, phase=phase, inputs=inputs, **kw)

    iso_off = {**DEFAULT_CASE_ISOLATION, "randomize_orientation": False}
    iso_on = {**DEFAULT_CASE_ISOLATION, "randomize_orientation": True}

    # 1. Equivariant solver gets identical scores (training, validation, sealed) with and without orientation randomization.
    with patch("tasks.copper_inversion.v2.service.Execution", CapturingExecution):
        svc_off = RiskOnlyService_with_checks(tmp_path, train, "svc-off", iso_off, slots=20)
        svc_on = RiskOnlyService_with_checks(tmp_path, train, "svc-on", iso_on, slots=20)
        reply_off = evaluate(svc_off, VALID)
        seen_solver_inputs.clear()
        reply_on = evaluate(svc_on, VALID)
        first_seen = list(seen_solver_inputs)
        # Replay of the exact same source hits the cached receipt deterministically with identical risk.
        replay_on = evaluate(svc_on, VALID)
        assert replay_on == reply_on
        assert reply_on["risk"] == pytest.approx(reply_off["risk"], abs=1e-12)

        # Verify distinct orientations were applied across cases and differ for a second candidate program.
        eval_key_1 = svc_on.evaluator.case_transformation_key(
            type("C", (), {"canonical_key": reply_on["source_sha256"]})())
        seen_solver_inputs.clear()
        reply_on_2 = evaluate(svc_on, VALID + "# second program\n")
        eval_key_2 = svc_on.evaluator.case_transformation_key(
            type("C", (), {"canonical_key": reply_on_2["source_sha256"]})())
        o1 = [case_orientation(eval_key_1, c["case_id"]) for c in train_cases]
        o2 = [case_orientation(eval_key_2, c["case_id"]) for c in train_cases]
        assert any(o != 0 for o in o1) and o1 != o2
        assert first_seen != seen_solver_inputs

    # Same equivariant solver through frozen validation + sealed evaluation keeps exact scores.
    from tasks.copper_inversion.v2 import selection
    solver_file = tmp_path / "equivariant.py"
    solver_file.write_text(VALID)
    svc_off.seal_training()
    svc_on.seal_training()
    val_and_test = {}
    for label, svc, flag in (("off", svc_off, False), ("on", svc_on, True)):
        with patch("tasks.copper_inversion.v2.selection.Execution", CapturingExecution), \
                patch("tasks.copper_inversion.v2.selection.InterfaceChecks", lambda root, ex: lambda p: True):
            sel = tmp_path / f"sel-{label}"
            selection.freeze(sel, [solver_file], val, [tid, tood], seconds=10, image=IMAGE,
                             training_run=svc.root, training_manifest=train,
                             anonymize_cases=True, canonical_origin=True, randomize_orientation=flag)
            v_rep = selection.run(sel, "validation")
            t_rep = selection.run(sel, "test")
            val_and_test[label] = (v_rep["candidates"][0]["metrics"], [r["metrics"] for r in t_rep["results"]])
    assert val_and_test["on"][0]["loss_total"] == pytest.approx(val_and_test["off"][0]["loss_total"], abs=1e-12)
    assert val_and_test["on"][0]["mean_iou"] == pytest.approx(val_and_test["off"][0]["mean_iou"], abs=1e-12)
    for m_on, m_off in zip(val_and_test["on"][1], val_and_test["off"][1]):
        assert m_on["mean_iou"] == pytest.approx(m_off["mean_iou"], abs=1e-12)
        assert m_on["loss_total"] == pytest.approx(m_off["loss_total"], abs=1e-12)

    # 2. Case-fingerprinting solver matches every case under randomize_orientation=False, but fails under True.
    fp_solver = (
        "import numpy as np\n"
        f"TABLE = {fp_table!r}\n"
        "def solve(problem, budget, seed):\n"
        "    key = (tuple(problem['shape']), tuple(np.round(problem['observations'][0]['points'][0], 4).tolist()))\n"
        "    if key in TABLE:\n"
        "        return np.where(np.asarray(TABLE[key], dtype=bool), 1.0, -1.0)\n"
        "    field = np.full(problem['shape'], -1.0)\n"
        "    field[0, 0, 0] = 1.0\n"
        "    return field\n"
    )
    # Also test a memorizer using an orientation-invariant key that returns an un-rotated memorized mask.
    inv_solver = (
        "import numpy as np\n"
        f"TABLE = {invariant_table!r}\n"
        "def solve(problem, budget, seed):\n"
        "    sp = problem['spacing']\n"
        "    key = (round(float(sp[0] * sp[1]), 6), round(float(problem['observations'][0]['points'][0, 2]), 4))\n"
        "    return np.where(np.asarray(TABLE[key], dtype=bool), 1.0, -1.0)\n"
    )
    with patch("tasks.copper_inversion.v2.service.Execution", CapturingExecution):
        svc_fp_off = RiskOnlyService_with_checks(tmp_path, train, "svc-fp-off", iso_off, slots=20)
        svc_fp_on = RiskOnlyService_with_checks(tmp_path, train, "svc-fp-on", iso_on, slots=20)
        r_fp_off = evaluate(svc_fp_off, fp_solver)
        r_fp_on = evaluate(svc_fp_on, fp_solver)
        r_inv_off = evaluate(svc_fp_off, inv_solver)
        r_inv_on = evaluate(svc_fp_on, inv_solver)

    def mean_iou(svc, reply):
        k = svc.evaluator.evaluation_attempt_usage_key(type("C", (), {"canonical_key": reply["source_sha256"]})())
        return json.loads((svc.root / "evaluations" / k / "result.json").read_text())["metrics"]["mean_iou"]

    assert mean_iou(svc_fp_off, r_fp_off) == pytest.approx(1.0, abs=1e-12)
    assert mean_iou(svc_fp_on, r_fp_on) < 0.5
    assert r_fp_on["risk"] > r_fp_off["risk"] + 0.05

    inv_key_on = svc_fp_on.evaluator.case_transformation_key(
        type("C", (), {"canonical_key": r_inv_on["source_sha256"]})())
    inv_orients = [case_orientation(inv_key_on, c["case_id"]) for c in train_cases]
    assert any(o != 0 for o in inv_orients)
    assert mean_iou(svc_fp_off, r_inv_off) == pytest.approx(1.0, abs=1e-12)
    # Every non-identity D4 orientation maps the corner truth to a disjoint quadrant or invalid shape (IoU = 0).
    assert mean_iou(svc_fp_on, r_inv_on) == pytest.approx(sum(o == 0 for o in inv_orients) / len(inv_orients), abs=1e-12)
    assert mean_iou(svc_fp_on, r_inv_on) < 1.0

    # 3. Resuming an existing 4-key service without randomize_orientation keeps its identity.json intact.
    four_key_cfg = {"manifest": str(train), "root": str(tmp_path / "svc-resume"), "image": IMAGE,
                    "seconds": 10, "slots": 10, "caller_uid": 0, "socket": str(tmp_path / "resume.sock"),
                    "case_isolation": {"anonymize_cases": True, "canonical_origin": True,
                                       "case_feedback": "none", "refuse_case_ids": True}}
    from tasks.copper_inversion.rsi.service import RiskOnlyService
    with patch("tasks.copper_inversion.v2.service.Execution", CapturingExecution):
        first_svc = RiskOnlyService(four_key_cfg)
        assert "randomize_orientation" not in first_svc.identity["configuration"]["case_isolation"]
        assert first_svc.case_isolation["randomize_orientation"] is False
        reopened = RiskOnlyService(four_key_cfg)
        assert reopened.public_contract == first_svc.public_contract
