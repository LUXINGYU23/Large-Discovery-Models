"""Risk-only service: whitelist, sentinel, blind scan, tiers, judges, import.

The service, suite evaluator, receipts and client are real. Only the container
boundary is replaced: solvers run in-process and the scorer phase calls the
real v2 scoring functions. No Docker, sidecar or model is used.
"""

import hashlib
import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from ldm_tts.contracts.evaluation import EvaluationPaused
from ldm_tts.engine.run_store import BudgetExceededError
from tasks.copper_inversion.core.execution import digest
from tasks.copper_inversion.rsi import exposure
from tasks.copper_inversion.rsi.audit import _simple_ratio, reconcile_budget, scan_blind
from tasks.copper_inversion.rsi.client import RiskClient
from tasks.copper_inversion.rsi.controller import Config, Controller
from tasks.copper_inversion.rsi.exposure import FIELDS
from tasks.copper_inversion.rsi.ledger import read
from tasks.copper_inversion.rsi.scripted import ScriptedAgent
from tasks.copper_inversion.rsi.selection import reserve_and_rank
from tasks.copper_inversion.rsi.service import RiskOnlyService
from tasks.copper_inversion.tests.test_v2_science import fixture
from tasks.copper_inversion.v2.data import read_problem, save_problem
from tasks.copper_inversion.v2.metrics import compare
from tasks.copper_inversion.v2.objective import score
from tasks.copper_inversion.v2.suite import SuiteEvaluator, aggregate, file_hash, load_manifest
from tasks.copper_inversion.v2.selection import freeze
from tasks.copper_inversion.v2.training_receipts import eligible

IMAGE = "sha256:" + "e" * 64
VALID = ("from tasks.copper_inversion.v2.baselines import initialize\n"
         "def solve(problem, budget, seed):\n    return initialize(problem)\n")
FIXED = ("import numpy as np\ndef solve(problem, budget, seed):\n"
         "    field = np.full(problem['shape'], -1.)\n    field[:3, :, :-1] = 1.\n    return field\n")
FAILING = "def solve(problem, budget, seed):\n    raise RuntimeError('synthetic failure')\n"
REJECTED = "# interface_reject\ndef solve(problem, budget, seed):\n    return None\n"


def truth_for(index):
    truth = np.zeros(fixture()["shape"], bool)
    truth[:4 + index % 3, :, :-1] = True
    truth[8, index % 8, (index // 8) % 6] = True  # distinct geometry per body
    return truth


class InProcessExecution:
    """Container stand-in: runs the candidate in-process, scores with real v2 code."""

    def __init__(self, root, image, account=None):
        self.root, self.image = Path(root), image
        self.root.mkdir(parents=True, exist_ok=True)

    def run(self, identity, *, phase, inputs, timeout, cpus=1, memory_mb=2048, source=None,
            model=None, science_module="core", truth=None):
        job = self.root / digest(identity)
        (job / "output").mkdir(parents=True, exist_ok=True)
        problem = read_problem(inputs)
        if phase == "solver":
            namespace = {}
            try:
                exec(compile((Path(source) / "solver.py").read_text(), "solver.py", "exec"), namespace)
                field = np.asarray(namespace["solve"](problem, {}, 47), dtype=float)
            except Exception:
                return job, dict(exit_code=1, reason=None, oom=False, wall_seconds=.5)
            np.save(job / "output/model.npy", field)
            return job, dict(exit_code=0, reason=None, oom=False, wall_seconds=1.5)
        field = np.load(Path(model) / "model.npy")
        metrics = score(problem, field)
        metrics.update(compare(problem, field, np.load(Path(truth) / "truth.npy")))
        hidden = score(read_problem(Path(truth) / "confirmation"), field)
        metrics.update({"confirmation_" + k: v for k, v in hidden.items()
                        if k in {"loss_binary", "loss_contact", "loss_total"}})
        (job / "output/result.json").write_text(json.dumps({"status": "succeeded", "metrics": metrics}))
        return job, dict(exit_code=0, reason=None, oom=False)


def write_suite(release, name, *, stage="development", bodies=4, packs=1, split="train",
                confirmation_shift=.01, seeds=(47,), offset=0):
    """<release>/<name>/manifest.json with one or two packs per synthetic body."""
    root = release / name
    rows = []
    for index in range(offset, offset + bodies):
        parent = hashlib.sha256(f"private-body-{index}".encode()).hexdigest()
        truth = root / "private" / f"body{index}"
        truth.mkdir(parents=True)
        np.save(truth / "truth.npy", truth_for(index))
        confirmation = fixture()
        confirmation["observations"][0]["points"][:, 0] += confirmation_shift
        save_problem(confirmation, truth / "confirmation")
        for modality in ("drill", "drill_outcrop")[:packs]:
            problem = fixture()
            problem["case_id"] = f"case{index}-{modality}"
            public = root / "public" / problem["case_id"]
            save_problem(problem, public)
            names = ("problem.json", "problem.npz", "truth.npy", "confirmation/problem.json",
                     "confirmation/problem.npz")
            rows.append({"case_id": problem["case_id"], "parent_id": parent, "modality": modality,
                         "public": str(public.relative_to(root)), "truth": str(truth.relative_to(root)),
                         "sha256": {n: file_hash((truth if n == "truth.npy" or n.startswith("confirmation/")
                                                  else public) / n) for n in names}})
    path = root / "manifest.json"
    path.write_text(json.dumps({"schema": "copper-suite-v2", "stage": stage, "split": split,
                                "seeds": list(seeds), "cases": rows}, sort_keys=True))
    return path


def tier(seconds, train, bodies, slots=321):
    return {"solver_seconds": seconds, "train_manifest": train, "train_bodies": bodies,
            "validation_manifest": "validation/manifest.json",
            "sealed_manifests": ["test_id/manifest.json", "test_ood/manifest.json"], "slots": slots}


def write_record(release, stage, **fields):
    name = "qualification.json" if stage == "formal" else "development-receipt.json"
    (release / name).write_text(json.dumps(fields, sort_keys=True))


def make_service(tmp_path, manifest, name="service", **options):
    configuration = {"manifest": str(manifest), "root": str(tmp_path / name), "image": IMAGE,
                     "seconds": options.pop("seconds", 10), "slots": options.pop("slots", 13),
                     "caller_uid": 0, "socket": str(tmp_path / (name + ".sock")), **options}
    with patch("tasks.copper_inversion.v2.service.Execution", InProcessExecution):
        service = RiskOnlyService(configuration)

    def checks(payload):
        if b"interface_reject" in (service.root / payload["source_path"]).read_bytes():
            service.slots.reserve(payload["sha256"], interface_rejected=True)
            raise ValueError("fixture interface failure")
        return True
    service.domain.checks = checks
    return service


class LoopbackClient(RiskClient):
    """The real client over an in-process JSON round trip instead of a socket."""

    def __init__(self, root, service):
        super().__init__(root, "unused.sock", service.public_contract, service_uid=0)
        self.service = service

    def request(self, payload):
        return json.loads(json.dumps(self.service.dispatch(json.loads(json.dumps(payload)))))


class RecordingAgent(ScriptedAgent):
    """Scripted researcher that preserves each message as a provider request file."""

    def __init__(self, rejected_branch=None, failing_branch=None):
        super().__init__()
        self.rejected_branch, self.failing_branch = rejected_branch, failing_branch

    def propose(self, *, branch, round_index, count, turn_root, message, **rest):
        path = Path(turn_root) / f"turns/round_{round_index:02d}/provider/r-provider-1.request.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(json.dumps({"model": "scripted", "input": [{"role": "user", "content": message}]}).encode())
        rows, usage = super().propose(branch=branch, round_index=round_index, count=count)
        if round_index == 1 and branch in (self.rejected_branch, self.failing_branch):
            body = REJECTED if branch == self.rejected_branch else FAILING
            for index, row in enumerate(rows):
                row["source"] = body.replace("synthetic", f"synthetic_{index}") + f"marker_{branch}_{index} = 1\n"
        return rows, usage


def scripted_select(pool, history, count):
    predictions = [{"program_id": p["program_id"], "branch": p["branch"], "mean": .5, "std": .1,
                    "acquisition": int(p["program_id"][:8], 16) / 0xffffffff} for p in pool]
    return {"predictions": predictions, "selected": reserve_and_rank(predictions, count)}


def run_controller(root, client, agent=None, rounds=3):
    controller = Controller(root, Config("ldm4geo", 7, rounds=rounds, proposals_per_branch=2, batch=4,
                                         reset_period=2, token_cap_input=0),
                            agent or RecordingAgent(), client, scripted_select,
                            "Fixed task brief: minimize the reported risk.", {"synthetic_test": True})
    return controller.run()


def evaluate(service, source):
    return service.dispatch({"operation": "evaluate", "source": source,
                             "contract_digest": service.public_contract})


@pytest.fixture
def train(tmp_path):
    return write_suite(tmp_path / "release", "train")


# AT-03 ---------------------------------------------------------------------

@pytest.mark.parametrize("source,status", [(VALID, "valid"), (REJECTED, "rejected"), (FAILING, "failed")])
def test_at03_real_service_returns_exactly_the_whitelist(tmp_path, train, source, status):
    service = make_service(tmp_path, train)
    reply = evaluate(service, source)
    assert set(reply) == FIELDS and reply["status"] == status
    assert set(reply["risk_by_case"]) == set(service.audit["case_ids"])
    text = json.dumps(reply)
    assert service.audit["sentinel"] not in text and "iou" not in text.lower()
    assert not any(parent in text for parent in service.audit["private_parent_ids"])
    private = json.loads((service.root / "rsi-feedback" / (reply["source_sha256"] + ".json")).read_text())
    # The sentinel and IoU were in the pre-projection record; the whitelist removed them.
    assert private["private_record"]["sentinel"] == service.audit["sentinel"]
    assert "mean_iou" in private["private_record"] and private["feedback"] == reply


# WP1 client ----------------------------------------------------------------

def test_client_pauses_on_non_allowlisted_keys_and_saves_nothing(tmp_path, train):
    service = make_service(tmp_path, train)
    client = LoopbackClient(tmp_path / "run", service)
    source = tmp_path / "solver.py"
    source.write_text(VALID)
    good = evaluate(service, VALID)
    for leaked in ({**good, "mean_iou": .9}, {**good, "sentinel": "x"}, {"status": "paused", "detail": "x"}):
        client.request = lambda payload, value=leaked: value
        with pytest.raises(EvaluationPaused, match="non-allowlisted"):
            client.evaluate_source(source)
    for refusal in ({"status": "paused"}, {"status": "infrastructure_error"}, {"status": "odd"}):
        client.request = lambda payload, value=refusal: value
        with pytest.raises(EvaluationPaused):
            client.evaluate_source(source)
    assert not (tmp_path / "run" / "feedback").exists()
    client.request = lambda payload: {"status": "budget_exhausted"}
    with pytest.raises(BudgetExceededError):
        client.evaluate_source(source)
    client = LoopbackClient(tmp_path / "run", service)
    assert client.evaluate_source(source) == good
    info = client.verify_contract(service.public_digest, 10, 13, objective="confirmation")
    assert info["objective"] == "confirmation" and info["tier"] is None
    with pytest.raises(EvaluationPaused):
        client.verify_contract(service.public_digest, 10, 13, objective="oracle_iou")


# AT-04 local part and AT-05 ------------------------------------------------

def test_at04_scripted_run_has_no_leak_and_whitelist_off_control_detects_one(tmp_path, train, monkeypatch):
    service = make_service(tmp_path, train)
    run = tmp_path / "run"
    result = run_controller(run, LoopbackClient(run, service))
    assert result["charged_records"] == 13
    report = scan_blind(run, service.root)
    assert report["provider_requests_checked"] == 12 and report["leaks"] == 0
    assert report["marker_hits"] == {"sentinel": 0, "private_parent_id": 0, "truth_hash": 0, "private_value": 0}
    assert report["iou_hits_by_kind"]["unexplained"] == 0 and report["status"] == "passed"
    assert any(p.parts[0] == "feedback" for p in (q.relative_to(run) for q in run.rglob("*.json")))
    assert reconcile_budget(run, service.root)["charged_sources"] == 13

    # Control: the same run with the whitelist disabled leaks the sentinel into
    # the scanned public directory, so the scan above could have found a leak.
    monkeypatch.setattr(exposure, "UNSAFE_DISABLE_WHITELIST", True)
    leaky = make_service(tmp_path, train, name="leaky")
    control = tmp_path / "control"
    with pytest.raises(ValueError, match="ledger record"):
        run_controller(control, LoopbackClient(control, leaky))
    monkeypatch.setattr(exposure, "UNSAFE_DISABLE_WHITELIST", False)
    detected = scan_blind(control, leaky.root, require_requests=False, expect_leak=True)
    assert detected["status"] == "control_detected" and detected["marker_hits"]["sentinel"] > 0
    assert detected["marker_hits"]["private_value"] > 0
    assert detected["iou_hits_by_kind"]["unexplained"] > 0
    with pytest.raises(ValueError, match="contaminated"):
        scan_blind(control, leaky.root, require_requests=False)
    with pytest.raises(ValueError, match="did not expose"):
        scan_blind(run, service.root, expect_leak=True)


def test_iou_hits_are_listed_and_only_task_wording_and_words_are_explained(tmp_path, train):
    service = make_service(tmp_path, train)
    run = tmp_path / "run"
    run_controller(run, LoopbackClient(run, service), rounds=1)
    request = next(run.rglob("*.request.bin"))
    task_line = "Hidden scoring is never IoU-based feedback."
    request.write_bytes(json.dumps({"input": "previous results.\n" + task_line,
                                    "id": "call_00_UFFVU367c0srmiOUaqeE4451"}).encode())
    report = scan_blind(run, service.root, allowed_text=[task_line + "\n"])
    kinds = [(hit["path"], hit["kind"]) for hit in report["iou_hits"]]
    assert (str(request.relative_to(run)), "english_word") in kinds
    assert (str(request.relative_to(run)), "identifier") in kinds
    assert (str(request.relative_to(run)), "task_description") in kinds
    # An unexplained word is listed for sign-off before unsealing, not a scan failure (A3).
    request.write_bytes(b'{"input": "best mean_iou=0.42"}')
    report = scan_blind(run, service.root, allowed_text=[task_line])
    assert report["status"] == "review_required" and report["review_required"] == 1 and report["leaks"] == 0
    assert report["review_sha256"] == scan_blind(run, service.root)["review_sha256"]


def test_hidden_iou_values_are_markers(tmp_path, train):
    service = make_service(tmp_path, train)
    run = tmp_path / "run"
    run_controller(run, LoopbackClient(run, service), rounds=1)
    values = [json.loads(p.read_text())["metrics"]["mean_iou"] for p in service.root.glob("evaluations/*/result.json")]
    value = next(v for v in values if type(v) is float and len(json.dumps(v)) >= 10 and not _simple_ratio(v))
    planted = run / "harness/epoch_00/branch_0/sessions/workspace/notes.txt"
    planted.parent.mkdir(parents=True)
    planted.write_text(f"score {json.dumps(value)} and a rounded {value:.3f}")
    with pytest.raises(ValueError, match="private_value"):
        scan_blind(run, service.root, require_requests=False)
    planted.write_text(f"only a rounded {value:.3f}")
    assert scan_blind(run, service.root, require_requests=False)["marker_hits"]["private_value"] == 0
    # Written before the service scored that value: a coincidence, listed but not a leak.
    planted.write_text(f"own loss {json.dumps(value)}")
    first = min(p.stat().st_mtime for p in service.root.glob("evaluations/*/*.json"))
    os.utime(planted, (first - 60, first - 60))
    report = scan_blind(run, service.root, require_requests=False)
    assert report["marker_hits"]["private_value"] == 0 and report["leaks"] == 0
    assert report["private_value_predated"] == [{"path": str(planted.relative_to(run)), "values": 1}]
    os.utime(planted, None)
    with pytest.raises(ValueError, match="private_value"):
        scan_blind(run, service.root, require_requests=False)


@pytest.mark.parametrize("marker", ["truth", "confirmation", "parent"])
def test_at05_truth_hashes_and_parent_ids_are_searched_in_the_whole_public_dir(tmp_path, train, marker):
    service = make_service(tmp_path, train)
    run = tmp_path / "run"
    run_controller(run, LoopbackClient(run, service), rounds=1)
    scan_blind(run, service.root)
    case = load_manifest(train)["cases"][0]
    value = {"truth": case["sha256"]["truth.npy"], "confirmation": case["sha256"]["confirmation/problem.npz"],
             "parent": case["parent_id"]}[marker]
    # Anywhere in the public side, not only requests, briefs and the ledger.
    planted = run / "harness/epoch_00/branch_0/sessions/workspace/notes.txt"
    planted.parent.mkdir(parents=True)
    planted.write_text("parent_id=" + value)
    with pytest.raises(ValueError, match="contaminated"):
        scan_blind(run, service.root, require_requests=False)


# Layout --------------------------------------------------------------------

def test_one_pack_per_body_follows_the_release_layout(tmp_path):
    release = tmp_path / "release"
    single = write_suite(release, "train", stage="formal", packs=1)
    double = write_suite(release, "validation", stage="formal", packs=2, split="validation")
    with pytest.raises(ValueError, match="exactly two"):
        load_manifest(single)
    assert load_manifest(double)
    assert load_manifest(single, packs_per_body=1)
    write_record(release, "formal", layout={"packs_per_body": 1})
    assert load_manifest(single)
    with pytest.raises(ValueError, match="exactly one"):
        load_manifest(double)
    write_record(release, "formal", layout={"packs_per_body": 3})
    with pytest.raises(ValueError, match="layout"):
        load_manifest(single)
    development = write_suite(tmp_path / "dev", "train", packs=1)
    assert load_manifest(development)


# AT-19 ---------------------------------------------------------------------

def tiered_release(root, stage):
    release = root / "release"
    write_suite(release, "train", stage=stage, bodies=8)
    write_suite(release, "train_s", stage=stage, bodies=4)
    # q25 judge variant: same bodies and public files, thinner confirmation data.
    shutil.copytree(release / "train_s", release / "train_s_q25")
    path = release / "train_s_q25/manifest.json"
    manifest = json.loads(path.read_text())
    for case in manifest["cases"]:
        hidden = fixture()
        hidden["observations"][0]["points"] = hidden["observations"][0]["points"][:1] + .02
        for name in ("labels", "groups"):
            hidden["observations"][0][name] = hidden["observations"][0][name][:1]
        folder = path.parent / case["truth"] / "confirmation"
        save_problem(hidden, folder)
        for name in ("problem.json", "problem.npz"):
            case["sha256"]["confirmation/" + name] = file_hash(folder / name)
    path.write_text(json.dumps(manifest, sort_keys=True))
    write_suite(release, "validation", stage=stage, bodies=2, split="validation", offset=10)
    write_suite(release, "train_s_q1seeds", stage=stage, bodies=4, seeds=(47, 48, 49))
    write_suite(release, "train_q1_both", stage=stage, bodies=4, packs=2)
    write_record(release, stage,
                 layout={"packs_per_body": 1, "packs_per_body_by_manifest": {"train_q1_both/manifest.json": 2}},
                 tiers={"S": tier(30, "train_s/manifest.json", 4), "F": tier(120, "train/manifest.json", 8)},
                 q1={"execution_seeds": [47, 48, 49], "execution_seeds_manifest": "train_s_q1seeds/manifest.json",
                     "both_packs_manifest": "train_q1_both/manifest.json", "solver_seconds": 30})
    return release


@pytest.mark.parametrize("stage", ["development", "formal"])
def test_at19_tier_table_sets_the_budget(tmp_path, stage):
    release = tiered_release(tmp_path, stage)
    with patch("tasks.copper_inversion.v2.qualification.verify_release"):
        short = make_service(tmp_path, release / "train_s/manifest.json", name="s30", seconds=30, slots=321)
        assert short.tier["name"] == "S" and short.identity["tier"]["solver_seconds"] == 30
        assert evaluate(short, VALID)["status"] == "valid"
        assert short.evaluator.seconds == 30
        with pytest.raises(ValueError, match="tier S requires 321 slots and 30 seconds"):
            make_service(tmp_path, release / "train_s/manifest.json", name="s60", seconds=60, slots=321)
        with pytest.raises(ValueError, match="tier S"):
            make_service(tmp_path, release / "train_s/manifest.json", name="s13", seconds=30, slots=13)
        assert make_service(tmp_path, release / "train/manifest.json", name="f", seconds=120, slots=321).tier["name"] == "F"
        with pytest.raises(ValueError, match="tier F"):
            make_service(tmp_path, release / "train/manifest.json", name="f30", seconds=30, slots=321)
        weak = make_service(tmp_path, release / "train_s_q25/manifest.json", name="q25", seconds=30, slots=321,
                            tier="S")
        assert weak.tier["name"] == "S" and weak.identity["suite_digest"] != short.identity["suite_digest"]
        with pytest.raises(ValueError, match="named tier"):
            make_service(tmp_path, release / "train_s/manifest.json", name="wrong", seconds=120, slots=321,
                         tier="F")
    validation = release / "validation/manifest.json"
    execution = InProcessExecution(tmp_path / "jobs", IMAGE)
    for seconds in (30, 120):
        assert SuiteEvaluator(tmp_path / f"v{seconds}", execution, validation, seconds).seconds == seconds
    with pytest.raises(ValueError, match="tier table"):
        SuiteEvaluator(tmp_path / "v60", execution, validation, 60)


def test_at19_releases_without_a_tier_table_keep_the_fixed_budget(tmp_path):
    release = tmp_path / "release"
    formal = write_suite(release, "train", stage="formal", packs=2)
    from tasks.copper_inversion.v2.service import EvaluatorService
    with patch("tasks.copper_inversion.v2.qualification.verify_release"):
        with patch("tasks.copper_inversion.v2.service.Execution", InProcessExecution):
            base = dict(manifest=str(formal), image=IMAGE, caller_uid=0, socket="s")
            assert EvaluatorService(dict(base, root=str(tmp_path / "a"), seconds=120, slots=321)).tier is None
            with pytest.raises(ValueError, match="321 slots and 120 seconds"):
                EvaluatorService(dict(base, root=str(tmp_path / "b"), seconds=30, slots=321))
            with pytest.raises(ValueError, match="no tier table"):
                EvaluatorService(dict(base, root=str(tmp_path / "c"), seconds=120, slots=321, tier="S"))
    development = write_suite(tmp_path / "dev", "train")
    make_service(tmp_path, development, name="d", seconds=120, slots=321)
    with pytest.raises(ValueError, match="fixed service budget"):
        make_service(tmp_path, development, name="e", seconds=121, slots=321)
    with pytest.raises(ValueError, match="fixed at 120"):
        SuiteEvaluator(tmp_path / "x", InProcessExecution(tmp_path / "jobs", IMAGE), formal, 30)


@pytest.mark.parametrize("stage", ["development", "formal"])
def test_at19_frozen_selection_uses_the_tier_budget(tmp_path, stage):
    release = tiered_release(tmp_path, stage)
    write_suite(release, "test_id", stage=stage, bodies=2, split="test_id", offset=20)
    write_suite(release, "test_ood", stage=stage, bodies=2, split="test_ood", offset=30)
    held_out = [release / f"{name}/manifest.json" for name in ("validation", "test_id", "test_ood")]
    source = tmp_path / "solver.py"
    source.write_text(VALID)
    with patch("tasks.copper_inversion.v2.qualification.verify_release"):
        short = make_service(tmp_path, release / "train_s/manifest.json", name="s30", seconds=30, slots=321)
        evaluate(short, VALID)
        provenance = dict(training_run=short.root, training_manifest=release / "train_s/manifest.json")
        frozen = freeze(tmp_path / "freeze30", [source], held_out[0], held_out[1:], seconds=30, image=IMAGE,
                        **provenance)
        assert frozen["seconds"] == 30
        with pytest.raises(ValueError, match="tier table"):
            freeze(tmp_path / "freeze60", [source], held_out[0], held_out[1:], seconds=60, image=IMAGE,
                   **provenance)
        # The training suite of S with F's 120 s is not one tier's protocol either.
        with pytest.raises(ValueError, match="tier table"):
            freeze(tmp_path / "freeze120", [source], held_out[0], held_out[1:], seconds=120, image=IMAGE,
                   **provenance)
        unlisted = write_suite(release, "validation_other", stage=stage, bodies=2, split="validation", offset=40)
        with pytest.raises(ValueError, match="one release tier"):
            freeze(tmp_path / "unlisted", [source], unlisted, held_out[1:], seconds=30, image=IMAGE, **provenance)
    # Old formal releases without a table keep the fixed 120 s.
    from tasks.copper_inversion.v2.selection import selection_seconds
    old = write_suite(tmp_path / "old", "train", stage="formal", packs=2)
    assert selection_seconds(old, load_manifest(old), [], []) == {120}


@pytest.mark.parametrize("stage", ["development", "formal"])
def test_q1_suites_need_the_explicit_measurement_purpose(tmp_path, stage):
    release = tiered_release(tmp_path, stage)
    seeds, both = release / "train_s_q1seeds/manifest.json", release / "train_q1_both/manifest.json"
    # Per-manifest layout overrides the release default.
    assert len(load_manifest(both)["cases"]) == 8 and load_manifest(seeds)["seeds"] == [47, 48, 49]
    record = release / ("qualification.json" if stage == "formal" else "development-receipt.json")
    fields = json.loads(record.read_text())
    write_record(release, stage, **{**fields, "layout": {"packs_per_body": 1}})
    with pytest.raises(ValueError, match="exactly one"):
        load_manifest(both)
    write_record(release, stage, **fields)
    execution = InProcessExecution(tmp_path / "jobs", IMAGE)
    for manifest in (seeds, both):
        evaluator = SuiteEvaluator(tmp_path / manifest.parent.name, execution, manifest, 30, purpose="q1")
        assert evaluator.context["purpose"] == "q1"
        with pytest.raises(ValueError, match="tier table"):
            SuiteEvaluator(tmp_path / "loop", execution, manifest, 30)
        with pytest.raises(ValueError, match="Q1 record"):
            SuiteEvaluator(tmp_path / "slow", execution, manifest, 120, purpose="q1")
    with pytest.raises(ValueError, match="not a Q1"):
        SuiteEvaluator(tmp_path / "s", execution, release / "train_s/manifest.json", 30, purpose="q1")
    write_record(release, stage, **{**fields, "q1": {**fields["q1"], "execution_seeds": [47, 48]}})
    with pytest.raises(ValueError, match="seeds differ"):
        SuiteEvaluator(tmp_path / "seeds", execution, seeds, 30, purpose="q1")
    write_record(release, stage, **fields)
    from tasks.copper_inversion.v2.service import EvaluatorService
    with patch("tasks.copper_inversion.v2.qualification.verify_release"), \
            patch("tasks.copper_inversion.v2.service.Execution", InProcessExecution):
        base = dict(manifest=str(both), image=IMAGE, caller_uid=0, socket="s", seconds=30, slots=10)
        service = EvaluatorService(dict(base, root=str(tmp_path / "q1service"), purpose="q1"))
        assert service.identity["purpose"] == "q1" and service.tier is None
        with pytest.raises(ValueError, match="one release tier"):
            EvaluatorService(dict(base, root=str(tmp_path / "noflag")))
        with pytest.raises(ValueError, match="no tier"):
            EvaluatorService(dict(base, root=str(tmp_path / "tiered"), purpose="q1", tier="S"))
        # Q1 never feeds a discovery loop: the risk-only service refuses it.
        with pytest.raises(ValueError, match="measurement-only"):
            make_service(tmp_path, seeds, name="rsi-q1", seconds=30, slots=10, purpose="q1")


# AT-23 ---------------------------------------------------------------------

def test_at23_oracle_risk_is_one_minus_iou_and_recorded_in_identity(tmp_path, train):
    oracle = make_service(tmp_path, train, name="oracle", objective="oracle_iou")
    standard = make_service(tmp_path, train, name="standard")
    reply = evaluate(oracle, FIXED)
    namespace = {}
    exec(FIXED, namespace)
    expected = {}
    for case in load_manifest(train)["cases"]:
        problem = read_problem(train.parent / case["public"])
        truth = np.load(train.parent / case["truth"] / "truth.npy")
        expected[case["case_id"]] = 1 - compare(problem, namespace["solve"](problem, {}, 47), truth)["iou"]
    assert len(set(expected.values())) > 1
    assert reply["risk_by_case"] == pytest.approx(expected, abs=1e-12)
    assert reply["risk"] == pytest.approx(sum(expected.values()) / len(expected), abs=1e-12)
    assert evaluate(standard, FIXED)["risk"] != pytest.approx(reply["risk"])
    failed = evaluate(oracle, FAILING)
    assert failed["status"] == "failed" and failed["risk"] == 1
    assert oracle.identity["objective"] == "oracle_iou" and standard.identity["objective"] == "confirmation"
    assert json.loads((oracle.root / "identity.json").read_text())["objective"] == "oracle_iou"
    assert oracle.public_contract != standard.public_contract
    assert oracle.dispatch({"operation": "info"})["objective"] == "oracle_iou"
    # Private receipts stay self-describing, so eligibility and sealing recompute 1 - IoU.
    records = eligible(oracle.root, train)
    assert all(row["metrics"]["loss_total"] == 1 - row["metrics"]["mean_iou"] for row in records.values())
    sealed = oracle.dispatch({"operation": "seal_training", "contract_digest": oracle.public_contract})
    assert sealed["status"] == "training_sealed" and sealed["source_sha256s"] == [reply["source_sha256"]]


def test_oracle_aggregate_is_body_weighted_and_failures_count_as_one():
    def row(parent, status, iou):
        return {"parent_id": parent, "status": status, "metrics": {"iou": iou}, "solver_wall_seconds": 1.,
                "objective": "oracle_iou"}
    metrics = aggregate([row("a", "succeeded", .8), row("a", "succeeded", .4), row("b", "failed", .9)])
    assert metrics["loss_total"] == pytest.approx(1 - (.6 + 0.) / 2)
    with pytest.raises(ValueError, match="mixed"):
        aggregate([row("a", "succeeded", .8), {**row("b", "succeeded", .5), "objective": "confirmation"}])


# Continuation import hook (for T6) ----------------------------------------

def test_import_charges_the_prefix_and_keeps_its_receipts_eligible(tmp_path, train):
    source = make_service(tmp_path, train, name="source")
    run = tmp_path / "run"
    run_controller(run, LoopbackClient(run, source), RecordingAgent(rejected_branch=0, failing_branch=1))
    rows = read(run / "ledger.jsonl")
    prefix_rows = [r for r in rows if r["round"] <= 2]
    assert {r["status"] for r in prefix_rows} == {"valid", "rejected", "failed"}
    prefix = tmp_path / "ledger-prefix.jsonl"
    prefix.write_bytes(b"".join(line + b"\n" for line in (run / "ledger.jsonl").read_bytes().splitlines()
                                [:len(prefix_rows)]))
    spec = {"source_root": str(source.root), "ledger_prefix": str(prefix),
            "ledger_prefix_sha256": hashlib.sha256(prefix.read_bytes()).hexdigest()}
    imported = make_service(tmp_path, train, name="continued", **{"import": spec})
    info = imported.dispatch({"operation": "info"})
    assert info["used_slots"] == len(prefix_rows) == 9
    # T6 compares this with sha256 of the checkpoint's ledger-prefix.jsonl bytes.
    assert info["import_ledger_prefix_sha256"] == hashlib.sha256(prefix.read_bytes()).hexdigest()
    assert source.dispatch({"operation": "info"})["import_ledger_prefix_sha256"] is None
    client = LoopbackClient(tmp_path / "continued-run", imported)
    expected = hashlib.sha256(prefix.read_bytes()).hexdigest()
    assert client.verify_contract(imported.public_digest, 10, 13, import_ledger_prefix_sha256=expected)
    for wrong in ("0" * 64, None):
        with pytest.raises(EvaluationPaused):
            client.verify_contract(imported.public_digest, 10, 13, import_ledger_prefix_sha256=wrong)
    plain = LoopbackClient(tmp_path / "source-run", source)
    assert plain.verify_contract(source.public_digest, 10, 13, import_ledger_prefix_sha256=None)
    with pytest.raises(EvaluationPaused):
        plain.verify_contract(source.public_digest, 10, 13, import_ledger_prefix_sha256=expected)
    assert info["contract_digest"] != source.public_contract
    measured = eligible(imported.root, train)
    assert set(measured) == {r["program_id"] for r in prefix_rows if r["status"] != "rejected"}
    # Restart is idempotent and the public prefix reconciles with the private budget.
    again = make_service(tmp_path, train, name="continued", **{"import": spec})
    assert again.public_contract == imported.public_contract
    assert again.dispatch({"operation": "info"})["import_ledger_prefix_sha256"] == expected
    assert again.ledger.counters["algorithm_feedback_slots"] == 9
    continuation = tmp_path / "continuation"
    continuation.mkdir()
    (continuation / "ledger.jsonl").write_bytes(prefix.read_bytes())
    assert reconcile_budget(continuation, imported.root)["charged_sources"] == 9
    sealed = imported.dispatch({"operation": "seal_training", "contract_digest": imported.public_contract})
    best = sorted((r for r in prefix_rows if r["status"] == "valid"),
                  key=lambda r: (r["risk"], -r["legal_rate"], r["seconds_per_solve"], r["program_id"]))[:3]
    assert sealed["source_sha256s"] == [r["program_id"] for r in best]
    assert sealed["used_slots"] == 9


def test_import_refuses_a_changed_prefix_or_contract(tmp_path, train):
    source = make_service(tmp_path, train, name="source")
    run = tmp_path / "run"
    run_controller(run, LoopbackClient(run, source), rounds=1)
    body = (run / "ledger.jsonl").read_bytes()
    rows = [json.loads(line) for line in body.splitlines()]
    rows[1]["risk_by_case"] = {k: 0. for k in rows[1]["risk_by_case"]}
    rows[1]["risk"] = 0.
    tampered = tmp_path / "tampered.jsonl"
    tampered.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    spec = {"source_root": str(source.root), "ledger_prefix": str(tampered),
            "ledger_prefix_sha256": hashlib.sha256(tampered.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match="public prefix differs"):
        make_service(tmp_path, train, name="bad", **{"import": spec})
    honest = tmp_path / "honest.jsonl"
    honest.write_bytes(body)
    spec = {"source_root": str(source.root), "ledger_prefix": str(honest),
            "ledger_prefix_sha256": "0" * 64}
    with pytest.raises(ValueError, match="pinned digest"):
        make_service(tmp_path, train, name="bad2", **{"import": spec})
    spec["ledger_prefix_sha256"] = hashlib.sha256(body).hexdigest()
    with pytest.raises(ValueError, match="different scientific contract"):
        make_service(tmp_path, train, name="bad3", objective="oracle_iou", **{"import": spec})


# Judge-specific validation selection (spec §3.3) ------------------------------

def slab(rows):
    return ("import numpy as np\ndef solve(problem, budget, seed):\n"
            f"    field = np.full(problem['shape'], -1.)\n    field[:{rows}, :, :-1] = 1.\n    return field\n")


def judge_release(tmp_path):
    release = tiered_release(tmp_path, "development")
    write_suite(release, "test_id", bodies=2, split="test_id", offset=20)
    write_suite(release, "test_ood", bodies=2, split="test_ood", offset=30)
    # Thinned confirmation: one hole whose label favours a deeper body than the full set.
    shutil.copytree(release / "validation", release / "validation_q25")
    path = release / "validation_q25/manifest.json"
    manifest = json.loads(path.read_text())
    for case in manifest["cases"]:
        hidden = fixture()
        obs = hidden["observations"][0]
        obs.update(points=np.array([[6.5, 5., 7.5]]), labels=np.array([1.]), groups=np.array([0]))
        folder = path.parent / case["truth"] / "confirmation"
        save_problem(hidden, folder)
        for name in ("problem.json", "problem.npz"):
            case["sha256"]["confirmation/" + name] = file_hash(folder / name)
    path.write_text(json.dumps(manifest, sort_keys=True))
    return release


PROTOCOL = {"tiers": {"S": tier(30, "train_s/manifest.json", 4), "F": tier(120, "train/manifest.json", 8)},
            "judges": {"standard": {"objective": "confirmation"},
                       "q25": {"objective": "confirmation", "train_manifest": "train_s_q25/manifest.json",
                               "validation_manifest": "validation_q25/manifest.json"},
                       "oracle": {"objective": "oracle_iou"}}}


def judged_run(tmp_path, release, judge, service_objective):
    from tasks.copper_inversion.rsi.ledger import append
    run_id = f"ldm4geo-{judge}-s201"
    train = release / PROTOCOL["judges"][judge].get("train_manifest", "train_s/manifest.json")
    service = make_service(tmp_path, train, name=f"active/private/pid/services/E3/{run_id}", seconds=30,
                           slots=321, objective=service_objective)
    run = tmp_path / "active/runs/pid/E3" / run_id
    for index, rows in enumerate((3, 4, 7)):
        feedback = evaluate(service, slab(rows))
        append(run / "ledger.jsonl", {"seq": index, "program_id": feedback["source_sha256"], "epoch": 0, "round": 1,
                                      "branch": index, "selection": "reserved",
                                      **{k: v for k, v in feedback.items() if k != "source_sha256"}})
    service.dispatch({"operation": "seal_training", "contract_digest": service.public_contract})
    (run / "manifest.json").write_text(json.dumps({"run_id": run_id, "experiment": "E3", "protocol_id": "pid",
                                                   "arm": "ldm4geo", "tier": "S", "judge": judge,
                                                   "blind": judge != "oracle"}))
    (run / "result.json").write_text("{}")
    return run_id


def expected_pick(release, validation, objective):
    """Independent recomputation of the judge's validation risk for each slab."""
    manifest = json.loads(validation.read_text())
    risks = {}
    for rows in (3, 4, 7):
        namespace = {}
        exec(slab(rows), namespace)
        values = []
        for case in manifest["cases"]:
            problem = read_problem(validation.parent / case["public"])
            field = namespace["solve"](problem, {}, 47)
            truth = validation.parent / case["truth"]
            if objective == "oracle_iou":
                values.append(1 - compare(problem, field, np.load(truth / "truth.npy"))["iou"])
            else:
                loss = score(read_problem(truth / "confirmation"), field)["loss_total"]
                values.append(loss / (1 + loss))
        risks[rows] = sum(values) / len(values)
    return min(risks, key=risks.get)


def test_validation_selection_uses_the_run_judge(tmp_path):
    from tasks.copper_inversion.experiments.rsi.freeze import freeze_run, validate_run
    release = judge_release(tmp_path)
    picks = {}
    with patch("tasks.copper_inversion.v2.selection.Execution", InProcessExecution), \
            patch("tasks.copper_inversion.v2.selection.InterfaceChecks", lambda root, execution: lambda payload: True):
        for judge, objective, validation in (("standard", "confirmation", "validation"),
                                             ("q25", "confirmation", "validation_q25"),
                                             ("oracle", "oracle_iou", "validation")):
            run_id = judged_run(tmp_path, release, judge, objective)
            record = freeze_run(tmp_path / "active", "pid", "E3", run_id, release=release, protocol=PROTOCOL)
            target = tmp_path / "active/private/pid/selection/E3" / run_id / "full"
            frozen = json.loads((target / "frozen.json").read_text())
            assert record["selection_objective"] == frozen["selection_objective"] == objective
            assert frozen["manifests"][0]["path"] == str((release / validation / "manifest.json").resolve())
            report = validate_run(tmp_path / "active", "pid", "E3", run_id)
            assert report["objective"] == objective
            by_rows = {hashlib.sha256(slab(rows).encode()).hexdigest(): rows for rows in (3, 4, 7)}
            picks[judge] = by_rows[report["selected_sha256"]]
            assert picks[judge] == expected_pick(release, release / validation / "manifest.json", objective)
            chosen = next(r for r in report["candidates"] if r["source_sha256"] == report["selected_sha256"])
            if objective == "oracle_iou":
                assert chosen["metrics"]["loss_total"] == pytest.approx(1 - chosen["metrics"]["mean_iou"])
    # The judge changes the pick: standard confirmation differs from both q25 and oracle here.
    assert picks["standard"] != picks["q25"] and picks["standard"] != picks["oracle"]


def test_freeze_refuses_a_judge_that_differs_from_the_training_service(tmp_path):
    from tasks.copper_inversion.experiments.rsi.freeze import freeze_run
    release = judge_release(tmp_path)
    run_id = judged_run(tmp_path, release, "oracle", "confirmation")
    with pytest.raises(ValueError, match="judge objective differs"):
        freeze_run(tmp_path / "active", "pid", "E3", run_id, release=release, protocol=PROTOCOL)


def test_simple_ratios_are_not_hidden_value_markers(tmp_path):
    """The M1 pilot (rsi-r1-a4dbd1606448faca) failed its scan on a hidden per-case IoU of 7/24 that
    also appeared as an agent's own count of ore samples (35 of 120); such ratios are ordinary numbers."""
    from tasks.copper_inversion.rsi.audit import _private_values
    evaluation = tmp_path / "evaluations" / "e1"
    evaluation.mkdir(parents=True)
    (evaluation / "result.json").write_text(json.dumps({"metrics": {"mean_iou": 0.12345678912}}))
    (evaluation / "private-case-results.json").write_text(json.dumps(
        [{"metrics": {"iou": 7 / 24}}, {"metrics": {"iou": 1234 / 5679}}]))
    assert _simple_ratio(7 / 24) and not _simple_ratio(1234 / 5679)
    assert _private_values(tmp_path) == sorted([json.dumps(0.12345678912).encode(), json.dumps(1234 / 5679).encode()])
