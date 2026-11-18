import json
import pickle
import random
import threading
import time
from types import SimpleNamespace

import pytest

from ldm_tts.harness.history import HistoryLog, HistoryRecord
from ldm_tts.harness.pi_research import proposal_validator
from tasks.antibody_rfd3.core.harness import AntibodyPiResearch, profile
from tasks.antibody_rfd3.core.worker import DesignWorker
from tasks.antibody_rfd3.core.workflow import parse_args
from tasks.antibody_rfd3.ldm_task.procedure import main


class FakeNativeAntBO:
    calls = 0

    def __init__(self, seed):
        import numpy as np
        import torch
        self.index = 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def propose(self, history, pending):
        import numpy as np
        import torch
        type(self).calls += 1
        self.index += 1
        return {"index": self.index, "random": random.random(), "numpy": float(np.random.random()),
                "torch": float(torch.rand(())), "history_size": len(history), "pending": sorted(pending)}


def test_two_round_offline_campaign(tmp_path):
    out = tmp_path / "fixture"
    assert main(["--method", "fixed", "--mock", "--out-dir", str(out)]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["successful_evaluation_count"] == 2
    assert set(summary["incumbent"]["evaluation"]["metrics"]) == {"mock_energy"}
    assert all((out / name).exists() for name in ("controller.sqlite3", "experiment-service.sqlite3",
                                                "events.jsonl", "budget.json", "history/log.jsonl"))
    import sqlite3
    with sqlite3.connect(out / "controller.sqlite3") as database:
        rows = database.execute("SELECT candidate_json FROM experiments ORDER BY decision_id").fetchall()
    assert len(rows) == 2
    assert all(len(json.loads(row[0])["payload"]["sequence"]) == 11 for row in rows)
    before = (out / "history/log.jsonl").read_text()
    assert main(["--method", "fixed", "--mock", "--out-dir", str(out), "--resume-run", str(out)]) == 0
    assert (out / "history/log.jsonl").read_text() == before


def test_backbone_reuse_calls_sequence_only(tmp_path, monkeypatch):
    args = parse_args(["--root", str(tmp_path), "--method", "fixed"])
    worker = DesignWorker(tmp_path / "run", args)
    worker.round = 1
    first = worker.jobs / "job-00000"
    first.mkdir()
    (first / "receipt.json").write_text(json.dumps({"round": 0, "candidates": [
        {"design_id": "job-00000/0", "backbone_id": first.name}]}))
    (tmp_path / "tmp").mkdir()
    seen = []
    fcntl = pytest.importorskip("fcntl")
    real_flock = fcntl.flock

    class Process:
        def __init__(self, cmd, **kwargs):
            seen.append(cmd)
            # The worker must retain its own GPU lock while the child is running,
            # while a different GPU remains free for an independent campaign.
            with (tmp_path / "tmp/antibody-rfd3-worker-gpu0.lock").open("a") as same:
                with pytest.raises(BlockingIOError):
                    real_flock(same, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with (tmp_path / "tmp/antibody-rfd3-worker-gpu1.lock").open("a") as other:
                real_flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                real_flock(other, fcntl.LOCK_UN)
            out = worker.jobs / "job-00001/output"
            out.mkdir()
            mapped = [{"source": {"chain": "B", "number": i+99, "insertion_code": ""},
                       "sequence_chain": "A", "sequence_position": i+1} for i in range(11)]
            candidate = {"full_sequences": {"A": "SNGDWGSPGTV"}, "residue_mapping": mapped,
                         "h3_residues": [m["source"] for m in mapped],
                         "heavy_chain": "B"}
            (out / "designs.json").write_text(json.dumps({"status": "success", "candidates": [candidate]}))
        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr("tasks.antibody_rfd3.core.worker.subprocess.Popen", Process)
    # Keep the old global lock occupied: a new GPU-specific worker must not use it.
    with (tmp_path / "tmp/antibody-rfd3-worker.lock").open("a") as legacy:
        real_flock(legacy, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setattr(fcntl, "flock", lambda fd, operation: real_flock(fd, operation | fcntl.LOCK_NB))
        rows = worker.execute("sequence_h3", 1, 0.5, first.name)
    assert rows[0]["sequence"] == "SNGDWGSPGTV" and rows[0]["backbone_id"] == first.name
    cmd, = seen
    assert cmd[cmd.index("-m") + 2] == "sequence"
    assert cmd[cmd.index("--generation") + 1] == str(first / "output/generation.json")
    assert cmd[0] == "taskset"


def test_budget_and_unknown_backbone(tmp_path):
    args = parse_args(["--mock", "--method", "fixed"])
    worker = DesignWorker(tmp_path, args)
    with pytest.raises(ValueError, match="Shared initialization"):
        worker.execute("design_h3", 1, 0.3)
    worker.round = 1
    with pytest.raises(ValueError, match="Unknown backbone"):
        worker.execute("sequence_h3", 1, 0.5, "../another-run")
    with pytest.raises(ValueError, match="budget"):
        worker.execute("design_h3", args.sequences_per_round + 1, 0.5)
    assert not list(worker.records())


def test_successful_generation_call_replays_and_failed_call_blocks(tmp_path):
    worker = DesignWorker(tmp_path, parse_args(["--method", "fixed", "--mock"]))
    first = worker.execute("design_h3", 16, 0.5, call_id="turn-1-call-1")
    assert worker.execute("design_h3", 16, 0.5, call_id="turn-1-call-1") == first
    assert len(list(worker.records())) == 1
    with pytest.raises(RuntimeError, match="different parameters"):
        worker.execute("design_h3", 15, 0.5, call_id="turn-1-call-1")
    path = worker.jobs / "job-00000/receipt.json"
    receipt = json.loads(path.read_text())
    receipt["status"] = "started"
    path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="not safely replayable"):
        worker.execute("design_h3", 16, 0.5, call_id="turn-1-call-1")
    with pytest.raises(RuntimeError, match="unresolved or failed"):
        worker.execute("design_h3", 16, 0.5, call_id="turn-1-call-2")


def test_antbo_checkpoint_restores_optimizer_and_reuses_same_generation(tmp_path, monkeypatch):
    from tasks.antibody_rfd3.core import baselines
    from tasks.antibody_rfd3.core.research import NativeResearch

    monkeypatch.setattr(baselines, "NativeAntBO", FakeNativeAntBO)
    monkeypatch.setattr(baselines, "prepare_antbo_source", lambda: None)
    FakeNativeAntBO.calls = 0
    args = parse_args(["--method", "antbo", "--mock"])
    worker = DesignWorker(tmp_path, args)
    history = [{"sequence": "SNGDWGSPGTV", "energy": -0.2}]
    first = NativeResearch(args, worker).antbo_proposal(2, history, set())
    assert first["index"] == 1 and FakeNativeAntBO.calls == 1
    assert NativeResearch(args, worker).antbo_proposal(2, history, set()) == first
    assert FakeNativeAntBO.calls == 1
    second = NativeResearch(args, worker).antbo_proposal(3, history, {"INFLIGHT"})
    assert second["index"] == 2 and second["pending"] == ["INFLIGHT"]
    assert FakeNativeAntBO.calls == 2
    assert NativeResearch(args, worker).antbo_proposal(3, history, {"INFLIGHT"}) == second
    assert FakeNativeAntBO.calls == 2
    with pytest.raises(RuntimeError, match="differs from its frozen inputs"):
        NativeResearch(args, worker).antbo_proposal(3, history, set())
    checkpoint = tmp_path / "antbo_checkpoints/g000003.pkl"
    with checkpoint.open("rb") as stream:
        state = pickle.load(stream)
    NativeResearch._write_checkpoint(checkpoint, {"status": "started", "input_sha256": state["input_sha256"]})
    with pytest.raises(RuntimeError, match="unresolved checkpoint"):
        NativeResearch(args, worker).antbo_proposal(3, history, {"INFLIGHT"})


def test_second_controller_cannot_stop_the_active_experiment_service(tmp_path):
    from ldm_tts.async_runtime import ESClient, RunLock, RunLockHeld
    from tasks.antibody_rfd3.core.campaign import experiment_service
    from tasks.antibody_rfd3.core.experiments import KIND

    directory = tmp_path / "campaign"
    directory.mkdir()
    args = parse_args(["--method", "fixed", "--mock"])
    with experiment_service(directory, args) as (process, url):
        with RunLock(directory / "campaign.lock"):
            with pytest.raises(RunLockHeld):
                main(["--method", "fixed", "--mock", "--resume-run", str(directory)])
            assert process.poll() is None
            assert ESClient(url).kind(KIND)


def test_resume_reuses_a_live_experiment_service(tmp_path):
    from tasks.antibody_rfd3.core.campaign import experiment_service

    directory = tmp_path / "campaign"
    directory.mkdir()
    args = parse_args(["--method", "fixed", "--mock"])
    with experiment_service(directory, args) as (first, first_url):
        with experiment_service(directory, args) as (resumed, resumed_url):
            assert (resumed.pid, resumed_url) == (first.pid, first_url)
            assert resumed.poll() is None


def test_failed_evaluation_consumes_budget(tmp_path):
    from experiment_service import BlockingExecutor
    from experiment_service.testing import running_service
    from ldm_tts.async_runtime import CampaignConfig, ESClient, PacedController, PacingConfig, campaign_extra
    from ldm_tts.async_runtime.history_bridge import HistoryBridge
    from ldm_tts.contracts import ObjectiveSet, ObjectiveSpec
    from tasks.antibody_rfd3.core.experiments import AntibodyCodec, KIND, history_projection
    from tasks.antibody_rfd3.core.research import InitializationPacing, NativeResearch

    def fail(_spec):
        raise RuntimeError("injected oracle failure")

    args = parse_args(["--method", "fixed", "--mock"])
    worker = DesignWorker(tmp_path / "run", args)
    pacing = PacingConfig.balanced(concurrency=1, reservoir_size=2, research_budget=1)
    config = CampaignConfig("failed", budget=1, concurrency=1, seed=42, extra=campaign_extra(pacing))
    with running_service([BlockingExecutor(KIND, "1", fail, workers=1)], autorun=True) as service:
        bridge = HistoryBridge(tmp_path / "history", history_projection("mock_energy"))
        with PacedController(tmp_path / "controller.sqlite3", config, codec=AntibodyCodec("1ADQ_A", "mock"),
            client=ESClient(service.base_url), research=NativeResearch(args, worker), history=bridge,
            pacing=InitializationPacing(), objectives=ObjectiveSet.from_specs([ObjectiveSpec("mock_energy", "minimize")])) as ctrl:
            deadline = time.monotonic() + 15
            while not ctrl.finished and time.monotonic() < deadline:
                ctrl.step()
                time.sleep(0.01)
            assert ctrl.finished
            observation, = ctrl.observations()
            assert observation.evaluation.status == "failed"
            assert ctrl.budget().consumed == 1
            assert ctrl.budget().free == 0


def test_initial_measurement_precedes_next_research_while_evaluation_is_delayed(tmp_path):
    from experiment_service import BlockingExecutor
    from experiment_service.testing import running_service
    from ldm_tts.async_runtime import (CampaignConfig, ESClient, PacedController, PacingConfig, RuntimeOptions,
                                       campaign_extra)
    from ldm_tts.async_runtime.history_bridge import HistoryBridge
    from ldm_tts.contracts import ObjectiveSet, ObjectiveSpec
    from tasks.antibody_rfd3.core.experiments import AntibodyCodec, KIND, history_projection, mock_energy
    from tasks.antibody_rfd3.core.research import InitializationPacing, NativeResearch

    started, release = threading.Event(), threading.Event()

    def evaluate(spec):
        started.set()
        if not release.wait(15):
            raise RuntimeError("test did not release the delayed oracle")
        return {"mock_energy": mock_energy(spec["antigen"], spec["sequence"])}

    args = parse_args(["--method", "fixed", "--mock", "--concurrency", "2"])
    worker = DesignWorker(tmp_path / "run", args)
    pacing = PacingConfig.balanced(concurrency=2, reservoir_size=2, dispatch_during_research="continue",
                                   research_budget=2)
    config = CampaignConfig("delayed", budget=2, concurrency=2, seed=42, extra=campaign_extra(pacing))
    executor = BlockingExecutor(KIND, "1", evaluate, workers=2, retry_safe=True)
    with running_service([executor], pools={"default": 2}, autorun=True) as service:
        bridge = HistoryBridge(tmp_path / "history", history_projection("mock_energy"))
        with PacedController(tmp_path / "controller.sqlite3", config, codec=AntibodyCodec("1ADQ_A", "mock"),
            client=ESClient(service.base_url), research=NativeResearch(args, worker), history=bridge,
            pacing=InitializationPacing(), objectives=ObjectiveSet.from_specs([ObjectiveSpec("mock_energy", "minimize")]),
            options=RuntimeOptions(poll_interval=0.01, long_poll_seconds=0)) as ctrl:
            try:
                deadline = time.monotonic() + 15
                while not started.is_set() and time.monotonic() < deadline:
                    ctrl.step()
                    time.sleep(0.01)
                assert started.is_set()
                for _ in range(20):
                    ctrl.step()
                    time.sleep(0.01)
                assert len(ctrl.generations()) == 1
                assert ctrl.view().head_ingest_seq == 0
                release.set()
                deadline = time.monotonic() + 15
                while not ctrl.finished and time.monotonic() < deadline:
                    ctrl.step()
                    time.sleep(0.01)
                assert ctrl.finished
                assert len(ctrl.observations()) == 2
                assert ctrl.generations()[1]["base_ingest_seq"] >= 1
            finally:
                release.set()


def test_shared_initialization_relocates_backbone_and_checks_h3(tmp_path):
    anchor = tmp_path / "anchor"
    source = anchor / "design_jobs/job-00000"
    out = source / "output"
    out.mkdir(parents=True)
    args = parse_args(["--initialization-run", str(anchor), "--samples-per-call", "1"])
    replica = DesignWorker(tmp_path / "replica", args)
    config = replica.template.copy()
    refs = config["h3_residues"]
    seq = "SNGKDGSPGVY"
    mapped = [{"source": ref, "sequence_chain": "A", "sequence_position": i}
              for i, ref in enumerate(refs, 1)]
    design = {"residue_mapping": mapped, "heavy_chain": "B", "h3_residues": refs,
              "full_sequences": {"A": seq}}
    receipt = {"status": "succeeded", "round": 0, "action": "design_h3", "count": 1,
               "temperature": args.temperature, "seed": args.seed, "mock_generator": False,
               "candidates": [{"sequence": seq}]}
    (source / "receipt.json").write_text(json.dumps(receipt))
    (out / "config.resolved.json").write_text(json.dumps(config))
    (out / "backbone.cif").write_text("data_fixture")
    (out / "generation.json").write_text(json.dumps({"status": "success", "structure": str(out / "backbone.cif")}))
    (out / "designs.json").write_text(json.dumps({"status": "success", "candidates": [design]}))
    row, = replica.execute("design_h3", 1, args.temperature, profile_id="explorer")
    assert (row["sequence"], row["backbone_id"]) == (seq, source.name)
    assert next(replica.records())["profile_id"] == "explorer"
    copied = replica.jobs / source.name / "output"
    assert json.loads((copied / "generation.json").read_text())["structure"] == str(copied / "backbone.cif")
    assert (copied / "backbone.cif").is_file()

    config["heavy_chain"] = "X"
    (out / "config.resolved.json").write_text(json.dumps(config))
    other = DesignWorker(tmp_path / "mismatched", args)
    with pytest.raises(ValueError, match="heavy_chain"):
        other.execute("design_h3", 1, args.temperature, profile_id="explorer")
    failed, = other.records()
    assert failed["status"] == "failed" and "candidates" not in failed


def test_pi_rejects_pending_measured_and_unissued_ids(tmp_path):
    worker = DesignWorker(tmp_path, parse_args(["--method", "fixed", "--mock"]))
    worker.round = 1
    records = worker.execute("design_h3", 16, 0.5)
    eligible = [r for r in records if not r["violations"]]
    measured, a, b = eligible[:3]
    agent = object.__new__(AntibodyPiResearch)
    agent.worker = worker
    agent.args = SimpleNamespace(method="ldm_harness")
    agent.required = 1
    log = HistoryLog()
    log.append([HistoryRecord("selected", "measured", measured["sequence"], "a", "selected", source="event:1"),
                HistoryRecord("result", "measured", measured["sequence"], "a", "succeeded", source="event:2"),
                HistoryRecord("selected", "pending", a["sequence"], "b", "selected", source="event:3")])
    validator = proposal_validator(agent.make_contract(), log, generation=2)
    for candidate in (measured["design_id"], a["design_id"], "forged"):
        assert validator(SimpleNamespace(submission={"design_ids": [candidate]}, history_seq=log.head)).decision == "retry"
    assert validator(SimpleNamespace(submission={"design_ids": [b["design_id"]]}, history_seq=log.head)).decision == "accept"
    assert len(profile().skill_dir_sha256[0]) == 64
