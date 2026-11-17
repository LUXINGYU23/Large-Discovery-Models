import json
from types import SimpleNamespace

import pytest

from ldm_tts.harness import canonical_sha256
from tasks.antibody_rfd3.core.harness import PiAgent, profile
from tasks.antibody_rfd3.core.worker import DesignWorker
from tasks.antibody_rfd3.core.workflow import parse_args
from tasks.antibody_rfd3.ldm_task.procedure import main


def test_two_round_offline_campaign(tmp_path):
    out = tmp_path / "fixture"
    assert main(["--method", "fixed", "--mock", "--out-dir", str(out)]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["successful_evaluation_count"] == 2
    assert set(summary["incumbent"]["evaluation"]["metrics"]) == {"mock_energy"}
    assert all((out / name).exists() for name in ("checkpoint.json", "events.jsonl", "budget.json"))


def test_backbone_reuse_calls_sequence_only(tmp_path, monkeypatch):
    args = parse_args(["--root", str(tmp_path)])
    worker = DesignWorker(tmp_path / "run", args)
    worker.round = 1
    first = worker.jobs / "job-00000"
    first.mkdir()
    (first / "receipt.json").write_text(json.dumps({"round": 0, "candidates": [
        {"design_id": "job-00000/0", "backbone_id": first.name}]}))
    (tmp_path / "tmp").mkdir()
    seen = []

    class Process:
        def __init__(self, cmd, **kwargs):
            seen.append(cmd)
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

    pytest.importorskip("fcntl")
    monkeypatch.setattr("tasks.antibody_rfd3.core.worker.subprocess.Popen", Process)
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
    row, = replica.execute("design_h3", 1, args.temperature)
    assert (row["sequence"], row["backbone_id"]) == (seq, source.name)
    copied = replica.jobs / source.name / "output"
    assert json.loads((copied / "generation.json").read_text())["structure"] == str(copied / "backbone.cif")
    assert (copied / "backbone.cif").is_file()

    config["heavy_chain"] = "X"
    (out / "config.resolved.json").write_text(json.dumps(config))
    other = DesignWorker(tmp_path / "mismatched", args)
    with pytest.raises(ValueError, match="heavy_chain"):
        other.execute("design_h3", 1, args.temperature)
    failed, = other.records()
    assert failed["status"] == "failed" and "candidates" not in failed


def test_pi_receives_energy_and_rejects_measured_or_unissued_ids(tmp_path):
    worker = DesignWorker(tmp_path, parse_args(["--method", "fixed", "--mock"]))
    worker.round = 1
    records = worker.execute("design_h3", 16, 0.5)
    eligible = [r for r in records if not r["violations"]]
    measured, a, b = eligible[:3]
    history = [{"sequence": measured["sequence"], "energy": -0.25, "oracle": "mock",
                "round": 0, "backbone_id": measured["backbone_id"]}]

    class Client:
        def run_turn(self, turns, submission_validator):
            turn, = turns
            assert json.loads(turn.message)["measured_history"] == history
            assert turn.history_digest == canonical_sha256(history)
            for ids in ([measured["design_id"], a["design_id"]], ["forged", "another"],
                        [a["design_id"], a["design_id"]]):
                assert submission_validator(SimpleNamespace(submission={"design_ids": ids})).decision == "retry"
            payload = {"design_ids": [a["design_id"], b["design_id"]]}
            assert submission_validator(SimpleNamespace(submission=payload)).decision == "accept"
            return [SimpleNamespace(turn_id=turn.turn_id, input_digest=turn.input_digest,
                                    submission_status="accepted", submission=payload, usage={})]

    agent = PiAgent(Client(), worker, SimpleNamespace(record=lambda *args: None,
                  consume_many=lambda *args, **kwargs: None))
    assert agent.propose(1, history, 2) == [a, b]
    assert len(profile().skill_dir_sha256[0]) == 64
