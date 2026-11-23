"""Sequence conditioning, durable proposals, and constraints at the worker boundary."""
import json
from types import SimpleNamespace

import pytest

from tasks.antibody_rfd3.core.sequence_design import mutation_options, mutate_sequences
from tasks.antibody_rfd3.core.worker import DesignWorker

SEED = "ASTYDGSPTAN"


def worker(tmp_path):
    args = SimpleNamespace(method="ldm_harness", backbones_per_round=4, sequences_per_round=128,
                           samples_per_call=16, research_candidates_per_role=8, seed=42,
                           temperature=0.5, mock=False)
    return DesignWorker(tmp_path, args)


def test_conditioned_mutations_are_unique_and_preserve_fixed_positions():
    options, groups = mutation_options(SEED, [1, 3], 1, 2, {"1": "AC", "3": "TY"})
    rows = mutate_sequences(SEED, 3, options, groups, 42)
    assert set(rows) == {"CSTYDGSPTAN", "ASYYDGSPTAN", "CSYYDGSPTAN"}
    with pytest.raises(ValueError, match="only 3"):
        mutate_sequences(SEED, 4, options, groups, 42)


@pytest.mark.parametrize("kwargs", [
    {"mutable_positions": [0]}, {"mutable_positions": [True]},
    {"mutable_positions": [2, 2]}, {"min_mutations": 0},
    {"max_mutations": 12}, {"allowed_amino_acids": []},
    {"allowed_amino_acids": {1: "AC"}}, {"allowed_amino_acids": {"1": "XX"}},
])
def test_bad_constraints_rejected(kwargs):
    fields = dict(seed_sequence=SEED, mutable_positions=None, min_mutations=1,
                  max_mutations=2, allowed_amino_acids=None)
    fields.update(kwargs)
    with pytest.raises(ValueError):
        mutation_options(**fields)


def test_replay_preserves_exact_conditioning_and_does_not_spend_twice(tmp_path):
    w = worker(tmp_path)
    params = dict(action="mutate_h3", seed_sequence=SEED, count=8,
                  mutable_positions=[1, 3, 7], min_mutations=1, max_mutations=2,
                  rationale="Probe positions near the measured parent", call_id="one", profile_id="refiner")
    rows = w.execute(**params)
    assert all(r["parent_sequence"] == SEED and r["backbone_id"] is None for r in rows)
    assert w.role_remaining("refiner")["sequences"] == 56
    assert w.execute(**params) == rows
    with pytest.raises(RuntimeError, match="different parameters"):
        w.execute(**{**params, "mutable_positions": [1, 4, 7]})
    assert w.role_remaining("refiner")["sequences"] == 56
    receipt = json.loads((w.jobs / "job-00000/receipt.json").read_text())
    assert receipt["sequence_design"]["seed_sequence"] == SEED
    assert receipt["status"] == "succeeded"


def test_direct_sequences_use_receipt_and_normal_admission(tmp_path):
    w = worker(tmp_path)
    rows = w.execute("register_h3", sequences=[SEED, "CCCCCCCCCCC"],
                     rationale="Explicit hypotheses", profile_id="explorer", call_id="direct")
    assert w.eligible(rows) == rows[:1]
    assert rows[0]["artifact"] == "design_jobs/job-00000/receipt.json"
    assert rows[0]["generation_action"] == "register_h3"
    assert w.remaining()["backbones"] == 4
    assert w.role_remaining("explorer")["sequences"] == 62
    with pytest.raises(ValueError, match="count must match"):
        w.execute("register_h3", count=1, sequences=[SEED, "ASYYDGSPTAN"],
                  rationale="Mismatch", profile_id="explorer")


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_count_does_not_create_receipts(tmp_path, count):
    w = worker(tmp_path)
    with pytest.raises(ValueError):
        w.execute("mutate_h3", count=count, seed_sequence=SEED, rationale="probe", profile_id="explorer")
    assert list(w.records()) == []


def test_synchronous_direct_submission_rejects_inconsistent_parameters(tmp_path):
    from tasks.antibody_rfd3.core.synchronous.worker import DesignWorker as SynchronousWorker
    from tasks.antibody_rfd3.core.synchronous.workflow import parse_args

    args = parse_args(["--method", "harness", "--seed-bank", str(tmp_path / "seed.json")])
    w = SynchronousWorker(tmp_path / "campaign", args)
    with pytest.raises(ValueError, match="count must match"):
        w.execute("register_h3", count=2, sequences=[SEED], rationale="probe")
    with pytest.raises(ValueError, match="cannot claim a generated backbone"):
        w.execute("register_h3", sequences=[SEED], backbone_id="other", rationale="probe")
    with pytest.raises(ValueError, match="Sequence design parameters"):
        w.execute("design_h3", count=1, seed_sequence=SEED)
    assert list(w.records()) == []
