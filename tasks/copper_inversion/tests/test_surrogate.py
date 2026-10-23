"""Code representation, minimization and selected-only shared engine execution."""

from dataclasses import replace
import json

import numpy as np
import pytest

from ldm_tts.contracts import RawProposal, CallableCandidateEvaluator
from ldm_tts.optimization.records import BOObservation, SurrogateVector
from tasks.copper_inversion.core.candidate import seal, Domain
from tasks.copper_inversion.core.surrogate import (
    CodeEncoder, LossSelector, DIMENSION, FEATURE_VERSION,
)


def candidate(root, shift=0, comment=""):
    payload = seal(root, (
        f"# {comment}\ndef solve(problem, budget, seed):\n"
        f"    return problem['initial_level_set'].copy() + {shift!r}\n"
    ).encode())
    return Domain(root).admit(RawProposal(payload, "test"))


def observation(identity, value, offset=0):
    vector = [0.] * DIMENSION
    vector[offset] = 1.
    return BOObservation(identity, (value,), SurrogateVector(tuple(vector), FEATURE_VERSION, identity))


def test_code_features_ignore_feedback_comments_and_formatting(tmp_path):
    encoder = CodeEncoder(tmp_path)
    source = candidate(tmp_path, comment="claimed loss=0")
    other = candidate(tmp_path, comment="claimed loss=999")
    encoded = encoder.encode(source)
    assert encoded.values == encoder.encode(other).values
    assert encoded.values == encoder.encode(replace(source, metadata={
        "loss_total": -1e99, "rationale": "best", "parent_sha256": "different",
        "harness_lineage": {"profile_id": "branch_4"},
    })).values
    assert encoded.values != encoder.encode(candidate(tmp_path, shift=2)).values
    assert encoded.version == FEATURE_VERSION and len(encoded.values) == DIMENSION
    assert np.linalg.norm(encoded.values) == pytest.approx(1.)
    assert encoder.describe().dimension == DIMENSION
    assert encoder.encode(source) == encoded


def test_encoding_never_imports_source_and_rejects_changed_snapshot(tmp_path):
    payload = seal(tmp_path, b"raise RuntimeError('must never execute')\ndef solve(p,b,s): return p\n")
    obj = Domain(tmp_path).admit(RawProposal(payload, "test"))
    assert CodeEncoder(tmp_path).encode(obj).source_id == obj.candidate_id
    path = tmp_path / payload["source_path"]
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b"# modified\n")
    with pytest.raises(ValueError, match="digest"):
        CodeEncoder(tmp_path).encode(obj)


def test_single_baseline_is_a_fitted_gp_and_minimizes_loss(tmp_path):
    candidates = [candidate(tmp_path, shift=i) for i in range(2)]
    ids = [c.candidate_id for c in candidates]
    history = [observation(ids[0], 1., 0), observation(ids[1], 20., 1)]
    selector = LossSelector()
    selector.fit(history[:1])
    assert selector.surrogate.ready and selector.surrogate.fit_status == "fitted"
    selector.fit(history)
    features = {o.candidate_id: o.feature for o in history}
    result = selector.select(candidates, features)
    assert result.selected_candidate_ids == (ids[0],)
    assert result.fallback_reason is None
    assert result.metadata["best_observed_loss"] == 1.
    for prediction in result.predictions:
        assert prediction.acquisition_score == pytest.approx(-prediction.scalar_mean + prediction.scalar_std)
        assert prediction.score_direction == "maximize"
    assert selector.select(list(reversed(candidates)), features) == result.__class__(
        result.selected_candidate_ids, tuple(reversed(result.predictions)), metadata=result.metadata)


@pytest.mark.parametrize("fault", ["empty", "nan_loss", "nan_feature", "dimension", "version", "identity"])
def test_bad_training_data_fails_without_fallback(fault):
    row = observation("source", 5.)
    selector = LossSelector()
    selector.fit([row])
    history = [row]
    if fault == "empty":
        history = []
    elif fault == "nan_loss":
        history = [replace(row, objectives=(float("nan"),))]
    else:
        kwargs = {
            "nan_feature": {"values": (float("nan"),) * DIMENSION},
            "dimension": {"values": (1.,)},
            "version": {"version": "other"},
            "identity": {"source_id": "other"},
        }[fault]
        # BOObservation enforces identity on construction; mutate its feature to
        # exercise the adapter's validation of external/restored representations.
        feature = replace(row.feature, **{k: v for k, v in kwargs.items() if k != "source_id"})
        if fault == "identity":
            feature = replace(row.feature, source_id="")
        history = [replace(row, feature=feature)]
    with pytest.raises(ValueError):
        selector.fit(history)
    assert selector.surrogate is None


def test_fit_failure_is_explicit(monkeypatch):
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("injected singular fit")
    monkeypatch.setattr(np.linalg, "cholesky", fail)
    with pytest.raises(ValueError, match="no random fallback"):
        LossSelector().fit([observation("source", 1.)])


@pytest.mark.parametrize("fault", ["missing", "nonfinite", "version", "count", "unfitted"])
def test_bad_query_never_falls_back(tmp_path, fault):
    source = candidate(tmp_path)
    selector = LossSelector()
    selector.fit([observation(source.candidate_id, 1.)])
    feature = CodeEncoder(tmp_path).encode(source)
    if fault == "nonfinite":
        feature = replace(feature, values=(float("inf"),) * DIMENSION)
    if fault == "version":
        feature = replace(feature, version="other")
    if fault == "unfitted":
        selector = LossSelector()
    features = {} if fault == "missing" else {source.candidate_id: feature}
    with pytest.raises(ValueError):
        selector.select([source], features, count=2 if fault == "count" else 1)


def test_shared_engine_selects_only_one_of_four_and_resume_is_idle(tmp_path):
    from ldm_tts.campaign import CampaignRequest, CampaignBudget
    from ldm_tts.engine import LDMEngine, LDMEngineState, InitialRoundReservoirExpander
    from ldm_tts.engine.expansion import CallableReservoirExpander, ExpansionResult
    from tasks.copper_inversion.core.workflow import parse_args
    from tasks.copper_inversion.core.task_spec import describe_ldm_task

    encoder, selector = CodeEncoder(tmp_path), LossSelector()
    spec = replace(describe_ldm_task(parse_args(["--mock"])),
                   surrogate=encoder.describe(), acquisition=selector.describe())
    pool = [candidate(tmp_path, shift=i) for i in range(13)]
    evaluated = []

    def expand(request):
        subset = pool[1 + 4 * (request.round_idx - 1):1 + 4 * request.round_idx]
        return ExpansionResult(tuple(RawProposal(c.payload, "pool") for c in subset))

    expander = InitialRoundReservoirExpander(
        initializer=CallableReservoirExpander(lambda _: ExpansionResult(
            (RawProposal(pool[0].payload, "baseline"),), selection_mode="reservoir_order")),
        search_expander=CallableReservoirExpander(expand), initial_reservoir_size=1,
    )

    def evaluate(c):
        evaluated.append(c.candidate_id)
        return {"loss_total": float(len(evaluated)), "research_utility": -float(len(evaluated))}

    budget = CampaignBudget(rounds=4, reservoir_size=4, batch_size=1,
                            target_observations=4, max_evaluation_attempts=4)

    def run(resume=False):
        runtime = CampaignRequest(run_dir=tmp_path, budget=budget, config={}, resume=resume).open_runtime(spec)
        state = LDMEngineState.from_checkpoint(runtime.load_checkpoint()) if resume else LDMEngineState()
        return LDMEngine(task_spec=spec, expander=expander, candidate_domain=Domain(tmp_path),
                         evaluator=CallableCandidateEvaluator(evaluate), runtime=runtime,
                         surrogate_encoder=encoder, selector=selector).run(budget.engine_config(), state=state)

    result = run()
    assert len(result.state.observations) == 4 and len(set(evaluated)) == 4
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    selections = [e["payload"] for e in events if e["event_type"] == "candidates_selected"]
    assert [len(s["predictions"]) for s in selections] == [0, 4, 4, 4]
    assert all(len(s["selected_candidate_ids"]) == 1 for s in selections)
    before = json.loads((tmp_path / "budget.json").read_text())["counters"]
    run(resume=True)
    assert len(evaluated) == 4
    assert json.loads((tmp_path / "budget.json").read_text())["counters"] == before
