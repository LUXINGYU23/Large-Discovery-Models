"""Scientific regression tests for occurrence sampling, refill and replay."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from ldm_tts.contracts import RawProposal, ReservoirBuilder
from ldm_tts.engine.expansion import ExpansionRequest
from ldm_tts.optimization.records import BOObservation
from ldm_tts.transport import ProposalResponse
from tasks.reasyn.core.candidate import ReaSynDomain
from tasks.reasyn.core.chemistry import MOCK_SMILES
from tasks.reasyn.core.proposals import ProposalExhausted, ReaSynExpander
from tasks.reasyn.core.sampling import (
    Q0_METADATA_KEY,
    attach_empirical_base_measure,
    empirical_base_masses,
    robust_z,
    tilted_distribution,
)
from tasks.reasyn.core.selection import AcquisitionTiltedSelector, TanimotoGPSelector
from tasks.reasyn.core.surrogate import MoleculeEncoder


class Sink:
    def append(self, *args, **kwargs):
        pass


class Targets:
    def __init__(self, batches):
        self.batches = iter(batches)
        self.requests = []

    def propose(self, request):
        self.requests.append(request)
        batch = next(self.batches)
        if isinstance(batch, Exception):
            raise batch
        return ProposalResponse(text=json.dumps({"candidates": [{"target_smiles": s} for s in batch]}))


class Projector:
    def __init__(self, products=None, run_dir=None):
        self.products = products or {}
        self.calls = []
        if run_dir is not None:
            self.run_dir = run_dir

    def project(self, targets, *, sampling_seed, identity):
        self.calls.append((targets, sampling_seed, identity))
        return [
            {"target": target, "target_index": index,
             "smiles": self.products.get(target, target),
             "synthesis": target, "pathway_verified": True}
            for index, target in enumerate(targets)
        ], identity + "/result.json"


def options(benchmark="tdc", **kwargs):
    values = dict(benchmark=benchmark, oracle="jnk3", mock=True, seed=7,
                  proposal_mode="openai", proposal_batch_size=2,
                  evaluations_per_round=2, max_replenishment_batches=4)
    values.update(kwargs)
    return SimpleNamespace(**values)


def product(smiles):
    return RawProposal({"smiles": smiles, "synthesis": smiles, "projection_artifact": "result.json"},
                       "reasyn_projector", {"pathway_verified": True})


def observed(smiles):
    candidate = ReaSynDomain("tdc", mock=True).admit(product(smiles))
    return SimpleNamespace(candidate=candidate, candidate_id=candidate.candidate_id,
                           canonical_key=candidate.canonical_key, metrics={"oracle_score": 0.5})


def test_independent_minibatches_preserve_product_occurrences_before_dedup():
    client = Targets([["CCO", "CCO"], ["CCN", "CCC"]])
    projector = Projector({"CCC": "CCO"})
    result = ReaSynExpander(options(), projector, Sink(), client=client).expand(ExpansionRequest(0, 4))
    assert len(client.requests) == 2
    assert [r.metadata["count"] for r in client.requests] == [2, 2]
    assert [r.metadata["minibatch_index"] for r in client.requests] == [0, 1]
    assert [r.metadata["history"] for r in client.requests] == [[], []]
    assert len(result.proposals) == 4
    reservoir = ReservoirBuilder(ReaSynDomain("tdc", mock=True)).build(result.proposals)
    assert len(reservoir.candidates) == 2
    by_smiles = {c.payload["smiles"]: c.metadata[Q0_METADATA_KEY] for c in reservoir.candidates}
    assert by_smiles["CCO"]["occurrence_count"] == 3
    assert by_smiles["CCO"]["probability"] == pytest.approx(0.75)
    assert by_smiles["CCN"]["probability"] == pytest.approx(0.25)
    assert projector.calls[0][1] + 2 == projector.calls[1][1]


def test_reconstruction_repeated_query_seeds_are_distinct_probability_identities():
    client = Targets([["CCO", "CCO"], ["CCO", "CCO"]])
    result = ReaSynExpander(options("reconstruction"), Projector(), Sink(), target="CCO", client=client).expand(ExpansionRequest(0, 4))
    reservoir = ReservoirBuilder(ReaSynDomain("reconstruction", mock=True)).build(result.proposals)
    assert len(reservoir.candidates) == 4
    assert {c.payload["sampling_seed"] for c in reservoir.candidates} == {7, 8, 9, 10}
    assert np.allclose(empirical_base_masses(reservoir.candidates), [0.25] * 4)
    assert all(c.metadata[Q0_METADATA_KEY]["identity_space"] == "canonical_query_and_sampling_seed" for c in reservoir.candidates)


def test_reconstruction_uniform_seed_q0_makes_alpha_invariant():
    # Independent seed identities preserve molecular multiplicity through the
    # number of trial entries, not through unequal per-entry q0 weights.
    client = Targets([["CCO", "CCO"], ["CCO", "CCN"]])
    result = ReaSynExpander(options("reconstruction"), Projector(), Sink(),
                           target="CCO", client=client).expand(ExpansionRequest(0, 4))
    candidates = ReservoirBuilder(ReaSynDomain("reconstruction", mock=True)).build(result.proposals).candidates
    q0 = empirical_base_masses(candidates)
    scores = [0.1 if c.payload["target_smiles"] == "CCO" else 0.9 for c in candidates]
    reference = tilted_distribution(q0, scores, alpha=1.0, eta=1.0)[0]
    for alpha in (0.0, 0.5, 2.0, 10.0):
        probabilities = tilted_distribution(q0, scores, alpha=alpha, eta=1.0)[0]
        np.testing.assert_allclose(probabilities, reference, rtol=1e-12, atol=1e-12)
    baseline = tilted_distribution(q0, scores, alpha=2.0, eta=0.0)[0]
    assert sum(p for p, c in zip(baseline, candidates) if c.payload["target_smiles"] == "CCO") == pytest.approx(0.75)
    assert not np.allclose(tilted_distribution(q0, scores, alpha=1.0, eta=0.5)[0], reference)


def test_full_history_product_rejections_refill_and_keep_current_round_occurrences():
    # First entry is outside the compact 12-observation prompt window.
    history = (observed("CCO"),) + tuple(observed("CCN") for _ in range(12))
    client = Targets([["CCC", "CCCO"], ["CCCO"], ["CCCC"]])
    projector = Projector({"CCC": "CCO"})
    result = ReaSynExpander(options(), projector, Sink(), client=client).expand(ExpansionRequest(1, 2, observations=history))
    assert len(client.requests) == 3
    assert len(client.requests[0].metadata["history"]) == 13
    assert client.requests[0].metadata["excluded_products"] == ["CCN", "CCO"]
    assert client.requests[1].metadata["rejection_feedback"][0]["reason"] == "already_evaluated"
    assert result.metadata["rejection_counts"]["already_evaluated_product"] == 1
    assert len(result.proposals) == 3
    probabilities = {p.payload["smiles"]: p.metadata[Q0_METADATA_KEY]["probability"] for p in result.proposals}
    assert probabilities == {"CCCO": pytest.approx(2 / 3), "CCCC": pytest.approx(1 / 3)}


def test_exhausted_refill_is_explicit_and_bounded(tmp_path):
    client = Targets([["CCO", "CCO"], ["CCO", "CCO"]])
    expander = ReaSynExpander(options(max_replenishment_batches=1), Projector(run_dir=tmp_path), Sink(), client=client)
    with pytest.raises(ProposalExhausted) as caught:
        expander.expand(ExpansionRequest(1, 2, observations=(observed("CCO"),)))
    assert len(client.requests) == 2
    assert caught.value.metadata["stop_reason"] == "proposal_replenishment_budget_exhausted"
    assert caught.value.metadata["targets_proposed"] == 4
    assert caught.value.metadata["valid_occurrences"] == 0
    assert caught.value.metadata["rejection_counts"]["already_evaluated_product"] == 4
    diagnostic = json.loads((tmp_path / "proposal_batches/round-000001/diagnostics.json").read_text())
    assert diagnostic["replenishment_batches"] == 1


def test_failed_second_minibatch_replays_first_response_without_endpoint_call(tmp_path):
    first = Targets([["CCO", "CCN"], RuntimeError("endpoint unavailable")])
    projector = Projector(run_dir=tmp_path)
    request = ExpansionRequest(0, 4)
    expander = ReaSynExpander(options("reconstruction"), projector, Sink(), client=first)
    with pytest.raises(RuntimeError, match="endpoint unavailable"):
        expander.expand(request)
    resumed = Targets([["CCC", "CCCO"]])
    result = ReaSynExpander(options("reconstruction"), projector, Sink(), client=resumed).expand(request)
    assert len(resumed.requests) == 1
    assert resumed.requests[0].metadata["minibatch_index"] == 1
    assert [p.payload["target_smiles"] for p in result.proposals] == ["CCO", "CCN", "CCC", "CCCO"]


def test_projection_occurrence_index_cannot_silently_merge_equal_queries():
    rows = [{"target": "CCO", "smiles": "CCO"}]
    with pytest.raises(ValueError, match="target_index"):
        ReaSynExpander._first_per_occurrence(rows, ["CCO", "CCO"])
    rows = [{"target": "CCO", "target_index": 0, "smiles": "CCO"},
            {"target": "CCO", "target_index": 1, "smiles": "CCN"},
            {"target": "CCO", "target_index": 0, "smiles": "CCC"}]
    assert [row["smiles"] for row in ReaSynExpander._first_per_occurrence(rows, ["CCO", "CCO"]).values()] == ["CCO", "CCN"]


def candidates_with_q0():
    proposals = attach_empirical_base_measure(
        tuple(product(s) for s in ["CCO", "CCO", "CCO", "CCN", "CCC", "CCCO"]),
        benchmark="tdc", mock=True,
    )
    return ReservoirBuilder(ReaSynDomain("tdc", mock=True)).build(proposals).candidates


def test_ldm_distribution_uses_q0_and_robust_ucb_not_best_of_n():
    q0 = np.array([0.75, 0.25])
    probability, logits, z = tilted_distribution(q0, [0.1, 0.9], alpha=1.0, eta=0.0)
    assert probability == pytest.approx(q0)
    assert logits == pytest.approx(np.log(q0 + 1e-12))
    uniform, _, _ = tilted_distribution(q0, [0.1, 0.9], alpha=0.0, eta=0.0)
    assert uniform == pytest.approx([0.5, 0.5])
    tilted, _, _ = tilted_distribution(q0, [0.1, 0.9], alpha=1.0, eta=2.0)
    assert tilted[1] > probability[1]
    assert robust_z([1.0, 1.0, 1.0]) == pytest.approx([0, 0, 0])
    assert np.abs(robust_z([0.0, 0.1, 0.2, 1000.0], clip=3.0)).max() <= 3.0


def test_independent_bo_pool_and_sampling_are_replayable():
    encoder = MoleculeEncoder(mock=True)
    candidates = candidates_with_q0()
    representations = {c.candidate_id: encoder.encode(c) for c in candidates}
    selector = AcquisitionTiltedSelector(
        TanimotoGPSelector(objective_name="oracle_score", feature_version=encoder.version),
        alpha=1, eta=0, seed=123, pool_size=2,
    )
    selector.fit([])
    first = selector.select(candidates, representations, count=1, round_idx=2)
    repeat = selector.select(tuple(reversed(candidates)), representations, count=1, round_idx=2)
    assert first.to_dict() == repeat.to_dict()
    assert first.metadata["proposal_reservoir_size"] == 4
    assert first.metadata["bo_pool_size"] == 2
    assert len(first.predictions) == 2
    assert sum(row["selection_probability"] for row in first.metadata["distribution"]) == pytest.approx(1.0)
    assert first.selected_candidate_ids[0] in first.metadata["bo_pool_candidate_ids"]
    with pytest.raises(ValueError, match="BO pool"):
        selector.select(candidates, representations, count=3)


def test_selector_applies_compiled_prior_and_weights():
    encoder = MoleculeEncoder(mock=True)
    candidates = candidates_with_q0()
    representations = {c.candidate_id: encoder.encode(c) for c in candidates}
    history = [BOObservation.scalar("past", 0.5, [0.1] * 32, feature_version=encoder.version, metadata={"round_idx": 0})]
    calls = []

    class Adapter:
        def build_selection_round(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(round_index=kwargs["round_index"])

    class Controller:
        def resolve(self, round_input):
            return SimpleNamespace(history_prior_mean=np.array([0.2]), query_prior_mean=np.ones(4),
                                   alpha=0.0, eta=0.0, epoch_id="e1", artifact_digest="abc",
                                   source="artifact", degraded=False, stage="active", metadata={})

        def record_predictions(self, round_idx, records):
            calls.append((round_idx, records))

    selector = AcquisitionTiltedSelector(
        TanimotoGPSelector(objective_name="oracle_score", feature_version=encoder.version),
        policy_controller=Controller(), policy_adapter=Adapter(),
    )
    selector.fit(history)
    result = selector.select(candidates, representations, round_idx=1)
    assert calls[0]["history"] == tuple(history)
    assert calls[0]["requested_evaluation_batch"] == 1
    assert result.metadata["compiled_policy"]["epoch_id"] == "e1"
    assert [row["selection_probability"] for row in result.metadata["distribution"]] == pytest.approx([0.25] * 4)
    assert all(row["active_mean"] > row["baseline_mean"] for row in result.metadata["distribution"])
    repeated = selector.select(candidates, representations, round_idx=1)
    assert repeated.to_dict() == result.to_dict()


def test_bo_uses_score_blind_local_pool_without_proposal_client():
    expander = ReaSynExpander(options(proposal_mode="bo", search_method="bo", bo_targets=["CCO", "CCN"]), Projector(), Sink())
    result = expander.expand(ExpansionRequest(0, 2))
    assert {p.payload["smiles"] for p in result.proposals} == {"CCO", "CCN"}
    assert result.attempts == ()
