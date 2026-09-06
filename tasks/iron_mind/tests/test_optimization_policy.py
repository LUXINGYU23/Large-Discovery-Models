"""Compiled-policy contracts at the Iron Mind task boundary."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from ldm_tts.contracts import Candidate
from ldm_tts.harness import (
    CompiledOptimizationPolicy,
    load_harness_mcp_config,
)
from ldm_tts.optimization import BOObservation, BOPrediction
from tasks.iron_mind.core.policy_diagnostics import prepare_ucb_policy_research
from tasks.iron_mind.core.candidate import IRON_MIND_Q0_METADATA_KEY
from tasks.iron_mind.core.ldm_selector import AcquisitionTiltedSelector
from tasks.iron_mind.core.history import condition_evidence
from tasks.iron_mind.core.optimization_policy import (
    IronMindOptimizationPolicyAdapter,
    policy_harness_profile,
)
from tasks.iron_mind.core.reaction_gp import ReactionCategoricalGPUCBSelector
from tasks.iron_mind.core.schema import (
    ReactionDatasetSchema,
    ReactionFactor,
    canonical_schema_payload,
    schema_sha256,
)
from tasks.iron_mind.core.surrogate import ReactionOneHotEncoder
from tasks.iron_mind.core.workflow import (
    _policy_harness_client,
    describe_ldm_task,
    parse_args,
)
from tasks.iron_mind.core.workflow_support import derived_budget, jsonable_args


def _schema() -> ReactionDatasetSchema:
    factors = (
        ReactionFactor("base", ("A", "B")),
        ReactionFactor("solvent", ("X", "Y")),
    )
    payload = canonical_schema_payload(
        dataset_id="compiled_policy_test",
        factors=factors,
        measurements=("yield",),
        objective="reaction_score",
        direction="maximize",
        observation_policy="single_row",
    )
    return ReactionDatasetSchema(
        "compiled_policy_test",
        factors,
        ("yield",),
        "reaction_score",
        "maximize",
        "single_row",
        schema_sha256(payload),
    )


def _candidate(
    schema: ReactionDatasetSchema,
    identifier: str,
    occurrence_count: int,
    *,
    base: str,
    solvent: str,
) -> Candidate:
    return Candidate(
        candidate_id=identifier,
        canonical_key=identifier,
        payload={
            "dataset_id": schema.dataset_id,
            "conditions": {"base": base, "solvent": solvent},
        },
        metadata={
            IRON_MIND_Q0_METADATA_KEY: {
                "occurrence_count": occurrence_count,
                "valid_occurrence_count": 4,
                "probability": occurrence_count / 4,
            }
        },
    )


def test_policy_features_match_schema_and_exclude_mean_leakage() -> None:
    schema = _schema()
    encoder = ReactionOneHotEncoder(schema)
    observed = _candidate(schema, "observed", 1, base="A", solvent="X")
    query = (
        _candidate(schema, "a", 3, base="A", solvent="Y"),
        _candidate(schema, "b", 1, base="B", solvent="X"),
    )
    history = (
        BOObservation.scalar(
            observed.candidate_id,
            4.0,
            encoder.encode(observed).values,
            feature_version=encoder.version,
            metadata={"round_idx": 0},
        ),
    )
    predictions = (
        BOPrediction.scalar("a", mean=3.0, std=1.0, acquisition_score=4.0),
        BOPrediction.scalar("b", mean=2.0, std=1.0, acquisition_score=3.0),
    )
    adapter = IronMindOptimizationPolicyAdapter(
        schema,
        seed=7,
        acquisition_beta=1.0,
        default_alpha=1.0,
        default_eta=2.0,
    )

    round_input = adapter.build_selection_round(
        history=history,
        candidates=query,
        representations={item.candidate_id: encoder.encode(item) for item in query},
        q0=np.asarray((0.75, 0.25)),
        baseline_predictions=predictions,
        valid_proposal_occurrences=4,
    )

    contract = adapter.capability_contract()
    assert contract.feature_names == (
        'base="A"',
        'base="B"',
        'solvent="X"',
        'solvent="Y"',
    )
    assert contract.feature_groups == {"base": (0, 2), "solvent": (2, 4)}
    assert round_input.history_features.tolist() == [[1.0, 0.0, 1.0, 0.0]]
    assert round_input.query_features.tolist() == [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 0.0],
    ]
    assert round_input.round_index == 1
    assert "round_index" not in round_input.execution_context["weight_context"]
    assert list(round_input.measured_observations) == [
        {
            "round_index": 0,
            "conditions": {"base": "A", "solvent": "X"},
            "reaction_score": 4.0,
        }
    ]
    mean_context = json.dumps(round_input.execution_context["mean_context"])
    assert all(
        forbidden not in mean_context
        for forbidden in ("q0", "acquisition", "candidate_id", "canonical_key")
    )
    assert "candidate_id" not in json.dumps(round_input.research_snapshot)
    assert round_input.research_snapshot["condition_evidence"]["factor_coverage"]["base"] == [
        {"option": "A", "measured_count": 1}, {"option": "B", "measured_count": 0},
    ]
    assert round_input.research_snapshot["fixed_optimization"] == {
        "surrogate": "factor-aware categorical ARD residual GP",
        "model_mismatch_variance": 0.04,
        "target_std_floor": 1.0,
        "acquisition": "GP-UCB",
        "acquisition_beta": 1.0,
        "selection": (
            "q0^alpha * exp(eta * robust_z(acquisition)), then Gumbel top-k"
        ),
        "editable_components": ["prior_mean", "alpha", "eta"],
    }


def test_condition_evidence_separates_factor_changes_replicates_and_confounding() -> None:
    observations = [
        {"round_index": index, "conditions": {"base": base, "solvent": solvent}, "reaction_score": score}
        for index, (base, solvent, score) in enumerate([
            ("A", "X", 21.0), ("B", "X", 18.0), ("B", "Y", 30.0), ("A", "X", 22.0),
        ])
    ]
    evidence = condition_evidence(observations, _schema())
    comparisons = {
        (row["earlier_round_index"], row["later_round_index"]): row
        for row in evidence["comparisons"]
    }
    changed = comparisons[(0, 1)]
    assert changed["comparison_type"] == "single_factor_change"
    assert changed["changed_factors"] == {"base": {"from": "A", "to": "B"}}
    assert changed["utility_difference"] == -3.0
    assert comparisons[(0, 3)]["comparison_type"] == "same_conditions"
    assert comparisons[(0, 3)]["changed_factors"] == {}
    assert (0, 2) not in comparisons
    assert evidence["factor_coverage"]["solvent"] == [
        {"option": "X", "measured_count": 3}, {"option": "Y", "measured_count": 1},
    ]


class _StaticPolicyController:
    def record_predictions(self, round_index, predictions):
        assert predictions and all("candidate_id" in row for row in predictions)

    def __init__(self) -> None:
        self.round_input = None

    def resolve(self, round_input):
        self.round_input = round_input
        return CompiledOptimizationPolicy(
            epoch_id="epoch_001",
            artifact_digest="a" * 64,
            history_prior_mean=np.zeros(len(round_input.history_features)),
            query_prior_mean=np.asarray((2.0, -2.0)),
            stage="exploit",
            alpha=0.0,
            eta=0.0,
            source="artifact",
            degraded=False,
            metadata={"action": "replace", "status": "accepted"},
        )


def test_policy_holdouts_use_training_prefix_and_restore_online_gp() -> None:
    schema = _schema()
    encoder = ReactionOneHotEncoder(schema)
    candidates = tuple(
        _candidate(schema, str(i), 1, base=base, solvent=solvent)
        for i, (base, solvent) in enumerate((("A", "X"), ("A", "Y"), ("B", "X"), ("B", "Y")))
    )
    history = tuple(BOObservation.scalar(
        item.candidate_id, score, encoder.encode(item).values,
        feature_version=encoder.version, metadata={"round_idx": i},
    ) for i, (item, score) in enumerate(zip(candidates[:3], (1.0, 5.0, 3.0), strict=True)))
    query = candidates[3:]
    representations = {item.candidate_id: encoder.encode(item) for item in query}
    gp = ReactionCategoricalGPUCBSelector(schema=schema, objective_name="reaction_score", feature_version=encoder.version)
    gp.fit(history)
    baseline = gp.select(query, representations)
    adapter = IronMindOptimizationPolicyAdapter(schema, seed=0, acquisition_beta=1.0, default_alpha=2.0, default_eta=0.25)
    round_input = adapter.build_selection_round(
        history=history, candidates=query, representations=representations,
        q0=np.ones(1), baseline_predictions=baseline.predictions, valid_proposal_occurrences=1,
    )
    prepared = prepare_ucb_policy_research(
        round_input, history=history, candidates=query, representations=representations,
        baseline=baseline.predictions, q0=np.ones(1), selector=gp, z_clip=2.0,
    )
    assert gp.select(query, representations).to_dict() == baseline.to_dict()
    for fold in prepared.execution_context["validation_folds"]:
        training = [history[i] for i in fold["train_indices"]]
        assert max(fold["train_indices"]) < min(fold["test_indices"])
        gp.fit(training)
        expected = gp.posterior_projection(training, np.asarray([history[i].feature_vector for i in fold["test_indices"]]))
        np.testing.assert_allclose(prepared.diagnostic_arrays[fold["prefix"] + "weights"], expected["weights"])


def test_compiled_policy_changes_gp_mean_and_ldm_weights() -> None:
    schema = _schema()
    encoder = ReactionOneHotEncoder(schema)
    observed = _candidate(schema, "observed", 1, base="A", solvent="X")
    candidates = (
        _candidate(schema, "a", 3, base="A", solvent="Y"),
        _candidate(schema, "b", 1, base="B", solvent="X"),
    )
    history = (
        BOObservation.scalar(
            observed.candidate_id,
            4.0,
            encoder.encode(observed).values,
            feature_version=encoder.version,
            metadata={"round_idx": 0},
        ),
    )
    controller = _StaticPolicyController()
    adapter = IronMindOptimizationPolicyAdapter(
        schema,
        seed=3,
        acquisition_beta=1.0,
        default_alpha=1.0,
        default_eta=1.0,
    )
    selector = AcquisitionTiltedSelector(
        ReactionCategoricalGPUCBSelector(
            schema=schema,
            objective_name="reaction_score",
            feature_version=encoder.version,
        ),
        alpha=1.0,
        eta=1.0,
        seed=3,
        pool_size=2,
        proposal_sample_count=4,
        policy_controller=controller,  # type: ignore[arg-type]
        policy_adapter=adapter,
    )
    selector.fit(history)

    result = selector.select(
        candidates,
        {item.candidate_id: encoder.encode(item) for item in candidates},
    )

    prediction = {item.candidate_id: item for item in result.predictions}
    assert prediction["a"].metadata["prior_mean_standardized"] == 2.0
    assert prediction["b"].metadata["prior_mean_standardized"] == -2.0
    assert prediction["a"].scalar_mean > prediction["b"].scalar_mean
    probabilities = [
        item.metadata["selection_probability"] for item in result.predictions
    ]
    assert probabilities == pytest.approx([0.5, 0.5])
    assert result.metadata["alpha_base_measure"] == 0.0
    assert result.metadata["eta_acquisition_tilt"] == 0.0
    assert result.metadata["compiled_policy"]["action"] == "replace"
    assert controller.round_input.round_index == 1


def test_compiled_method_declares_policy_pool_and_separate_budget() -> None:
    args = parse_args(
        [
            "--mock",
            "--proposal-mode",
            "none",
            "--search-method",
            "ldm_harness_compiled",
            "--initialization-mode",
            "shared_random",
            "--iterations",
            "3",
        ]
    )

    budget = derived_budget(args, domain_size=100)
    task_spec = describe_ldm_task(args)

    assert args.policy_capability == ["ldm_weights@1", "prior_mean@1"]
    assert budget["proposal_attempts"] == 8
    assert budget["harness_turns"] == 8
    assert budget["policy_harness_turns"] == 2
    assert task_spec.acquisition.parameters["optimization_policy"] == (
        "persistent_harness_compiled"
    )
    assert task_spec.proposal_search.parameters["skills_loaded"] is False
    assert task_spec.proposal_search.parameters["policy_skills_loaded"] is True
    assert task_spec.proposal_search.parameters["optimization_policy"] == (
        "persistent_harness_compiled"
    )
    profile = policy_harness_profile()[0]
    assert profile.profile_id == "policy_architect"
    assert profile.skill_dirs[0].as_posix() == "/resources/skills/compile-ldm-policy"
    assert len(profile.skill_dir_sha256[0]) == 64


def test_existing_methods_do_not_emit_compiled_policy_arguments() -> None:
    args = parse_args(
        [
            "--mock",
            "--proposal-mode",
            "none",
            "--search-method",
            "ldm_harness",
        ]
    )

    assert not any(key.startswith("policy_") for key in jsonable_args(args))


def test_policy_sidecar_mounts_only_public_task_resources(tmp_path) -> None:
    from tasks.iron_mind.core import workflow

    args = parse_args(
        [
            "--mock",
            "--proposal-mode",
            "none",
            "--search-method",
            "ldm_harness_compiled",
            "--initialization-mode",
            "shared_random",
            "--harness-cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    table = workflow._load_mock_table(workflow._schema_for("buchwald_hartwig"))
    client = _policy_harness_client(
        args,
        SimpleNamespace(run_dir=tmp_path / "run", run_id="campaign-test"),
        SimpleNamespace(
            base_url="https://example.invalid",
            model="test-model",
            api_key="test-key",
        ),
        table,
        load_harness_mcp_config(None),
    )

    command = " ".join(client.command)
    assert "mock_oracle.csv" not in command
    assert "dst=/resources,readonly" in command
    assert "dst=/skills" not in command
    assert "/public/reaction_schemas.json" in command
    assert client.config.submission_contract.tool_name == "submit_optimization_policy"
    assert client.config.profiles[0].skill_dirs
    assert client.config.mcp_servers[-1].server_id == "ldm_policy"
