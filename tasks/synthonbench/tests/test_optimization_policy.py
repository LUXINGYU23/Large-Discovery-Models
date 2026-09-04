"""Compiled-policy contracts at the SynthonBench task boundary."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from ldm_tts.contracts import Candidate
from ldm_tts.harness import CompiledOptimizationPolicy, load_harness_mcp_config
from ldm_tts.optimization import BOObservation, BOPrediction
from tasks.synthonbench.core.constants import OBJECTIVE_NAME, Q0_METADATA_KEY
from tasks.synthonbench.core.ldm_selector import AcquisitionTiltedSelector
from tasks.synthonbench.core.nystrom_encoder import SynthonNystromEncoder
from tasks.synthonbench.core.optimization_policy import (
    SynthonOptimizationPolicyAdapter,
    policy_harness_profile,
)
from tasks.synthonbench.core.policy_features import SynthonPolicyFeatureEncoder
from tasks.synthonbench.core.tanimoto_gp import (
    SynthonTanimotoGPUCBSelector,
    TanimotoGPUCBConfig,
)
from tasks.synthonbench.core.workflow import (
    _load_benchmark,
    _policy_harness_client,
    describe_ldm_task,
    parse_args,
)
from tasks.synthonbench.core.workflow_support import campaign_budget, jsonable_args


class _Space:
    def __init__(self, *, first_smiles: str = "C[U]") -> None:
        self._records = {
            ("r1", 1, 1): first_smiles,
            ("r1", 1, 2): "CC[U]",
            ("r1", 2, 11): "[U]N",
            ("r1", 2, 12): "[U]O",
            ("r2", 1, 21): "c1cc([Np])ccc1",
            ("r2", 1, 22): "CC[Np]",
        }

    def positions(self, reaction_id: str) -> tuple[int, ...]:
        return tuple(sorted({key[1] for key in self._records if key[0] == reaction_id}))

    def synthon_ids(self, reaction_id: str, position: int) -> tuple[int, ...]:
        return tuple(
            sorted(
                key[2]
                for key in self._records
                if key[:2] == (reaction_id, position)
            )
        )

    def synthon_smiles(self, reaction_id: str, position: int, synthon_id: int) -> str:
        return self._records[(reaction_id, position, synthon_id)]

    def product_count_estimate(self, reaction_id: str) -> int:
        count = 1
        for position in self.positions(reaction_id):
            count *= len(self.synthon_ids(reaction_id, position))
        return count


def _candidate(
    reaction_id: str,
    synthon_ids: tuple[int, ...],
    occurrences: int = 1,
    total: int = 1,
) -> Candidate:
    product_id = f"{reaction_id}|{'_'.join(str(value) for value in synthon_ids)}"
    return Candidate(
        candidate_id=f"synthonbench:{product_id}",
        payload={"reaction_id": reaction_id, "synthon_ids": list(synthon_ids)},
        canonical_key=product_id,
        source="test",
        metadata={
            Q0_METADATA_KEY: {
                "occurrence_count": occurrences,
                "valid_occurrence_count": total,
                "probability": occurrences / total,
            }
        },
    )


def _adapter(features: SynthonPolicyFeatureEncoder) -> SynthonOptimizationPolicyAdapter:
    return SynthonOptimizationPolicyAdapter(
        features,
        target="kif11",
        seed=7,
        acquisition_beta=1.0,
        gp_signal_std=1.0,
        gp_mean_std=1.0,
        gp_observation_noise_std=1.0,
        gp_reaction_weight=0.25,
        fingerprint_bits=128,
        gp_landmarks=3,
        gp_kernel_jitter=1.0e-8,
        default_alpha=1.0,
        default_eta=2.0,
    )


def test_policy_features_are_deterministic_finite_and_space_versioned() -> None:
    first = SynthonPolicyFeatureEncoder(_Space(), ("r2", "r1"))
    second = SynthonPolicyFeatureEncoder(_Space(), ("r1", "r2"))
    changed = SynthonPolicyFeatureEncoder(_Space(first_smiles="N[U]"), ("r1", "r2"))

    vector = first.encode_tuple("r1", (1, 11))

    np.testing.assert_allclose(vector, second.encode_tuple("r1", (1, 11)))
    assert np.isfinite(vector).all()
    assert first.feature_names == second.feature_names
    assert first.version == second.version
    assert changed.version != first.version
    assert first.descriptor_statistics()["space_sha256"] == first.space_digest


def test_policy_adapter_separates_research_data_from_deployed_mean_inputs() -> None:
    space = _Space()
    policy_features = SynthonPolicyFeatureEncoder(space, ("r1", "r2"))
    surrogate = SynthonNystromEncoder(
        space,
        ("r1", "r2"),
        landmark_count=3,
        seed=2,
        fingerprint_bits=128,
    )
    observed = _candidate("r1", (1, 11))
    candidates = (
        _candidate("r1", (2, 11), 3, 4),
        _candidate("r2", (21,), 1, 4),
    )
    history = (
        BOObservation.scalar(
            observed.candidate_id,
            4.0,
            surrogate.encode(observed).values,
            feature_version=surrogate.version,
            metadata={"round_idx": 0},
        ),
    )
    predictions = (
        BOPrediction.scalar(
            candidates[0].candidate_id,
            mean=3.0,
            std=1.0,
            acquisition_score=4.0,
        ),
        BOPrediction.scalar(
            candidates[1].candidate_id,
            mean=2.0,
            std=1.0,
            acquisition_score=3.0,
        ),
    )

    round_input = _adapter(policy_features).build_selection_round(
        history=history,
        candidates=candidates,
        baseline_predictions=predictions,
        valid_proposal_occurrences=4,
    )

    assert round_input.round_index == 1
    assert "round_index" not in round_input.execution_context["weight_context"]
    assert round_input.history_features.shape == (1, policy_features.dimension)
    assert round_input.query_features.shape == (2, policy_features.dimension)
    assert round_input.research_snapshot["new_measured_observations"][0][
        OBJECTIVE_NAME
    ] == 4.0
    assert "smiles" in json.dumps(round_input.research_snapshot)
    mean_context = json.dumps(round_input.execution_context["mean_context"])
    assert all(
        term not in mean_context
        for term in ("q0", "acquisition", "candidate_id", "synthon_ids", "smiles")
    )


class _StaticPolicyController:
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


def test_compiled_policy_changes_gp_mean_and_ldm_weights() -> None:
    space = _Space()
    surrogate = SynthonNystromEncoder(
        space,
        ("r1", "r2"),
        landmark_count=3,
        seed=3,
        fingerprint_bits=128,
    )
    observed = _candidate("r1", (1, 11))
    candidates = (
        _candidate("r1", (2, 11), 3, 4),
        _candidate("r2", (21,), 1, 4),
    )
    history = (
        BOObservation.scalar(
            observed.candidate_id,
            4.0,
            surrogate.encode(observed).values,
            feature_version=surrogate.version,
            metadata={"round_idx": 0},
        ),
    )
    controller = _StaticPolicyController()
    selector = AcquisitionTiltedSelector(
        SynthonTanimotoGPUCBSelector(
            objective_name=OBJECTIVE_NAME,
            feature_dimension=surrogate.dimension,
            feature_version=surrogate.version,
            config=TanimotoGPUCBConfig(),
        ),
        alpha=1.0,
        eta=1.0,
        z_clip=5.0,
        seed=3,
        pool_size=2,
        proposal_sample_count=4,
        policy_controller=controller,  # type: ignore[arg-type]
        policy_adapter=_adapter(SynthonPolicyFeatureEncoder(space, ("r1", "r2"))),
    )
    selector.fit(history)

    result = selector.select(
        candidates,
        {candidate.candidate_id: surrogate.encode(candidate) for candidate in candidates},
    )

    priors = {
        prediction.metadata["prior_mean_standardized"]
        for prediction in result.predictions
    }
    assert priors == {2.0, -2.0}
    assert [
        prediction.metadata["selection_probability"]
        for prediction in result.predictions
    ] == pytest.approx([0.5, 0.5])
    assert result.metadata["alpha_base_measure"] == 0.0
    assert result.metadata["eta_acquisition_tilt"] == 0.0
    assert result.metadata["compiled_policy"]["action"] == "replace"
    assert controller.round_input.round_index == 1


def test_compiled_method_declares_policy_pool_budget_and_public_mounts(
    tmp_path,
) -> None:
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
            "--harness-cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    benchmark = _load_benchmark(args)
    client = _policy_harness_client(
        args,
        SimpleNamespace(run_dir=tmp_path / "run", run_id="campaign-test"),
        SimpleNamespace(
            base_url="https://example.invalid",
            model="test-model",
            api_key="test-key",
        ),
        benchmark,
        load_harness_mcp_config(None),
    )
    spec = describe_ldm_task(args, benchmark)
    budget = campaign_budget(args, None)
    command = " ".join(client.command)

    assert args.policy_capability == ["ldm_weights@1", "prior_mean@1"]
    assert budget["harness_turns"] == 8
    assert budget["policy_harness_turns"] == 2
    assert spec.acquisition.parameters["optimization_policy"] == (
        "persistent_harness_compiled"
    )
    assert spec.proposal_search.parameters["policy_skills_loaded"] is True
    profile = policy_harness_profile()[0]
    assert profile.profile_id == "policy_architect"
    assert profile.skill_dirs[0].as_posix() == "/resources/skills/compile-ldm-policy"
    assert len(profile.skill_dir_sha256[0]) == 64
    assert "dst=/resources,readonly" in command
    assert "dst=/skills" not in command
    assert "/public/verification_record.json" in command
    assert "score_table" not in command
    assert client.config.submission_contract.tool_name == "submit_optimization_policy"
    assert client.config.mcp_servers[-1].server_id == "ldm_policy"


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
