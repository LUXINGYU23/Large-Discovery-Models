"""Compiled policy research and pending-aware selection for the antibody campaign."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import os
from pathlib import Path

import numpy as np

from ldm_tts.async_runtime.policy import PolicyPrior, arrival_generations
from ldm_tts.async_runtime.selectors import ReferenceSelector
from ldm_tts.harness import (DockerPolicyExecutor, HarnessClient, HarnessLimits, HarnessProfile,
                             HarnessSubmissionError, PolicyCapabilityContract, PolicyResearchController,
                             directory_sha256, file_sha256, policy_submission_contract)
from ldm_tts.harness.container import pi_sidecar_command, resolve_container_user
from ldm_tts.harness.pi import PiHarnessConfig, load_pi_guest_runtime, policy_mcp_server
from ldm_tts.harness.ucb_diagnostics import optimization_progress, prediction_feedback, with_feedback
from ldm_tts.optimization.gp import RBFGPSurrogate
from ldm_tts.optimization.pending import SelectionRule

from .experiments import ENERGY_UTILITY

RESOURCES = Path(__file__).resolve().parents[1] / "resources/harness"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
H3_LENGTH = 11
FEATURE_VERSION = "cdrh3-one-hot-v1"
FEATURE_NAMES = tuple(f"h3_{position + 1}_{aa}" for position in range(H3_LENGTH) for aa in AMINO_ACIDS)
POLICY_TURN_SECONDS = 900


def featurize(candidate):
    """Position-major one-hot encoding; all distinct amino acids are equidistant."""
    sequence = candidate.payload["sequence"]
    if len(sequence) != H3_LENGTH or any(aa not in AMINO_ACIDS for aa in sequence):
        raise ValueError("policy features require an 11-residue canonical CDRH3")
    return tuple(float(aa == residue) for residue in sequence for aa in AMINO_ACIDS)


def make_selector():
    """The reference GP maximizes utility; the codec freezes utility = -energy."""
    return ReferenceSelector(featurize=featurize, objective_name=ENERGY_UTILITY,
        feature_version=FEATURE_VERSION, weights_from_policy=True,
        rule=SelectionRule(alpha=1.0, eta=1.0, beta=1.0, z_clip=3.0,
                           pending_handling="kriging_believer"),
        lengthscale=3.0, noise=1e-4, feature_scale_floor=1.0, target_scale_floor=0.01)


class AntibodyPolicyAdapter:
    def __init__(self):
        self.contract = PolicyCapabilityContract(
            task_id="antibody_rfd3", api_version=1,
            enabled_capabilities=("prior_mean@1", "ldm_weights@1"),
            feature_names=FEATURE_NAMES, feature_groups={"cdrh3_one_hot": (0, len(FEATURE_NAMES))},
            mean_clip=3.0, default_alpha=1.0, default_eta=1.0)

    def capability_contract(self):
        return self.contract

    def with_feedback(self, round_input, records):
        return with_feedback(round_input, records)

    def validate_task_execution(self, execution, round_input):
        errors = []
        for name, size in (("history_prior_mean", len(round_input.history_features)),
                           ("query_prior_mean", len(round_input.query_features))):
            values = getattr(execution, name)
            if values.shape != (size,) or not np.isfinite(values).all():
                errors.append(HarnessSubmissionError("/outputs/" + name, "antibody_prior_alignment",
                    "Return one finite standardized prior per authoritative CDRH3 feature row."))
        return tuple(errors)


class AntibodyPolicyPrior(PolicyPrior):
    """Use the same utility and immutable feature contract in turns and applications."""

    version = "2"

    def context(self, generation, location, scale):
        context = super().context(generation, location, scale)
        context["mean_context"].update({"feature_version": FEATURE_VERSION,
            "amino_acid_order": AMINO_ACIDS, "objective": ENERGY_UTILITY,
            "target_definition": "energy_utility = -energy; maximize utility, equivalently minimize energy"})
        context["weight_context"].update({"objective": ENERGY_UTILITY,
            "target_definition": "energy_utility = -energy; maximize utility, equivalently minimize energy",
            "acquisition": "pending-aware upper confidence bound on utility"})
        return context

    def application_context(self, generation, location, scale, *, history, head_ingest_seq, generation_bases, feedback):
        context = super().application_context(generation, location, scale, history=history,
            head_ingest_seq=head_ingest_seq, generation_bases=generation_bases, feedback=feedback)
        rounds = arrival_generations([o.metadata["ingest_seq"] for o in history],
                                     [*generation_bases, head_ingest_seq])
        utilities = [float(o.metrics[ENERGY_UTILITY]) for o in history]
        context["weight_context"].update({
            "optimization_progress": optimization_progress(rounds, utilities),
            "prediction_feedback": prediction_feedback([o.candidate_id for o in history], rounds, utilities,
                                                       feedback, [o.decision_id for o in history]),
        })
        return context


def prepare_policy_turn(selector, round_input, info):
    """Annotate the frozen research snapshot with the objective and plain-GP predictions.

    The future reservoir's q0 is not yet known: these are unconditioned baseline
    predictions on the frozen query set, not the eventual selection distribution.
    """
    rows, _ = selector.training_rows(info["history"])
    surrogate = RBFGPSurrogate(rows, feature_version=selector.feature_version, **selector.gp_settings)
    predictions = [surrogate.predict_record(candidate.candidate_id, selector.vector(candidate).values,
                                            beta=selector.rule.beta) for candidate in info["query"]]
    snapshot = {**round_input.research_snapshot,
        "objective": {"energy_direction": "minimize", "utility": ENERGY_UTILITY,
                      "utility_definition": "-energy", "utility_direction": "maximize"},
        "features": {"version": FEATURE_VERSION, "sequence_length": H3_LENGTH,
                     "amino_acid_order": AMINO_ACIDS, "layout": "position-major one-hot"},
        "selection": {"rule": selector.rule.to_dict(), "weights_from_policy": True,
                      "future_q0": "unknown until proposals are merged and selection occurs",
                      "sparse_history": "the residual prior affects GP means once the GP has at least two observations"},
        "baseline_predictions": [{"candidate_id": p.candidate_id, "utility_mean": p.scalar_mean,
                                  "utility_std": p.scalar_std, "ucb": p.acquisition_score}
                                 for p in predictions]}
    return replace(round_input, research_snapshot=snapshot)


@contextmanager
def open_policy(args, directory, campaign_id, selector):
    """Start a separate policy session and bind it to the asynchronous prior lifecycle."""
    url, model = os.environ.get("LLM_BASE_URL"), os.environ.get("LLM_MODEL_NAME")
    key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not (url and model and key):
        raise ValueError("Policy Pi requires LLM_BASE_URL, LLM_MODEL_NAME and LLM_API_KEY")
    root = Path(directory) / "policy_harness"
    root.mkdir(parents=True, exist_ok=True)
    role, skill = Path("policy/AGENTS.md"), Path("policy/compile-ldm-policy")
    profile = HarnessProfile("policy_architect", Path("/resources") / role,
        (Path("/resources") / skill,), file_sha256(RESOURCES / role), (directory_sha256(RESOURCES / skill),))
    config = PiHarnessConfig(artifact_root=Path("/artifacts"), profiles=(profile,),
        campaign_id=campaign_id, task_id="antibody_rfd3", case_id=f"{args.antigen}:policy", seed=args.seed,
        submission_contract=policy_submission_contract(),
        limits=HarnessLimits(wall_time_seconds=POLICY_TURN_SECONDS,
            tool_call_budgets={"web_search": 0, "web_fetch": 0}),
        base_url=url.removesuffix("/responses"), model=model,
        thinking=os.environ.get("LLM_REASONING_EFFORT", "off"), context7_enabled=False,
        force_first_tool_call=False, guest_runtime=load_pi_guest_runtime("antibody_rfd3", RESOURCES / "image"),
        mcp_servers=(policy_mcp_server(),))
    command = pi_sidecar_command(root, args.sidecar_image, cache_root=args.guest_cache,
                                 mounts=((RESOURCES, "/resources", True),))
    with HarnessClient(command, api_key=key, config=config) as client:
        controller = PolicyResearchController(client=client, adapter=AntibodyPolicyAdapter(),
            executor=DockerPolicyExecutor(args.sidecar_image, container_user=resolve_container_user(None)), root=root,
            recovery_budget=lambda: 2.0 * POLICY_TURN_SECONDS)
        yield AntibodyPolicyPrior(controller,
            prepare_turn=lambda round_input, info: prepare_policy_turn(selector, round_input, info))
