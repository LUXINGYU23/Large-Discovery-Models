"""Task-local assembly around the shared LDM engine."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from ldm_tts.contracts import (
    AcquisitionSpec,
    CandidateDomainSpec,
    LDMTaskSpec,
    ObjectiveSpec,
    ProposalSearchSpec,
    ReservoirExpansionSpec,
    ReservoirSpec,
    ResponseSpaceSpec,
)
from ldm_tts.data import DataCollectionSink
from ldm_tts.engine import LDMEngine
from ldm_tts.engine.run_store import CampaignRuntime
from ldm_tts.transport import ProposalClient
from tasks.synthonbench.core.candidate import SynthonCandidateDomain
from tasks.synthonbench.core.catalog import SynthonProposalCatalog
from tasks.synthonbench.core.constants import OBJECTIVE_NAME, TASK_ID
from tasks.synthonbench.core.evaluator import OfficialSynthonEvaluator
from tasks.synthonbench.core.ldm_selector import AcquisitionTiltedSelector
from tasks.synthonbench.core.nystrom_encoder import SynthonNystromEncoder
from tasks.synthonbench.core.prompting import (
    DEFAULT_PROMPT_POLICY,
    validate_prompt_policy,
)
from tasks.synthonbench.core.proposals import SynthonBenchProposalExpander
from tasks.synthonbench.core.tanimoto_gp import TanimotoGPUCBConfig, SynthonTanimotoGPUCBSelector


@dataclass(frozen=True)
class CampaignComponentOptions:
    """Dependencies and immutable choices for one SynthonBench campaign."""

    client: ProposalClient
    official_task: object
    runtime: CampaignRuntime
    sink: DataCollectionSink
    target: str
    proposal_samples: int
    bo_pool_size: int
    proposal_max_workers: int
    slate_size: int
    reaction_allocation: str
    selection_seed: int
    fingerprint_bits: int
    gp_landmarks: int
    gp_kernel_jitter: float
    gp_signal_std: float
    gp_mean_std: float
    gp_observation_noise_std: float
    gp_reaction_weight: float
    acquisition_beta: float
    alpha: float
    eta: float
    z_clip: float
    prompt_policy: str = DEFAULT_PROMPT_POLICY
    before_requests: Callable[[int], None] | None = None

    def __post_init__(self) -> None:
        if self.proposal_samples <= self.bo_pool_size:
            raise ValueError("proposal_samples must exceed bo_pool_size")
        if self.proposal_max_workers < 1 or self.slate_size < 1:
            raise ValueError("worker and slate sizes must be positive")
        for name in ("acquisition_beta", "alpha", "eta"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        validate_prompt_policy(self.prompt_policy)


@dataclass(frozen=True)
class CampaignComponents:
    """All task adapters connected to a single shared LDMEngine."""

    task_spec: LDMTaskSpec
    domain: SynthonCandidateDomain
    encoder: SynthonNystromEncoder
    selector: AcquisitionTiltedSelector
    evaluator: OfficialSynthonEvaluator
    engine: LDMEngine


def build_campaign_components(options: CampaignComponentOptions) -> CampaignComponents:
    """Assemble task-local components without reading global state or credentials."""

    official_task = options.official_task
    allowed_reactions = tuple(official_task.allowed_reactions)
    encoder = SynthonNystromEncoder(
        official_task.space,
        allowed_reactions,
        landmark_count=options.gp_landmarks,
        seed=options.selection_seed,
        fingerprint_bits=options.fingerprint_bits,
        kernel_jitter=options.gp_kernel_jitter,
        reaction_weight=options.gp_reaction_weight,
    )
    selector = build_synthon_selector(
        encoder=encoder,
        selection_seed=options.selection_seed,
        gp_signal_std=options.gp_signal_std,
        gp_mean_std=options.gp_mean_std,
        gp_observation_noise_std=options.gp_observation_noise_std,
        acquisition_beta=options.acquisition_beta,
        alpha=options.alpha,
        eta=options.eta,
        z_clip=options.z_clip,
        bo_pool_size=options.bo_pool_size,
        proposal_samples=options.proposal_samples,
    )
    task_spec = build_synthon_task_spec(
        encoder=encoder,
        acquisition=selector.describe(),
        proposal_samples=options.proposal_samples,
        bo_pool_size=options.bo_pool_size,
        proposal_max_workers=options.proposal_max_workers,
        slate_size=options.slate_size,
        reaction_allocation=options.reaction_allocation,
        prompt_policy=options.prompt_policy,
    )
    domain = SynthonCandidateDomain(official_task.space, allowed_reactions, options.target, options.sink)
    catalog = SynthonProposalCatalog(official_task.space, allowed_reactions=allowed_reactions,
                                    slate_size=options.slate_size, seed=options.selection_seed,
                                    reaction_allocation=options.reaction_allocation)
    expander = SynthonBenchProposalExpander(options.client, domain, catalog, target=options.target,
                                            before_requests=options.before_requests,
                                            max_workers=options.proposal_max_workers,
                                            prompt_policy=options.prompt_policy)
    evaluator = OfficialSynthonEvaluator(official_task)
    engine = LDMEngine(task_spec=task_spec, expander=expander, candidate_domain=domain,
                       evaluator=evaluator, runtime=options.runtime, selector=selector,
                       surrogate_encoder=encoder)
    return CampaignComponents(task_spec, domain, encoder, selector, evaluator, engine)


def build_synthon_task_spec(
    *,
    encoder: SynthonNystromEncoder,
    acquisition: AcquisitionSpec,
    proposal_samples: int,
    bo_pool_size: int,
    proposal_max_workers: int,
    slate_size: int,
    reaction_allocation: str,
    prompt_policy: str,
) -> LDMTaskSpec:
    """Describe the exact finite tuple, response, surrogate, and LDM policy contract."""

    return LDMTaskSpec(
        task=TASK_ID,
        candidate_domain=CandidateDomainSpec(
            name="Source-pinned synthon tuples",
            kind="reaction_synthon_tuple",
            dimension=None,
            representation="One reaction ID and one ordered valid synthon ID per official reaction slot.",
        ),
        objectives=(ObjectiveSpec(OBJECTIVE_NAME, "maximize", "Official fixed-oracle SynthonBench utility."),),
        response_spaces=(_response_space(),),
        acquisition=acquisition,
        reservoir=_reservoir_spec(proposal_samples, proposal_max_workers),
        surrogate=encoder.describe(),
        proposal_search=ProposalSearchSpec(
            name="parallel_independent_requests",
            breadth=proposal_samples,
            evaluation_policy="empirical_q0_maintained_acquisition_tilted",
            parameters={"max_workers": proposal_max_workers, "one_candidate_per_request": True},
        ),
        metadata={
            "proposal_samples": proposal_samples,
            "bo_pool_size": bo_pool_size,
            "slate_size": slate_size,
            "reaction_allocation": reaction_allocation,
            "prompt_policy": prompt_policy,
            "sampling_mode": "local_concurrent_independent_requests",
        },
    )


def build_synthon_selector(
    *,
    encoder: SynthonNystromEncoder,
    selection_seed: int,
    gp_signal_std: float,
    gp_mean_std: float,
    gp_observation_noise_std: float,
    acquisition_beta: float,
    alpha: float,
    eta: float,
    z_clip: float,
    bo_pool_size: int,
    proposal_samples: int,
) -> AcquisitionTiltedSelector:
    base = SynthonTanimotoGPUCBSelector(
        objective_name=OBJECTIVE_NAME,
        feature_dimension=encoder.dimension,
        feature_version=encoder.version,
        config=TanimotoGPUCBConfig(
            beta=acquisition_beta,
            signal_std=gp_signal_std,
            mean_std=gp_mean_std,
            observation_noise_std=gp_observation_noise_std,
        ),
    )
    return AcquisitionTiltedSelector(base, alpha=alpha, eta=eta, z_clip=z_clip,
                                     seed=selection_seed, pool_size=bo_pool_size,
                                     proposal_sample_count=proposal_samples)


def _response_space() -> ResponseSpaceSpec:
    return ResponseSpaceSpec(
        name="synthon_tuple_json",
        output_kind="json_object",
        parser="tasks.synthonbench.core.proposal_parsing:parse_synthon_response",
        description="One source-valid tuple chosen only from a supplied finite public synthon slate.",
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["reaction_id", "synthon_ids"],
            "properties": {"reaction_id": {"type": "string"}, "synthon_ids": {"type": "array"}},
        },
    )


def _reservoir_spec(proposal_samples: int, max_workers: int) -> ReservoirSpec:
    return ReservoirSpec(
        name="synthon_tuple_reservoir",
        expansions=(ReservoirExpansionSpec(
            name="independent_synthon_tuple_proposals",
            action_kind="emit_candidate",
            response_space="synthon_tuple_json",
            produces_candidates=True,
            description=f"Launch {proposal_samples} independent one-candidate requests with at most {max_workers} workers.",
        ),),
        candidate_validator="tasks.synthonbench.core.candidate:SynthonCandidateDomain",
        deduplication_key="official SynthonBench product_id",
        max_size=proposal_samples,
    )


__all__ = [
    "CampaignComponentOptions",
    "CampaignComponents",
    "build_campaign_components",
    "build_synthon_selector",
    "build_synthon_task_spec",
]
