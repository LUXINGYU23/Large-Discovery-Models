"""Lazy resolution of historical package-root imports during migration.

Every group below targets an ``ldm_tts.*`` module so the compatibility layer
keeps working from the built wheel, which ships only the ``ldm_tts`` package.
Do not add ``tasks.*`` modules here: task-specific symbols (for example
``tasks.nanogpt.core.expansion_schema``) are intentionally not re-exported from
the package root, because ``tasks`` is not part of the wheel and the shared
package must not depend on one task's internals. Import such symbols directly
from their owning task module instead.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_GROUPS: tuple[tuple[str, frozenset[str]], ...] = (
    (
        "ldm_tts.optimization.acquisition",
        frozenset(
            {
                "AcquisitionConfig",
                "AcquisitionFunction",
                "PosteriorAcquisition",
                "make_acquisition",
            }
        ),
    ),
    (
        "ldm_tts.optimization.records",
        frozenset(
            {
                "AcquisitionSelector",
                "BOObservation",
                "BOPrediction",
                "BOSelectionResult",
                "FeatureVector",
                "SurrogateEncoder",
                "SurrogateVector",
            }
        ),
    ),
    (
        "ldm_tts.engine.run_store",
        frozenset(
            {
                "BudgetExceededError",
                "BudgetLedger",
                "CampaignEvent",
                "CampaignRuntime",
                "CampaignStatus",
                "unique_run_dir",
            }
        ),
    ),
    (
        "ldm_tts.contracts.candidate",
        frozenset(
            {
                "Candidate",
                "CandidateAdmission",
                "CandidateDomainAdapter",
                "CandidateRejection",
                "RawProposal",
                "ReservoirBuildResult",
                "ReservoirBuilder",
            }
        ),
    ),
    (
        "ldm_tts.data",
        frozenset(
            {
                "AugmentationReport",
                "DataCollectionPaths",
                "DataCollectionSink",
                "EXPERT_JUSTIFICATION_SYSTEM_PROMPT",
                "ExpertJustificationPipeline",
                "ExpertJustifier",
                "JustificationRequest",
                "LDMDataCollectionError",
                "OpenAICompatibleExpert",
                "dataset_info_payload",
                "make_complete_design_ir",
                "make_parameter_edit_ir",
                "normalize_task_id",
                "render_prose",
                "render_record",
                "smallmol_ir_from_prompt_response",
                "smallmol_irs_from_round_record",
                "validate_ir_record",
            }
        ),
    ),
    (
        "ldm_tts.engine.runtime",
        frozenset({"LDMSearchLoopResult", "LDMSearchRoundResult", "run_budgeted_search"}),
    ),
    (
        "ldm_tts.transport.openai",
        frozenset(
            {
                "EndpointCircuitBreaker",
                "EndpointCircuitOpen",
                "EndpointRequestError",
                "OpenAICompatibleProposalClient",
                "call_with_circuit_breaker",
                "chat_completions_url",
                "models_url",
                "preflight_openai_chat",
                "preflight_openai_endpoint",
                "request_openai_chat",
                "request_openai_chat_response",
                "request_openai_models",
            }
        ),
    ),
    (
        "ldm_tts.engine",
        frozenset({"LDMEngine", "LDMEngineConfig", "LDMEngineResult", "LDMEngineState"}),
    ),
    (
        "ldm_tts.contracts.evaluation",
        frozenset(
            {
                "CallableCandidateEvaluator",
                "CandidateEvaluator",
                "EvaluationResult",
                "Observation",
            }
        ),
    ),
    (
        "ldm_tts.engine.expansion",
        frozenset(
            {
                "CallableReservoirExpander",
                "DirectEmissionExpander",
                "ExpansionRequest",
                "ExpansionResult",
                "ReservoirExpander",
            }
        ),
    ),
    (
        "ldm_tts.registration.experiment",
        frozenset(
            {
                "ExperimentContract",
                "ExperimentContractError",
                "ExperimentProfile",
                "load_active_experiment_contract",
                "load_experiment_contract",
                "snapshot_experiment_contract",
                "validate_profile_args",
            }
        ),
    ),
    (
        "ldm_tts.optimization.gp",
        frozenset(
            {
                "GPPrediction",
                "RBFGPSurrogate",
                "RBFGPUCBSelector",
                "SearchObservation",
                "select_max_ucb",
                "select_max_ucb_record",
            }
        ),
    ),
    ("ldm_tts.contracts.evaluation", frozenset({"ObjectiveSet"})),
    (
        "ldm_tts.transport.parsing",
        frozenset(
            {
                "extract_json_object_text",
                "load_json_object",
                "reject_keys",
                "require_allowed_keys",
                "require_list",
                "require_nonnegative_int",
                "require_number",
                "require_str",
                "strip_json_fence",
            }
        ),
    ),
    (
        "ldm_tts.transport",
        frozenset(
            {
                "CallableProposalClient",
                "ProposalClient",
                "ProposalRequest",
                "ProposalResponse",
            }
        ),
    ),
    (
        "ldm_tts.contracts.evaluation",
        frozenset({"as_float", "best_item", "finite_or_none", "is_finite_number", "ranked_items"}),
    ),
    (
        "ldm_tts.contracts",
        frozenset(
            {
                "AcquisitionSpec",
                "CandidateDomainSpec",
                "CandidateSpaceSpec",
                "LDMTaskSpec",
                "ObjectiveSpec",
                "ProposalSearchSpec",
                "ReservoirExpansionSpec",
                "ReservoirSpec",
                "ResponseSpaceSpec",
                "SurrogateSpaceSpec",
            }
        ),
    ),
    (
        "ldm_tts.engine.run_store",
        frozenset({"AtomicJsonLog", "JsonlTrajectoryRecorder", "load_jsonl"}),
    ),
    (
        "ldm_tts.engine.run_store",
        frozenset({"CandidateTraceRecord", "LDMRoundTrace"}),
    ),
)

COMPAT_EXPORTS = frozenset(name for _, names in _EXPORT_GROUPS for name in names)


def resolve(name: str) -> Any:
    """Import and return one historical package-root export on demand."""

    if name == "CandidateSpaceSpec":
        from ldm_tts.contracts import CandidateDomainSpec

        return CandidateDomainSpec
    for module_name, names in _EXPORT_GROUPS:
        if name in names:
            return getattr(import_module(module_name), name)
    raise AttributeError(name)
