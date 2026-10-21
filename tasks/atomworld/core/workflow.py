"""One shared LDM campaign over a fixed schedule of benchmark questions and revisions."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ldm_tts.campaign import (
    CampaignBudget,
    CampaignRecipe,
    CampaignRequest,
    run_campaign,
)
from ldm_tts.data import DataCollectionSink
from ldm_tts.engine.run_store import unique_run_dir
from ldm_tts.harness import HarnessError
from ldm_tts.registration.experiment import (
    load_active_experiment_contract,
    load_experiment_contract,
)
from ldm_tts.transport import CallableProposalClient
from ldm_tts.transport.openai import (
    EndpointCircuitBreaker,
    EndpointRequestError,
    OpenAICompatibleProposalClient,
    generation_body,
)
from tasks.atomworld.core.data import (
    TASK_ROOT,
    load_prepared,
    write_json,
)
from tasks.atomworld.core.evaluator import AtomWorldEvaluator, load_official_evaluator
from tasks.atomworld.core.proposals import AtomWorldDomain, BlindRefinementExpander
from tasks.atomworld.core.task_spec import describe_ldm_task
from tasks.atomworld.core.methods import LDM_METHODS, HARNESS_METHODS
from .provenance import bind_schedule
from .reporting import project_result
from . import workflow_args
from tasks.atomworld.core.selection import AtomWorldLDMSelector, GeometryEncoder


def _provider_api_key():
    return os.environ.get(
        "LLM_API_KEY",
        os.environ.get("LDM_LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
    )


def _harness_provider_preflight(args):
    if not args.llm_url or not args.llm_model_name:
        raise EndpointRequestError(
            "Set LLM_BASE_URL and LLM_MODEL_NAME or the corresponding CLI options"
        )
    if not _provider_api_key():
        raise EndpointRequestError(
            "Set LLM_API_KEY, LDM_LLM_API_KEY, or OPENAI_API_KEY for Harness proposals"
        )
    client = OpenAICompatibleProposalClient(
        url=args.llm_url,
        model=args.llm_model_name,
        api_key=_provider_api_key(),
        timeout_seconds=args.harness_response_timeout,
        max_tokens=args.llm_max_tokens,
        temperature=args.llm_temperature,
        max_retries=0,
        wire_api="responses",
        extra_body=generation_body(wire_api=args.llm_wire_api, reasoning=args.llm_reasoning,
                                   extra_body=json.loads(args.llm_extra_body_json)),
    )
    return {"backend": "harness", "wire_api": "responses", **client.preflight()}


def _load_mock():
    fixture = json.loads((TASK_ROOT / "resources/mock_fixture.json").read_text())
    samples = fixture["public"]
    targets = {row["sample_id"]: row["target_cif"] for row in fixture["private"]}
    outputs = fixture["mock_outputs"]
    return (
        samples,
        targets,
        {
            "dataset_kind": "synthetic_mock",
            "sample_count": len(samples),
            "paper_split_verified": False,
        },
        outputs,
    )


def run(args, *, client=None, harness_client_factory=None, policy_executor=None):
    if args.mock:
        samples, targets, manifest, outputs = _load_mock()
    else:
        samples, targets, manifest = load_prepared(args.data_dir)
        outputs = None
    if args.sample_id is not None:
        samples = [
            sample for sample in samples if sample["sample_id"] == args.sample_id
        ]
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        raise ValueError("Dataset has no selected samples")
    run_dir = (
        args.out_dir.resolve()
        if args.resume
        else unique_run_dir(args.out_dir).resolve()
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    harness_mode = args.search_method in HARNESS_METHODS
    ldm_mode = args.search_method in LDM_METHODS
    proposal_samples = len(args.harness_profile) if harness_mode else args.proposal_samples if ldm_mode else 1
    harness_provider_preflight_required = harness_mode and (
        not args.mock or harness_client_factory is None
    )
    if client is None and not harness_mode:
        if args.mock:
            client = CallableProposalClient(
                lambda request: outputs[request.metadata["sample_id"]][
                    min(
                        request.metadata["round_idx"] % args.attempts_per_sample,
                        len(outputs[request.metadata["sample_id"]]) - 1,
                    )
                ]
            )
        else:
            if not args.llm_url or not args.llm_model_name:
                raise ValueError(
                    "Set LLM_BASE_URL and LLM_MODEL_NAME or the corresponding CLI options"
                )
            extra = json.loads(args.llm_extra_body_json)
            if not isinstance(extra, dict):
                raise ValueError("llm-extra-body-json must be a JSON object")
            client = OpenAICompatibleProposalClient(
                url=args.llm_url,
                model=args.llm_model_name,
                api_key=_provider_api_key(),
                timeout_seconds=args.llm_timeout,
                max_tokens=args.llm_max_tokens,
                temperature=args.llm_temperature,
                max_retries=0,
                wire_api=args.llm_wire_api,
                extra_body=generation_body(wire_api=args.llm_wire_api, reasoning=args.llm_reasoning, extra_body=extra),
                breaker=EndpointCircuitBreaker(failure_threshold=1),
            )
    official = None if args.mock else load_official_evaluator(args.upstream_root)
    evaluator = AtomWorldEvaluator(targets, run_dir, official=official, mock=args.mock)
    if harness_mode:
        from tasks.atomworld.core.harness import AtomWorldHarnessExpander, make_client

        expander = AtomWorldHarnessExpander(
            samples,
            args,
            run_dir,
            client_factory=harness_client_factory or make_client,
            policy_executor=policy_executor,
        )
    elif ldm_mode:
        from tasks.atomworld.core.ldm_proposals import AtomWorldLDMExpander

        expander = AtomWorldLDMExpander(samples, args, run_dir, client)
    else:
        expander = BlindRefinementExpander(
            samples,
            client,
            attempts_per_sample=args.attempts_per_sample,
            run_dir=run_dir,
            mock=args.mock,
            operations=args.proposal_format == "operations",
        )
    sink = DataCollectionSink.from_env(default_root=run_dir / "ldm_data")
    domain = AtomWorldDomain(samples, sink=sink, mock=args.mock)
    selector = AtomWorldLDMSelector(args, samples, run_dir, expander=expander) if ldm_mode else None
    encoder = GeometryEncoder(samples, mock=args.mock) if ldm_mode else None
    rounds = len(samples) * args.attempts_per_sample
    contract, profile = load_active_experiment_contract()
    contract = contract or load_experiment_contract(TASK_ROOT / "experiment.json")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if not k.startswith("_")}
    config.update(
        evaluations_per_round=1,
        proposal_samples=proposal_samples,
        harness_candidates_per_session=1 if harness_mode else 0,
    )
    bind_schedule(args, run_dir, samples, manifest, config,
                  proposal_samples=proposal_samples, encoder=encoder,
                  ldm_mode=ldm_mode, harness_mode=harness_mode)
    write_json(run_dir / "dataset_manifest.json", manifest)
    runtime_ref = []

    def initialize(runtime):
        runtime_ref.append(runtime)
        expander.runtime = runtime
        checkpoint = runtime.load_checkpoint() if args.resume else None
        completed = bool(checkpoint and checkpoint.get("next_round", 0) >= rounds)
        if not completed and (
            harness_provider_preflight_required
            or (not args.mock and hasattr(client, "preflight"))
        ):
            runtime.consume("endpoint_preflights")
            try:
                report = (
                    _harness_provider_preflight(args)
                    if harness_provider_preflight_required
                    else client.preflight()
                )
                write_json(run_dir / "endpoint_preflight.json", report)
                runtime.record("endpoint_preflight_succeeded", report)
            except EndpointRequestError:
                runtime.pause(
                    "paused_endpoint",
                    phase="preflight",
                    message="Endpoint preflight failed; configure service and resume this run",
                )
                raise

    try:
        campaign = run_campaign(
            CampaignRequest(
                run_dir=run_dir,
                budget=CampaignBudget(
                    rounds=rounds,
                    reservoir_size=proposal_samples if ldm_mode else 1,
                    batch_size=1,
                    target_successful_evaluations=rounds,
                    max_evaluation_attempts=rounds,
                    max_empty_reservoir_rounds=rounds + 1,
                    extra_limits={
                        "proposal_attempts": rounds * proposal_samples + args.service_retry_allowance,
                        **(
                            {}
                            if harness_mode
                            else {
                                "llm_requests": 0
                                if args.mock
                                else rounds * proposal_samples + args.service_retry_allowance
                            }
                        ),
                        "mock_model_requests": rounds * proposal_samples if args.mock else 0,
                        "endpoint_preflights": 0
                        if args.mock and not harness_provider_preflight_required
                        else args.service_retry_allowance + 1,
                        "geometry_tool_calls": rounds
                        if args.proposal_format == "operations"
                        else 0,
                        "outer_iterations": rounds + args.service_retry_allowance,
                        "harness_turns": (rounds + args.service_retry_allowance)
                        * (len(args.harness_profile) + int(args._research_strategy.requires_main))
                        if harness_mode
                        else 0,
                        "main_research_turns": rounds + args.service_retry_allowance if args._research_strategy.requires_main else 0,
                        "policy_harness_turns": rounds
                        if args.search_method in ("harness_public_audit", "ldm_harness_compiled")
                        else 0,
                    },
                ),
                config=config,
                resume=args.resume,
                contract_snapshot=contract.to_dict(),
                contract_sha256=contract.digest,
                contract_profile=profile,
                runtime_hook=initialize,
                artifact_projector=lambda runtime, result: project_result(
                    runtime.run_dir,
                    samples,
                    args.attempts_per_sample,
                    result,
                    mock=args.mock,
                    dataset_manifest=manifest,
                    proposal_format=args.proposal_format,
                    search_method=args.search_method,
                    runtime=runtime,
                ),
            ),
            CampaignRecipe(describe_ldm_task(args), expander, domain, evaluator,
                           selector=selector, surrogate_encoder=encoder),
        )
    except EndpointRequestError:
        if runtime_ref:
            status_path = runtime_ref[0].run_dir / "status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            if not (
                status.get("status") == "paused_endpoint"
                and status.get("phase") == "preflight"
            ):
                runtime_ref[0].pause(
                    "paused_endpoint",
                    phase="proposal",
                    message="Endpoint unavailable; resume the same run after recovery",
                )
        raise
    except HarnessError:
        if runtime_ref:
            runtime_ref[0].pause(
                "paused_harness",
                phase="proposal",
                message="Harness unavailable; resume the same run to continue persistent sessions",
            )
        raise
    finally:
        if harness_mode:
            expander.close()
    return campaign


def main(argv=None):
    args = workflow_args.parse_args(argv)
    if args.dry_run:
        print(json.dumps(describe_ldm_task(args).to_dict(), indent=2))
        return 0
    result = run(args)
    print(
        json.dumps(
            {
                "task": "atomworld",
                "run_dir": str(result.runtime.run_dir),
                "mode": "mock" if args.mock else "real",
                "schedule_completed": result.projected["schedule_completed"],
                "one_shot_accuracy": result.projected["one_shot_accuracy"],
                "extended_final_accuracy": result.projected["extended_final_accuracy"],
                "official_paper_score_claimed": False,
            },
            indent=2,
        )
    )
    return 0 if result.projected["schedule_completed"] else 1
