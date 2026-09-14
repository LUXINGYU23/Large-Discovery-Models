"""One shared LDM campaign over a fixed schedule of benchmark questions and revisions."""

from __future__ import annotations

import argparse
import csv
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
)
from tasks.atomworld.core.data import (
    DEFAULT_UPSTREAM,
    TASK_ROOT,
    load_prepared,
    sha256_file,
    write_json,
)
from tasks.atomworld.core.evaluator import AtomWorldEvaluator, load_official_evaluator
from tasks.atomworld.core.proposals import AtomWorldDomain, BlindRefinementExpander
from tasks.atomworld.core.task_spec import describe_ldm_task


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=Path(os.environ.get("ATOMWORLD_UPSTREAM_ROOT", DEFAULT_UPSTREAM)),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs/atomworld"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument(
        "--proposal-mode",
        choices=("mock", "real", "live", "none", "callable", "openai"),
        default=None,
    )
    parser.add_argument("--attempts-per-sample", type=int, default=1)
    parser.add_argument(
        "--search-method",
        choices=("llm", "harness", "blind_harness_compiled"),
        default="llm",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Pilot alias for scheduled attempts per sample",
    )
    parser.add_argument("--campaign-index", type=int, default=0)
    parser.add_argument(
        "--initialization-mode",
        choices=("shared_start", "independent"),
        default="shared_start",
    )
    parser.add_argument("--service-retry-allowance", type=int, default=3)
    parser.add_argument(
        "--harness-profile", nargs="+", action="extend", default=None
    )
    parser.add_argument("--harness-sidecar-image", default="ldm-pi-harness:latest")
    parser.add_argument(
        "--harness-cache-dir", type=Path, default=Path.home() / ".cache/ldm-gondolin"
    )
    parser.add_argument("--harness-docker-host", default="")
    parser.add_argument("--harness-container-user", default=None)
    parser.add_argument(
        "--harness-thinking",
        choices=("off", "minimal", "low", "medium", "high", "xhigh", "max"),
        default="high",
    )
    parser.add_argument("--harness-wall-time-seconds", type=int, default=1800)
    parser.add_argument("--harness-response-timeout", type=float, default=2100)
    parser.add_argument("--harness-recovery-seconds", type=float, default=0)
    parser.add_argument("--harness-max-submission-attempts", type=int, default=3)
    parser.add_argument(
        "--harness-tool-budget",
        nargs="+",
        action="extend",
        default=None,
    )
    default_harness_tool_budgets = [
        "web_search=8",
        "fetch_content=16",
        "get_search_content=16",
        "resolve-library-id=4",
        "query-docs=8",
        "bash=64",
    ]
    parser.add_argument("--harness-mcp-config", type=Path, default=None)
    parser.add_argument("--policy-runner-image", default="ldm-pi-harness:latest")
    parser.add_argument(
        "--limit", type=int, default=0, help="0 evaluates every prepared sample"
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        help="Select one exact public sample ID for a pilot case",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--proposal-format", choices=("cif", "operations"), default="cif"
    )
    parser.add_argument(
        "--tools-root",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--llm-url",
        default=os.environ.get(
            "LLM_BASE_URL",
            os.environ.get("LDM_LLM_URL", os.environ.get("OPENAI_BASE_URL", "")),
        ),
    )
    parser.add_argument(
        "--llm-model-name",
        default=os.environ.get(
            "LLM_MODEL_NAME",
            os.environ.get("LDM_LLM_MODEL", os.environ.get("OPENAI_MODEL", "")),
        ),
    )
    parser.add_argument("--llm-max-tokens", type=int, default=8192)
    parser.add_argument("--llm-timeout", type=float, default=180.0)
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument(
        "--llm-extra-body-json",
        default="{}",
        help="Provider reasoning controls; never put credentials here",
    )
    args = parser.parse_args(argv)
    # The shared config runner repeats list-valued flags. Preserve all entries
    # both in that form and in a grouped, interactive command-line invocation.
    if args.harness_profile is None:
        args.harness_profile = ["geometry_research", "structure_audit"]
    if args.harness_tool_budget is None:
        args.harness_tool_budget = default_harness_tool_budgets
    if args.proposal_mode in {"mock", "callable"}:
        args.mock = True
    if args.resume_from is not None:
        args.out_dir, args.resume = args.resume_from, True
    elif args.run_name is not None:
        if Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
            parser.error("run-name must be a single safe path component")
        args.out_dir = args.out_dir / args.run_name
    if args.iterations is not None:
        args.attempts_per_sample = args.iterations
    if args.attempts_per_sample < 1 or args.limit < 0:
        parser.error("attempts-per-sample must be positive and limit nonnegative")
    if args.campaign_index < 0 or args.service_retry_allowance < 0:
        parser.error("campaign-index and service-retry-allowance must be nonnegative")
    if args.tools_root is not None:
        parser.error(
            "--tools-root is obsolete: bounded geometry tools now ship inside this task"
        )
    if args.search_method != "llm":
        from ldm_tts.harness import parse_tool_call_budgets
        from tasks.atomworld.core.harness import harness_profiles

        try:
            harness_profiles(args.harness_profile)
            if "policy_architect" in args.harness_profile:
                raise ValueError(
                    "policy_architect is reserved for the independent policy session"
                )
            parse_tool_call_budgets(
                args.harness_tool_budget,
                excluded_tools=("submit_answer", "submit_optimization_policy"),
            )
        except ValueError as exc:
            parser.error(str(exc))
        if (
            min(
                args.harness_wall_time_seconds,
                args.harness_response_timeout,
                args.harness_max_submission_attempts,
            )
            <= 0
            or args.harness_recovery_seconds < 0
        ):
            parser.error(
                "Harness deadlines and submission count must be positive; recovery nonnegative"
            )
        if args.proposal_format != "cif":
            parser.error(
                "Harness submits CIF answers; execute geometry operations inside its guest"
            )
    if not args.mock and args.data_dir is None:
        parser.error(
            "--data-dir is required for real runs; prepare the released data first"
        )
    try:
        extra = json.loads(args.llm_extra_body_json)
    except ValueError:
        parser.error("llm-extra-body-json must be a JSON object")
    if not isinstance(extra, dict):
        parser.error("llm-extra-body-json must be a JSON object")
    if any(
        any(
            word in key.lower()
            for word in ("token_key", "api_key", "authorization", "password", "secret")
        )
        for key in extra
    ):
        parser.error(
            "Credentials must be supplied through environment variables, not extra-body JSON"
        )
    if args.mock and args.proposal_format != "cif":
        parser.error(
            "Mock fixture uses CIF output; operations mode is verified with the real local geometry tool"
        )
    return args


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


def project_result(
    run_dir: Path,
    samples: list[dict],
    attempts: int,
    result,
    *,
    mock: bool,
    dataset_manifest: dict,
    proposal_format="cif",
    search_method="llm",
) -> dict:
    # Never select engine.best: it is a hidden-answer oracle. Fixed last-submission reporting.
    by_key = {
        observation.canonical_key: observation.evaluation
        for observation in result.state.observations
    }
    rows = []
    for sample_index, sample in enumerate(samples):
        details = []
        for attempt in range(attempts):
            ordinal = sample_index * attempts + attempt
            path = run_dir / "attempts" / f"{ordinal:06d}.json"
            raw = json.loads(path.read_text()) if path.exists() else None
            evaluation = by_key.get(raw["canonical_key"]) if raw else None
            details.append(
                {
                    "attempt": attempt + 1,
                    "submitted": raw is not None,
                    "correct": bool(
                        evaluation and evaluation.metrics.get("correct") == 1
                    ),
                    "candidate_id": evaluation.candidate_id if evaluation else None,
                    "error": None if evaluation else "missing_or_rejected_submission",
                }
            )
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "action_name": sample["action_name"],
                "one_shot_correct": details[0]["correct"],
                "final_correct": details[-1]["correct"],
                "oracle_any_attempt_correct": any(item["correct"] for item in details),
                "attempts": details,
            }
        )
    count = len(rows)
    macro = {}
    for action in sorted({row["action_name"] for row in rows}):
        group = [row for row in rows if row["action_name"] == action]
        macro[action] = {
            "n": len(group),
            "one_shot_accuracy": sum(r["one_shot_correct"] for r in group) / len(group),
            "final_accuracy": sum(r["final_correct"] for r in group) / len(group),
        }
    payload = {
        "task": "atomworld",
        "mode": "mock" if mock else "real",
        "proposal_format": proposal_format,
        "search_method": search_method,
        "selection_protocol": "final_submission",
        "dataset_kind": dataset_manifest["dataset_kind"],
        "paper_split_verified": dataset_manifest.get("paper_split_verified", False),
        "sample_count": count,
        "attempts_per_sample": attempts,
        "one_shot_accuracy": sum(row["one_shot_correct"] for row in rows) / count,
        "extended_final_accuracy": sum(row["final_correct"] for row in rows) / count
        if attempts > 1
        else None,
        "oracle_any_attempt_accuracy_diagnostic": sum(
            row["oracle_any_attempt_correct"] for row in rows
        )
        / count,
        "final_selection": "last scheduled answer; missing/rejected final answers count as incorrect",
        "official_paper_score_claimed": False,
        "by_action": macro,
        "samples": rows,
        "engine_summary": result.summary,
        "warning": "Synthetic fixture only; not an LLM benchmark score"
        if mock
        else "Released local subset; not a verified paper test split",
    }
    write_json(run_dir / "result.json", payload)
    with (run_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "action_name",
                "round",
                "attempt",
                "correct",
                "submitted",
                "candidate_id",
            ],
        )
        writer.writeheader()
        for sample_index, row in enumerate(rows):
            for item in row["attempts"]:
                writer.writerow(
                    {
                        "sample_id": row["sample_id"],
                        "action_name": row["action_name"],
                        "round": sample_index * attempts + item["attempt"] - 1,
                        **{
                            key: item[key]
                            for key in (
                                "attempt",
                                "correct",
                                "submitted",
                                "candidate_id",
                            )
                        },
                        "correct": int(item["correct"]),
                    }
                )
    return payload


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
    harness_mode = args.search_method != "llm"
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
                api_key=os.environ.get(
                    "LLM_API_KEY",
                    os.environ.get(
                        "LDM_LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "")
                    ),
                ),
                timeout_seconds=args.llm_timeout,
                max_tokens=args.llm_max_tokens,
                temperature=args.llm_temperature,
                max_retries=0,
                extra_body=extra,
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
    rounds = len(samples) * args.attempts_per_sample
    contract, profile = load_active_experiment_contract()
    contract = contract or load_experiment_contract(TASK_ROOT / "experiment.json")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(
        evaluations_per_round=1,
        proposal_samples=len(args.harness_profile) if harness_mode else 1,
        harness_candidates_per_session=1 if harness_mode else 0,
    )
    # These immutable values prevent resuming against a different task schedule or endpoint policy.
    immutable = {
        "samples": samples,
        "attempts_per_sample": args.attempts_per_sample,
        "dataset_manifest": manifest,
        "mock": args.mock,
        "proposal_format": args.proposal_format,
        "tool_source_sha256": {
            str(p.relative_to(TASK_ROOT)): sha256_file(p)
            for p in sorted(
                (TASK_ROOT / "resources/harness/image/atomworld_tools").glob("*.py")
            )
        }
        if args.proposal_format == "operations"
        else {},
        "llm_model_name": args.llm_model_name,
        "llm_max_tokens": args.llm_max_tokens,
        "llm_temperature": args.llm_temperature,
        "llm_extra_body_json": args.llm_extra_body_json,
        "search_method": args.search_method,
        "campaign_index": args.campaign_index,
        "service_retry_allowance": args.service_retry_allowance,
    }
    if harness_mode:
        from ldm_tts.harness import canonical_sha256

        resource_root = TASK_ROOT / "resources/harness"
        immutable["harness_resources_sha256"] = canonical_sha256(
            {
                str(path.relative_to(resource_root)): sha256_file(path)
                for path in sorted(resource_root.rglob("*"))
                if path.is_file() and "__pycache__" not in path.parts
            }
        )
        immutable["harness_settings"] = {
            key: value
            for key, value in config.items()
            if key.startswith(("harness_", "policy_"))
        }
    immutable_path = run_dir / "schedule.json"
    if args.resume and immutable_path.exists():
        if json.loads(immutable_path.read_text()) != immutable:
            raise ValueError(
                "Resume requires the same dataset, schedule, model and generation settings"
            )
    else:
        write_json(immutable_path, immutable)
    write_json(run_dir / "dataset_manifest.json", manifest)
    runtime_ref = []

    def initialize(runtime):
        runtime_ref.append(runtime)
        expander.runtime = runtime
        checkpoint = runtime.load_checkpoint() if args.resume else None
        completed = bool(checkpoint and checkpoint.get("next_round", 0) >= rounds)
        if (
            not args.mock
            and not harness_mode
            and not completed
            and hasattr(client, "preflight")
        ):
            runtime.consume("endpoint_preflights")
            try:
                report = client.preflight()
                write_json(run_dir / "endpoint_preflight.json", report)
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
                    reservoir_size=1,
                    batch_size=1,
                    target_observations=rounds,
                    max_evaluation_attempts=rounds,
                    max_empty_reservoir_rounds=rounds + 1,
                    extra_limits={
                        "proposal_attempts": rounds,
                        **(
                            {}
                            if harness_mode
                            else {
                                "llm_requests": 0
                                if args.mock
                                else rounds + args.service_retry_allowance
                            }
                        ),
                        "mock_model_requests": rounds if args.mock else 0,
                        "endpoint_preflights": 0
                        if args.mock or harness_mode
                        else args.service_retry_allowance + 1,
                        "geometry_tool_calls": rounds
                        if args.proposal_format == "operations"
                        else 0,
                        "outer_iterations": rounds + args.service_retry_allowance,
                        "harness_turns": (rounds + args.service_retry_allowance)
                        * len(args.harness_profile)
                        if harness_mode
                        else 0,
                        "policy_harness_turns": rounds
                        if args.search_method == "blind_harness_compiled"
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
                ),
            ),
            CampaignRecipe(describe_ldm_task(args), expander, domain, evaluator),
        )
    except EndpointRequestError:
        if runtime_ref:
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
    args = parse_args(argv)
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
                "one_shot_accuracy": result.projected["one_shot_accuracy"],
                "extended_final_accuracy": result.projected["extended_final_accuracy"],
                "official_paper_score_claimed": False,
            },
            indent=2,
        )
    )
    return 0
