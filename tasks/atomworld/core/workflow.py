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
    parser.add_argument("--attempts-per-sample", type=int, default=1)
    parser.add_argument(
        "--limit", type=int, default=0, help="0 evaluates every prepared sample"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--proposal-format", choices=("cif", "operations"), default="cif"
    )
    parser.add_argument(
        "--tools-root",
        type=Path,
        default=DEFAULT_UPSTREAM.parent / "atomworld-agentic-reproduction",
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
    if args.attempts_per_sample < 1 or args.limit < 0:
        parser.error("attempts-per-sample must be positive and limit nonnegative")
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
                "attempt",
                "correct",
                "submitted",
                "candidate_id",
            ],
        )
        writer.writeheader()
        for row in rows:
            for item in row["attempts"]:
                writer.writerow(
                    {
                        "sample_id": row["sample_id"],
                        "action_name": row["action_name"],
                        **{
                            key: item[key]
                            for key in (
                                "attempt",
                                "correct",
                                "submitted",
                                "candidate_id",
                            )
                        },
                    }
                )
    return payload


def run(args, *, client=None):
    if args.mock:
        samples, targets, manifest, outputs = _load_mock()
    else:
        samples, targets, manifest = load_prepared(args.data_dir)
        outputs = None
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
    if client is None:
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
    expander = BlindRefinementExpander(
        samples,
        client,
        attempts_per_sample=args.attempts_per_sample,
        run_dir=run_dir,
        mock=args.mock,
        tools_root=args.tools_root if args.proposal_format == "operations" else None,
    )
    sink = DataCollectionSink.from_env(default_root=run_dir / "ldm_data")
    domain = AtomWorldDomain(samples, sink=sink, mock=args.mock)
    rounds = len(samples) * args.attempts_per_sample
    contract, profile = load_active_experiment_contract()
    contract = contract or load_experiment_contract(TASK_ROOT / "experiment.json")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    # These immutable values prevent resuming against a different task schedule or endpoint policy.
    immutable = {
        "samples": samples,
        "attempts_per_sample": args.attempts_per_sample,
        "dataset_manifest": manifest,
        "mock": args.mock,
        "proposal_format": args.proposal_format,
        "tool_source_sha256": {
            str(p.relative_to(args.tools_root)): sha256_file(p)
            for p in sorted(args.tools_root.rglob("*.py"))
        }
        if args.proposal_format == "operations"
        else {},
        "llm_model_name": args.llm_model_name,
        "llm_max_tokens": args.llm_max_tokens,
        "llm_temperature": args.llm_temperature,
        "llm_extra_body_json": args.llm_extra_body_json,
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
        if not args.mock and not completed and hasattr(client, "preflight"):
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
                        "llm_requests": 0 if args.mock else rounds + 1,
                        "mock_model_requests": rounds if args.mock else 0,
                        "endpoint_preflights": 0 if args.mock else 2,
                        "geometry_tool_calls": rounds
                        if args.proposal_format == "operations"
                        else 0,
                        "outer_iterations": rounds + 1,
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
