"""Workflow assembly for one Iron Mind LDM campaign."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from ldm_tts.contracts import LDMTaskSpec
from ldm_tts.data import DataCollectionSink
from ldm_tts.engine import LDMEngineConfig
from ldm_tts.engine.run_store import CampaignRuntime, unique_run_dir
from ldm_tts.optimization.gp import RBFGPUCBSelector
from ldm_tts.registration.experiment import (
    load_active_experiment_contract,
    load_experiment_contract,
    snapshot_experiment_contract,
)
from ldm_tts.transport import CallableProposalClient, ProposalClient
from ldm_tts.transport.openai_http import EndpointRequestError

from tasks.iron_mind.core.data import FrozenReactionTable
from tasks.iron_mind.core.dependencies import load_pinned_reaction_table
from tasks.iron_mind.core.factory import (
    OBJECTIVE_NAME,
    CampaignComponentOptions,
    build_campaign_components,
    build_reaction_task_spec,
)
from tasks.iron_mind.core.mock import load_mock_table
from tasks.iron_mind.core.mock import mock_response as _mock_response
from tasks.iron_mind.core.proposals import build_deepseek_reaction_client
from tasks.iron_mind.core.reporting import write_campaign_reports
from tasks.iron_mind.core.schema import ReactionDatasetSchema, load_reaction_schemas
from tasks.iron_mind.core.surrogate import ReactionOneHotEncoder
from tasks.iron_mind.core.workflow_support import (
    derived_budget,
    jsonable_args,
    load_campaign_state,
)

TASK_ID = "iron_mind"
TASK_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = TASK_ROOT / "resources" / "reaction_schemas.json"
MOCK_ORACLE_PATH = TASK_ROOT / "resources" / "mock_oracle.csv"

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Iron Mind LDM task.")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--proposal-mode", choices=("callable", "openai"), default="callable")
    parser.add_argument("--dataset-id", default="buchwald_hartwig")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--reservoir-size", type=int, default=4)
    parser.add_argument("--evaluations-per-round", type=int, default=1)
    parser.add_argument("--acquisition-beta", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--llm-url", default=os.environ.get("LDM_LLM_URL", ""))
    parser.add_argument("--llm-model-name", default=os.environ.get("LDM_LLM_MODEL", ""))
    parser.add_argument("--llm-timeout", type=float, default=120.0)
    parser.add_argument("--llm-max-tokens", type=int, default=2048)
    parser.add_argument("--llm-temperature", type=float, default=0.7)
    parser.add_argument("--campaign-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)

def describe_ldm_task(args: argparse.Namespace) -> LDMTaskSpec:
    """Describe the fixed-schema reaction task before campaign assembly."""

    schema = _load_table(args).schema if not args.mock else _schema_for(args.dataset_id)
    return _task_spec(args, schema)

def _task_spec(args: argparse.Namespace, schema: ReactionDatasetSchema) -> LDMTaskSpec:
    encoder = ReactionOneHotEncoder(schema)
    selector = RBFGPUCBSelector(
        objective_name=OBJECTIVE_NAME,
        beta=args.acquisition_beta,
        feature_version=encoder.version,
    )
    return build_reaction_task_spec(schema, selector.describe())

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    table = _load_table(args)
    task_spec = _task_spec(args, table.schema)
    contract, profile_name = load_active_experiment_contract()
    if contract is None:
        contract = load_experiment_contract(TASK_ROOT / "experiment.json")
    payload = _run_payload(args, task_spec, contract.digest, profile_name)
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    return _run_campaign(args, table, task_spec, contract, profile_name, payload)

def _run_campaign(
    args, table, task_spec, contract, profile_name: str, payload: dict[str, Any]
) -> int:
    schema = table.schema
    runtime = _open_runtime(args, task_spec, contract, profile_name)
    try:
        client = _proposal_client(args, table)
    except KeyError:
        _pause_endpoint(
            runtime,
            args,
            payload,
            "LDM_LLM_API_KEY is required for OpenAI proposal mode.",
        )
        return 2
    if args.proposal_mode == "openai":
        if not _preflight_endpoint(client, runtime, args, payload):
            return 2
    sink = DataCollectionSink.from_env(default_root=runtime.run_dir / "ldm_data")
    before_request = None if args.proposal_mode == "callable" else lambda: runtime.consume("llm_requests")
    components = build_campaign_components(
        CampaignComponentOptions(
            client=client,
            schema=schema,
            table=table,
            sink=sink,
            runtime=runtime,
            before_request=before_request,
            acquisition_beta=args.acquisition_beta,
        )
    )
    return _finish_campaign(args, components, runtime, payload)

def _finish_campaign(
    args,
    components,
    runtime: CampaignRuntime,
    payload: dict[str, Any],
) -> int:
    state = load_campaign_state(runtime, args.resume_from is not None)
    try:
        config = LDMEngineConfig(args.iterations, args.reservoir_size, args.evaluations_per_round)
        result = components.engine.run(
            config,
            state=state,
        )
    except EndpointRequestError as exc:
        _pause_endpoint(
            runtime,
            args,
            payload,
            str(exc),
            phase="reservoir_expansion",
        )
        return 2
    write_campaign_reports(runtime, objective_name=OBJECTIVE_NAME)
    payload.update(engine_summary=result.summary, run_dir=str(runtime.run_dir.resolve()))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.summary["successful_evaluation_count"] else 1
def _validate_args(args: argparse.Namespace) -> None:
    if args.iterations < 0:
        raise SystemExit("--iterations must be non-negative")
    if args.reservoir_size != 4:
        raise SystemExit("Iron Mind requires --reservoir-size=4")
    if args.evaluations_per_round != 1:
        raise SystemExit("Iron Mind requires --evaluations-per-round=1")
    if args.acquisition_beta < 0:
        raise SystemExit("--acquisition-beta must be non-negative")
    if not math.isfinite(args.llm_temperature) or not 0.0 <= args.llm_temperature <= 2.0:
        raise SystemExit("--llm-temperature must be finite and between 0 and 2")
    if args.campaign_index < 0:
        raise SystemExit("--campaign-index must be non-negative")
    if args.mock and args.dataset_id != "buchwald_hartwig":
        raise SystemExit("Mock campaigns require --dataset-id=buchwald_hartwig")
    if not args.mock and args.proposal_mode != "openai":
        raise SystemExit("Non-mock Iron Mind campaigns require --proposal-mode=openai")
    if not args.mock and args.data_dir is None:
        raise SystemExit("Non-mock Iron Mind campaigns require --data-dir")

def _run_payload(args: argparse.Namespace, task_spec: LDMTaskSpec, contract_sha256: str, profile_name: str) -> dict[str, Any]:
    return {
        "task": TASK_ID,
        "mode": "mock" if args.mock else "real",
        "proposal_mode": args.proposal_mode,
        "dataset_id": args.dataset_id,
        "objective": OBJECTIVE_NAME,
        "contract_profile": profile_name,
        "contract_sha256": contract_sha256,
        "ldm_task_spec": task_spec.to_dict(),
    }

def _open_runtime(args, task_spec, contract, profile_name: str) -> CampaignRuntime:
    default_name = (
        "mock" if args.mock else f"{args.dataset_id}-campaign-{args.campaign_index:02d}"
    )
    run_dir = args.resume_from.resolve() if args.resume_from else unique_run_dir(args.out_dir / (args.run_name or default_name))
    profile = contract.profile(profile_name) if profile_name else None
    runtime = CampaignRuntime.open(
        run_dir,
        task=TASK_ID,
        config=jsonable_args(args),
        task_spec=task_spec,
        budget_limits=dict(profile.budget) if profile else derived_budget(args),
        contract_snapshot=contract.to_dict(),
        contract_sha256=contract.digest,
        contract_profile=profile_name,
        resume=args.resume_from is not None,
    )
    if args.resume_from is None:
        snapshot_experiment_contract(contract, run_dir, profile=profile_name)
    return runtime

def _schema_for(dataset_id: str) -> ReactionDatasetSchema:
    try:
        return load_reaction_schemas(SCHEMA_PATH)[dataset_id]
    except KeyError as exc:
        raise SystemExit(f"Unknown --dataset-id: {dataset_id}") from exc

def _load_mock_table(schema: ReactionDatasetSchema) -> FrozenReactionTable:
    return load_mock_table(schema, MOCK_ORACLE_PATH)

def _load_table(args: argparse.Namespace) -> FrozenReactionTable:
    if args.mock:
        return _load_mock_table(_schema_for(args.dataset_id))
    assert args.data_dir is not None
    return load_pinned_reaction_table(dataset_id=args.dataset_id, data_root=args.data_dir)

def _proposal_client(args: argparse.Namespace, table: FrozenReactionTable) -> ProposalClient:
    if args.proposal_mode == "callable":
        return CallableProposalClient(lambda _request: _mock_response(table))
    return build_deepseek_reaction_client(
        base_url=args.llm_url,
        model=args.llm_model_name,
        api_key=os.environ["LDM_LLM_API_KEY"],
        timeout_seconds=args.llm_timeout,
        max_tokens=args.llm_max_tokens,
        temperature=args.llm_temperature,
    )

def _preflight_endpoint(
    client: ProposalClient, runtime: CampaignRuntime, args: argparse.Namespace, payload: dict[str, Any]
) -> bool:
    if not args.llm_url or not args.llm_model_name:
        return _pause_endpoint(runtime, args, payload, "OpenAI proposal mode requires an endpoint URL and model name.")
    try:
        preflight = client.preflight()  # type: ignore[attr-defined]
    except EndpointRequestError as exc:
        return _pause_endpoint(runtime, args, payload, str(exc))
    runtime.record("endpoint_preflight_succeeded", preflight)
    payload["endpoint_preflight"] = preflight
    return True

def _pause_endpoint(
    runtime: CampaignRuntime,
    args: argparse.Namespace,
    payload: dict[str, Any],
    message: str,
    *,
    phase: str = "endpoint_preflight",
) -> bool:
    runtime.pause(
        "paused_endpoint_unavailable",
        phase=phase,
        message=message,
        details={"model": args.llm_model_name},
    )
    payload.update(run_dir=str(runtime.run_dir.resolve()), status="paused_endpoint_unavailable")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return False
