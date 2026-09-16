"""ReaSyn Science Benchmark assembly through ldm_tts.campaign.run_campaign."""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import sys
import math
from copy import copy
from pathlib import Path
from ldm_tts.campaign import (
    CampaignBudget,
    CampaignRecipe,
    CampaignRequest,
    run_campaign,
)
from ldm_tts.contracts import (
    AcquisitionSpec,
    CandidateDomainSpec,
    LDMTaskSpec,
    ObjectiveSpec,
    ProposalSearchSpec,
    ReservoirExpansionSpec,
    ReservoirSpec,
    ResponseSpaceSpec,
    SurrogateSpaceSpec,
)
from ldm_tts.data import DataCollectionSink
from ldm_tts.engine.runtime import consume_proposal_attempts
from ldm_tts.engine.run_store import atomic_json_write, unique_run_dir, BudgetExceededError
from ldm_tts.engine.reporting import load_successful_observations
from .selection import TanimotoGPSelector, AcquisitionTiltedSelector
from ldm_tts.registration.experiment import (
    load_active_experiment_contract,
    load_experiment_contract,
    snapshot_experiment_contract,
)
from ldm_tts.transport.openai import (
    OpenAICompatibleProposalClient,
    EndpointRequestError,
    WIRE_APIS, REASONING_LEVELS, generation_body,
)
from .candidate import ReaSynDomain
from .chemistry import canonicalize
from .evaluator import ReconstructionEvaluator, TDCOracleEvaluator
from .metrics import ORACLES, WIDTH_ONE, reconstruction_metrics, top_auc, top_mean
from .projector import Projector, ProjectionInterruptedError
from .proposals import DEFAULT_PROPOSAL_RECOVERY_SEED_SPAN, ReaSynExpander, ProposalExhausted
from .surrogate import MoleculeEncoder
from ldm_tts.harness import HarnessError, PolicyResearchController, DockerPolicyExecutor
from .harness import HarnessTargetSource, create_client
from .optimization_policy import ReaSynPolicyAdapter

TASK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_ROOT.parents[1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="LDM-guided frozen ReaSyn reconstruction and released 13-oracle TDC optimization."
    )
    p.add_argument("--mock", action="store_true")
    p.add_argument(
        "--benchmark", choices=("reconstruction", "tdc"), default="reconstruction"
    )
    p.add_argument(
        "--proposal-mode",
        choices=("auto", "mock", "callable", "none", "openai", "baseline", "harness"),
        default="auto",
    )
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--reservoir-size", "--proposal-samples", dest="reservoir_size", type=int, default=4)
    p.add_argument("--search-method", choices=("auto", "baseline", "ldm", "bo", "llm", "harness", "ldm_harness", "ldm_harness_compiled"), default="auto")
    p.add_argument("--proposal-batch-size", type=int, default=2)
    p.add_argument("--bo-pool-size", type=int)
    p.add_argument("--bo-targets-file", type=Path, default=os.environ.get("REASYN_BO_TARGETS"))
    p.add_argument("--max-replenishment-batches", type=int, default=4)
    p.add_argument("--recovery-attempts", type=int, default=3)
    p.add_argument("--projection-retry-targets", type=int)
    p.add_argument("--initialization-mode", choices=("none", "shared_start"), default="none")
    p.add_argument("--run-name", default="")
    p.add_argument("--evaluations-per-round", type=int, default=1)
    p.add_argument("--max-oracle-calls", type=int, default=10000)
    p.add_argument("--oracle", choices=ORACLES, default="jnk3")
    p.add_argument("--seed", "--campaign-index", dest="seed", type=int, default=0)
    p.add_argument(
        "--upstream-root",
        type=Path,
        default=Path(
            os.environ.get("REASYN_ROOT", str(REPO_ROOT.parent / "ReaSyn-reasyn_v2"))
        ),
    )
    p.add_argument("--model-paths", default=os.environ.get("REASYN_MODEL_PATHS", ""))
    p.add_argument("--fpindex", type=Path)
    p.add_argument("--rxn-matrix", type=Path)
    p.add_argument("--additional-fpindex", type=Path)
    p.add_argument(
        "--dataset",
        choices=("zinc250k", "enamine", "chembl", "custom"),
        default="zinc250k",
    )
    p.add_argument("--targets-file", type=Path)
    p.add_argument("--target-smiles", default="")
    p.add_argument("--target-limit", type=int, default=1)
    p.add_argument("--search-width", type=int, default=0)
    p.add_argument("--exhaustiveness", type=int, default=4)
    p.add_argument("--num-cycles", type=int, default=1)
    p.add_argument("--num-editflow-samples", type=int, default=4)
    p.add_argument("--max-results", type=int, default=100)
    p.add_argument("--projection-time-limit", type=int, default=1000)
    p.add_argument("--projection-timeout", type=int, default=3600,
        help="Single-target process allowance including loading; each additional serial target adds projection-time-limit seconds.")
    p.add_argument(
        "--evaluator-python", default=os.environ.get("REASYN_PYTHON", sys.executable)
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--acquisition-beta", type=float, default=1.0)
    p.add_argument("--acquisition-alpha", type=float, default=1.0)
    p.add_argument("--acquisition-eta", type=float, default=1.0)
    p.add_argument("--acquisition-z-clip", type=float, default=5.0)
    p.add_argument("--harness-sessions", type=int, default=4)
    p.add_argument("--harness-sidecar-image", default="ldm-pi-harness:latest")
    p.add_argument("--harness-docker-host", default="")
    p.add_argument("--harness-container-user", default="auto")
    p.add_argument("--harness-cache-dir", type=Path)
    p.add_argument("--harness-thinking", choices=("off", "minimal", "low", "medium", "high", "xhigh", "max"))
    p.add_argument("--harness-wall-time-seconds", type=int, default=1800)
    p.add_argument("--harness-response-timeout", type=int, default=2100)
    p.add_argument("--harness-mcp-config", type=Path)
    p.add_argument("--harness-tool-budget", action="append", default=[])
    p.add_argument("--policy-runner-image", default="ldm-pi-harness:latest")
    p.add_argument("--gp-history-limit", type=int, default=256)
    p.add_argument(
        "--llm-url",
        default=os.environ.get(
            "LLM_BASE_URL",
            os.environ.get("LDM_LLM_URL", os.environ.get("OPENAI_BASE_URL", "")),
        ),
    )
    p.add_argument(
        "--llm-model",
        default=os.environ.get("LLM_MODEL_NAME", os.environ.get("LDM_LLM_MODEL", "")),
    )
    p.add_argument("--llm-max-tokens", type=int, default=4096)
    p.add_argument("--llm-wire-api", choices=WIRE_APIS, default="responses")
    p.add_argument("--llm-reasoning", choices=REASONING_LEVELS)
    p.add_argument("--llm-temperature", type=float, default=0.7)
    p.add_argument("--llm-extra-body-json", default="{}")
    p.add_argument("--out-dir", type=Path, default=Path("runs/reasyn"))
    p.add_argument("--resume-from", type=Path)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if args.search_method == "auto":
        args.search_method = "baseline" if args.proposal_mode == "baseline" else "ldm"
    if args.proposal_mode == "callable":
        args.proposal_mode = "mock"
    if args.proposal_mode == "none":
        args.proposal_mode = "auto"
    harness_method = args.search_method in ("harness", "ldm_harness", "ldm_harness_compiled")
    if args.llm_reasoning is not None and args.harness_thinking is not None and args.llm_reasoning != args.harness_thinking:
        p.error("llm-reasoning and harness-thinking must agree")
    args.llm_reasoning = args.llm_reasoning or args.harness_thinking or "off"
    args.harness_thinking = "off" if args.llm_reasoning == "none" else args.llm_reasoning
    try:
        extra = json.loads(args.llm_extra_body_json)
        if not isinstance(extra, dict):
            raise ValueError("llm-extra-body-json must be an object")
        generation_body(wire_api=args.llm_wire_api, reasoning=args.llm_reasoning, extra_body=extra)
        args.llm_extra_body_json = json.dumps(extra, sort_keys=True)
        if not math.isfinite(args.llm_temperature) or not 0 <= args.llm_temperature <= 2:
            raise ValueError("llm-temperature must be finite and between 0 and 2")
    except ValueError as exc:
        p.error(str(exc))
    if harness_method and args.llm_wire_api != "responses":
        p.error("Harness requires llm-wire-api responses")
    if harness_method:
        if args.proposal_mode not in ("auto", "harness"):
            p.error("Harness methods require proposal-mode auto or harness; mock controls the evaluator only")
        args.proposal_mode = "harness"
    elif args.proposal_mode == "harness":
        p.error("proposal-mode harness requires a Harness search-method")
    if args.bo_pool_size is None:
        args.bo_pool_size = args.reservoir_size
    if args.projection_retry_targets is None:
        args.projection_retry_targets = args.reservoir_size * args.recovery_attempts
    if args.run_name and (Path(args.run_name).name != args.run_name or args.run_name in (".", "..")):
        p.error("run-name must be a single directory name")
    args.upstream_root = args.upstream_root.expanduser().resolve()
    if args.harness_cache_dir:
        args.harness_cache_dir = args.harness_cache_dir.expanduser().resolve()
    if args.harness_mcp_config:
        args.harness_mcp_config = args.harness_mcp_config.expanduser().resolve()
    if args.proposal_mode == "auto":
        args.proposal_mode = "mock" if args.mock else "openai"
    if args.proposal_mode == "mock" and not args.mock:
        p.error("mock proposals require --mock; they are not scientific results")
    if args.proposal_mode == "baseline" and args.benchmark == "tdc":
        p.error("use upstream optimize_tdc.py for Graph GA baseline")
    if not args.search_width:
        args.search_width = (
            (1 if args.oracle in WIDTH_ONE else 2) if args.benchmark == "tdc" else 8
        )
    raw = (
        args.model_paths
        or "data/trained_model/nv-reasyn-ar-166m-v2.ckpt,data/trained_model/nv-reasyn-eb-174m-v2.ckpt"
    )

    def upstream_path(value):
        value = Path(value).expanduser()
        return (
            value.resolve()
            if value.is_absolute()
            else (args.upstream_root / value).resolve()
        )

    args.model_paths = [upstream_path(v.strip()) for v in raw.split(",")]
    args.fpindex = upstream_path(args.fpindex or "data/processed/comp_2048/fpindex.pkl")
    args.rxn_matrix = upstream_path(
        args.rxn_matrix or "data/processed/comp_2048/matrix.pkl"
    )
    if args.targets_file:
        args.targets_file = upstream_path(args.targets_file)
    if args.additional_fpindex:
        args.additional_fpindex = upstream_path(args.additional_fpindex)
    elif args.benchmark == "reconstruction" and args.dataset == "zinc250k":
        args.additional_fpindex = upstream_path(
            "data/processed/zinc250k_2048/fpindex.pkl"
        )
    if (
        args.iterations < 1
        or args.reservoir_size < 1
        or not 1 <= args.evaluations_per_round <= args.reservoir_size
    ):
        p.error("positive iterations and 1 <= evaluation batch <= reservoir required")
    if not 1 <= args.max_oracle_calls <= 10000:
        p.error("oracle call cap must be in [1,10000]")
    if any(
        v < 1
        for v in (
            args.search_width,
            args.exhaustiveness,
            args.num_cycles,
            args.num_editflow_samples,
            args.max_results,
            args.target_limit,
            args.gp_history_limit,
            args.projection_time_limit,
            args.projection_timeout,
        )
    ):
        p.error("search, dataset and timing limits must be positive")
    if args.seed < 0 or any(not math.isfinite(v) or v < 0 for v in (args.acquisition_beta, args.acquisition_alpha, args.acquisition_eta)):
        p.error("seed and acquisition parameters must be finite and nonnegative")
    if (args.proposal_batch_size < 1 or args.bo_pool_size < args.evaluations_per_round
            or args.max_replenishment_batches < 0 or args.recovery_attempts < 0
            or args.projection_retry_targets < 0 or args.harness_sessions < 1
            or args.harness_wall_time_seconds < 1 or args.harness_response_timeout < 1
            or not math.isfinite(args.acquisition_z_clip) or args.acquisition_z_clip <= 0):
        p.error("Invalid sampling, recovery, or Harness limits")
    return args


def describe_ldm_task(args):
    recon = args.benchmark == "reconstruction"
    objective = "similarity" if recon else "oracle_score"
    direct = args.search_method in ("llm", "harness", "baseline")
    return LDMTaskSpec(
        task="reasyn",
        candidate_domain=CandidateDomainSpec(
            name="ReaSyn projection queries"
            if recon
            else "ReaSyn synthesis-verified molecules",
            kind="molecular_projection_query" if recon else "synthesizable_molecule",
            dimension=None,
            representation="Canonical target SMILES and controlled restart seed"
            if recon
            else "Canonical product SMILES with replay-verified stock/reaction trace",
            constraints={"benchmark": args.benchmark, "mock": args.mock},
        ),
        objectives=(
            ObjectiveSpec(
                objective,
                "maximize",
                "Morgan r2/4096 similarity to original target"
                if recon
                else "Released TDC oracle score",
            ),
        ),
        response_spaces=(
            ResponseSpaceSpec(
                name="projection_targets_json",
                output_kind="json",
                schema={
                    "type": "object",
                    "required": ["candidates"],
                    "properties": {
                        "candidates": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["target_smiles"],
                                "properties": {"target_smiles": {"type": "string"}},
                                "additionalProperties": False,
                            },
                        }
                    },
                    "additionalProperties": False,
                },
                parser="tasks.reasyn.core.proposals:parse_targets",
            ),
        ),
        acquisition=AcquisitionSpec(
            name="direct_proposal_order" if direct else "empirical_q0_tanimoto_gp_ucb" if args.search_method.startswith("ldm") else "tanimoto_gp_ucb",
            objective_names=(objective,),
            score_direction="maximize",
            selection_rule="Empirical group q0 with alpha*log(group_q0+epsilon)-log(group_trial_count)+eta*robust_z(UCB) sampling" if args.search_method.startswith("ldm") else "GP-UCB" if args.search_method == "bo" else "Direct proposal order",
            parameters={
                "beta": args.acquisition_beta,
                "history_limit": args.gp_history_limit,
                "alpha": args.acquisition_alpha, "eta": args.acquisition_eta,
                "bo_pool_size": args.bo_pool_size,
            },
        ),
        reservoir=ReservoirSpec(
            name="projection_query_reservoir"
            if recon
            else "projected_product_reservoir",
            expansions=(
                ReservoirExpansionSpec(
                    name="feedback_conditioned_targets",
                    action_kind="emit_candidate",
                    response_space="projection_targets_json",
                    produces_candidates=True,
                    description="LDM proposes targets; frozen ReaSyn supplies actual valid synthesis pathways",
                ),
            ),
            candidate_validator="tasks.reasyn.core.candidate:ReaSynDomain",
            deduplication_key="Canonical query plus controlled restart seed"
            if recon
            else "Canonical isomeric product SMILES SHA256",
            max_size=args.reservoir_size,
        ),
        surrogate=SurrogateSpaceSpec(
            kind="none",
            representation="Direct proposal order without a surrogate",
            dimension_policy="none",
        ) if direct else MoleculeEncoder(mock=args.mock).describe(),
        proposal_search=ProposalSearchSpec(
            name="independent_minibatch_empirical_q0" if args.search_method.startswith("ldm") else args.search_method,
            breadth=args.reservoir_size,
            evaluation_policy="gp_ucb_selected_frozen_projection"
            if recon
            else "gp_ucb_selected_tdc_oracle",
        ),
        metadata={
            "benchmark": args.benchmark,
            "policy_capabilities": ["prior_mean@1", "ldm_weights@1"],
            "alpha_configurable": True,
            "mock": args.mock,
            "proposal_mode": args.proposal_mode,
            "search_method": args.search_method,
            "official_suite_oracles": list(ORACLES),
        },
    )


def _jsonable(args):
    def convert(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, list):
            return [convert(x) for x in v]
        return v

    return {k: convert(v) for k, v in vars(args).items() if k != "provider_api_key"}


def _direct_provider_api_key():
    return os.environ.get(
        "LLM_API_KEY",
        os.environ.get("LDM_LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
    )


def _harness_provider_api_key():
    return _direct_provider_api_key()


def _preflight_harness_provider(args):
    if not args.llm_url or not args.llm_model:
        raise ValueError("Set LLM_BASE_URL and LLM_MODEL_NAME for real Harness proposals")
    if not getattr(args, "provider_api_key", ""):
        raise EndpointRequestError(
            "Set LLM_API_KEY or OPENAI_API_KEY for real Harness proposals"
        )
    client = OpenAICompatibleProposalClient(
        url=args.llm_url,
        model=args.llm_model,
        api_key=args.provider_api_key,
        timeout_seconds=args.harness_response_timeout,
        max_tokens=args.llm_max_tokens,
        max_retries=0,
        wire_api="responses",
        temperature=args.llm_temperature,
        extra_body=generation_body(wire_api=args.llm_wire_api, reasoning=args.llm_reasoning,
                                   extra_body=json.loads(args.llm_extra_body_json)),
    )
    return {"backend": "harness", "wire_api": "responses", **client.preflight()}


def _targets(args):
    if args.benchmark == "tdc":
        return [""]
    if args.target_smiles:
        return [canonicalize(args.target_smiles, mock=args.mock, stereo=False)]
    if args.mock:
        return ["CCO"]
    names = {
        "zinc250k": "test_zinc250k.txt",
        "enamine": "enamine_smiles_1k.txt",
        "chembl": "chembl_filtered_1k.txt",
    }
    path = args.targets_file or args.upstream_root / "data" / names.get(
        args.dataset, "targets.txt"
    )
    lines = [
        s.strip().split(",")[0] for s in path.read_text().splitlines() if s.strip()
    ]
    if lines and lines[0].lower() == "smiles":
        lines = lines[1:]
    if len(lines) < args.target_limit:
        raise ValueError(
            f"target-limit requests {args.target_limit} targets, but {path} contains "
            f"only {len(lines)}; use a complete dataset or an explicit smaller subset"
        )
    targets = [canonicalize(s, stereo=False) for s in lines[: args.target_limit]]
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("requested targets must be nonempty and unique")
    return targets


def _file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def _directory_hash(root):
    root = Path(root).resolve()
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        entries.append({"path": path.relative_to(root).as_posix(), "sha256": _file_hash(path)})
    return hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _harness_component_digests(args):
    if args.proposal_mode != "harness":
        return {}
    digests = {
        "resource_tree": _directory_hash(TASK_ROOT / "resources/harness"),
        "harness_mcp_config": None,
    }
    if args.harness_mcp_config:
        digests["harness_mcp_config"] = _file_hash(args.harness_mcp_config)
    return digests


def _scientific_identity(args, configuration, target, contract):
    identity_keys = (
        "benchmark",
        "mock",
        "oracle",
        "seed",
        "proposal_mode",
        "num_cycles",
        "search_width",
        "exhaustiveness",
        "num_editflow_samples",
        "model_paths",
        "fpindex",
        "rxn_matrix",
        "additional_fpindex",
        "reservoir_size",
        "evaluations_per_round",
        "max_oracle_calls",
        "llm_model",
        "llm_max_tokens",
        "llm_wire_api", "llm_reasoning", "llm_temperature", "llm_extra_body_json",
        "max_results",
        "projection_time_limit",
        "projection_timeout",
        "device",
        "evaluator_python",
        "gp_history_limit",
        "acquisition_beta", "acquisition_alpha", "acquisition_eta", "acquisition_z_clip",
        "search_method", "proposal_batch_size", "bo_pool_size", "max_replenishment_batches",
        "recovery_attempts", "projection_retry_targets", "initialization_mode",
        "proposal_recovery_seed_span",
        "harness_sessions", "harness_thinking", "harness_tool_budget",
        "harness_sidecar_image", "policy_runner_image", "harness_mcp_config",
        "harness_wall_time_seconds", "harness_response_timeout",
    )
    scientific = {k: configuration[k] for k in identity_keys}
    scientific["bo_targets"] = getattr(args, "bo_targets", [])
    scientific["original_target"] = target
    scientific["source_archive_digest"] = contract.benchmark["source_commit"]
    scientific["asset_digests"] = args.asset_digests
    scientific["harness_component_digests"] = configuration["harness_component_digests"]
    if args.benchmark == "reconstruction":
        scientific["q0_identity_version"] = "canonical_query_group_trial_allocation_v1"
    return scientific


def _status_recovery_pass(status_payload):
    details = status_payload.get("details", {}) if isinstance(status_payload, dict) else {}
    value = details.get("proposal_recovery_pass") if isinstance(details, dict) else None
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("status proposal_recovery_pass must be a nonnegative integer")
    return value


def _proposal_recovery_details(expander):
    value = getattr(expander, "recovery_pass", 0)
    if value:
        return {"proposal_recovery_pass": value}
    return None


def _proposal_recovery_seed_span(args, run_dir, *, resume):
    config_path = Path(run_dir) / "config.json"
    if resume and config_path.exists():
        original = json.loads(config_path.read_text())
        value = original.get("proposal_recovery_seed_span", original.get("iterations"))
    else:
        value = max(
            DEFAULT_PROPOSAL_RECOVERY_SEED_SPAN,
            args.iterations,
            args.max_oracle_calls,
        )
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("proposal_recovery_seed_span must be a positive integer")
    return value


def _check_assets(args):
    args.asset_digests = {}
    if args.mock:
        return
    paths = [
        args.upstream_root / "reasyn/sampler/sampler.py",
        args.fpindex,
        args.rxn_matrix,
        *args.model_paths,
    ]
    if args.additional_fpindex:
        paths.append(args.additional_fpindex)
    missing = [str(p) for p in paths if not p.is_file()]
    if len(args.model_paths) != 2:
        raise ValueError("exactly two checkpoints required: AR then EB")
    if missing:
        raise FileNotFoundError("Missing ReaSyn assets: " + ", ".join(missing))

    # Hash once before all target campaigns. Same-path asset replacement cannot
    # silently reuse cached products or resume an incompatible scientific run.
    for path in sorted(set(paths) | set((args.upstream_root / "reasyn").rglob("*.py"))):
        args.asset_digests[str(path)] = _file_hash(path)
    pinned = json.loads((TASK_ROOT / "resources/source_provenance.json").read_text())[
        "files"
    ]
    prefix = "ReaSyn-reasyn_v2/"
    for name, expected in pinned.items():
        if name.startswith(prefix) and name.endswith((".py", ".yml")):
            path = args.upstream_root / name[len(prefix) :]
            actual = _file_hash(path)
            if actual != expected:
                raise ValueError(
                    "Supplied upstream source differs from pinned archive: " + name
                )
    args.asset_digests["adapter_projection_worker"] = _file_hash(
        TASK_ROOT / "core/projection_worker.py"
    )


def main(argv=None):
    args = parse_args(argv)
    spec = describe_ldm_task(args)
    contract, profile = load_active_experiment_contract()
    contract = contract or load_experiment_contract(TASK_ROOT / "experiment.json")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "task": "reasyn",
                    "contract_profile": profile,
                    "configuration": _jsonable(args),
                    "ldm_task_spec": spec.to_dict(),
                },
                indent=2,
            )
        )
        return 0
    _check_assets(args)
    if args.search_method == "bo" or args.initialization_mode == "shared_start":
        from .chemistry import MOCK_SMILES
        if args.mock:
            args.bo_targets = list(MOCK_SMILES)
        elif args.bo_targets_file:
            args.bo_targets = [canonicalize(line.strip()) for line in args.bo_targets_file.expanduser().read_text().splitlines() if line.strip()]
        elif args.benchmark == "reconstruction" and args.proposal_mode == "baseline":
            args.bo_targets = []
        else:
            raise ValueError("Real BO/shared_start requires a score-blind --bo-targets-file pool")
        if args.bo_targets_file and not args.bo_targets:
            raise ValueError("BO target pool is empty")
    targets = _targets(args)
    root = (
        args.resume_from.resolve()
        if args.resume_from
        else ((args.out_dir / args.run_name).resolve() if args.run_name else unique_run_dir(args.out_dir.resolve()))
    )
    if args.run_name and not args.resume_from and root.exists():
        raise FileExistsError("Named run directory already exists; use --resume-from")
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        "benchmark": args.benchmark,
        "oracle": args.oracle if args.benchmark == "tdc" else None,
        "targets": targets,
        "mock": args.mock,
    }
    manifest = root / "targets_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("resume target manifest mismatch")
    atomic_json_write(manifest, identity)
    outputs = []
    exit_code = 0
    for index, target in enumerate(targets):
        run_dir = root if len(targets) == 1 else root / f"target-{index:05d}"
        result, code = _run_one(
            args,
            spec,
            contract,
            profile,
            run_dir,
            target,
            resume=args.resume_from is not None
            and (run_dir / "campaign.json").exists(),
        )
        outputs.append(result)
        exit_code = max(exit_code, code)
        if code == 2:
            break  # Endpoint pause must not burn through other targets.
    if args.benchmark == "reconstruction":
        all_rows = []
        for item in outputs:
            all_rows.extend(item.get("reconstruction_rows", []))
        aggregate = reconstruction_metrics(targets, all_rows, mock=args.mock)
        aggregate.update(
            task="reasyn",
            benchmark="reconstruction",
            mock=args.mock,
            qualification="draft",
            completed_campaigns=len(outputs),
            requested_campaigns=len(targets),
            comparison="LDM adaptation; projection budget/config must match baseline",
        )
        atomic_json_write(root / "benchmark_result.json", aggregate)
    print(
        json.dumps(
            {
                "task": "reasyn",
                "benchmark": args.benchmark,
                "mock": args.mock,
                "run_dir": str(root),
                "campaigns": len(outputs),
                "exit_code": exit_code,
            },
            indent=2,
        )
    )
    return exit_code


def _run_one(args, spec, contract, profile, run_dir, target, *, resume):
    run_dir.mkdir(parents=True, exist_ok=True)
    previous_status_payload = (
        json.loads((run_dir / "status.json").read_text())
        if resume and (run_dir / "status.json").exists()
        else {}
    )
    previous_status = previous_status_payload.get("status", "")
    previous_proposal_recovery_pass = _status_recovery_pass(previous_status_payload)
    # Preserve scientific config across resume; only iterations may extend.
    configuration = _jsonable(args)
    proposal_recovery_seed_span = _proposal_recovery_seed_span(args, run_dir, resume=resume)
    args.proposal_recovery_seed_span = proposal_recovery_seed_span
    configuration.update(proposal_samples=args.reservoir_size,
        proposal_candidates_per_request=args.proposal_batch_size,
        proposal_counting="bounded_minibatches",
        proposal_recovery_seed_span=proposal_recovery_seed_span,
        harness_component_digests=_harness_component_digests(args))
    scientific = _scientific_identity(args, configuration, target, contract)
    identity_path = run_dir / "scientific_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != scientific:
        raise ValueError("resume scientific configuration mismatch")
    atomic_json_write(identity_path, scientific)
    snapshot_experiment_contract(contract, run_dir, profile=profile)
    projector = Projector(args, run_dir)
    sink = DataCollectionSink.from_env(default_root=run_dir / "ldm_data")
    client = None
    if args.proposal_mode == "harness" and (not args.llm_url or not args.llm_model):
        raise ValueError("Set LLM_BASE_URL and LLM_MODEL_NAME for real Harness proposals")
    if args.proposal_mode == "openai" and args.search_method != "bo":
        if not args.llm_url or not args.llm_model:
            raise ValueError(
                "Set LLM_BASE_URL and LLM_MODEL_NAME for real LDM proposals"
            )
        client = OpenAICompatibleProposalClient(
            url=args.llm_url,
            model=args.llm_model,
            api_key=_direct_provider_api_key(),
            max_tokens=args.llm_max_tokens,
            max_retries=0,
            wire_api=args.llm_wire_api,
            temperature=args.llm_temperature,
            extra_body=generation_body(wire_api=args.llm_wire_api, reasoning=args.llm_reasoning,
                                       extra_body=json.loads(args.llm_extra_body_json)),
        )
    expander = ReaSynExpander(args, projector, sink, target=target, client=client)
    evaluator = (
        ReconstructionEvaluator(args, projector, target)
        if target
        else TDCOracleEvaluator(args, run_dir)
    )
    encoder = MoleculeEncoder(mock=args.mock)
    selector = TanimotoGPSelector(
        objective_name=spec.objectives[0].name,
        beta=args.acquisition_beta,
        feature_version=encoder.version,
        history_limit=args.gp_history_limit,
    )
    cap = args.iterations * args.evaluations_per_round
    if not target:
        cap = min(cap, args.max_oracle_calls)
    checkpoint_path = run_dir / "checkpoint.json"
    resuming_complete = (
        resume
        and checkpoint_path.exists()
        and len(load_successful_observations(checkpoint_path)) >= cap
    )
    batches_per_round = math.ceil(args.reservoir_size / args.proposal_batch_size) + args.max_replenishment_batches
    request_cap = (args.iterations + args.recovery_attempts) * batches_per_round
    proposal_cap = args.reservoir_size + args.max_replenishment_batches * args.proposal_batch_size
    projection_rounds = args.iterations if target else args.iterations + args.recovery_attempts
    projection_cap = (cap if target else projection_rounds * proposal_cap) + args.projection_retry_targets
    uses_harness = args.proposal_mode == "harness"
    provider_preflight_required = client is not None or uses_harness
    extra = {
        "llm_requests": request_cap if client else 0,
        "proposal_request_attempts": request_cap if (client or uses_harness) else 0,
        "proposal_attempts": request_cap if (client or uses_harness) else 0,
        "recovery_attempts": args.recovery_attempts,
        "harness_turns": request_cap * args.harness_sessions if uses_harness else 0,
        "oracle_calls": args.max_oracle_calls if not target else 0,
        "projection_targets": projection_cap,
        "projection_cycle_allowance": projection_cap * args.num_cycles,
        "endpoint_preflight_requests": (
            json.loads((run_dir / "budget.json").read_text())
            .get("counters", {})
            .get("endpoint_preflight_requests", 0)
            + (0 if resuming_complete else 1)
            if resume and (run_dir / "budget.json").exists()
            else 1
        )
        if provider_preflight_required
        else 0,
    }
    budget = CampaignBudget(
        rounds=args.iterations,
        reservoir_size=args.reservoir_size,
        batch_size=args.evaluations_per_round,
        target_observations=cap,
        max_evaluation_attempts=cap,
        extra_limits=extra,
    )
    if args.search_method.startswith("ldm"):
        selector = AcquisitionTiltedSelector(selector, alpha=args.acquisition_alpha,
            eta=args.acquisition_eta, z_clip=args.acquisition_z_clip, seed=args.seed,
            pool_size=args.bo_pool_size)
    elif args.search_method in ("llm", "harness", "baseline"):
        selector, encoder = None, None
    holder = {}
    harness_clients = []
    search_expander = expander
    if args.initialization_mode == "shared_start":
        initial_args = copy(args)
        initial_args.search_method = "bo"
        initial_args.proposal_mode = "mock" if args.mock else "baseline"
        initializer = ReaSynExpander(initial_args, projector, sink, target=target)
        class PairedStartExpander:
            def expand(self, request):
                if request.round_idx == 0:
                    from dataclasses import replace
                    return replace(initializer.expand(request), selection_mode="reservoir_order")
                return search_expander.expand(request)
        expander = PairedStartExpander()

    def runtime_hook(runtime):
        holder["runtime"] = runtime
        if resuming_complete:
            return  # Completed campaigns can be reopened without live services.
        projector.before_project = lambda count: runtime.consume_many(
            {
                "projection_targets": count,
                "projection_cycle_allowance": count * args.num_cycles,
            }
        )
        proposal_recovery_pass = 0
        if resume and (previous_status.startswith("paused") or previous_status == "failed"):
            consumed = runtime.consume("recovery_attempts")
            if previous_status == "paused_proposal_exhausted":
                proposal_recovery_pass = int(consumed)
                runtime.record(
                    "proposal_recovery_pass_started",
                    {"proposal_recovery_pass": proposal_recovery_pass},
                )
            else:
                proposal_recovery_pass = previous_proposal_recovery_pass
        search_expander.recovery_pass = proposal_recovery_pass
        def before_request():
            amounts = {"proposal_request_attempts": 1}
            if client:
                amounts["llm_requests"] = 1
            runtime.consume_many(amounts)
        search_expander.before_request = before_request
        if uses_harness:
            provider_args = copy(args)
            provider_args.provider_api_key = _harness_provider_api_key()
            runtime.consume("endpoint_preflight_requests")
            preflight = _preflight_harness_provider(provider_args)
            runtime.record("endpoint_preflight_succeeded", preflight)
            harness_client = create_client(provider_args, run_dir / "harness", runtime.run_id, target)
            harness_clients.append(harness_client)
            harness_client.start()
            search_expander.client = HarnessTargetSource(harness_client, run_dir / "harness", args,
                target=target, account=runtime.consume_many)
            if args.search_method == "ldm_harness_compiled":
                policy_client = create_client(provider_args, run_dir / "policy_harness", runtime.run_id, target, policy=True)
                harness_clients.append(policy_client)
                policy_client.start()
                adapter = ReaSynPolicyAdapter(encoder, benchmark=args.benchmark, alpha=args.acquisition_alpha, eta=args.acquisition_eta, seed=args.seed,
                    history_limit=args.gp_history_limit, beta=args.acquisition_beta, z_clip=args.acquisition_z_clip)
                selector.policy_adapter = adapter
                selector.policy_controller = PolicyResearchController(client=policy_client, adapter=adapter,
                    executor=DockerPolicyExecutor(args.policy_runner_image, args.harness_docker_host,
                        resolve_policy_user(args)), root=run_dir / "policy_harness", account=runtime.consume_many)

        if not target:
            evaluator.before_oracle = lambda: runtime.consume("oracle_calls")
        if client:
            # Preflight is separately reported and runs on every new/resumed process.
            runtime.consume("endpoint_preflight_requests")
            preflight = client.preflight()
            runtime.record("endpoint_preflight_succeeded", preflight)

    try:
        result = run_campaign(
            CampaignRequest(
                run_dir=run_dir,
                budget=budget,
                config=configuration,
                resume=resume,
                contract_sha256=contract.digest,
                contract_profile=profile,
                runtime_hook=runtime_hook,
                context={
                    "original_target": target,
                    "oracle": args.oracle,
                    "mock": args.mock,
                },
                artifact_projector=lambda runtime, engine: _report(
                    args, runtime, target
                ),
            ),
            CampaignRecipe(
                task_spec=spec,
                expander=expander,
                candidate_domain=ReaSynDomain(args.benchmark, mock=args.mock),
                evaluator=evaluator,
                surrogate_encoder=encoder,
                selector=selector,
            ),
        )
    except (EndpointRequestError, HarnessError, ProjectionInterruptedError) as exc:
        runtime = holder.get("runtime")
        status = "paused_projection_interrupted" if isinstance(exc, ProjectionInterruptedError) else "paused_endpoint_unavailable"
        if runtime:
            runtime.pause(status, phase="proposal_or_projection",
                message="Resolve the interrupted service, then resume this directory within the separate recovery allowance.",
                details=_proposal_recovery_details(search_expander))
            report = _report(args, runtime, target)
            report["status"] = status
            atomic_json_write(run_dir / "result.json", report)
        else:
            report = {"status": status}
        return report, 2
    except ProposalExhausted as exc:
        runtime = holder.get("runtime")
        runtime.record("proposal_replenishment_exhausted", exc.metadata)
        if exc.attempts:
            consume_proposal_attempts(runtime, exc.attempts)
        runtime.pause(
            "paused_proposal_exhausted",
            phase="proposal_replenishment",
            message=str(exc),
            details=_proposal_recovery_details(search_expander),
        )
        report = _report(args, runtime, target)
        report.update(status="paused_proposal_exhausted", proposal_diagnostics=exc.metadata)
        atomic_json_write(run_dir / "result.json", report)
        return report, 1
    except BudgetExceededError as exc:
        runtime = holder.get("runtime")
        runtime.pause(
            "paused_resource_budget",
            phase="resource_budget",
            message=str(exc),
            details=_proposal_recovery_details(search_expander),
        )
        report = _report(args, runtime, target)
        report["status"] = "paused_resource_budget"
        atomic_json_write(run_dir / "result.json", report)
        return report, 1
    finally:
        for harness_client in reversed(harness_clients):
            harness_client.close()
    complete = result.engine.summary["successful_evaluation_count"] == cap
    if not complete:
        result.runtime.status.update("stopped", phase="incomplete_scientific_budget", budget=result.runtime.budget)
    return result.projected, 0 if complete else 1


def resolve_policy_user(args):
    from ldm_tts.harness.container import resolve_container_user
    return resolve_container_user(args.harness_container_user, args.harness_docker_host)


def _report(args, runtime, target):
    if args.proposal_mode == "harness":
        kinds = ("harness", "policy_harness") if args.search_method == "ldm_harness_compiled" else ("harness",)
        atomic_json_write(runtime.run_dir / "harness_provenance.json", {
            "schema_version": 1, "pools": {kind: [f"{kind}/manifest.json"] for kind in kinds},
        })
    checkpoint = runtime.run_dir / "checkpoint.json"
    observations = load_successful_observations(checkpoint) if checkpoint.exists() else []
    events = runtime.events()
    atomic_json_write(
        runtime.run_dir / "search_manifest.json",
        {"rounds": [e for e in events if e.get("event_type") == "reservoir_built"]},
    )
    atomic_json_write(
        runtime.run_dir / "selection_record.json",
        {
            "selections": [
                e for e in events if e.get("event_type") == "candidates_selected"
            ]
        },
    )
    rows = []
    for i, o in enumerate(observations):
        rows.append(
            {
                "evaluation": i + 1,
                "candidate_id": o["candidate"]["candidate_id"],
                "score": o["evaluation"]["metrics"][
                    "similarity" if target else "oracle_score"
                ],
            }
        )
    with (runtime.run_dir / "trajectory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("evaluation", "candidate_id", "score")
        )
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "task": "reasyn",
        "benchmark": args.benchmark,
        "mock": args.mock,
        "qualification": "draft",
        "successful_evaluations": len(observations),
        "budget": json.loads((runtime.run_dir / "budget.json").read_text()),
    }
    if target:
        product_rows = []
        for o in observations:
            artifact = o["evaluation"]["artifacts"].get("projection")
            if artifact:
                data = json.loads((runtime.run_dir / artifact).read_text())
                product_rows.extend(
                    {**r, "target": target, "projection_query": r["target"]}
                    for r in data["rows"]
                )
        report["reconstruction_rows"] = product_rows
        report["metrics"] = reconstruction_metrics(
            [target], product_rows, mock=args.mock
        )
    else:
        cache_path = runtime.run_dir / "oracle_cache.json"
        entries = (
            json.loads(cache_path.read_text())["entries"] if cache_path.exists() else []
        )
        scores = [e["score"] for e in entries if e["status"] == "completed"]
        full = len(scores) == args.max_oracle_calls and len(entries) == len(scores)
        report.update(
            oracle=args.oracle,
            oracle_calls=len(entries),
            completed_oracle_calls=len(scores),
            oracle_budget_exhausted=full,
            official_budget=args.max_oracle_calls == 10000,
            metrics={
                "top10": top_mean(scores),
                "auc_top10_observed": top_auc(scores, max_calls=args.max_oracle_calls),
                "auc_top10": top_auc(scores, max_calls=args.max_oracle_calls)
                if full
                else None,
            },
        )
        # No automatic early-stop padding: an interrupted/pilot campaign is not
        # upstream scientific convergence. Aggregate only full-budget runs.
    atomic_json_write(runtime.run_dir / "result.json", report)
    return report
