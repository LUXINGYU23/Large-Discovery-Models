"""ReaSyn Science Benchmark assembly through ldm_tts.campaign.run_campaign."""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import sys
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
)
from ldm_tts.data import DataCollectionSink
from ldm_tts.engine.run_store import atomic_json_write, unique_run_dir
from ldm_tts.engine.reporting import load_successful_observations
from .selection import TanimotoGPSelector
from ldm_tts.registration.experiment import (
    load_active_experiment_contract,
    load_experiment_contract,
    snapshot_experiment_contract,
)
from ldm_tts.transport.openai import (
    OpenAICompatibleProposalClient,
    EndpointRequestError,
)
from .candidate import ReaSynDomain
from .chemistry import canonicalize
from .evaluator import ReconstructionEvaluator, TDCOracleEvaluator
from .metrics import ORACLES, WIDTH_ONE, reconstruction_metrics, top_auc, top_mean
from .projector import Projector
from .proposals import ReaSynExpander
from .surrogate import MoleculeEncoder

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
        choices=("auto", "mock", "openai", "baseline"),
        default="auto",
    )
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--reservoir-size", type=int, default=4)
    p.add_argument("--evaluations-per-round", type=int, default=1)
    p.add_argument("--max-oracle-calls", type=int, default=10000)
    p.add_argument("--oracle", choices=ORACLES, default="jnk3")
    p.add_argument("--seed", type=int, default=0)
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
    p.add_argument("--projection-timeout", type=int, default=3600)
    p.add_argument(
        "--evaluator-python", default=os.environ.get("REASYN_PYTHON", sys.executable)
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--acquisition-beta", type=float, default=1.0)
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
    p.add_argument("--out-dir", type=Path, default=Path("runs/reasyn"))
    p.add_argument("--resume-from", type=Path)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    args.upstream_root = args.upstream_root.expanduser().resolve()
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
    if args.seed < 0 or args.acquisition_beta < 0:
        p.error("seed and acquisition beta must be nonnegative")
    return args


def describe_ldm_task(args):
    recon = args.benchmark == "reconstruction"
    objective = "similarity" if recon else "oracle_score"
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
            name="tanimoto_gp_ucb",
            objective_names=(objective,),
            score_direction="maximize",
            selection_rule="Highest shared UCB of Tanimoto GP with bounded recent training history",
            parameters={
                "beta": args.acquisition_beta,
                "history_limit": args.gp_history_limit,
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
        surrogate=MoleculeEncoder(mock=args.mock).describe(),
        proposal_search=ProposalSearchSpec(
            name="multi_round_feedback_best_of_n",
            breadth=args.reservoir_size,
            evaluation_policy="gp_ucb_selected_frozen_projection"
            if recon
            else "gp_ucb_selected_tdc_oracle",
        ),
        metadata={
            "benchmark": args.benchmark,
            "mock": args.mock,
            "proposal_mode": args.proposal_mode,
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

    return {k: convert(v) for k, v in vars(args).items()}


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

    def file_hash(path):
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1048576), b""):
                h.update(chunk)
        return h.hexdigest()

    # Hash once before all target campaigns. Same-path asset replacement cannot
    # silently reuse cached products or resume an incompatible scientific run.
    for path in sorted(set(paths) | set((args.upstream_root / "reasyn").rglob("*.py"))):
        args.asset_digests[str(path)] = file_hash(path)
    pinned = json.loads((TASK_ROOT / "resources/source_provenance.json").read_text())[
        "files"
    ]
    prefix = "ReaSyn-reasyn_v2/"
    for name, expected in pinned.items():
        if name.startswith(prefix) and name.endswith((".py", ".yml")):
            path = args.upstream_root / name[len(prefix) :]
            actual = file_hash(path)
            if actual != expected:
                raise ValueError(
                    "Supplied upstream source differs from pinned archive: " + name
                )
    args.asset_digests["adapter_projection_worker"] = file_hash(
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
    targets = _targets(args)
    root = (
        args.resume_from.resolve()
        if args.resume_from
        else unique_run_dir(args.out_dir.resolve())
    )
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
    # Preserve scientific config across resume; only iterations may extend.
    configuration = _jsonable(args)
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
        "max_results",
        "projection_time_limit",
        "projection_timeout",
        "device",
        "evaluator_python",
        "gp_history_limit",
        "acquisition_beta",
    )
    scientific = {k: configuration[k] for k in identity_keys}
    scientific["original_target"] = target
    scientific["source_archive_digest"] = contract.benchmark["source_commit"]
    scientific["asset_digests"] = args.asset_digests
    identity_path = run_dir / "scientific_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != scientific:
        raise ValueError("resume scientific configuration mismatch")
    atomic_json_write(identity_path, scientific)
    snapshot_experiment_contract(contract, run_dir, profile=profile)
    projector = Projector(args, run_dir)
    sink = DataCollectionSink.from_env(default_root=run_dir / "ldm_data")
    client = None
    if args.proposal_mode == "openai":
        if not args.llm_url or not args.llm_model:
            raise ValueError(
                "Set LLM_BASE_URL and LLM_MODEL_NAME for real LDM proposals"
            )
        client = OpenAICompatibleProposalClient(
            url=args.llm_url,
            model=args.llm_model,
            api_key=os.environ.get(
                "LLM_API_KEY",
                os.environ.get("LDM_LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
            ),
            max_tokens=args.llm_max_tokens,
            max_retries=0,
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
    projection_cap = cap if target else args.iterations * args.reservoir_size
    extra = {
        "llm_requests": args.iterations if client else 0,
        "proposal_attempts": args.iterations if client else 0,
        "oracle_calls": args.max_oracle_calls if not target else 0,
        "projection_targets": projection_cap,
        "projection_cycle_allowance": projection_cap * args.num_cycles,
        "endpoint_preflight_requests": (
            json.loads((run_dir / "budget.json").read_text())
            .get("counters", {})
            .get("endpoint_preflight_requests", 0)
            + 1
            if resume and (run_dir / "budget.json").exists()
            else 1
        )
        if client
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
    holder = {}

    def runtime_hook(runtime):
        holder["runtime"] = runtime
        projector.before_project = lambda count: runtime.consume_many(
            {
                "projection_targets": count,
                "projection_cycle_allowance": count * args.num_cycles,
            }
        )
        expander.before_request = lambda: runtime.consume("llm_requests")
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
    except EndpointRequestError:
        runtime = holder.get("runtime")
        if runtime:
            runtime.pause(
                "paused_endpoint_unavailable",
                phase="proposal_preflight_or_expansion",
                message="Configured proposal endpoint unavailable; resolve service and resume this directory.",
            )
        return {"status": "paused_endpoint_unavailable"}, 2
    return result.projected, 0 if result.engine.summary[
        "successful_evaluation_count"
    ] else 1


def _report(args, runtime, target):
    observations = load_successful_observations(runtime.run_dir / "checkpoint.json")
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
