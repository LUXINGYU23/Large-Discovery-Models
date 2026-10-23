"""Assemble one shared runtime/engine and task-owned science and execution adapters."""

import argparse
import csv
import json
import os
from pathlib import Path
from contextlib import ExitStack
from ldm_tts.campaign import CampaignRequest, CampaignBudget
from ldm_tts.engine import LDMEngine, LDMEngineState, InitialRoundReservoirExpander
from ldm_tts.engine.expansion import CallableReservoirExpander, ExpansionResult
from ldm_tts.contracts import RawProposal
from ldm_tts.contracts.evaluation import EvaluationPaused
from ldm_tts.harness import HarnessError
from ldm_tts.engine.run_store import atomic_json_write
from .data import CASES, load_problem, RESOURCES
from .candidate import seal, Domain
from .harness import Expander
from .task_spec import describe_ldm_task
from ldm_tts.harness.research_strategy import configure_research_strategy


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Agent-designed geological inversion")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--search-method", choices=("harness", "ldm_harness"), default="harness")
    p.add_argument("--harness-research-strategy-config", type=Path)
    p.add_argument("--case-id", choices=CASES, default=CASES[0])
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--out-dir", type=Path, default=Path("runs/copper"))
    p.add_argument("--solver-seconds", type=int, default=None)
    p.add_argument("--evaluator-image", default="ldm-copper-evaluator:20261023")
    p.add_argument("--sidecar-image", default="ldm-pi-harness:copper-20261023")
    p.add_argument(
        "--harness-cache-dir",
        type=Path,
        default=Path("/mnt/data1/Large-Discovery-Models/cache/gondolin"),
    )
    p.add_argument("--harness-seconds", type=int, default=900)
    p.add_argument("--llm-url", default="https://api.deepseek.com")
    p.add_argument("--llm-model", default="deepseek-flash")
    p.add_argument("--api-key-file", type=Path, default=None)
    p.add_argument("--thinking", default="low")
    a = p.parse_args(argv)
    if a.iterations < 0 or a.harness_seconds < 1:
        p.error("invalid budget")
    a.solver_seconds = a.solver_seconds or (60 if a.case_id.endswith("2d") else 120)
    if a.solver_seconds < 1:
        p.error("invalid solver seconds")
    configure_research_strategy(a, enabled=a.search_method == "ldm_harness", measurement_supported=False)
    return a


def main(argv=None):
    args = parse_args(argv)
    spec = describe_ldm_task(args)
    if args.dry_run:
        print(json.dumps(spec.to_dict()))
        return 0
    root = args.out_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    from .execution import source_identity

    config = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if not k.startswith("_") and k not in ("resume", "out_dir", "api_key_file", "dry_run")
    }
    config.update(
        implementation_sha256=source_identity(),
        llm_url=args.llm_url or os.environ.get("LLM_BASE_URL"),
        llm_model=args.llm_model or os.environ.get("LLM_MODEL_NAME"),
    )
    # Environment image tags are resolved before scientific resume identity is frozen.
    if not args.mock:
        from .execution import docker

        config["evaluator_image"] = docker(
            "image", "inspect", args.evaluator_image, "--format", "{{.Id}}"
        ).stdout.strip()
        config["sidecar_image"] = docker(
            "image", "inspect", args.sidecar_image, "--format", "{{.Id}}"
        ).stdout.strip()
    researchers = 4 if args.search_method == "ldm_harness" else 1
    main_turns = args.iterations if args._research_strategy.requires_main else 0
    budget = CampaignBudget(
        rounds=args.iterations + 1,
        reservoir_size=researchers,
        batch_size=1,
        target_observations=args.iterations + 1,
        max_evaluation_attempts=args.iterations + 1,
        extra_limits={
            "benchmark_jobs": 2 * (args.iterations + 1),
            "harness_turns": researchers * args.iterations + main_turns,
            "main_research_turns": main_turns,
            "interface_jobs": 6 * (researchers * args.iterations + 1),
            "solver_jobs": args.iterations + 1,
            "scorer_jobs": args.iterations + 1,
        },
    )
    from ldm_tts.registration.experiment import load_active_experiment_contract

    contract, profile = load_active_experiment_contract()
    request = CampaignRequest(
        run_dir=root,
        budget=budget,
        config=config,
        resume=args.resume,
        contract_snapshot=contract.to_dict() if contract else None,
        contract_sha256=contract.digest if contract else "",
        contract_profile=profile,
    )
    runtime = request.open_runtime(spec)
    try:
        with ExitStack() as stack:
            problem = load_problem(args.case_id, fixture=args.mock)
            if args.mock:
                from .mock import MockEvaluator

                checks = lambda payload: True
                evaluator = MockEvaluator()
            else:
                from .execution import Execution
                from .interface_checks import InterfaceChecks
                from .evaluator import Evaluator

                execution = Execution(
                    root / "jobs", config["evaluator_image"], runtime.consume_many
                )
                checks = InterfaceChecks(root, execution, args.case_id)
                evaluator = Evaluator(root, execution, problem, args.solver_seconds)
                args.sidecar_image = config["sidecar_image"]
            from .harness_runtime import open_search
            search = open_search(args, root, problem, stack, checks, runtime)
            seed = seal(root, (RESOURCES / "seeds/solver.py").read_bytes())

            def initialize(_):
                checks(seed)
                return ExpansionResult(
                    proposals=(RawProposal(seed, "fixed_seed"),),
                    selection_mode="reservoir_order",
                )

            expander = InitialRoundReservoirExpander(
                initializer=CallableReservoirExpander(initialize),
                search_expander=search,
                initial_reservoir_size=1,
            )
            state = LDMEngineState()
            if args.resume:
                checkpoint = runtime.load_checkpoint()
                if checkpoint is not None:
                    state = LDMEngineState.from_checkpoint(checkpoint)
            from .surrogate import CodeEncoder, LossSelector
            guided = args.search_method == "ldm_harness"
            engine = LDMEngine(
                task_spec=spec,
                expander=expander,
                candidate_domain=Domain(root),
                evaluator=evaluator,
                runtime=runtime,
                surrogate_encoder=CodeEncoder(root) if guided else None,
                selector=LossSelector() if guided else None,
            )
            outcome = engine.run(budget.engine_config(), state=state)
            if args._research_strategy.requires_main and args.iterations:
                from ldm_tts.harness.research_runtime import write_fork_provenance
                write_fork_provenance(root / "harness", compiled_policy=False)
            observations = outcome.state.observations
            successful = [o for o in observations if o.evaluation.succeeded]
            best = min(successful, key=lambda o: o.metrics["loss_total"], default=None)
            report = {
                "task": "copper_inversion",
                "mock": args.mock,
                "case_id": args.case_id,
                "observations": len(observations),
                "successful": len(successful),
                "best": best.to_dict() if best else None,
            }
            atomic_json_write(root / "result.json", report)
            if best:
                (root / "best_solver.py").write_bytes(
                    (root / best.candidate.payload["source_path"]).read_bytes()
                )
            with (root / "trajectory.csv").open("w") as f:
                w = csv.writer(f)
                w.writerow(["round", "candidate", "status", "loss_total"])
                for o in observations:
                    w.writerow(
                        [
                            o.round_idx,
                            o.canonical_key,
                            o.evaluation.status,
                            o.metrics.get("loss_total", ""),
                        ]
                    )
            print(
                json.dumps(
                    {
                        "task": "copper_inversion",
                        "run_dir": str(root),
                        "observations": len(observations),
                        "best_loss": best.metrics["loss_total"] if best else None,
                    }
                )
            )
            return 0 if len(observations) == args.iterations + 1 and successful else 1
    except HarnessError as exc:
        runtime.pause("paused_endpoint_unavailable", phase="harness", message=str(exc))
        raise
    except EvaluationPaused as exc:
        runtime.status.update(
            "paused_outcome_unknown",
            phase="evaluation",
            details={"error": str(exc)},
            budget=runtime.budget,
        )
        raise
