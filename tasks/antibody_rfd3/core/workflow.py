"""Run the shared CDRH3 optimization campaign with structure-based proposals."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ldm_tts.campaign import CampaignBudget, CampaignRecipe, CampaignRequest, run_campaign
from ldm_tts.contracts import (AcquisitionSpec, CandidateDomainSpec, EvaluationResult, LDMTaskSpec,
    ObjectiveSpec, ProposalSearchSpec, RawProposal, ReservoirExpansionSpec, ReservoirSpec,
    ResponseSpaceSpec, SurrogateSpaceSpec)
from ldm_tts.engine.expansion import ExpansionResult
from ldm_tts.engine.run_store import unique_run_dir
from ldm_tts.registration.experiment import load_active_experiment_contract
from tasks.antibody.core.engine_adapters import (AntibodyCandidateDomain, AntibodyEvaluator,
    AntibodyGPSelector, AntibodySurrogateEncoder, SEQUENCE_CONSTRAINTS)
from tasks.antibody.core.ldm_light.ldm_acq import AbsolutEvaluator
from tasks.antibody.core.absolut_oracle import TARGETS
from .evaluator import IsolatedAbsolutEvaluator
from .worker import DesignWorker

PI_METHODS = ("harness", "ldm_harness")
METHODS = (*PI_METHODS, "fixed", "antbo")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--oracle", choices=("mock", "absolut"), default="mock")
    p.add_argument("--mock", action="store_true", help="Use deterministic generator fixtures for an offline campaign")
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--research-instructions", default="", help="Additional research guidance, recorded in each Pi turn")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--initialization-run", type=Path,
                   help="Replay the initial design jobs from a completed real-generator campaign for paired comparisons")
    p.add_argument("--pool-size", type=int, default=2)
    p.add_argument("--backbones-per-round", type=int, default=2)
    p.add_argument("--sequences-per-round", type=int, default=32)
    p.add_argument("--samples-per-call", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--share-gpu", action="store_true")
    p.add_argument("--enable-boltzgen", action="store_true", help="Enable optional host-configured, fixed-framework BoltzGen H3 proposals in Pi rounds")
    p.add_argument("--cpu-limit", type=int, default=2)
    p.add_argument("--antigen", default="1ADQ_A")
    p.add_argument("--root", type=Path, default=Path("/Aidata/shared/ldm/antibody-poc"))
    p.add_argument("--absolut-path", type=Path, default=os.environ.get("ABSOLUT_PATH"))
    p.add_argument("--absolut-resources", action="append", default=[])
    p.add_argument("--absolut-timeout", type=int, default=600)
    p.add_argument("--sidecar-image", default=os.environ.get("LDM_PI_IMAGE", "ldm-pi-harness:latest"))
    p.add_argument("--guest-cache", type=Path, default=os.environ.get("LDM_GONDOLIN_CACHE"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/antibody_rfd3"))
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if a.method is None:
        a.method = "fixed" if a.mock else "ldm_harness"
    if a.absolut_timeout <= 0:
        p.error("absolut-timeout must be positive")
    if min(a.iterations, a.cpu_limit, a.pool_size, a.samples_per_call, a.backbones_per_round) < 1:
        p.error("iterations, CPU and generation limits must be positive")
    if a.pool_size > a.sequences_per_round or a.samples_per_call > a.sequences_per_round:
        p.error("pool-size and samples-per-call must fit the sequence budget")
    if a.seed < 0 or a.gpu < 0 or not 0.01 <= a.temperature <= 1:
        p.error("seed and GPU index must be nonnegative and temperature in [0.01,1]")
    if a.mock and (a.method != "fixed" or a.oracle != "mock"):
        p.error("--mock is the offline fixed-method fixture. Use --oracle mock alone for real tools and Pi.")
    if a.enable_boltzgen and a.method not in PI_METHODS:
        p.error("BoltzGen is an optional Pi proposal tool; select harness or ldm_harness")
    return a


def metric(a):
    return "mock_energy" if a.oracle == "mock" else "absolut_energy"


def describe_ldm_task(args):
    gp = args.method == "ldm_harness"
    actions = ("design_h3", "sequence_h3", "sequence_h3_soluble") if args.method in PI_METHODS else ("design_h3",)
    if getattr(args, "enable_boltzgen", False) and args.method in PI_METHODS:
        actions += ("design_h3_boltzgen",)
    if args.method == "antbo":
        actions += ("antbo_suggest",)
    return LDMTaskSpec("antibody_rfd3",
        CandidateDomainSpec("CDRH3", "sequence", 11, "canonical amino acids", SEQUENCE_CONSTRAINTS),
        (ObjectiveSpec(metric(args), "minimize"),), (ResponseSpaceSpec("designs", "json"),),
        AcquisitionSpec("ei" if gp else args.method, (metric(args),), "maximize",
                        "maximum expected improvement" if gp else "first admissible submitted H3"),
        ReservoirSpec("h3_designs", tuple(ReservoirExpansionSpec(action, "emit_candidate", "designs", True)
                      for action in actions), "AntibodyCandidateDomain", "H3 sequence within one antigen"),
        AntibodySurrogateEncoder(11).describe() if gp else SurrogateSpaceSpec("none", "no campaign GP", "none"),
        ProposalSearchSpec(args.method, parameters={"oracle": args.oracle, "mock_generator": args.mock,
                           "skill_ids": ["cdrh3-design"] if args.method in PI_METHODS else []}),
        metadata={"antigen": args.antigen, "template": "1N8Z B1-120", "h3": "B99-B109"})


class MockOracle:
    def __init__(self, antigen):
        self.antigen = antigen

    def evaluate(self, candidate):
        seq = candidate.payload["sequence"]
        digest = hashlib.sha256(f"mock-v1/{self.antigen}/{seq}".encode()).digest()
        value = -int.from_bytes(digest[:4], "big") / 2**32
        return EvaluationResult(candidate.candidate_id, "succeeded", {"mock_energy": value},
                                metadata={"oracle": "mock-v1", "sequence": seq})


class Expander:
    def __init__(self, args, worker):
        self.args, self.worker = args, worker
        self.agent = self.native = None

    def expand(self, request):
        a = self.args
        self.worker.round = request.round_idx
        self.worker.evaluated = {o.candidate.canonical_key for o in request.observations}
        history = [dict(round=o.round_idx, sequence=o.candidate.payload["sequence"],
                        backbone_id=o.candidate.payload.get("backbone_id"),
                        design_id=o.candidate.payload.get("design_id"),
                        result_artifacts=dict(o.evaluation.artifacts),
                        energy=o.evaluation.metrics[metric(a)], oracle=a.oracle)
                   for o in request.observations if o.evaluation.succeeded]
        count = a.pool_size if history and a.method == "ldm_harness" else 1
        if self.agent:
            rows = self.agent.propose(request.round_idx, history, count)
        elif history and a.method == "antbo":
            from .baselines import NativeAntBO
            if self.native is None:
                self.native = NativeAntBO(a.seed)
            rows = [self.native.propose(history)]
        else:
            seen = {o.candidate.canonical_key for o in request.observations}
            rows = []
            while not rows and self.worker.remaining()["backbones"] > 0:
                count_left = min(a.samples_per_call, self.worker.remaining()["sequences"])
                if count_left < 1:
                    break
                generated = self.worker.execute("design_h3", count_left, a.temperature)
                rows = [r for r in generated if not r["violations"] and r["sequence"] not in seen][:1]
            if not rows:
                raise RuntimeError("Fixed proposal budget exhausted without an admissible H3")
        return ExpansionResult(tuple(RawProposal(row, a.method) for row in rows),
                               selection_mode="acquisition" if history and a.method == "ldm_harness" else "reservoir_order",
                               metadata={"oracle": a.oracle, "history_count": len(history),
                                         "generation_remaining": self.worker.remaining()})


def run(a):
    if a.dry_run:
        print(json.dumps(describe_ldm_task(a).to_dict(), indent=2))
        return 0
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:a.cpu_limit])
    directory = unique_run_dir(a.out_dir.resolve())
    if a.oracle == "mock":
        evaluator = MockOracle(a.antigen)
    elif TARGETS.get(a.antigen, a.antigen) in TARGETS.values():
        if not a.absolut_path:
            raise ValueError("Set --absolut-path or ABSOLUT_PATH")
        evaluator = IsolatedAbsolutEvaluator(a, directory)
    else:
        if not a.absolut_path or not (a.absolut_path / "src/bin/Absolut").is_file():
            raise ValueError("Set --absolut-path or ABSOLUT_PATH to the installation with antigen data")
        evaluator = AntibodyEvaluator(AbsolutEvaluator(dict(antigen=a.antigen, process=1,
            startTask=min(os.sched_getaffinity(0)), path=str(a.absolut_path.resolve())), f"h3-{os.getpid()}"))
    worker = DesignWorker(directory, a)
    expander = Expander(a, worker)
    selector = encoder = None
    if a.method == "ldm_harness":
        settings = argparse.Namespace(acq="ei", acq_beta=1.0, acq_xi=0.001, n_init=1, device="cpu")
        selector = AntibodyGPSelector(args=settings, method_spec={"reduction": "max"},
                                      rng=np.random.default_rng(a.seed), objective_name=metric(a))
        encoder = AntibodySurrogateEncoder(11)
    contract, profile = load_active_experiment_contract()
    with ExitStack() as stack:
        def setup(runtime):
            worker.runtime = runtime
            if a.method in PI_METHODS:
                from .harness import open_agent
                try:
                    expander.agent = stack.enter_context(open_agent(a, runtime, worker))
                except Exception as exc:
                    runtime.fail(exc)
                    raise
        budget = CampaignBudget(a.iterations, a.pool_size, target_observations=a.iterations,
                                max_evaluation_attempts=a.iterations, extra_limits={
                                    "backbones_requested": a.iterations * a.backbones_per_round,
                                    "sequences_requested": a.iterations * a.sequences_per_round})
        request = CampaignRequest(directory, budget,
                                  config={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                                  runtime_hook=setup, contract_snapshot=contract.to_dict() if contract else None,
                                  contract_sha256=contract.digest if contract else "", contract_profile=profile)
        result = run_campaign(request, CampaignRecipe(describe_ldm_task(a), expander,
                              AntibodyCandidateDomain(11), evaluator, selector, encoder))
    print(json.dumps({"run_dir": str(directory), "oracle": a.oracle, "stop_reason": result.engine.stop_reason}))
    return int(sum(o.evaluation.succeeded for o in result.engine.state.observations) != a.iterations)
