"""Run the CDRH3 optimization campaign with asynchronous evaluation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ldm_tts.contracts import (AcquisitionSpec, CandidateDomainSpec, LDMTaskSpec, ObjectiveSpec,
    ProposalSearchSpec, ReservoirExpansionSpec, ReservoirSpec, ResponseSpaceSpec, SurrogateSpaceSpec)
from tasks.antibody.core.engine_adapters import SEQUENCE_CONSTRAINTS
from .experiments import ENERGY_UTILITY
from .policy import FEATURE_VERSION, H3_LENGTH, FEATURE_NAMES

PI_METHODS = ("harness", "ldm_harness")
METHODS = (*PI_METHODS, "fixed", "antbo")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--oracle", choices=("mock", "absolut"), default="mock")
    p.add_argument("--mock", action="store_true", help="Use deterministic sequence generator fixtures")
    p.add_argument("--concurrency", type=int, default=2, help="Maximum concurrent evaluations")
    p.add_argument("--iterations", type=int, default=2, help="Total evaluation budget")
    p.add_argument("--research-rounds", type=int,
                   help="Research generation budget (default: evaluation budget)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--initialization-run", type=Path,
                   help="Replay initial design jobs from a real-generator campaign for paired comparisons")
    p.add_argument("--pool-size", type=int, default=2)
    p.add_argument("--backbones-per-round", type=int, default=2)
    p.add_argument("--sequences-per-round", type=int, default=32)
    p.add_argument("--samples-per-call", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--share-gpu", action="store_true")
    p.add_argument("--cpu-limit", type=int, default=2)
    p.add_argument("--antigen", default="1ADQ_A")
    p.add_argument("--root", type=Path, default=Path("/Aidata/shared/ldm/antibody-poc"))
    p.add_argument("--absolut-path", type=Path, default=os.environ.get("ABSOLUT_PATH"))
    p.add_argument("--sidecar-image", default=os.environ.get("LDM_PI_IMAGE", "ldm-copper-pi:antibody-async-20261118"))
    p.add_argument("--guest-cache", type=Path, default=os.environ.get("LDM_GONDOLIN_CACHE"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/antibody_rfd3"))
    p.add_argument("--resume-run", type=Path, help="Resume an existing asynchronous campaign directory")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if a.method is None:
        a.method = "fixed" if a.mock else "ldm_harness"
    if a.research_rounds is None:
        a.research_rounds = a.iterations
    if min(a.iterations, a.concurrency, a.cpu_limit, a.pool_size, a.samples_per_call, a.backbones_per_round) < 1:
        p.error("iterations, concurrency, CPU and generation limits must be positive")
    if not 1 <= a.research_rounds <= a.iterations:
        p.error("research-rounds must be positive and cannot exceed the evaluation budget")
    if a.pool_size > a.sequences_per_round or a.samples_per_call > a.sequences_per_round:
        p.error("pool-size and samples-per-call must fit the sequence budget")
    if a.seed < 0 or a.gpu < 0 or not 0.01 <= a.temperature <= 1:
        p.error("seed and GPU index must be nonnegative and temperature in [0.01,1]")
    if a.mock and a.initialization_run:
        p.error("mock generation cannot replay a real-generator initialization")
    return a


def metric(a):
    return "mock_energy" if a.oracle == "mock" else "absolut_energy"


def describe_ldm_task(args):
    gp = args.method == "ldm_harness"
    actions = ("design_h3", "sequence_h3") if args.method in PI_METHODS else ("design_h3",)
    skills = (["cdrh3-design", "cdrh3-experiment-design", "cdrh3-evidence-review", "compile-ldm-policy"]
              if gp else ["cdrh3-design"] if args.method == "harness" else [])
    roles = (["explorer", "refiner", "evidence_reviewer", "policy_architect"] if gp
             else ["cdrh3_research"] if args.method == "harness" else [])
    if args.method == "antbo":
        actions += ("antbo_suggest",)
    return LDMTaskSpec("antibody_rfd3",
        CandidateDomainSpec("CDRH3", "sequence", H3_LENGTH, "canonical amino acids", SEQUENCE_CONSTRAINTS),
        (ObjectiveSpec(metric(args), "minimize"),), (ResponseSpaceSpec("designs", "json"),),
        AcquisitionSpec("ucb" if gp else args.method, (ENERGY_UTILITY if gp else metric(args),), "maximize",
                        "pending-aware sampling with a compiled utility prior and policy weights" if gp
                        else "first admissible submitted H3",
                        metadata={"utility_definition": f"{ENERGY_UTILITY} = -{metric(args)}"} if gp else {}),
        ReservoirSpec("h3_designs", tuple(ReservoirExpansionSpec(action, "emit_candidate", "designs", True)
                      for action in actions), "AntibodyCandidateDomain", "H3 sequence within one antigen"),
        SurrogateSpaceSpec("vector", "position-major CDRH3 one-hot amino acid features", "fixed",
                           dimension=len(FEATURE_NAMES), encoder="tasks.antibody_rfd3.core.policy.featurize",
                           version=FEATURE_VERSION) if gp else SurrogateSpaceSpec("none", "no campaign GP", "none"),
        ProposalSearchSpec(args.method, parameters={"oracle": args.oracle, "mock_generator": args.mock,
                           "concurrency": args.concurrency, "research_rounds": args.research_rounds,
                           "skill_ids": skills, "roles": roles}),
        metadata={"antigen": args.antigen, "template": "1N8Z B1-120", "h3": "B99-B109"})


def run(args):
    task = describe_ldm_task(args)
    if args.dry_run:
        print(json.dumps(task.to_dict(), indent=2))
        return 0
    from .campaign import run as run_campaign
    return run_campaign(args, task)
