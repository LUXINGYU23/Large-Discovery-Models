"""Shared runner interface for the AlphaBench T3 campaign."""

import argparse
import json
from pathlib import Path

from ldm_tts.contracts import (AcquisitionSpec, CandidateDomainSpec, LDMTaskSpec, ObjectiveSpec,
    ProposalSearchSpec, ReservoirExpansionSpec, ReservoirSpec, ResponseSpaceSpec, SurrogateSpaceSpec)
from tasks.alphabench.core.protocol import LDM_METHODS, METHODS, PROFILES, T3Protocol
from tasks.alphabench.core.selection import FactorEncoder, FactorSelector
from tasks.alphabench.core.workflow import run


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Complete AlphaBench T3 task")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--protocol-profile", choices=PROFILES)
    parser.add_argument("--backend", choices=("qlib", "assay"))
    parser.add_argument("--market")
    parser.add_argument("--init-mode", choices=("cold", "alpha158", "file", "import_pool"))
    for name in ("iterations", "evaluations", "batch-size", "sessions", "candidates-per-session", "random-seed"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--resume-run", type=Path)
    for name in ("seed-file", "import-pool", "initialization-bundle", "upstream-root", "data-manifest", "protocol-file"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--oracle-url")
    return parser.parse_args(argv)


def resolve_protocol(args):
    overrides = {key: getattr(args, key) for key in ("method", "backend", "market", "init_mode", "evaluations",
        "batch_size", "sessions", "candidates_per_session", "random_seed") if getattr(args, key) is not None}
    if args.iterations is not None:
        overrides["rounds"] = args.iterations
    if args.protocol_profile is not None:
        overrides["profile"] = args.protocol_profile
    if args.protocol_file:
        payload = json.loads(args.protocol_file.read_text(encoding="utf-8"))
        if any(payload.get(key) != value for key, value in overrides.items()):
            raise ValueError("CLI overrides differ from the frozen protocol file")
        return T3Protocol(**payload)
    return T3Protocol(**overrides)


def describe_ldm_task(args):
    protocol = resolve_protocol(args)
    objective = ("mock_" if args.mock else "") + protocol.objective
    guided = protocol.method in LDM_METHODS
    return LDMTaskSpec(task="alphabench",
        candidate_domain=CandidateDomainSpec("factor_expression", "symbolic", None,
            "Full pinned T3 Qlib/Assay expressions; safe canonical AST identity"),
        objectives=(ObjectiveSpec(objective, "maximize", "Signed search-period cross-sectional daily correlation"),),
        response_spaces=(ResponseSpaceSpec("factors", "json", "Named expressions in a candidates array"),),
        acquisition=FactorSelector(objective).describe() if guided else AcquisitionSpec("reservoir_order", (objective,), "maximize", "Accepted response order"),
        reservoir=ReservoirSpec("factor_reservoir", (ReservoirExpansionSpec("factor_proposal", "emit_candidate", "factors", True,
            "Generate and repair complete causal factor expressions"),), "Static grammar followed by metered backend checks", "SHA256(grammar,dialect,canonical expression)"),
        surrogate=FactorEncoder().describe() if guided else SurrogateSpaceSpec("none", "No surrogate", "none"),
        proposal_search=ProposalSearchSpec(protocol.method), metadata={"mock": args.mock, "protocol_digest": protocol.identity})


def main(argv=None):
    args = parse_args(argv)
    protocol = resolve_protocol(args)
    spec = describe_ldm_task(args)
    if args.dry_run:
        print(json.dumps({"task": "alphabench", "protocol": protocol.to_dict(), "ldm_task_spec": spec.to_dict()}, indent=2))
        return 0
    return run(args, protocol, spec)


if __name__ == "__main__":
    raise SystemExit(main())
