"""Candidate-program admission and public program features for the GP."""

from __future__ import annotations

import ast
from typing import Any

from ldm_tts.contracts import Candidate, CandidateRejection, RawProposal, SurrogateSpaceSpec
from ldm_tts.optimization.records import SurrogateVector

from .cases import CaseSpec

FEATURE_VERSION = "rg_program_ast_family_v2"
# Method families are read from called functions and attributes. Together with
# six AST size statistics they describe what a program does, never who it is:
# no content hash, candidate identity, or proposal frequency enters the vector.
METHOD_FAMILIES = (
    ("confidence", ("entropy", "softmax", "log_softmax", "logsumexp", "nll_loss", "cross_entropy", "kl_div")),
    ("selection", ("quantile", "topk", "argsort", "sort", "masked_select", "where", "nonzero", "median", "percentile")),
    ("geometry", ("normalize", "layer_norm", "batch_norm", "std", "var", "cov", "cdist", "pdist", "whiten")),
    ("memory", ("register_buffer", "deepcopy", "clone", "detach", "ema", "momentum", "state_dict", "load_state_dict")),
    ("views", ("dropout", "flip", "crop", "interpolate", "augment", "noise", "rand_like", "randn_like")),
    ("control", ("zero_grad", "step", "clip_grad_norm_", "clip_grad_value_", "param_groups", "requires_grad_",
                 "no_grad", "enable_grad")),
)
SIZE_FEATURES = ("program_chars", "function_defs", "loops", "branches", "calls", "numeric_constants")
FEATURE_NAMES = SIZE_FEATURES + tuple(f"family_{name}" for name, _ in METHOD_FAMILIES)
FEATURE_GROUPS = {"program_size": (0, len(SIZE_FEATURES)), "method_families": (len(SIZE_FEATURES), len(FEATURE_NAMES))}


def program_features(program: str, *, max_chars: int) -> tuple[float, ...]:
    tree = ast.parse(program)
    nodes = list(ast.walk(tree))

    def count(*types):
        return sum(isinstance(node, types) for node in nodes)

    names = []
    for node in nodes:
        if isinstance(node, ast.Call):
            func = node.func
            names.append(func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", ""))
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
    lowered = [name.lower() for name in names]
    families = tuple(min(sum(any(key in name for key in keys) for name in lowered) / 6.0, 1.0)
                     for _, keys in METHOD_FAMILIES)
    numeric = sum(isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)
                  for n in nodes)
    return (
        min(len(program) / float(max_chars), 1.0),
        min(count(ast.FunctionDef, ast.AsyncFunctionDef) / 10.0, 1.0),
        min(count(ast.For, ast.While) / 10.0, 1.0),
        min(count(ast.If) / 10.0, 1.0),
        min(count(ast.Call) / 60.0, 1.0),
        min(numeric / 40.0, 1.0),
        *families,
    )


class ProgramDomain:
    """Task-owned admission with the same rules as Harness submission validation."""

    def __init__(self, case: CaseSpec) -> None:
        self.case = case

    def normalize(self, payload: Any) -> tuple[str, str]:
        program = payload.get("program") if isinstance(payload, dict) else None
        errors = self.case.check_program(program)
        if errors:
            raise ValueError("; ".join(f"{e['code']}: {e['message']}" for e in errors))
        program = program.strip() + "\n"
        return program, self.case.canonical_key(program)

    def admit(self, proposal: RawProposal) -> Candidate | CandidateRejection:
        if not isinstance(proposal.payload, dict) or set(proposal.payload) != {"program"}:
            return CandidateRejection("invalid_program_payload", "payload must contain only program", proposal.source)
        try:
            program, key = self.normalize(proposal.payload)
        except ValueError as exc:
            return CandidateRejection("invalid_program", str(exc), proposal.source)
        return Candidate(f"rg-{key[:16]}", {"program": program}, key, proposal.source,
                         metadata=dict(proposal.metadata))


class ProgramEncoder:
    version = FEATURE_VERSION

    def __init__(self, case: CaseSpec) -> None:
        self.max_chars = case.rules["max_program_chars"]

    def describe(self) -> SurrogateSpaceSpec:
        return SurrogateSpaceSpec(
            kind="vector",
            representation="Six AST size statistics and six method-family indicators over called names",
            dimension_policy="fixed",
            dimension=len(FEATURE_NAMES),
            encoder="tasks.researchgym.core.candidate:ProgramEncoder",
            version=FEATURE_VERSION,
        )

    def encode(self, candidate: Candidate) -> SurrogateVector:
        return SurrogateVector(program_features(candidate.payload["program"], max_chars=self.max_chars),
                               FEATURE_VERSION, candidate.candidate_id)
