"""Small deterministic assets for the shared-engine mock campaign."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from itertools import product

from ldm_tts.contracts import (
    AcquisitionSpec,
    LDMTaskSpec,
    ObjectiveSpec,
    ProposalSearchSpec,
    RawProposal,
    SurrogateSpaceSpec,
)
from ldm_tts.engine import InitialRoundReservoirExpander
from ldm_tts.engine.expansion import (
    CallableReservoirExpander,
    ExpansionRequest,
    ExpansionResult,
)
from tasks.nucleobench.core.candidate import MutationContext
from tasks.nucleobench.core.cases import NucleoBenchCase
from tasks.nucleobench.core.task_spec import build_task_spec

MOCK_START_SEQUENCE = "AAAAAAAA"
MOCK_EDITABLE_POSITIONS = (0, 2, 4, 6)
MOCK_CASE = NucleoBenchCase(
    case_id="mock_dna",
    model_family="mock",
    model_name="deterministic_base_energy",
    target="mock_utility",
    model_selector={},
    sequence_length=len(MOCK_START_SEQUENCE),
    editable_position_count=len(MOCK_EDITABLE_POSITIONS),
    max_seconds=1,
    state="planned",
)
MOCK_CONTEXT = MutationContext(
    case=MOCK_CASE,
    start_set_digest=hashlib.sha256(
        json.dumps([MOCK_START_SEQUENCE], separators=(",", ":")).encode()
    ).hexdigest(),
    start_index=0,
    start_sequence=MOCK_START_SEQUENCE,
    editable_positions=MOCK_EDITABLE_POSITIONS,
)


def build_mock_expander() -> InitialRoundReservoirExpander:
    return InitialRoundReservoirExpander(
        initializer=CallableReservoirExpander(_initial_expansion),
        search_expander=CallableReservoirExpander(_active_expansion),
        initial_reservoir_size=1,
    )


def build_mock_task_spec() -> LDMTaskSpec:
    task_spec = build_task_spec(MOCK_CASE, search_method="llm")
    metadata = {
        key: value
        for key, value in task_spec.metadata.items()
        if key
        not in {
            "model_requests_per_round",
            "candidates_per_model_request",
            "search_breadth",
        }
    }
    return replace(
        task_spec,
        acquisition=AcquisitionSpec(
            name="deterministic_mock_order",
            objective_names=("utility",),
            score_direction="maximize",
            selection_rule="Deterministic reservoir order for contract verification.",
        ),
        objectives=(
            ObjectiveSpec(
                name="utility",
                direction="maximize",
                description="Negative deterministic mock energy.",
            ),
        ),
        surrogate=SurrogateSpaceSpec(
            kind="none",
            representation="Disabled for deterministic contract verification.",
            dimension_policy="none",
        ),
        proposal_search=ProposalSearchSpec(
            name="deterministic_mock_enumeration",
            evaluation_policy="task evaluator boundary with deterministic mock energy",
        ),
        reservoir=replace(task_spec.reservoir, max_size=None),
        metadata={
            **metadata,
            "execution_profile": "mock",
            "search_method": "mock",
        },
    )


def mock_energies(sequences: list[str] | tuple[str, ...]) -> tuple[float, ...]:
    weights = {"A": 0.0, "C": 1.0, "G": 2.0, "T": 0.5}
    return tuple(
        -sum((index + 1) * weights[base] for index, base in enumerate(sequence))
        for sequence in sequences
    )


def _initial_expansion(request: ExpansionRequest) -> ExpansionResult:
    return ExpansionResult(
        proposals=(
            RawProposal(
                {"mutations": [{"position": 0, "base": "C"}]},
                "mock_shared_initialization",
                {"collectable": True, "round_idx": request.round_idx},
            ),
        ),
        metadata={"phase": "shared_initialization"},
        selection_mode="reservoir_order",
    )


def _active_expansion(request: ExpansionRequest) -> ExpansionResult:
    evaluated = {
        json.dumps(
            observation.candidate.payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        for observation in request.observations
    }
    proposals = []
    for bases in product("ACGT", repeat=len(MOCK_EDITABLE_POSITIONS)):
        mutations = [
            {"position": position, "base": base}
            for position, base in zip(MOCK_EDITABLE_POSITIONS, bases, strict=True)
            if base != "A"
        ]
        if not mutations:
            continue
        payload = {"mutations": mutations}
        key = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if key in evaluated:
            continue
        proposals.append(
            RawProposal(
                payload,
                "mock_active_search",
                {"collectable": True, "round_idx": request.round_idx},
            )
        )
        if len(proposals) == request.reservoir_size:
            break
    if len(proposals) != request.reservoir_size:
        raise ValueError("mock mutation domain cannot fill the requested reservoir")
    return ExpansionResult(tuple(proposals), metadata={"phase": "active_search"})


__all__ = [
    "MOCK_CASE",
    "MOCK_CONTEXT",
    "MOCK_EDITABLE_POSITIONS",
    "MOCK_START_SEQUENCE",
    "build_mock_expander",
    "build_mock_task_spec",
    "mock_energies",
]
