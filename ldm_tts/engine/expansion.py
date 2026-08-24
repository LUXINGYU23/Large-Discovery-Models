"""Behavioral reservoir-expansion interface and reusable adapters."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

from ldm_tts.contracts import Candidate, Observation, RawProposal
from ldm_tts.transport import ProposalClient, ProposalRequest, ProposalResponse


@dataclass(frozen=True)
class ExpansionRequest:
    """Task-neutral context supplied to one reservoir expansion step."""

    round_idx: int
    reservoir_size: int
    observations: tuple[Observation, ...] = ()
    parent: Candidate | None = None
    expansion_schema: dict[str, Any] = field(default_factory=dict)
    acquisition_feedback: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.round_idx < 0:
            raise ValueError("expansion round_idx must be non-negative")
        if self.reservoir_size < 1:
            raise ValueError("expansion reservoir_size must be positive")


@dataclass(frozen=True)
class ExpansionResult:
    """Raw proposals and optional schema update emitted by an expander."""

    proposals: tuple[RawProposal, ...] = ()
    schema_update: dict[str, Any] | None = None
    attempts: tuple[ProposalResponse, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.proposals and self.schema_update is None and not self.attempts:
            raise ValueError("expansion must emit proposals or update the expansion schema")


@runtime_checkable
class ReservoirExpander(Protocol):
    """Task-owned behavioral seam for reservoir expansion."""

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        ...


class CallableReservoirExpander:
    """Local adapter around deterministic task or test expansion logic."""

    def __init__(self, operation: Callable[[ExpansionRequest], ExpansionResult]) -> None:
        self.operation = operation

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        return self.operation(request)


class InitialRoundReservoirExpander:
    """Use a deterministic initializer once, then delegate to the active search expander."""

    def __init__(
        self,
        *,
        initializer: ReservoirExpander,
        search_expander: ReservoirExpander,
        initial_reservoir_size: int,
    ) -> None:
        if initial_reservoir_size < 1:
            raise ValueError("initial reservoir size must be positive")
        self.initializer = initializer
        self.search_expander = search_expander
        self.initial_reservoir_size = initial_reservoir_size

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        if request.round_idx == 0:
            result = self.initializer.expand(
                replace(request, reservoir_size=self.initial_reservoir_size)
            )
            metadata = {"phase": "shared_initialization", **result.metadata}
            return ExpansionResult(
                proposals=result.proposals,
                schema_update=result.schema_update,
                attempts=result.attempts,
                metadata=metadata,
            )
        return self.search_expander.expand(request)


class DirectEmissionExpander:
    """Use one proposal client turn to emit raw candidate payloads directly."""

    def __init__(
        self,
        *,
        client: ProposalClient,
        build_request: Callable[[ExpansionRequest], ProposalRequest],
        parse_response: Callable[[ProposalResponse], Iterable[RawProposal | Any]],
        source: str = "llm_direct_emission",
    ) -> None:
        self.client = client
        self.build_request = build_request
        self.parse_response = parse_response
        self.source = source

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        response = self.client.propose(self.build_request(request))
        proposals = tuple(
            item if isinstance(item, RawProposal) else RawProposal(item, self.source)
            for item in self.parse_response(response)
        )
        if not proposals:
            raise ValueError("direct-emission response produced no raw proposals")
        return ExpansionResult(
            proposals=proposals,
            attempts=(response,),
            metadata={"mode": "direct_emission"},
        )


__all__ = [
    "CallableReservoirExpander",
    "DirectEmissionExpander",
    "ExpansionRequest",
    "ExpansionResult",
    "InitialRoundReservoirExpander",
    "ReservoirExpander",
]
