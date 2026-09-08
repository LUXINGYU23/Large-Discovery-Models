"""Shared initialization-expander behavior."""

from __future__ import annotations

from ldm_tts.contracts import RawProposal
from ldm_tts.engine.expansion import (
    ExpansionRequest,
    ExpansionResult,
    InitialRoundReservoirExpander,
)


class RecordingExpander:
    def __init__(self, source: str) -> None:
        self.source = source
        self.requests = []

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        self.requests.append(request)
        return ExpansionResult(proposals=(RawProposal({"source": self.source}, self.source),))


def test_initial_round_expander_uses_the_explicit_initial_size_once() -> None:
    initializer = RecordingExpander("init")
    search = RecordingExpander("search")
    expander = InitialRoundReservoirExpander(
        initializer=initializer,
        search_expander=search,
        initial_reservoir_size=3,
    )

    initial = expander.expand(ExpansionRequest(round_idx=0, reservoir_size=64))
    later = expander.expand(ExpansionRequest(round_idx=1, reservoir_size=64))

    assert initializer.requests[0].reservoir_size == 3
    assert search.requests[0].reservoir_size == 64
    assert initial.metadata["phase"] == "shared_initialization"
    assert initial.selection_mode == "reservoir_order"
    assert later.proposals[0].source == "search"
