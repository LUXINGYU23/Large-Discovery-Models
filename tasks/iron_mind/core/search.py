"""Task-local deterministic initialization and finite-domain BO expansion."""

from __future__ import annotations

import random
from collections.abc import Sequence

from ldm_tts.contracts import RawProposal
from ldm_tts.engine.expansion import ExpansionRequest, ExpansionResult

from tasks.iron_mind.core.candidate import (
    IRON_MIND_Q0_METADATA_KEY,
    canonical_candidate_key,
)
from tasks.iron_mind.core.data import FrozenReactionTable


INITIALIZATION_MODES = ("none", "shared_random")
SEARCH_METHODS = ("ldm", "ldm_harness", "bo", "llm", "harness")


class IronMindInitializationExpander:
    """Draw a fixed, seed-controlled initial design from the official finite table."""

    def __init__(self, table: FrozenReactionTable, *, seed: int, attach_q0: bool) -> None:
        self.payloads = _canonical_payloads(table)
        self.seed = seed
        self.attach_q0 = attach_q0

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        sample_size = min(request.reservoir_size, len(self.payloads))
        selected = random.Random(self.seed).sample(self.payloads, sample_size)
        proposals = tuple(
            RawProposal(payload, "iron_mind_shared_initialization", self._metadata(sample_size))
            for payload in selected
        )
        return ExpansionResult(proposals=proposals, metadata={"initialization_size": sample_size})

    def _metadata(self, sample_size: int) -> dict[str, object]:
        metadata: dict[str, object] = {"collectable": False}
        if self.attach_q0:
            metadata[IRON_MIND_Q0_METADATA_KEY] = {
                "occurrence_count": 1,
                "valid_occurrence_count": sample_size,
                "probability": 1.0 / sample_size,
            }
        return metadata


class FullReactionDomainExpander:
    """Expose every unseen finite reaction condition to the task-local BO selector."""

    def __init__(self, table: FrozenReactionTable) -> None:
        self.payloads = _canonical_payloads(table)

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        proposals = tuple(
            RawProposal(payload, "iron_mind_full_domain_bo", {"collectable": False})
            for payload in self.payloads
        )
        return ExpansionResult(
            proposals=proposals,
            metadata={"search_space": "full_unseen_finite_reaction_domain", "domain_size": len(proposals)},
        )


def finite_domain_size(table: FrozenReactionTable) -> int:
    """Return the number of unique condition tuples available to search."""

    return len(table.rows_by_conditions)


def _canonical_payloads(table: FrozenReactionTable) -> tuple[dict[str, object], ...]:
    payloads = _unique_payloads(table)
    return tuple(sorted(payloads, key=canonical_candidate_key))


def _unique_payloads(table: FrozenReactionTable) -> Sequence[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for conditions in table.rows_by_conditions:
        payloads.append(
            {
                "dataset_id": table.schema.dataset_id,
                "conditions": {
                    name: conditions[index]
                    for index, name in enumerate(table.schema.factor_names)
                },
            }
        )
    return payloads


__all__ = [
    "FullReactionDomainExpander",
    "INITIALIZATION_MODES",
    "IronMindInitializationExpander",
    "SEARCH_METHODS",
    "finite_domain_size",
]
