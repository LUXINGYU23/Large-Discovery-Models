"""Task-local randomized initialization and score-blind BO candidate pools."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Sequence
from typing import Any

from ldm_tts.contracts import RawProposal
from ldm_tts.engine.expansion import ExpansionRequest, ExpansionResult

from tasks.synthonbench.core.constants import Q0_METADATA_KEY


INITIALIZATION_MODES = ("none", "shared_random")
SEARCH_METHODS = ("ldm", "bo", "llm")


class SynthonInitializationExpander:
    """Sample a shared, product-uniform initial design from the official space."""

    def __init__(self, space: Any, reactions: Sequence[str], *, seed: int, attach_q0: bool) -> None:
        self.space = space
        self.reactions = tuple(reactions)
        self.seed = seed
        self.attach_q0 = attach_q0

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        payloads = sample_unique_products(
            self.space, self.reactions, count=request.reservoir_size, seed=self.seed
        )
        metadata = _initial_metadata(len(payloads), self.attach_q0)
        return ExpansionResult(
            proposals=tuple(RawProposal(item, "synthonbench_shared_initialization", metadata) for item in payloads),
            metadata={"initialization_size": len(payloads), "sampling": "product_uniform"},
        )


class RandomSynthonPoolExpander:
    """Offer a score-blind random unseen product pool to the base GP-UCB comparator."""

    def __init__(self, space: Any, reactions: Sequence[str], *, seed: int) -> None:
        self.space = space
        self.reactions = tuple(reactions)
        self.seed = seed

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        observed = {item.canonical_key for item in request.observations}
        payloads = sample_unique_products(
            self.space,
            self.reactions,
            count=request.reservoir_size,
            seed=_round_seed(self.seed, request.round_idx),
            excluded_product_ids=observed,
        )
        return ExpansionResult(
            proposals=tuple(
                RawProposal(item, "synthonbench_score_blind_random_bo", {"collectable": False})
                for item in payloads
            ),
            metadata={"sampling": "score_blind_random_unseen_pool", "pool_size": len(payloads)},
        )


def sample_unique_products(
    space: Any,
    reactions: Sequence[str],
    *,
    count: int,
    seed: int,
    excluded_product_ids: Iterable[str] = (),
) -> tuple[dict[str, object], ...]:
    """Sample exact product tuples without evaluating scores or mutating task state."""

    if count < 1:
        raise ValueError("product sample count must be positive")
    reaction_ids, cumulative = _reaction_distribution(space, reactions)
    excluded = set(excluded_product_ids)
    if count > cumulative[-1] - len(excluded):
        raise ValueError("requested product pool exceeds the remaining official search space")
    rng = random.Random(seed)
    accepted: list[dict[str, object]] = []
    accepted_ids = set(excluded)
    max_attempts = max(1000, count * 100)
    for _ in range(max_attempts):
        payload = _sample_product(space, reaction_ids, cumulative, rng)
        identifier = _product_id(payload)
        if identifier in accepted_ids:
            continue
        accepted_ids.add(identifier)
        accepted.append(payload)
        if len(accepted) == count:
            return tuple(accepted)
    raise RuntimeError("could not draw the requested unique product pool without replacement")


def _reaction_distribution(space: Any, reactions: Sequence[str]) -> tuple[tuple[str, ...], tuple[int, ...]]:
    ids = tuple(str(item) for item in reactions)
    if not ids:
        raise ValueError("at least one allowed reaction is required")
    counts = tuple(int(space.product_count_estimate(item)) for item in ids)
    if any(item < 1 for item in counts):
        raise ValueError("official reaction product counts must be positive")
    total = 0
    cumulative = []
    for item in counts:
        total += item
        cumulative.append(total)
    return ids, tuple(cumulative)


def _sample_product(space: Any, reactions: Sequence[str], cumulative: Sequence[int], rng: random.Random):
    draw = rng.randrange(cumulative[-1])
    reaction_index = next(index for index, bound in enumerate(cumulative) if draw < bound)
    reaction_id = reactions[reaction_index]
    synthons = [
        int(rng.choice(tuple(space.synthon_ids(reaction_id, position))))
        for position in space.positions(reaction_id)
    ]
    return {"reaction_id": reaction_id, "synthon_ids": synthons}


def _initial_metadata(sample_size: int, attach_q0: bool) -> dict[str, object]:
    metadata: dict[str, object] = {"collectable": False}
    if attach_q0:
        metadata[Q0_METADATA_KEY] = {
            "occurrence_count": 1,
            "valid_occurrence_count": sample_size,
            "probability": 1.0 / sample_size,
        }
    return metadata


def _product_id(payload: dict[str, object]) -> str:
    from synthonbench.ids import product_id

    return str(product_id(payload["reaction_id"], payload["synthon_ids"]))


def _round_seed(seed: int, round_idx: int) -> int:
    raw = f"{seed}:{round_idx}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


__all__ = [
    "INITIALIZATION_MODES", "RandomSynthonPoolExpander", "SEARCH_METHODS",
    "SynthonInitializationExpander", "sample_unique_products",
]
