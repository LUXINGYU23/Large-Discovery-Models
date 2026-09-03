"""Task-local public features for compiled NucleoBench policy means."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence

import numpy as np

from tasks.nucleobench.core.candidate import MutationContext
from tasks.nucleobench.core.hamming_gp import BASE_CODE

_POSITION_BIN_COUNT = 8
BASES = ("A", "C", "G", "T")
_TRANSITIONS = frozenset((("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")))


class NucleoPolicyFeatureEncoder:
    """Summarize mutation patches without reading evaluator or hidden benchmark data."""

    def __init__(self, context: MutationContext) -> None:
        self.context = context
        self.positions = tuple(context.editable_positions)
        self.start_bases = tuple(context.start_sequence[position] for position in self.positions)
        self.start_codes = np.asarray([BASE_CODE[base] for base in self.start_bases])
        self.feature_names = (
            "mutation_fraction",
            "log_mutation_fraction",
            "absolute_position_mean",
            "absolute_position_std",
            "editable_rank_mean",
            "editable_rank_std",
            "adjacent_edit_fraction",
            "gc_delta_per_edit",
            "transition_fraction",
            *(f"target_base_fraction_{base}" for base in BASES),
            *(f"editable_region_{index}_fraction" for index in range(_POSITION_BIN_COUNT)),
        )
        self.feature_groups = {
            "edit_burden": (0, 2),
            "position_summary": (2, 7),
            "substitution_summary": (7, 13),
            "editable_regions": (13, 13 + _POSITION_BIN_COUNT),
        }
        identity = {
            "case_id": context.case.case_id,
            "start_set_digest": context.start_set_digest,
            "start_index": context.start_index,
            "editable_positions": list(self.positions),
            "feature_names": list(self.feature_names),
        }
        self.space_digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.version = f"nucleobench_policy_features_{self.space_digest[:16]}_v1"

    def encode_vector(self, values: Sequence[float]) -> np.ndarray:
        codes = np.asarray(values, dtype=float)
        if codes.shape != self.start_codes.shape or not np.isfinite(codes).all():
            raise ValueError("Nucleo policy input must align with the editable mask")
        if np.any(codes < 0) or np.any(codes > 3) or not np.allclose(codes, np.rint(codes)):
            raise ValueError("Nucleo policy input must contain categorical DNA codes")
        changed = np.flatnonzero(codes != self.start_codes)
        result = np.zeros(len(self.feature_names), dtype=float)
        if not len(changed):
            return result

        count = len(changed)
        editable_count = len(self.positions)
        absolute = np.asarray([self.positions[index] for index in changed], dtype=float)
        rank = changed.astype(float) / max(editable_count - 1, 1)
        sequence_scale = max(self.context.case.sequence_length - 1, 1)
        result[0] = count / editable_count
        result[1] = math.log1p(count) / math.log1p(editable_count)
        result[2] = float(absolute.mean() / sequence_scale)
        result[3] = float(absolute.std() / sequence_scale)
        result[4] = float(rank.mean())
        result[5] = float(rank.std())
        if count > 1:
            result[6] = float(np.count_nonzero(np.diff(changed) == 1) / (count - 1))

        target_bases = [BASES[int(codes[index])] for index in changed]
        source_bases = [self.start_bases[index] for index in changed]
        gc_delta = sum(base in {"G", "C"} for base in target_bases) - sum(
            base in {"G", "C"} for base in source_bases
        )
        result[7] = gc_delta / count
        result[8] = sum(
            (source, target) in _TRANSITIONS
            for source, target in zip(source_bases, target_bases, strict=True)
        ) / count
        for offset, base in enumerate(BASES, start=9):
            result[offset] = target_bases.count(base) / count
        bins = np.minimum(
            (changed * _POSITION_BIN_COUNT) // editable_count,
            _POSITION_BIN_COUNT - 1,
        )
        for bin_index in range(_POSITION_BIN_COUNT):
            result[13 + bin_index] = np.count_nonzero(bins == bin_index) / count
        return result

    def describe_vector(self, values: Sequence[float]) -> dict[str, object]:
        codes = np.asarray(values, dtype=float)
        self.encode_vector(codes)
        mutations = [
            {"position": self.positions[index], "base": BASES[int(codes[index])]}
            for index in np.flatnonzero(codes != self.start_codes)
        ]
        return {"mutations": mutations, "mutation_count": len(mutations)}


__all__ = ["NucleoPolicyFeatureEncoder"]
