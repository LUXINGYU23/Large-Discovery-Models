"""Experimental coverage of editable-position distance shells around the incumbent."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ldm_tts.optimization.measurement import (
    EvaluationGroup, MeasurementContext, SelectionRequirements,
)
from tasks.nucleobench.core.hamming_gp import ENCODER_ALGORITHM
from tasks.nucleobench.core.hamming_posterior import code_matrix


@dataclass(frozen=True)
class HammingShellCoverage:
    boundaries: tuple[float, ...] = (1 / 12, 1 / 3, 2 / 3)
    max_regions: int = 3
    pool_per_region: int = 4

    def __post_init__(self) -> None:
        boundaries = tuple(self.boundaries)
        if (not boundaries or any(type(value) not in (int, float) for value in boundaries)
                or any(not np.isfinite(value) or not 0 < value < 1 for value in boundaries)
                or any(a >= b for a, b in zip(boundaries, boundaries[1:]))):
            raise ValueError("Hamming shell boundaries must be strictly increasing fractions in (0, 1)")
        for value in (self.max_regions, self.pool_per_region):
            if type(value) is not int or value < 1:
                raise ValueError("coverage region counts must be positive integers")
        object.__setattr__(self, "boundaries", boundaries)

    def identity(self) -> dict:
        return {"name": "hamming_shell_coverage", "version": 1,
                **asdict(self), "boundaries": list(self.boundaries)}

    def plan(self, context: MeasurementContext) -> SelectionRequirements:
        if not context.history:
            return SelectionRequirements(metadata={"reason": "no_measured_incumbent"})
        scores = np.asarray([item.scalar_score for item in context.history])
        if not np.isfinite(scores).all():
            raise ValueError("coverage requires finite measured utility")
        incumbent = min(context.history, key=lambda item: (-item.scalar_score, item.candidate_id))
        features = [item.feature for item in context.history]
        features.extend(context.representations[item.candidate_id] for item in context.candidates)
        if any(feature is None for feature in features):
            raise ValueError("coverage requires encoded observations and candidates")
        versions = {feature.version for feature in features}
        if len(versions) != 1 or not next(iter(versions)).startswith(ENCODER_ALGORITHM + ":"):
            raise ValueError("coverage requires one editable-position Hamming feature version")
        codes = code_matrix([feature.values for feature in features])
        reference = code_matrix(incumbent.feature.values)[0]
        distances = np.mean(codes != reference, axis=1)
        shells = np.searchsorted(self.boundaries, distances, side="left")
        history_shells = shells[:len(context.history)]
        candidate_shells = shells[len(context.history):]
        available = {
            shell: tuple(item.candidate_id for item, region in zip(context.candidates, candidate_shells)
                         if region == shell)
            for shell in range(1, len(self.boundaries) + 1)
        }
        regions = []
        for shell, members in available.items():
            if members:
                measured_indices = np.flatnonzero(history_shells == shell)
                regions.append((len(measured_indices), int(measured_indices[-1]) if len(measured_indices) else -1,
                                shell, members))
        regions.sort()
        regions = regions[:min(self.max_regions, context.selection_count, context.pool_size)]
        extra_pool_slots = context.pool_size - len(regions)
        groups = []
        for _, _, shell, members in regions:
            extra = min(self.pool_per_region, len(members)) - 1
            extra = min(extra, extra_pool_slots)
            extra_pool_slots -= extra
            groups.append(EvaluationGroup(f"shell_{shell}", members, 1 + extra, 1))
        return SelectionRequirements(tuple(groups), metadata={
            "incumbent_candidate_id": incumbent.candidate_id,
            "feature_version": incumbent.feature.version,
            "boundaries": list(self.boundaries),
            "distance": "fraction_of_editable_positions_different_from_current_incumbent",
            "candidate_shells": {item.candidate_id: int(shell)
                                 for item, shell in zip(context.candidates, candidate_shells)},
            "successful_history_counts": {
                f"shell_{shell}": int(np.sum(history_shells == shell))
                for shell in range(len(self.boundaries) + 1)
            },
            "region_order": "fewest_successful_measurements_then_oldest_measurement_then_shell",
        })
