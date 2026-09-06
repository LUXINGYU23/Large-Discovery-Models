"""Exact condition comparisons derived only from campaign measurements."""

from collections.abc import Mapping, Sequence
from typing import Any

from tasks.iron_mind.core.constants import OBJECTIVE_NAME
from tasks.iron_mind.core.schema import ReactionDatasetSchema


def condition_evidence(
    observations: Sequence[Mapping[str, Any]], schema: ReactionDatasetSchema,
) -> dict[str, Any]:
    comparisons = []
    for index, current in enumerate(observations):
        for earlier_index, earlier in enumerate(observations[:index]):
            changes = {
                name: {"from": earlier["conditions"][name], "to": current["conditions"][name]}
                for name in schema.factor_names
                if earlier["conditions"][name] != current["conditions"][name]
            }
            if len(changes) > 1:
                continue
            comparisons.append({
                "earlier_history_index": earlier_index,
                "later_history_index": index,
                "earlier_round_index": earlier["round_index"],
                "later_round_index": current["round_index"],
                "comparison_type": "single_factor_change" if changes else "same_conditions",
                "changed_factors": changes,
                "utility_difference": current[OBJECTIVE_NAME] - earlier[OBJECTIVE_NAME],
            })
    return {
        "scope": (
            "Measured history only. A replicate requires equality of every condition. "
            "Single-factor differences are local observations, not general causal effects; "
            "pairs changing multiple factors are omitted."
        ),
        "comparisons": comparisons,
        "factor_coverage": {
            factor.name: [
                {"option": option, "measured_count": sum(
                    item["conditions"][factor.name] == option for item in observations
                )}
                for option in factor.options
            ]
            for factor in schema.factors
        },
    }
