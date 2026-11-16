"""The four executable, equal-budget inheritance policies."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Arm:
    clear: str
    boundary_payload: str
    round_feed: str


ARMS = {
    "ldm4geo": Arm("every_k", "history", "own"),
    "accumulated": Arm("never", "history", "own"),
    "restart_best": Arm("every_k", "best_program", "own"),
    "independent_sampling": Arm("every_round", "initial_program", "none"),
}


def policy(name):
    try:
        return ARMS[name]
    except KeyError as exc:
        raise ValueError("unsupported controller arm") from exc
