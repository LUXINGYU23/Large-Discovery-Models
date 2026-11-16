"""Controller definitions (contracts §2.2); the controller reads fields, never arm names."""

from dataclasses import asdict, dataclass, fields


CHOICES = {"clear": ("never", "every_k", "every_round"),
           "boundary_payload": ("history", "best_program", "initial_program", "parents"),
           "round_feed": ("own", "all", "none"),
           "budget_split": ("shared", "separate")}


@dataclass(frozen=True)
class Arm:
    branches: int = 4
    turns_per_round: int = 1
    clear: str = "every_k"
    boundary_payload: str = "history"
    round_feed: str = "own"
    budget_split: str = "shared"

    def __post_init__(self):
        for name in ("branches", "turns_per_round"):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= 64:
                raise ValueError("invalid arm setting: " + name)
        for name, allowed in CHOICES.items():
            if getattr(self, name) not in allowed:
                raise ValueError("invalid arm setting: " + name)

    @classmethod
    def from_dict(cls, value):
        """Strict: exactly the six contract fields, as in protocol["arms"][name]."""
        names = {f.name for f in fields(cls)}
        if not isinstance(value, dict) or set(value) != names:
            raise ValueError("arm definition requires exactly: " + ", ".join(sorted(names)))
        return cls(**value)

    def to_dict(self):
        return asdict(self)


ARMS = {
    "ldm4geo": Arm(4, 1, "every_k", "history", "own", "shared"),
    "accumulated": Arm(4, 1, "never", "history", "own", "shared"),
    "restart_best": Arm(4, 1, "every_k", "best_program", "own", "shared"),
    "independent_sampling": Arm(4, 1, "every_round", "initial_program", "none", "shared"),
    "single_conversation": Arm(1, 4, "never", "history", "all", "shared"),
    "separate_runs": Arm(4, 1, "every_k", "history", "own", "separate"),
    "program_evolution": Arm(4, 1, "every_round", "parents", "none", "shared"),
}


def policy(name):
    try:
        return ARMS[name]
    except KeyError as exc:
        raise ValueError("unsupported controller arm") from exc
