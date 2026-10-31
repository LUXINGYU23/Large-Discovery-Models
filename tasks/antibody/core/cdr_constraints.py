"""CDR3 sequence constraints used by the active antibody search."""

import re
from itertools import groupby

import numpy as np


_AA = "ACDEFGHIKLMNPQRSTVWY"
_IDX_TO_AA = dict(enumerate(_AA))
_N_GLYCOSYLATION_PATTERN = "N[^P][ST][^P]"


def check_cdr_constraints_all(x: np.ndarray) -> tuple[int, int, int]:
    sequence = "".join(_IDX_TO_AA[int(aa)] for aa in x)
    repeat_count = max(sum(1 for _ in group) for _, group in groupby(sequence))
    charge = sum(
        int(aa in "RK") + 0.1 * int(aa == "H") - int(aa in "DE")
        for aa in sequence
    )
    return (
        int(repeat_count > 5),
        int(charge > 2.0 or charge < -2.0),
        int(bool(re.search(_N_GLYCOSYLATION_PATTERN, sequence))),
    )


def check_cdr_constraints(x: np.ndarray) -> bool:
    return not np.any(check_cdr_constraints_all(x))
