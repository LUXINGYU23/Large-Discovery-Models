"""Untrusted worker entrypoint: the controller never trusts its score or exit code alone."""

import importlib.util
import json
import os
from pathlib import Path
import numpy as np
from .data import read_problem


from ..core.runner import input_digest, remaining_seconds


def main():
    deadline = float(os.environ['COPPER_WORKLOAD_DEADLINE_MONOTONIC'])
    p = read_problem("/input")
    original = input_digest(p)
    spec = importlib.util.spec_from_file_location("candidate", "/submitted/solver.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    limits = json.loads(Path("/input/budget.json").read_text())
    limits['wall_time_seconds'] = remaining_seconds(limits['wall_time_seconds'], deadline)
    result = m.solve(p, limits, limits["seed"])
    if input_digest(p) != original:
        raise ValueError("solve modified the public problem")
    if np.ma.isMaskedArray(result) or not isinstance(result, np.ndarray):
        raise TypeError("solve must return an unmasked ndarray")
    np.save("/output/model.npy", result, allow_pickle=False)


if __name__ == "__main__":
    main()
