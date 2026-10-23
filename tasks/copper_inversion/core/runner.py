"""Untrusted worker entrypoint: the controller never trusts its score or exit code alone."""

import importlib.util
import json
from pathlib import Path
import numpy as np
from .data import read_problem


def main():
    p = read_problem("/input")
    spec = importlib.util.spec_from_file_location("candidate", "/submitted/solver.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    limits = json.loads(Path("/input/budget.json").read_text())
    result = m.solve(p, limits, limits["seed"])
    if not isinstance(result, np.ndarray):
        raise TypeError("solve must return ndarray")
    np.save("/output/model.npy", result, allow_pickle=False)


if __name__ == "__main__":
    main()
