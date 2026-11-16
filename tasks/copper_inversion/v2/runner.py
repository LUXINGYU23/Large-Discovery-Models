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
    # The batch supervisor points each run at its own case; defaults are the single-run mounts.
    inputs = Path(os.environ.get("COPPER_INPUT", "/input"))
    output = Path(os.environ.get("COPPER_OUTPUT", "/output"))
    source = os.environ.get("COPPER_SOURCE", "/submitted/solver.py")
    p = read_problem(inputs)
    original = input_digest(p)
    spec = importlib.util.spec_from_file_location("candidate", source)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    limits = json.loads((inputs / "budget.json").read_text())
    limits['wall_time_seconds'] = remaining_seconds(limits['wall_time_seconds'], deadline)
    result = m.solve(p, limits, limits["seed"])
    if input_digest(p) != original:
        raise ValueError("solve modified the public problem")
    if np.ma.isMaskedArray(result) or not isinstance(result, np.ndarray):
        raise TypeError("solve must return an unmasked ndarray")
    np.save(output / "model.npy", result, allow_pickle=False)


if __name__ == "__main__":
    main()
