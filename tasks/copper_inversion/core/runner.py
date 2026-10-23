"""Untrusted worker entrypoint: the controller never trusts its score or exit code alone."""

import importlib.util
import hashlib
import json
from pathlib import Path
import numpy as np
from .data import read_problem


def input_digest(value):
    """Detect accidental input mutation; this is not the scoring trust boundary."""
    def describe(item):
        if isinstance(item, np.ndarray):
            return {"shape": item.shape, "dtype": item.dtype.str,
                    "writeable": item.flags.writeable,
                    "sha256": hashlib.sha256(item.tobytes(order="C")).hexdigest()}
        if isinstance(item, dict):
            return {key: describe(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [describe(child) for child in item]
        return item
    return hashlib.sha256(json.dumps(describe(value), sort_keys=True).encode()).hexdigest()


def main():
    p = read_problem("/input")
    original = input_digest(p)
    spec = importlib.util.spec_from_file_location("candidate", "/submitted/solver.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    limits = json.loads(Path("/input/budget.json").read_text())
    result = m.solve(p, limits, limits["seed"])
    if input_digest(p) != original:
        raise ValueError("solve modified the public problem")
    if np.ma.isMaskedArray(result) or not isinstance(result, np.ndarray):
        raise TypeError("solve must return an unmasked ndarray")
    np.save("/output/model.npy", result, allow_pickle=False)


if __name__ == "__main__":
    main()
