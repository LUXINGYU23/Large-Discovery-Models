"""Public-only serialization. Hidden truth is never a problem field."""

import json
from pathlib import Path
import numpy as np
from .contract import validate


def save_problem(problem, directory):
    validate(problem)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in problem.items() if k not in {"domain_mask", "forbidden_mask", "observations"}}
    arrays = {k: problem[k] for k in ("domain_mask", "forbidden_mask")}
    meta["observations"] = []
    for i, obs in enumerate(problem["observations"]):
        meta["observations"].append({k: v for k, v in obs.items() if k not in {"points", "labels", "groups"}})
        for key in ("points", "labels", "groups"):
            arrays[f"{key}_{i}"] = obs[key]
    (directory / "problem.json").write_text(json.dumps(meta, sort_keys=True, allow_nan=False))
    np.savez(directory / "problem.npz", **arrays)


def read_problem(directory):
    directory = Path(directory)
    problem = json.loads((directory / "problem.json").read_text())
    with np.load(directory / "problem.npz", allow_pickle=False) as arrays:
        for key in ("domain_mask", "forbidden_mask"):
            problem[key] = arrays[key]
        for i, obs in enumerate(problem["observations"]):
            for key in ("points", "labels", "groups"):
                obs[key] = arrays[f"{key}_{i}"]
    validate(problem)
    for arr in [problem["domain_mask"], problem["forbidden_mask"],
                *[o[k] for o in problem["observations"] for k in ("points", "labels", "groups")]]:
        arr.flags.writeable = False
    return problem
