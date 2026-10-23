"""Trusted process: no submitted source is mounted or imported."""

import json
from pathlib import Path
from .data import read_problem
from .geometry import read_model
from .objective import score


def main():
    p = read_problem("/input")
    try:
        metrics = score(p, read_model("/model/model.npy", p["shape"]))
        result = {"status": "succeeded", "metrics": metrics}
    except (ValueError, FileNotFoundError, OSError) as exc:
        result = {"status": "invalid", "error": str(exc)}
    Path("/output/result.json").write_text(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
