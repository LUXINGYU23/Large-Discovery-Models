"""Read a frozen posterior and score task-validated categorical vectors; no oracle."""

import json
from pathlib import Path
import sys

import numpy as np

from hamming_posterior import residual_moments


def main():
    request = json.load(sys.stdin)
    snapshot = json.loads(Path(request["snapshot_file"]).read_text())
    if snapshot["snapshot_id"] != request["snapshot_id"]:
        raise ValueError("Surrogate snapshot changed; use the snapshot_id supplied for this turn")
    posterior = snapshot["posterior"]
    residual, deviation = residual_moments(
        request["codes"], posterior["training_codes"], posterior["cholesky"],
        posterior["alpha"], posterior["length_scale"],
    )
    means = posterior["target_mean"] + posterior["target_scale"] * residual
    deviations = posterior["target_scale"] * deviation
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(deviations)):
        raise ValueError("Surrogate returned a non-finite posterior")
    json.dump([
        {"mean": float(mean), "std": float(std),
         "acquisition_score": float(mean + posterior["beta"] * std)}
        for mean, std in zip(means, deviations, strict=True)
    ], sys.stdout, allow_nan=False)


if __name__ == "__main__":
    main()
