"""Independent v2 scorer: no candidate imports, optional sealed truth mount."""

import json
from pathlib import Path
import numpy as np
from .data import read_problem
from .objective import score
from .metrics import compare
from ..core.geometry import read_model


def main():
    problem = read_problem("/input")
    # Bad trusted data is an infrastructure failure, never a candidate penalty.
    truth_path = Path("/truth/truth.npy")
    truth = np.load(truth_path, allow_pickle=False) if truth_path.exists() else None
    if truth is not None:
        if truth.shape != tuple(problem["shape"]) or truth.dtype != bool:
            raise ValueError("invalid sealed truth")
        if np.any(truth & (~problem["domain_mask"] | problem["forbidden_mask"])):
            raise ValueError("sealed truth outside feasible region")
    confirmation_path=Path("/truth/confirmation")
    confirmation=read_problem(confirmation_path) if confirmation_path.exists() else None
    if confirmation is not None:
        if confirmation['priors']!=problem['priors']:
            raise ValueError('confirmation geological priors differ')
        if any(confirmation[key]!=problem[key] for key in ("shape","spacing","origin")):
            raise ValueError("confirmation geometry differs")
        for key in ("domain_mask","forbidden_mask"):
            if not np.array_equal(confirmation[key],problem[key]):raise ValueError("confirmation masks differ")
    try:
        field = read_model("/model/model.npy", problem["shape"])
        metrics = score(problem, field)
        if truth is not None:
            metrics.update(compare(problem, field, truth))
        if confirmation is not None:
            hidden=score(confirmation,field)
            metrics.update({"confirmation_"+key:value for key,value in hidden.items()
                            if key in {"loss_binary","loss_contact","loss_total"} or key.endswith(("_label_error","_contact_rmse"))})
        result = dict(status="succeeded", metrics=metrics)
    except ValueError as exc:
        result = dict(status="invalid", error=str(exc), metrics={"iou": 0.} if truth is not None else {})
    Path("/output/result.json").write_text(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
