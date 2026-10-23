"""Public immutable interface fixture. This convenience check is not an admission receipt."""

import importlib.util
import sys
import numpy as np

spec = importlib.util.spec_from_file_location("candidate", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
import json
from pathlib import Path

meta = json.loads((Path(__file__).parent / "problem.json").read_text())
shape = (12, 9) if len(meta["shape"]) == 2 else (9, 8, 7)
x = np.indices(shape, dtype=float)
initial = 1 - sum(((x[i] - (n - 1) / 2) / (n / 3)) ** 2 for i, n in enumerate(shape))
y = np.full(shape, np.nan)
y[(0,) * len(shape)] = 0
y[tuple(n // 2 for n in shape)] = 1
p = dict(
    schema_version=1,
    case_id=meta["case_id"],
    shape=shape,
    axis_names=meta["axis_names"],
    origin=[0.0] * len(shape),
    spacing=[1.0] * len(shape),
    initial_level_set=initial,
    observations=[dict(name="drillhole", values=y, mask=np.isfinite(y), weight=1.0)],
    sketch=None,
    active_mask=None,
    objective_version="objective_v1",
)
f = m.solve(p, dict(wall_time_seconds=30, cpu_count=1, memory_mb=2048), 47)
assert isinstance(f, np.ndarray) and f.shape == shape and f.dtype.kind == "f"
assert np.isfinite(f).all() and (f > 0).any() and (f < 0).any()
print("public interface selfcheck passed (not a formal score)")
