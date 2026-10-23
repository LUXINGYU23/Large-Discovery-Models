"""Public geological inputs; UBC coordinates are decoded by discretize."""

from pathlib import Path
import hashlib
import json
import numpy as np

RESOURCES = Path(__file__).resolve().parents[1] / "resources"
CASES = ("drillholes_sketch_2d", "drillholes_3d", "drillholes_outcrops_3d")


def load_problem(case_id, *, fixture=False):
    if case_id not in CASES:
        raise ValueError("unknown geological case")
    if fixture:
        shape = (12, 9) if case_id.endswith("2d") else (9, 8, 7)
        coords = np.indices(shape, dtype=float)
        initial = 1 - sum(
            ((coords[i] - (n - 1) / 2) / (n / 3)) ** 2 for i, n in enumerate(shape)
        )
        values = np.full(shape, np.nan)
        values[(0,) * len(shape)] = 0
        values[tuple(n // 2 for n in shape)] = 1
        obs = [("drillhole", values, 1.0)]
        sketch, active = None, None
        origin = [0.0] * len(shape)
        spacing = [1.0] * len(shape)
    else:
        root = RESOURCES / "cases" / case_id
        manifest = json.loads((RESOURCES / "provenance.json").read_text())["files"]
        for path in root.glob("*.txt"):
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != manifest[str(path.relative_to(RESOURCES))]
            ):
                raise ValueError("case data digest mismatch: " + path.name)
        if case_id.endswith("2d"):
            initial = np.loadtxt(root / "initial_model.txt") - 0.5
            shape = initial.shape
            obs = [("drillhole", np.loadtxt(root / "drillhole_vertical.txt"), 1.0)]
            sketch = np.loadtxt(root / "sketch.txt")
            active = None
            origin, spacing = [0.0, 0.0], [1.0, 1.0]
        else:
            from discretize import TensorMesh

            mesh = TensorMesh.read_UBC(root / "mesh.txt")
            shape = tuple(mesh.shape_cells)

            def read(name):
                return mesh.read_model_UBC(root / name).reshape(shape, order="F")

            initial = np.nan_to_num(read("initial_model.txt"), nan=0.0) - 0.5
            obs = [("drillhole", read("drillholes.txt"), 1.0)]
            if "outcrops" in case_id:
                obs.append(("outcrop", read("outcrops.txt"), 0.1))
            sketch, active = None, read("ind_active.txt").astype(bool)
            origin = mesh.origin.tolist()
            spacing = [float(h[0]) for h in mesh.h]
            if any(not np.allclose(h, h[0]) for h in mesh.h):
                raise ValueError("objective_v1 requires uniform spacing per axis")
    return dict(
        schema_version=1,
        case_id=case_id,
        shape=tuple(shape),
        axis_names=["x", "z"] if len(shape) == 2 else ["x", "y", "z"],
        origin=origin,
        spacing=spacing,
        initial_level_set=initial,
        observations=[
            dict(name=n, values=v, mask=np.isfinite(v), weight=w) for n, v, w in obs
        ],
        sketch=sketch,
        active_mask=active,
        objective_version="objective_v1",
    )


def save_problem(problem, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meta = {
        k: v
        for k, v in problem.items()
        if k not in ("initial_level_set", "sketch", "active_mask", "observations")
    }
    arrays = {"initial_level_set": problem["initial_level_set"]}
    for name in ("sketch", "active_mask"):
        if problem[name] is not None:
            arrays[name] = problem[name]
    meta["observations"] = []
    for i, o in enumerate(problem["observations"]):
        meta["observations"].append({"name": o["name"], "weight": o["weight"]})
        arrays[f"values_{i}"] = o["values"]
        arrays[f"mask_{i}"] = o["mask"]
    (directory / "problem.json").write_text(json.dumps(meta, sort_keys=True))
    np.savez(directory / "problem.npz", **arrays)


def read_problem(directory):
    directory = Path(directory)
    p = json.loads((directory / "problem.json").read_text())
    with np.load(directory / "problem.npz", allow_pickle=False) as a:
        for k in ("initial_level_set", "sketch", "active_mask"):
            p[k] = a[k] if k in a else None
        for i, o in enumerate(p["observations"]):
            o.update(values=a[f"values_{i}"], mask=a[f"mask_{i}"])
    p["shape"] = tuple(p["shape"])
    for arr in [
        p["initial_level_set"],
        p["sketch"],
        p["active_mask"],
        *[v for o in p["observations"] for v in (o["values"], o["mask"])],
    ]:
        if arr is not None:
            arr.flags.writeable = False
    return p
