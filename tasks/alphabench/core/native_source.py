"""Load the pinned upstream algorithms with small, auditable runtime patches."""

from contextlib import contextmanager
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

from ldm_tts.engine.run_store import atomic_json_write


SOURCE_HASHES = {
    "__init__.py": "edef85599507fbe85d1669f4f2edc0e63b043fe20a19b0e2e5b85af020b10b52",
    "base.py": "f8717e37290350cd1d936ced0f1a737223d3beec431371b3e07c94e69d8b5bfe",
    "cot.py": "6626f8f06369703dec99b6378da4406ffa64f2583553f27bc60e6a95156854d6",
    "tot.py": "029a39c250909984beb44c916d096e0d438e17ba8eedc0c90d75718a145f4e61",
    "ea.py": "024d9e0335645e80e39bed058cae25ed9a969d01633e5f59ec58e20a00f1768f",
}


def patched_source(name, text):
    def replace(before, after):
        nonlocal text
        if text.count(before) != 1:
            raise ValueError(f"pinned {name} patch does not match")
        text = text.replace(before, after)

    for module in ("time", "uuid", "threading"):
        text = text.replace(f"\nimport {module}\n", f"\nfrom .runtime import {module}\n")
    text = text.replace("from concurrent.futures import ", "from .runtime import ")
    if name in {"cot.py", "tot.py", "ea.py"}:
        text = text.replace("from .base import BaseAlgo", "from .base import BaseAlgo\nfrom .runtime import native_state")
    if name == "cot.py":
        snapshot = 'native_state("cot.chain", {"chain": chain})\n'
        replace("        for r in range(1, rounds + 1):", "        " + snapshot + "\n        for r in range(1, rounds + 1):")
        replace("                continue\n\n            cand_name", "                " + snapshot + "                continue\n\n            cand_name")
        replace("\n        summary = {", "\n            " + snapshot + "\n        summary = {")
    if name == "tot.py":
        before = ('            if self._is_better(best_global.get("metrics", {}), metrics):\n'
                  '                best_global = {"name": name, "expression": expr, "metrics": metrics}\n')
        replace(before, "            with self._seen_lock:\n" +
                "".join("    " + line for line in before.splitlines(keepends=True)))
        replace("            if depth >= rounds or not survivor_nodes:",
                '            native_state("tot.node", branch_history[-1])\n\n'
                "            if depth >= rounds or not survivor_nodes:")
    if name == "ea.py":
        replace("        current_pool = _rank_by_rank_ic(pool)\n", "        current_pool = _rank_by_rank_ic(pool)\n"
                '        native_state("ea.pool", {"round": 0, "pool": current_pool, "candidates": []})\n')
        replace("                current_pool = _rank_by_rank_ic(combined)[:pool_size]\n",
                "                current_pool = _rank_by_rank_ic(combined)[:pool_size]\n"
                '                native_state("ea.pool", {"round": r, "pool": current_pool, "candidates": unique_candidates})\n')
    return text


@contextmanager
def load_algorithms(source_root, private_root, scheduler):
    source_root, private_root = Path(source_root), Path(private_root)
    destination = private_root / "native_source"
    files, manifest = {}, {}
    for name, expected in SOURCE_HASHES.items():
        raw = (source_root / "searcher" / "algo" / name).read_bytes()
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f"pinned AlphaBench source changed: searcher/algo/{name}")
        body = patched_source(name, raw.decode("utf-8")).encode("utf-8")
        files[name] = body
        manifest[name] = {"source_sha256": expected, "patched_sha256": sha256(body).hexdigest()}
    manifest = {"files": manifest, "runtime_sha256": sha256(
        Path(__file__).with_name("native_runtime.py").read_bytes()).hexdigest(),
        "loader_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    manifest_path = destination / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("native runtime patches changed across resume")
    destination.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (destination / name).write_bytes(body)
    atomic_json_write(manifest_path, manifest)
    # One Host campaign per process; never install or mutate upstream packages.
    package_name = "_alphabench_native"
    if package_name in sys.modules:
        raise RuntimeError("native algorithms are already loaded in this process")
    spec = importlib.util.spec_from_file_location(
        package_name, destination / "__init__.py", submodule_search_locations=[str(destination)])
    package = importlib.util.module_from_spec(spec)
    runtime = ModuleType(package_name + ".runtime")
    runtime.__dict__.update(scheduler.bindings())
    sys.modules[package_name] = package
    sys.modules[runtime.__name__] = runtime
    try:
        spec.loader.exec_module(package)
        yield package
    finally:
        for name in list(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                del sys.modules[name]
