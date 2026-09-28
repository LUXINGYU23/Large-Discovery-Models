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


def verified_sources(root, hashes):
    files = {}
    for name, expected in hashes.items():
        raw = (Path(root) / name).read_bytes()
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f"pinned AlphaBench source changed: {name}")
        files[name] = raw.decode("utf-8")
    return files


def patched_source(name, text, *, matched=False):
    def replace(before, after):
        nonlocal text
        if text.count(before) != 1:
            raise ValueError(f"pinned {name} patch does not match")
        text = text.replace(before, after)

    for module in ("time", "uuid", "threading"):
        text = text.replace(f"\nimport {module}\n", f"\nfrom .runtime import {module}\n")
    text = text.replace("from concurrent.futures import ", "from .runtime import ")
    if name in {"cot.py", "tot.py", "ea.py"}:
        text = text.replace("from .base import BaseAlgo", "from .base import BaseAlgo\nfrom .runtime import native_state, native_seed")
    if name == "cot.py":
        call = 'seed_metrics = self._safe_eval_batch(seed["name"], seed["expression"])["metrics"]'
        replace("        " + call, "        with native_seed():\n            " + call)
        snapshot = 'native_state("cot.chain", {"chain": chain})\n'
        replace("        for r in range(1, rounds + 1):", "        " + snapshot + "\n        for r in range(1, rounds + 1):")
        if matched:
            before = ('            if cand is None:\n'
                  '                self._log("  No candidate returned; skipping round.")\n'
                  '                chain.append({\n'
                  '                    "round":        r,\n'
                  '                    "name":         best_name,\n'
                  '                    "expression":   best_expr,\n'
                  '                    "metrics":      best_metrics,\n'
                  '                    "instruction":  instruction,\n'
                  '                    "elapsed_time": llm_elapsed,\n'
                  '                    "generated":    {},\n'
                  '                })\n'
                  '                continue\n')
            after = ('            while cand is None:\n'
                 '                self._log("  No candidate returned; retrying this round.")\n'
                 '                retry_started = time.time()\n'
                 '                payload = self.search_fn(\n'
                 '                    instruction=instruction, model=self.model, N=1, max_try=5,\n'
                 '                    avoid_repeat=False, verbose=False, debug_mode=False,\n'
                 '                    temperature=self.temperature, enable_reason=self.enable_reason,\n'
                 '                    local=self.local, local_port=self.local_port,\n'
                 '                )\n'
                 '                llm_elapsed += time.time() - retry_started\n'
                 '                cand = self._extract_single_candidate(payload)\n')
            replace(before, after)
        else:
            replace("                continue\n\n            cand_name",
                    "                " + snapshot + "                continue\n\n            cand_name")
        replace("\n        summary = {", "\n            " + snapshot + "\n        summary = {")
    if name == "tot.py":
        call = 'seed_batch   = self._safe_eval_batch([{"name": seed["name"], "expression": seed["expression"]}])'
        replace("        " + call, "        with native_seed():\n            " + call)
        before = ('            if self._is_better(best_global.get("metrics", {}), metrics):\n'
                  '                best_global = {"name": name, "expression": expr, "metrics": metrics}\n')
        replace(before, "            with self._seen_lock:\n" +
                "".join("    " + line for line in before.splitlines(keepends=True)))
        replace("            if depth >= rounds or not survivor_nodes:",
                '            native_state("tot.node", branch_history[-1])\n\n'
                "            if depth >= rounds or not survivor_nodes:")
        replace('                return {"history": branch_history, "best": parent}',
                '                native_state("tot.node", branch_history[-1])\n'
                '                return {"history": branch_history, "best": parent}')
    if name == "ea.py":
        replace("        current_pool = _rank_by_rank_ic(pool)\n", "        current_pool = _rank_by_rank_ic(pool)\n"
                '        native_state("ea.pool", {"round": 0, "pool": current_pool, "candidates": []})\n')
        replace("                current_pool = _rank_by_rank_ic(combined)[:pool_size]\n",
                "                current_pool = _rank_by_rank_ic(combined)[:pool_size]\n"
                '                native_state("ea.pool", {"round": r, "pool": current_pool, "candidates": unique_candidates})\n')
        if matched:
            replace("                unique_candidates = _get_unique_set(candidates)\n",
                "                unique_candidates = _get_unique_set(candidates)\n"
                "                while len(unique_candidates) < N:\n"
                "                    extra, _, elapsed = self._run_llm_round(\n"
                "                        _seed_block_json(_select_seed_pool(current_pool, top_k=self.seeds_top_k)),\n"
                "                        N - len(unique_candidates), 0, r)\n"
                "                    llm_elapsed += elapsed\n"
                "                    mut_candidates.extend(extra)\n"
                "                    candidates.extend(extra)\n"
                "                    unique_candidates = _get_unique_set(candidates)\n")
    return text


@contextmanager
def load_algorithms(source_root, private_root, scheduler, *, matched=False):
    source_root, private_root = Path(source_root), Path(private_root)
    destination = private_root / "native_source"
    files, manifest = {}, {}
    for name, source in verified_sources(source_root / "searcher/algo", SOURCE_HASHES).items():
        body = patched_source(name, source, matched=matched).encode("utf-8")
        files[name] = body
        manifest[name] = {"source_sha256": SOURCE_HASHES[name], "patched_sha256": sha256(body).hexdigest()}
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
