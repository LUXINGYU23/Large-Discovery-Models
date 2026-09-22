"""Execute the pinned example entry with repaired call signatures and shared seams."""

import ast
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace
import typing

from ldm_tts.engine.run_store import atomic_json_write
from .initialization import alpha158_library
from .native_source import verified_sources


ENTRY = "benchmark/engine/searching/benchmark_searching.py"
SOURCE_HASHES = {
    ENTRY: "477e86571bc03d8bc752569616d0c1428ff247bff9e437b5d619e46a28d41312",
    "factors/lib/alpha158/__init__.py": "836ad459c041f5bf410e38d6725ef38a1882fb12e073554b1f84518666f78fb6",
    "factors/lib/alpha158/qlib_compile_product.json": "1537b5b8612938a124ca9d665ac8182ed2d947a68ded494f7626d83629091b94",
    "factors/lib/alpha158/kbar.json": "e9abedf35bb71962de9d2768693e1ae440ca7c5c798d9486c95f80ce78cd9294",
    "factors/lib/alpha158/price.json": "8af498533386d98bc4d80b74d5dd9eb9429c6b0e24db6b97b26d2fabf5cbb8f9",
    "factors/lib/alpha158/rolling.json": "270a261280235c8f629d8841dae0ccc0f8def54ba77aae1cf49b191035d344d5",
}


def benchmark_source(source):
    names = {"to_cot_kwargs", "to_tot_kwargs", "run_batch", "benchmark_main"}
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError("pinned benchmark entry definitions changed")
    body = "\n\n".join(ast.get_source_segment(source, node) for node in nodes) + "\n"

    def patch(old, new, count=1):
        nonlocal body
        if body.count(old) != count:
            raise ValueError("pinned benchmark patch does not match: " + old)
        body = body.replace(old, new)

    patch("evaluate_factor_fn=evaluate_factor_via_api,", "evaluate_fn=evaluate_factor_via_api,", 2)
    patch("batch_evaluate_factors_fn=batch_evaluate_factors_via_api,",
          "batch_evaluate_fn=batch_evaluate_factors_via_api,", 2)
    patch("        tot_searcher = ToTSearcher(\n            evaluate_fn=evaluate_factor_via_api,\n"
          "            batch_evaluate_fn=batch_evaluate_factors_via_api,",
          "        tot_searcher = ToTSearcher(\n            evaluate_fn=evaluate_factor_via_api,\n"
          "            batch_evaluate_fn=batch_evaluate_fn_dict,")
    patch('common_cot = {"rounds": rounds, "verbose": True, "save_dir": cot_save_dir}',
          'common_cot = {"rounds": rounds, "save_dir": cot_save_dir}')
    patch('            "verbose": False,\n', "")
    patch("            verbose=True,\n", "")
    patch("    factor_performance = batch_evaluate_factors_via_api(\n",
          "    factor_performance = initialization_fn(\n")
    patch('    model = config["model"]["name"]',
          '    native_state("benchmark.config", {"config": config, "enable_reason": enable_reason})\n'
          '    model = config["model"]["name"]')
    patch('    Exec = ThreadPoolExecutor if executor == "thread" else ProcessPoolExecutor',
          '    if executor != "thread":\n        raise ValueError("native benchmark requires recorded threads")\n'
          '    Exec = ThreadPoolExecutor')
    body += ('\n    return {"baseline": filtered_performance,\n'
             '            "cot": {"results": results, "errors": errors} if config.get("cot", {}).get("enable", True) else None,\n'
             '            "tot": {"results": results_tot, "errors": errors_tot} if config.get("tot", {}).get("enable", True) else None,\n'
             '            "ea": summary if config.get("ea", {}).get("enable", True) else None}\n')
    return body


def load_benchmark(source_root, private_root, scheduler, algorithms, *, initialization_fn,
                   search_fn, evaluate_fn, batch_evaluate_fn, batch_evaluate_fn_dict):
    sources = verified_sources(source_root, SOURCE_HASHES)
    body = benchmark_source(sources[ENTRY])
    directory = Path(private_root) / "native_benchmark"
    manifest = {"sources": SOURCE_HASHES, "patched_sha256": sha256(body.encode()).hexdigest(),
                "adapter_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    path = directory / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("native benchmark entry changed across resume")
    atomic_json_write(path, manifest)
    target = directory / "benchmark.py"
    target.write_text(body, encoding="utf-8")
    factors = alpha158_library(source_root)
    namespace = {"os": os, "Path": Path, "Any": typing.Any, "Optional": typing.Optional,
        "Dict": typing.Dict, "Callable": typing.Callable, "Iterable": typing.Iterable,
        "ThreadPoolExecutor": scheduler.executor, "as_completed": scheduler.completed,
        "native_state": scheduler.bindings()["native_state"],
        "load_factors_alpha158": factors.load_factors_alpha158,
        "load_factors_alpha158_names": factors.load_factors_alpha158_names,
        "CoTSearcher": algorithms.cot.CoTSearcher, "ToTSearcher": algorithms.tot.ToTSearcher,
        "EA_Searcher": algorithms.ea.EA_Searcher,
        "call_qlib_search": search_fn, "evaluate_factor_via_api": evaluate_fn,
        "batch_evaluate_factors_via_api": batch_evaluate_fn,
        "batch_evaluate_fn_dict": batch_evaluate_fn_dict, "initialization_fn": initialization_fn,
        "print": lambda *args, **kwargs: None,
        "tqdm": lambda **kwargs: SimpleNamespace(update=lambda *_: None, close=lambda: None)}
    exec(compile(body, str(target), "exec"), namespace)
    return namespace["benchmark_main"]
