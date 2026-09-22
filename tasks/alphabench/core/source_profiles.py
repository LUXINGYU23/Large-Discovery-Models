"""Resolve the fixed native entry points without running a model or backtester."""

import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from ldm_tts.engine.run_store import atomic_json_write
from .initialization import alpha158_seeds, parse_seeds
from .native_benchmark import SOURCE_HASHES as BENCHMARK_SOURCES
from .native_source import SOURCE_HASHES as ALGORITHM_SOURCES, verified_sources
from .protocol import digest


CONFIGS = {
    "searcher/config.yaml": "44acb2a620befa29725597432c9c18a724ff83bade7f2b8ebdfcffbb93bb4d36",
    "searcher/configs/cot_config.yaml": "350786ef69f1c10d4dc28ffbfe90af3f8eeef2ecca79f3e9c59f2f739cf4bec5",
    "searcher/configs/tot_config.yaml": "30e8b2c327438266a623ea0737817bf61107cfd976f7b3683cf66ff424c36d07",
    "searcher/configs/ea_config.yaml": "b7183544de6849ed2d2ceda856d1b7aaaa858e8c3340a7a004d4116ec4bd78db",
    "example/search/configs/search_csi300.yaml": "fd73cc1c7d8ea580a85b3c2840ed709b3c65f16faf03d9a9dc4edd7b1170a15c",
}
SOURCES = {
    "searcher/config/config.py": "4bfa115b15d0ff381afb6c6916e61f7bff1ca2eeedda9c264e5f8baec56fd81b",
    "searcher/pipeline.py": "a7ec8b90a6efa861f0bbefc82b3a655aae94b927edca09e7be93972138ab2367",
    "searcher/backtester.py": "d1337de271c4fcd6290f775162724ebff5977627f18a16dd2d94739dce40e5ea",
    "searcher/run_test_backtest.py": "28341ba70fc0d648cb6773897132688e4cf695567a2e582520732a75734bc38d",
    "searcher/seed_factors.json": "c9c15b05d315cb633807dedf1958e22564c775a98ad98beac9bd0b9892813451",
    "example/search/run_t3_search.py": "9a3141e04454838021007cf070e41c256683f65c68153842b5af92597881df06",
    "ffo/client/factor_eval_client.py": "11a29ba56e72427b3c4c776440889e65a7bed93dfc29f7767fa59db709d12c51",
}


def definition(source, name, parent=None):
    nodes = ast.parse(source).body
    if parent:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == parent).body
    return next(node for node in nodes if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)


def signature_defaults(node):
    pairs = list(zip(node.args.args[-len(node.args.defaults):], node.args.defaults)) if node.args.defaults else []
    pairs.extend(zip(node.args.kwonlyargs, node.args.kw_defaults))
    return {arg.arg: ast.literal_eval(value) for arg, value in pairs if value is not None}


def parser_defaults(node):
    return {call.args[0].value: ast.literal_eval(keyword.value)
            for call in ast.walk(node) if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and call.func.attr == "add_argument" and call.args and isinstance(call.args[0], ast.Constant)
            for keyword in call.keywords if keyword.arg == "default"}


def resolve_source(source_root, config_path, method, *, market=None):
    if config_path not in CONFIGS or method not in {"cot", "tot", "ea"}:
        raise ValueError("select a pinned source config and cot/tot/ea method")
    hashes = {config_path: CONFIGS[config_path], **SOURCES, **BENCHMARK_SOURCES,
              **{"searcher/algo/" + name: value for name, value in ALGORITHM_SOURCES.items()}}
    sources = verified_sources(source_root, hashes)
    declared = yaml.safe_load(sources[config_path])
    searcher = config_path.startswith("searcher/")
    profile = "upstream_searcher_v1" if searcher else "upstream_benchmark_v1"
    origins = {}

    def origin(key, file, field):
        origins[key] = {"file": file, "field": field}

    algo_file = "searcher/algo/" + method + ".py"
    if searcher:
        if declared["searching"]["algo"]["name"] != method:
            raise ValueError("method differs from the selected searcher YAML")
        model = declared["searching"]["model"]
        model.pop("key", None)
        algorithm = dict(declared["searching"]["algo"]["param"])
        for key, value, field in (("model", model["name"], "searching.model.name"),
                                  ("temperature", model["temperature"], "searching.model.temperature"),
                                  ("accept_threshold", declared["backtesting"]["accept_threshold"], "backtesting.accept_threshold")):
            if key not in algorithm:
                algorithm[key] = value
                origin("algorithm." + key, config_path, field)
        run = definition(sources[algo_file], "run", {"cot": "CoTAlgo", "tot": "ToTAlgo", "ea": "EAAlgo"}[method])
        context = {"self": SimpleNamespace(config=algorithm), "seeds": []}
        values = {}
        for node in run.body:
            if not isinstance(node, ast.Assign) or "self.config.get(" not in ast.unparse(node):
                continue
            exec(compile(ast.Module(body=[node], type_ignores=[]), algo_file, "exec"), context)
            key = node.targets[0].id
            values[key] = context[key]
            if "algorithm." + key not in origins:
                origin("algorithm." + key, config_path if key in algorithm else algo_file,
                       "searching.algo.param." + key if key in algorithm else f"{run.name}:{node.lineno}")
        algorithm = values
        seed_file = declared["searching"]["algo"].get("seed_file", "")
        initialization = {"mode": "file" if seed_file else "cold", "seed_file": seed_file or None,
            "count": len(parse_seeds(sources[seed_file], Path(seed_file).suffix)) if seed_file else 30,
            "algorithm_seed_selection": "entire evaluated pool" if method == "ea" else "top workers by IC"}
        origin("initialization", config_path if seed_file else "searcher/pipeline.py",
               "searching.algo.seed_file" if seed_file else "cold_start_generate.N")
        backtest = dict(declared["backtesting"])
        verification = dict(declared["verification"])
        client = signature_defaults(definition(sources["ffo/client/factor_eval_client.py"], "evaluate_factor", "FactorEvalClient"))
        final = parser_defaults(definition(sources["searcher/run_test_backtest.py"], "main"))
        requested_market = market or backtest["market"]
        search = {"start": backtest["search_start"], "end": backtest["search_end"],
            "end_inclusive": True, "purge_split_labels": False, "market": requested_market,
            "fast": backtest["fast"], "stock_topk": backtest["top_k"], "stock_n_drop": backtest["n_drop"],
            "oracle_workers": backtest["n_jobs"], "http_timeout": backtest["timeout"],
            "worker_timeout": client["timeout"], "label": client["label"], "forward_n": client["forward_n"]}
        validation = {"enabled": verification["enabled"], "start": verification["val_start"],
            "end": verification["val_end"], "end_inclusive": True, "fast": True,
            "declared_forward_n": verification["verification_forward_n"], "effective_forward_n": client["forward_n"]}
        test = {"start": verification["test_start"], "end": verification["test_end"], "end_inclusive": True,
            "factor_select_n": final["--top-n"], "stock_topk": final["--top-n"], "stock_n_drop": final["--drop-k"],
            "fast": final["--fast"], "rank_by": final["--rank-by"],
            "pool": "native_final_pool", "ranking": "validation preferred, search fallback"}
        origin("search", config_path, "backtesting; effective label/forward_n/worker timeout: FFO client defaults")
        origin("validation", config_path, "verification; effective forward_n: FFO client default")
        origin("test", "searcher/run_test_backtest.py", "main parser defaults and rank_factors")
    else:
        model = declared["model"]
        requested_market = market or declared["market"]
        klass = {"cot": "CoTSearcher", "tot": "ToTSearcher", "ea": "EA_Searcher"}[method]
        algorithm = signature_defaults(definition(sources[algo_file], "__init__", klass))
        algorithm = {key: value for key, value in algorithm.items()
                     if key not in {"batch_evaluate_fn", "logger", "save_dir"}}
        algorithm.update(model=model["name"], temperature=model["temperature"], enable_reason=False,
                         local=model["local"], local_port=model["local_port"])
        settings = declared[method]
        mapping = {"generations": "rounds", "generate_size": "N", "population_size": "pool_size"} if method == "ea" else {"size": "N"}
        algorithm.update({mapping.get(key, key): value for key, value in settings.items() if key != "enable"})
        if method != "ea":
            algorithm["workers"] = 4
        for key in algorithm:
            origin("algorithm." + key, config_path if key in set(mapping.values()) | set(settings) | {"model", "temperature", "local", "local_port"} else algo_file,
                   "model/" + method if key != "workers" else "benchmark_main.num_workers")
        origin("algorithm.enable_reason", "benchmark/engine/searching/benchmark_searching.py", "benchmark_main default")
        if method != "ea":
            origin("algorithm.workers", "benchmark/engine/searching/benchmark_searching.py", "benchmark_main.num_workers")
        else:
            origin("algorithm.seeds_top_k", "benchmark/engine/searching/benchmark_searching.py", "benchmark_main.EA_Searcher.seeds_top_k")
        groups = ("rolling",) if method == "ea" else ("kbar", "price")
        baseline = alpha158_seeds(source_root, ("kbar", "rolling", "price"))
        seeds = alpha158_seeds(source_root, groups)
        initialization = {"mode": "alpha158", "baseline_groups": ["kbar", "rolling", "price"],
            "baseline_count": len(baseline), "algorithm_groups": list(groups), "count": len(seeds),
            "algorithm_seed_selection": "all group members in source order"}
        defaults = signature_defaults(definition(sources["ffo/client/factor_eval_client.py"], "batch_evaluate_factors_via_api"))
        client = signature_defaults(definition(sources["ffo/client/factor_eval_client.py"], "evaluate_factor", "FactorEvalClient"))
        search = {"start": defaults["start_date"], "end": defaults["end_date"],
            "end_inclusive": True, "purge_split_labels": False, "market": defaults["market"],
            "initialization_market": requested_market, "fast": defaults["fast"],
            "stock_topk": defaults["topk"], "stock_n_drop": defaults["n_drop"],
            "oracle_workers": defaults["max_workers"], "worker_timeout": defaults["timeout"],
            "label": defaults["label"], "forward_n": client["forward_n"]}
        validation, test = None, None
        origin("initialization", "benchmark/engine/searching/benchmark_searching.py", "benchmark_main baseline and group filtering")
        origin("search", "ffo/client/factor_eval_client.py", "evaluate_factor_via_api/batch_evaluate_factors_via_api defaults")
        origin("validation", "benchmark/engine/searching/benchmark_searching.py", "absent from this entry")
        origin("test", "benchmark/engine/searching/benchmark_searching.py", "absent from this entry")

    if requested_market not in {"csi300", "csi500", "csi1000", "sp500", "nasdaq100"}:
        raise ValueError("unknown source-profile market")
    effective = {"algorithm": algorithm, "initialization": initialization,
                 "search": search, "validation": validation, "test": test}
    deltas = [{"field": "model", "source": algorithm["model"], "adapter": "deepseek-flash",
               "reason": "user-selected provider: Responses API, reasoning.effort=max"},
              {"field": "temperature", "source": algorithm["temperature"], "adapter": None,
               "reason": "requested value is sent but ignored by DeepSeek thinking mode"}]
    if requested_market != (declared["backtesting"]["market"] if searcher else declared["market"]):
        deltas.append({"field": "market", "source": declared["backtesting"]["market"] if searcher else declared["market"],
                       "adapter": requested_market, "reason": "explicit source CLI market selection"})
    if not searcher:
        deltas.extend([
            {"field": "method_enabled", "source": declared[method]["enable"], "adapter": True,
             "reason": "one explicitly selected method per native task"},
            {"field": "evaluator_shape", "source": "nested list results and obsolete constructor keywords",
             "adapter": "named list / single dict / dict by name at the actual algorithm seams",
             "reason": "repair incompatible fixed entry APIs; no score or candidate filtering change"},
            {"field": "search.market", "source": search["market"], "adapter": requested_market,
             "reason": "bind every callback to the selected market, including SP500"},
        ])
    output = {"schema_version": 1, "profile": profile, "config": config_path, "method": "alphabench_" + method,
        "source_commit": "31bb94bbb7744177c51c9d011e07c31e3092b93e", "sources": hashes,
        "declared": declared, "effective_source": effective, "parameter_sources": origins,
        "requested_market": requested_market, "protocol_delta": deltas,
        "execution_ready": False,
        "pending": ["source-profile oracle intervals/filter semantics and native final-pool wiring",
                    "frozen complete budgets and data/environment identities"] +
                   ([] if searcher else ["explicit validation/test extension; neither exists in the example entry"])}
    output["digest"] = digest(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--config", choices=CONFIGS, required=True)
    parser.add_argument("--method", choices=("cot", "tot", "ea"), required=True)
    parser.add_argument("--market")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    resolved = resolve_source(args.upstream_root, args.config, args.method, market=args.market)
    if args.output.exists() and json.loads(args.output.read_text(encoding="utf-8")) != resolved:
        raise ValueError("source profile output already contains a different resolution")
    atomic_json_write(args.output, resolved)
    print(json.dumps({"output": str(args.output.resolve()), "digest": resolved["digest"], "execution_ready": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
