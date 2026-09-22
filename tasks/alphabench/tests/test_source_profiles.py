import copy
import json
import shutil

import pytest
import yaml

from ldm_tts.engine.run_store import BudgetExceededError
from ldm_tts.contracts.evaluation import EvaluationPaused
from tasks.alphabench.core.native_benchmark import ENTRY, load_benchmark
from tasks.alphabench.core.native_runtime import NativeStop
from tasks.alphabench.core.native_source import load_algorithms
from tasks.alphabench.core.protocol import digest
from tasks.alphabench.core.source_profiles import main, resolve_source
from tasks.alphabench.tests.test_native import callbacks, metrics, scheduler, source


EXAMPLE = "example/search/configs/search_csi300.yaml"


@pytest.mark.parametrize("config,method,rounds,temperature,workers,N,seeds", [
    ("searcher/config.yaml", "tot", 10, .5, 1, 6, 30),
    ("searcher/configs/cot_config.yaml", "cot", 10, 1.75, 4, None, 30),
    ("searcher/configs/tot_config.yaml", "tot", 10, .5, 4, 6, 30),
    ("searcher/configs/ea_config.yaml", "ea", 10, .7, None, 20, 125),
    (EXAMPLE, "cot", 10, 1.5, 4, None, 13),
    (EXAMPLE, "tot", 3, 1.5, 4, 6, 13),
    (EXAMPLE, "ea", 10, 1.5, None, 30, 29),
])
def test_source_configs_keep_their_actual_defaults_and_seed_semantics(source, config, method, rounds, temperature, workers, N, seeds):
    resolved = resolve_source(source, config, method, market="sp500")
    effective = resolved["effective_source"]
    algorithm = effective["algorithm"]
    assert (algorithm["rounds"], algorithm["temperature"], algorithm.get("workers"), algorithm.get("N")) == (rounds, temperature, workers, N)
    assert effective["initialization"]["count"] == seeds
    assert all("algorithm." + key in resolved["parameter_sources"] for key in algorithm)
    assert resolved["execution_ready"] is False
    assert digest({key: value for key, value in resolved.items() if key != "digest"}) == resolved["digest"]
    if config == EXAMPLE:
        assert effective["initialization"]["baseline_count"] == 42
        assert effective["search"]["market"] == "csi300" and effective["search"]["initialization_market"] == "sp500"
        assert (effective["search"]["start"], effective["search"]["end"], effective["search"]["fast"]) == ("2023-01-01", "2024-01-01", False)
        assert effective["validation"] is None and effective["test"] is None
        assert algorithm["enable_reason"] is False
    else:
        assert effective["search"]["market"] == "sp500"
        assert (effective["search"]["start"], effective["search"]["end"], effective["search"]["fast"]) == ("2016-01-01", "2021-01-01", True)
        assert effective["validation"]["start"] == "2021-01-01" and effective["test"]["start"] == "2022-01-01"
        assert algorithm["enable_reason"] is True


def test_resolved_source_is_immutable_and_requires_unmodified_pinned_inputs(tmp_path, source):
    output = tmp_path / "resolved.json"
    arguments = ["--upstream-root", str(source), "--config", EXAMPLE, "--method", "ea", "--output", str(output)]
    assert main(arguments) == 0
    before = output.read_bytes()
    assert main(arguments) == 0 and output.read_bytes() == before
    with pytest.raises(ValueError, match="different resolution"):
        main(arguments + ["--market", "sp500"])
    with pytest.raises(ValueError, match="method differs"):
        resolve_source(source, "searcher/config.yaml", "ea")
    clone = tmp_path / "source"
    for path in json.loads(output.read_text())["sources"]:
        target = clone / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / path, target)
    with (clone / ENTRY).open("a") as handle:
        handle.write("\n# changed entry\n")
    with pytest.raises(ValueError, match="pinned AlphaBench source changed"):
        resolve_source(clone, EXAMPLE, "ea")


def config(source, destination, method):
    value = yaml.safe_load((source / EXAMPLE).read_text())
    for name in ("cot", "tot", "ea"):
        value[name]["enable"] = name == method
    value["cot"]["rounds"] = value["tot"]["rounds"] = value["ea"]["generations"] = 2
    value["tot"]["size"] = value["ea"]["generate_size"] = 3
    value["ea"]["population_size"] = 5
    value["market"], value["save_dir"] = "sp500", str(destination)
    return value


@pytest.mark.parametrize("method", ["cot", "tot", "ea"])
def test_example_runs_every_group_seed_and_replays_the_complete_entry(tmp_path, source, method):
    run = tmp_path / "campaign"
    sched = scheduler(run)
    called = []
    operations = callbacks()
    def baseline(factors, *, market):
        assert len(factors) == 42 and market == "sp500"
        return [{**factor, **metrics(factor["expression"])} for factor in factors]
    operations["initialization_fn"] = baseline
    def wrap(kind, function):
        def call(identity, *args, **kwargs):
            called.append((kind, identity))
            return function(*args, **kwargs)
        return sched.callback(kind, call)
    args = config(source, run / "native_output", method)
    with load_algorithms(source, run / "private", sched) as algorithms:
        entry = load_benchmark(source, run / "private", sched, algorithms,
            **{name: wrap(name, operation) for name, operation in operations.items()})
        result = sched.run(lambda: entry(copy.deepcopy(args)))
    assert len(result["baseline"]) == 42
    assert sum(kind == "initialization_fn" for kind, _ in called) == 1
    if method == "ea":
        assert len(result["ea"]["history"]) == 2 and len(result["ea"]["final_pool"]) == 5
        initial = next(event["payload"]["output"]["state"] for event in sched.campaign.events()
                       if event["event_type"] == "native_boundary" and event["payload"]["kind"] == "state"
                       and event["payload"]["output"]["kind"] == "ea.pool")
        assert initial["round"] == 0 and len(initial["pool"]) == 29
    else:
        assert len(result[method]["results"]) == 13 and not result[method]["errors"]
        assert sum(kind in {"evaluate_fn", "batch_evaluate_fn_dict"} for kind, _ in called) >= 13
    sched = scheduler(run)
    before = (run / "events.jsonl").read_bytes()
    def forbidden(*args, **kwargs):
        pytest.fail("completed example callback dispatched again")
    with load_algorithms(source, run / "private", sched) as algorithms:
        entry = load_benchmark(source, run / "private", sched, algorithms,
            **{name: sched.callback(name, forbidden) for name in operations})
        assert sched.run(lambda: entry(copy.deepcopy(args))) == result
    assert (run / "events.jsonl").read_bytes() == before
    sched = scheduler(run)
    changed = copy.deepcopy(args)
    changed["model"]["temperature"] = .2
    with load_algorithms(source, run / "private", sched) as algorithms:
        entry = load_benchmark(source, run / "private", sched, algorithms,
            **{name: sched.callback(name, forbidden) for name in operations})
        with pytest.raises(EvaluationPaused, match="native boundary changed"):
            sched.run(lambda: entry(changed))


@pytest.mark.parametrize("method", ["cot", "tot", "ea"])
def test_example_does_not_turn_budget_stop_into_worker_failure(tmp_path, source, method):
    sched = scheduler(tmp_path)
    operations = callbacks()
    operations["initialization_fn"] = lambda factors, **_: [{**factor, **metrics(factor["expression"])} for factor in factors]
    def stop(*args, **kwargs):
        raise NativeStop(BudgetExceededError("provider budget exhausted"))
    operations["search_fn"] = stop
    with load_algorithms(source, tmp_path / "private", sched) as algorithms:
        entry = load_benchmark(source, tmp_path / "private", sched, algorithms,
            **{name: sched.callback(name, lambda identity, *args, operation=operation, **kwargs: operation(*args, **kwargs))
               for name, operation in operations.items()})
        with pytest.raises(BudgetExceededError, match="provider budget exhausted"):
            sched.run(lambda: entry(config(source, tmp_path / "output", method)))
