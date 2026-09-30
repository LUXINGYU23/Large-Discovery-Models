from contextlib import contextmanager
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from threading import Barrier
from threading import Lock
import time

import pytest

from ldm_tts.contracts.evaluation import EvaluationPaused
from ldm_tts.engine.run_store import BudgetExceededError, CampaignRuntime
from tasks.alphabench.core.host import HostDispatcher
from tasks.alphabench.core.native_runtime import NativeRuntime
from tasks.alphabench.core.native_source import load_algorithms
from tasks.alphabench.core.protocol import digest
from tasks.alphabench.core.receipts import Receipts


SOURCE = Path(os.environ["ALPHABENCH_SOURCE_ROOT"]) if os.environ.get("ALPHABENCH_SOURCE_ROOT") else None


@pytest.fixture
def source():
    if SOURCE is None or not (SOURCE / "searcher/algo/__init__.py").exists():
        pytest.skip("pinned AlphaBench source checkout is required")
    return SOURCE


def scheduler(root):
    runtime = CampaignRuntime.open(root, task="alphabench", resume=(root / "campaign.json").exists())
    return NativeRuntime(runtime, HostDispatcher())


def metrics(expression):
    value = int(sha256(expression.encode()).hexdigest()[:4], 16) / 65535
    return {"success": True, "metrics": {"ic": value / 2, "rank_ic": value, "icir": value * 2}}


SEEDS = [{"name": f"seed{i}", "expression": expr, "metrics": metrics(expr)["metrics"]}
         for i, expr in enumerate(("$open", "$close", "$volume"))]
CONFIG = {"rounds": 3, "workers": 2, "N": 3, "top_k": 2, "pool_size": 4,
          "seeds_top_k": 2, "mutation_rate": .5, "crossover_rate": .5, "accept_threshold": .25}


def search_fixture(instruction, model, N, **kwargs):
    # UUID suffixes are display names, not part of this deterministic generator.
    prompt = re.sub(r"_[0-9a-f]{6}\b", "_UUID", instruction)
    number = int(sha256(prompt.encode()).hexdigest()[:5], 16)
    time.sleep(.001 * (number % 4))
    # Includes within-batch duplicates and partial generation without cross-branch races.
    return {"success": True, "factors": [
        {"name": f"factor{number}_{i}", "expression": f"Mean($close,{2 + number * 10 + i % 2})"}
        for i in range(max(1, N - (number % 2)))], "quality": {"fixture": True}}


def callbacks():
    return {"search_fn": search_fixture,
            "evaluate_fn": metrics,
            "batch_evaluate_fn": lambda factors: [metrics(f["expression"]) for f in factors],
            "batch_evaluate_fn_dict": lambda factors: {f["name"]: metrics(f["expression"]) for f in factors}}


@contextmanager
def original_algorithms(source):
    name = "_alphabench_original_fixture"
    spec = importlib.util.spec_from_file_location(name, source / "searcher/algo/__init__.py",
                                                  submodule_search_locations=[str(source / "searcher/algo")])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        for key in list(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]


def scientific(value, method):
    if isinstance(value, dict):
        result = {key: scientific(item, method) for key, item in value.items()
                  if "elapsed" not in key and key not in {"save_path"}}
        if method == "tot":
            for key in ("history", "final_pool"):
                if key in result:
                    result[key].sort(key=lambda item: json.dumps(item, sort_keys=True))
        return result
    if isinstance(value, (tuple, list)):
        return [scientific(item, method) for item in value]
    if isinstance(value, str):
        return re.sub(r"_[0-9a-f]{6}\b", "_UUID", value)
    return value


@pytest.mark.parametrize("method", ["cot", "tot", "ea"])
def test_fixed_native_algorithms_match_upstream_and_replay_without_callbacks(tmp_path, source, method):
    with original_algorithms(source) as original:
        expected = original.create_algo(method, CONFIG, **callbacks()).run(SEEDS, str(tmp_path / "original"))
    run = tmp_path / "campaign"
    sched = scheduler(run)
    called = []
    def wrap(kind, function):
        def operation(identity, *args, **kwargs):
            called.append((kind, identity))
            return function(*args, **kwargs)
        return sched.callback(kind, operation)
    with load_algorithms(source, run / "private", sched) as module:
        algo = module.create_algo(method, CONFIG, **{kind: wrap(kind, function) for kind, function in callbacks().items()})
        actual = sched.run(lambda: algo.run(SEEDS, str(run / "native")))
    assert actual["history"] and called
    assert scientific(actual, method) == scientific(expected, method)
    journal_before = (run / "events.jsonl").read_bytes()
    replay = NativeRuntime(sched.campaign, HostDispatcher())
    def forbidden(*args, **kwargs):
        pytest.fail("completed native callback was dispatched again")
    with load_algorithms(source, run / "private", replay) as module:
        algo = module.create_algo(method, CONFIG, **{kind: replay.callback(kind, forbidden) for kind in callbacks()})
        replayed = replay.run(lambda: algo.run(SEEDS, str(run / "native")))
    assert actual == replayed  # Names, completion order, timing and every tree/chain/pool entry.
    assert (run / "events.jsonl").read_bytes() == journal_before


@pytest.mark.parametrize("method", ["cot", "ea"])
def test_matched_native_generation_fills_each_round_after_empty_or_duplicate_output(tmp_path, source, method):
    sched = scheduler(tmp_path / "campaign")
    lock, calls, evaluated = Lock(), 0, []

    def search(instruction, model, N, **kwargs):
        nonlocal calls
        with lock:
            calls += 1
            index = calls
        expression = ("$close" if index <= (1 if method == "cot" else 2) else "$open")
        return {"success": True, "factors": [] if method == "cot" and index == 1 else
                [{"name": f"candidate{index}", "expression": expression}]}

    def evaluate(factors):
        evaluated.extend(factor["expression"] for factor in factors)
        return [metrics(factor["expression"]) for factor in factors]

    config = {**CONFIG, "rounds": 1, "workers": 1, "N": 2, "mutation_rate": .5, "crossover_rate": .5}
    operations = {"search_fn": sched.callback("search_fn", lambda identity, *args, **kwargs: search(*args, **kwargs)),
                  "batch_evaluate_fn": sched.callback("batch_evaluate_fn", lambda identity, factors: evaluate(factors)),
                  "evaluate_fn": sched.callback("evaluate_fn", lambda identity, expression: metrics(expression))}
    with load_algorithms(source, tmp_path / "campaign/private", sched, matched=True) as module:
        algo = module.create_algo(method, config, **operations)
        sched.run(lambda: algo.run(SEEDS[:1], str(tmp_path / "native")))
    assert calls == (2 if method == "cot" else 3)
    assert evaluated == (["$open", "$open"] if method == "cot" else ["$close", "$open"])


def test_scheduler_keeps_parallel_io_but_replays_recorded_delivery_order(tmp_path):
    sched = scheduler(tmp_path)
    barrier, calls = Barrier(2), []
    def work(identity, index):
        calls.append(index)
        barrier.wait(timeout=3)
        if index == 0:
            time.sleep(.02)
        return index
    def stage(current, function):
        with current.executor(max_workers=2) as pool:
            futures = [pool.submit(function, i) for i in range(2)]
            return [future.result() for future in current.completed(futures)]
    first = sched.run(lambda: stage(sched, sched.callback("io", work)))
    assert first == [1, 0]
    replay = NativeRuntime(sched.campaign, HostDispatcher())
    assert replay.run(lambda: stage(replay, replay.callback("io", work))) == first
    assert sorted(calls) == [0, 1]


@pytest.mark.parametrize("method", ["cot", "tot", "ea"])
@pytest.mark.parametrize("phase", ["search", "evaluation"])
def test_native_budget_signal_escapes_upstream_catchers_and_stops_other_workers(tmp_path, source, method, phase):
    sched = scheduler(tmp_path)
    failure = BudgetExceededError("fixture", 1, 0)
    calls = []
    def stop(identity, *args, **kwargs):
        assert sched.stopped is None
        calls.append(identity)
        raise failure
    with load_algorithms(source, tmp_path / "private", sched) as module:
        operations = {kind: sched.callback(kind, lambda identity, *args, _fn=function, **kwargs: _fn(*args, **kwargs))
                      for kind, function in callbacks().items()}
        target = "search_fn" if phase == "search" else "batch_evaluate_fn_dict" if method == "tot" else "batch_evaluate_fn"
        operations[target] = sched.callback(target, stop)
        algo = module.create_algo(method, CONFIG, **operations)
        with pytest.raises(BudgetExceededError) as error:
            sched.run(lambda: algo.run(SEEDS, str(tmp_path / "native")))
    assert error.value is failure
    assert 1 <= len(calls) <= 2  # Both workers may have entered before the first failure is delivered.


def test_replay_mismatch_pauses_before_new_work_and_releases_waiters(tmp_path):
    sched = scheduler(tmp_path)
    calls = []
    def stage(current, changed=False):
        def work(index):
            return current.callback("io", lambda identity, item: calls.append(item) or item)(
                index + (10 if changed and index == 1 else 0))
        with current.executor(max_workers=2) as pool:
            return [future.result() for future in [pool.submit(work, i) for i in range(2)]]
    sched.run(lambda: stage(sched))
    with pytest.raises(EvaluationPaused, match="boundary changed"):
        replay = NativeRuntime(sched.campaign, HostDispatcher())
        replay.run(lambda: stage(replay, True))
    assert sorted(calls) == [0, 1]


def test_source_integrity_is_checked_before_execution(tmp_path, source):
    import shutil
    copy = tmp_path / "source"
    shutil.copytree(source / "searcher/algo", copy / "searcher/algo")
    (copy / "searcher/algo/cot.py").write_text("raise AssertionError('must not execute')\n")
    with pytest.raises(ValueError, match="source changed"):
        with load_algorithms(copy, tmp_path / "private", scheduler(tmp_path / "run")):
            pytest.fail("tampered source was loaded")


def crash_fixture(root, method, crash):
    sched = scheduler(root)
    receipts = Receipts(root / "receipts")
    physical = root / "physical"
    physical.mkdir(exist_ok=True)
    record, hits = sched.campaign.record, 0
    def crashing_record(kind, payload=None, **kwargs):
        nonlocal hits
        event = record(kind, payload, **kwargs)
        target = payload and (payload.get("kind") == crash or
                              (crash == "state" and payload.get("kind") == "state"))
        if kind == "native_boundary" and target:
            hits += 1
            in_flight = any(json.loads(file.read_text())["state"] != "completed"
                            for file in receipts.root.glob("*.json"))
            if hits >= (2 if crash == "state" and method != "tot" else 1) and not in_flight:
                os._exit(86)  # No Python exception unwinding or executor shutdown.
        return event
    sched.campaign.record = crashing_record
    def search(instruction, model, N, **kwargs):
        prompt = re.sub(r"_[0-9a-f]{6}\b", "_UUID", instruction)
        number = int(sha256(prompt.encode()).hexdigest()[:8], 16)
        return {"success": True, "factors": [
            {"name": f"factor{number}_{i}", "expression": f"Mean($close,{number * 10 + i + 2})"}
            for i in range(N)]}
    def wrap(kind, function):
        def operation(identity, *args, **kwargs):
            request = {"args": args, "kwargs": kwargs}
            def send():
                with (physical / (digest(identity) + ".txt")).open("a") as handle:
                    handle.write("dispatched\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                return function(*args, **kwargs)
            return receipts.execute(identity, request,
                reserve=lambda: sched.campaign.consume_many({"fixture_calls": 1}, usage_key=identity),
                operation=send, owner=sched.host)
        return sched.callback(kind, operation)
    operations = callbacks()
    operations["search_fn"] = search
    with load_algorithms(SOURCE, root / "private", sched) as module:
        algo = module.create_algo(method, CONFIG, **{key: wrap(key, value) for key, value in operations.items()})
        answer = sched.run(lambda: algo.run(SEEDS, str(root / "native")))
    (root / "answer.json").write_text(json.dumps(answer))


@pytest.mark.parametrize("method", ["cot", "tot", "ea"])
@pytest.mark.parametrize("boundary", ["search_fn.end", "evaluation", "state"])
def test_actual_process_death_resumes_native_algorithm_without_duplicate_dispatch(tmp_path, source, method, boundary):
    target = ("batch_evaluate_fn_dict.end" if method == "tot" else "batch_evaluate_fn.end") if boundary == "evaluation" else boundary
    run, baseline = tmp_path / "killed", tmp_path / "baseline"
    def child(root, crash):
        return subprocess.run([sys.executable, "-m", "tasks.alphabench.tests.test_native",
                               str(root), method, crash], capture_output=True, text=True, timeout=30)
    killed = child(run, target)
    assert killed.returncode == 86, killed.stderr
    prefix = (run / "events.jsonl").read_bytes()
    resumed = child(run, "none")
    assert resumed.returncode == 0, resumed.stderr
    assert (run / "events.jsonl").read_bytes().startswith(prefix)
    answer = json.loads((run / "answer.json").read_text())
    replayed = child(run, "none")
    assert replayed.returncode == 0, replayed.stderr
    assert json.loads((run / "answer.json").read_text()) == answer
    assert all(file.read_text() == "dispatched\n" for file in (run / "physical").iterdir())
    assert child(baseline, "none").returncode == 0
    assert scientific(json.loads((baseline / "answer.json").read_text()), method) == scientific(answer, method)


if __name__ == "__main__":
    crash_fixture(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
