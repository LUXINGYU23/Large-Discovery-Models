"""Execute the real bridge with fake heavyweight modules and chemistry states."""

import json
import builtins
import io
import pickle
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
import pytest
from tasks.reasyn.core import projection_worker


class Mol:
    def __init__(self, s):
        self.csmiles = s


class Stack:
    def __init__(self):
        self.items = []

    def push_mol(self, mol, idx):
        self.items.append(mol)

    def push_rxn(self, rxn, idx):
        if len(self.items) < 2:
            return False
        self.items = self.items[:-2] + [Mol("CCC")]
        return True

    def get_stack_depth(self):
        return len(self.items)

    def get_top(self):
        return self.items[-1:]


@pytest.fixture
def worker_fixture(tmp_path, monkeypatch):
    captures = []
    controls = {"fail_on_call": None, "empty_targets": set()}
    original_import = builtins.__import__

    def import_without_sklearn(name, *args, **kwargs):
        if name == "sklearn" or name.startswith("sklearn."):
            raise AssertionError("mock worker assets must not import optional sklearn")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_sklearn)

    class Model:
        def __init__(self, config):
            captures.append(("model", config))

        def to(self, device):
            captures.append(("device", device))
            return self

        def load_state_dict(self, weights):
            captures.append(("weights", weights))

        def eval(self):
            return self

        def parameters(self):
            return [SimpleNamespace(numel=lambda: 5)]

    class Frame:
        empty = False

        def head(self, n):
            captures.append(("max_results", n))
            return self

        def to_dict(self, orient):
            return [
                {"smiles": "CCO", "synthesis": "CCO", "num_steps": 0, "score": 1},
                {
                    "smiles": "CCC",
                    "synthesis": "CCO;CCN;R0",
                    "num_steps": 1,
                    "score": 0.5,
                },
                {"smiles": "COC", "synthesis": "COC", "num_steps": 0, "score": 1},
                {
                    "smiles": "CCC",
                    "synthesis": "CCO;CCN;R9",
                    "num_steps": 1,
                    "score": 1,
                },
                {
                    "smiles": "CCCC",
                    "synthesis": "CCO;CCN;R0",
                    "num_steps": 1,
                    "score": 1,
                },
            ]

    class Sampler:
        def __init__(self, **kwargs):
            captures.append(("sampler", kwargs))
            self.target = kwargs["mol"].csmiles

        def evolve(self, **kwargs):
            captures.append(("evolve", kwargs))
            if sum(name == "evolve" for name, _ in captures) == controls["fail_on_call"]:
                raise RuntimeError("interrupted projection")

        def get_dataframe(self):
            if self.target in controls["empty_targets"]:
                return SimpleNamespace(empty=True)
            return Frame()

    def module(name, **attrs):
        m = ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        monkeypatch.setitem(sys.modules, name, m)

    def load(path, **kwargs):
        captures.append(("load", kwargs))
        return {
            "hyper_parameters": {"config": {"model": "frozen"}},
            "state_dict": {"model.weight": 3},
        }

    module(
        "torch",
        load=load,
        manual_seed=lambda s: captures.append(("seed", s)),
        cuda=SimpleNamespace(is_available=lambda: False),
        inference_mode=nullcontext,
    )
    module(
        "omegaconf", OmegaConf=SimpleNamespace(create=lambda c: SimpleNamespace(**c))
    )
    module("reasyn.models.reasyn", ReaSyn=Model)
    module("reasyn.chem.mol", Molecule=Mol)
    module("reasyn.chem.stack", Stack=Stack)
    module("reasyn.sampler.sampler", Sampler=Sampler)
    module("reasyn.utils.sample_utils", TimeLimit=lambda limit: ("seconds", limit))
    fpindex = tmp_path / "fp.pkl"
    matrix = tmp_path / "matrix.pkl"
    fpindex.write_bytes(
        pickle.dumps(SimpleNamespace(_molecules=[Mol("CCO"), Mol("CCN")]))
    )
    matrix.write_bytes(pickle.dumps(SimpleNamespace(reactions=["reaction-zero"])))
    settings = {
        "search_width": 8,
        "exhaustiveness": 4,
        "exact_break": True,
        "time_limit": 120,
        "num_cycles": 16,
        "num_editflow_samples": 100,
        "max_results": 100,
    }
    request = {
        "model_paths": ["ar.ckpt", "eb.ckpt"],
        "device": "cpu",
        "fpindex": str(fpindex),
        "rxn_matrix": str(matrix),
        "targets": ["CCO"],
        "sampling_seed": 7,
        "settings": settings,
    }
    inp = tmp_path / "request.json"
    out = tmp_path / "output.json"
    inp.write_text(json.dumps(request))
    monkeypatch.setattr(
        sys, "argv", ["projection_worker", "--request", str(inp), "--output", str(out)]
    )
    return SimpleNamespace(
        request=request, input=inp, output=out, captures=captures, controls=controls
    )


def test_worker_loads_frozen_pair_applies_limits_and_replays_routes(worker_fixture):
    captures = worker_fixture.captures
    assert projection_worker.main() == 0
    data = json.loads(worker_fixture.output.read_text())
    assert [(r["smiles"], r["num_steps"]) for r in data["rows"]] == [
        ("CCO", 0),
        ("CCC", 1),
    ]
    assert all(r["pathway_verified"] for r in data["rows"])
    assert data["parameter_count"] == 10
    assert data["complete"] is True
    assert data["completed_target_indices"] == [0]
    assert all(r["target_index"] == 0 and r["sampling_seed"] == 7 for r in data["rows"])
    assert sum(name == "load" for name, _ in captures) == 2
    assert all(
        value == {"map_location": "cpu", "weights_only": False}
        for name, value in captures
        if name == "load"
    )
    evolve = next(value for name, value in captures if name == "evolve")
    assert evolve["num_cycles"] == 16 and evolve["num_editflow_samples"] == 100
    assert evolve["num_editflow_steps"] == 100 and evolve["max_evolve_steps"] == 8
    assert evolve["time_limit"] == ("seconds", 120)
    sampler = next(value for name, value in captures if name == "sampler")
    assert (
        sampler["factor"] == 8
        and sampler["max_active_states"] == 4
        and sampler["exact_break"] is True
    )


def test_worker_resumes_completed_and_empty_occurrences_without_resampling(worker_fixture):
    fixture = worker_fixture
    fixture.request["targets"] = ["CCO", "CCN", "CCO"]
    fixture.input.write_text(json.dumps(fixture.request))
    fixture.controls.update(fail_on_call=3, empty_targets={"CCN"})
    with pytest.raises(RuntimeError, match="interrupted projection"):
        projection_worker.main()
    partial = json.loads(fixture.output.read_text())
    assert partial["complete"] is False
    assert partial["completed_target_indices"] == [0, 1]
    assert {row["target_index"] for row in partial["rows"]} == {0}
    assert projection_worker.main() == 0
    complete = json.loads(fixture.output.read_text())
    assert complete["complete"] is True
    assert complete["completed_target_indices"] == [0, 1, 2]
    assert {row["target_index"] for row in complete["rows"]} == {0, 2}
    assert [value for name, value in fixture.captures if name == "seed"] == [7, 8, 9, 9]
    loads = sum(name == "load" for name, _ in fixture.captures)
    assert projection_worker.main() == 0
    assert sum(name == "load" for name, _ in fixture.captures) == loads


def test_worker_rejects_changed_request_before_loading_models(worker_fixture):
    fixture = worker_fixture
    assert projection_worker.main() == 0
    fixture.request["sampling_seed"] += 1
    fixture.input.write_text(json.dumps(fixture.request))
    with pytest.raises(ValueError, match="request mismatch"):
        projection_worker.main()
    assert sum(name == "load" for name, _ in fixture.captures) == 2


def test_released_sklearn_metric_alias_is_installed_only_when_unpickled(monkeypatch):
    sklearn = ModuleType("sklearn")
    metrics = ModuleType("sklearn.metrics")
    distance = ModuleType("sklearn.metrics._dist_metrics")
    distance.ManhattanDistance = type("ManhattanDistance", (), {})
    sklearn.metrics = metrics
    metrics._dist_metrics = distance
    for module in (sklearn, metrics, distance):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    encoded = b"csklearn.metrics._dist_metrics\nManhattanDistance64\n."
    assert projection_worker._AssetUnpickler(io.BytesIO(encoded)).load() is distance.ManhattanDistance
