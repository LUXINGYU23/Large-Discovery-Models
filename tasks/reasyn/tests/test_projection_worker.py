"""Execute the real bridge with fake heavyweight modules and chemistry states."""

import json
import pickle
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
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


def test_worker_loads_frozen_pair_applies_limits_and_replays_routes(
    tmp_path, monkeypatch
):
    captures = []

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

        def evolve(self, **kwargs):
            captures.append(("evolve", kwargs))

        def get_dataframe(self):
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
    assert projection_worker.main() == 0
    data = json.loads(out.read_text())
    assert [(r["smiles"], r["num_steps"]) for r in data["rows"]] == [
        ("CCO", 0),
        ("CCC", 1),
    ]
    assert all(r["pathway_verified"] for r in data["rows"])
    assert data["parameter_count"] == 10
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
