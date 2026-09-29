from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tasks.antibody.core.cdr_constraints import check_cdr_constraints, check_cdr_constraints_all
from tasks.antibody.core.ldm.acquisition.parallel_search import (
    execute_atoms,
    execute_sampling_atoms,
    parallel_local_search,
)
from tasks.antibody.core.ldm.dsl.alphabet import AA_TO_IDX
from tasks.antibody.core.ldm.dsl.search_space import (
    LatinHyperCubeSampling,
    LocalSearch,
    NeighborSampling,
    Or,
)


CENTER = "ARDYGNYWYFD"
DEVICE = torch.device("cpu")
CONFIG = np.full(11, 20)


@pytest.mark.parametrize(
    ("sequence", "violations"),
    [
        ("ARDYGNYWYFD", (0, 0, 0)),
        ("AAAAAAAAAAA", (1, 0, 0)),
        ("RRRACFGHIKL", (0, 1, 0)),
        ("ACDNSTFGHIK", (0, 0, 1)),
    ],
)
def test_cdr_constraints_reject_repeat_charge_and_glycosylation(sequence, violations):
    encoded = np.asarray([AA_TO_IDX[aa] for aa in sequence])
    assert check_cdr_constraints_all(encoded) == violations
    assert check_cdr_constraints(encoded) is (violations == (0, 0, 0))


class FakeGP:
    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        return values

    def likelihood(self, values: torch.Tensor) -> SimpleNamespace:
        totals = values.sum(dim=1)
        return SimpleNamespace(
            mean=totals / 10,
            stddev=torch.full_like(totals, 0.5),
        )


def acquisition(values: torch.Tensor) -> torch.Tensor:
    return values.sum(dim=1)


def test_sampling_atoms_return_scored_candidates_with_bias():
    atoms = [LatinHyperCubeSampling(num=3)]

    results = execute_sampling_atoms(
        atoms,
        FakeGP(),
        acquisition,
        lambda seq: 2.0 if seq[0] >= 0 else 0.0,
        0.25,
        np.random.default_rng(7),
        1.0,
        DEVICE,
        acq_name="ucb",
    )

    assert len(results) == 3
    for result in results:
        assert set(result) == {"seq", "ucb", "mu", "sigma", "bias", "bias+ucb"}
        assert result["sigma"] == pytest.approx(0.5)
        assert result["bias"] == pytest.approx(2.0)
        assert result["bias+ucb"] == pytest.approx(result["ucb"] + 0.5)


def test_sampling_atoms_accept_empty_input():
    assert execute_sampling_atoms(
        [], FakeGP(), acquisition, None, 0.0,
        np.random.default_rng(7), 1.0, DEVICE,
    ) == []


def test_local_search_reports_centers_and_hill_climb_steps():
    np.random.seed(4)
    atom = LocalSearch(CENTER, radius=2, restart=2, steps=3)

    results = parallel_local_search(
        [atom], FakeGP(), acquisition, None, 0.0,
        CONFIG, False, DEVICE, timeout_s=1.0,
    )

    assert len(results) == 8
    assert sum("step=0" in result["source"] for result in results) == 2
    assert all(result["source"].startswith(f"LocalSearch(center={CENTER}") for result in results)
    assert all(len(result["seq"]) == 11 for result in results)
    assert all("bias+ei" in result for result in results)


def test_local_search_honors_fixed_positions_and_radius():
    np.random.seed(8)
    atom = LocalSearch(CENTER, fixed=".**********", radius=1, restart=1, steps=5)

    results = parallel_local_search(
        [atom], FakeGP(), acquisition, None, 0.0,
        CONFIG, False, DEVICE, timeout_s=1.0,
    )

    center = atom.center_idx
    assert len(results) == 6
    assert all(result["seq"][0] == center[0] for result in results)
    assert all(sum(a != b for a, b in zip(result["seq"], center)) <= 1 for result in results)


def test_local_search_with_expired_timeout_only_evaluates_center():
    atom = LocalSearch(CENTER, restart=1, steps=4)

    results = parallel_local_search(
        [atom], FakeGP(), acquisition, None, 0.0,
        CONFIG, False, DEVICE, timeout_s=-1.0,
    )

    assert len(results) == 1
    assert "step=0" in results[0]["source"]


def test_local_search_accepts_no_workers():
    assert parallel_local_search(
        [], FakeGP(), acquisition, None, 0.0,
        CONFIG, False, DEVICE,
    ) == []


def test_execute_atoms_dispatches_union_members():
    np.random.seed(3)
    search = Or(
        LocalSearch(CENTER, radius=0, restart=1, steps=1),
        NeighborSampling(CENTER, radius=1, budget=2),
    )

    results = execute_atoms(
        search, FakeGP(), acquisition, None, 0.0,
        CONFIG, False, np.random.default_rng(12), 1.0, DEVICE,
    )

    assert len(results) == 3
    assert sum("source" in result for result in results) == 1
    assert sum("source" not in result for result in results) == 2
