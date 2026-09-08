from io import BytesIO, StringIO
from pathlib import Path
import importlib
import json

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from tasks.atomworld.core.geometry import (
    OperationError,
    OPERATIONS_SCHEMA,
    execute_operations,
)


def cif(atoms):
    stream = BytesIO()
    write(stream, atoms, format="cif")
    return stream.getvalue().decode("latin-1")


@pytest.fixture
def atoms():
    return Atoms(
        "HHeLiBe",
        positions=[[1, 1, 1], [3, 1, 1], [1, 2, 3], [8, 8, 8]],
        cell=[10, 10, 10],
        pbc=True,
    )


def execute(atoms, operations):
    return read(StringIO(execute_operations(cif(atoms), operations)), format="cif")


def test_add_change_remove_swap_and_sequential_indices(atoms):
    result = execute(
        atoms,
        [
            {"op": "add", "symbol": "Fe", "position": [2, 3, 4]},
            {"op": "change", "index": 0, "symbol": "C"},
            {"op": "swap", "index1": 0, "index2": 1},
            {"op": "remove", "index": 2},
            {"op": "move", "index": 2, "d_pos": [0, 0, -1]},
        ],
    )
    assert result.get_chemical_symbols() == ["He", "C", "Be", "Fe"]
    np.testing.assert_allclose(
        result.positions, [[1, 1, 1], [3, 1, 1], [8, 8, 7], [2, 3, 4]]
    )
    assert atoms.get_chemical_symbols() == ["H", "He", "Li", "Be"]


def test_move_and_insert_use_direct_vector_across_periodic_boundary():
    original = Atoms(
        "HHe", positions=[[0.5, 1, 1], [9.5, 1, 1]], cell=[10, 10, 10], pbc=True
    )
    moved = execute(
        original, [{"op": "move_towards", "index1": 0, "index2": 1, "distance": 1}]
    )
    np.testing.assert_allclose(moved.positions[0], [1.5, 1, 1])
    inserted = execute(
        original,
        [
            {
                "op": "insert_between",
                "index1": 0,
                "index2": 1,
                "symbol": "Fe",
                "distance": 4.5,
            }
        ],
    )
    np.testing.assert_allclose(inserted.positions[-1], [5, 1, 1])


def test_cartesian_displacement_in_triclinic_cell():
    original = Atoms(
        "HHe",
        positions=[[1, 2, 3], [4, 5, 6]],
        cell=[[10, 0, 0], [2, 9, 0], [1, 3, 8]],
        pbc=True,
    )
    result = execute(original, [{"op": "move", "index": 0, "d_pos": [1, -0.5, 0.75]}])
    np.testing.assert_allclose(result.positions[0], [2, 1.5, 3.75], atol=1e-12)
    np.testing.assert_allclose(result.positions[1], [4, 5, 6], atol=1e-12)
    np.testing.assert_allclose(result.cell, original.cell, atol=1e-12)


def test_delete_below_preserves_reference_and_equal_z(atoms):
    result = execute(atoms, [{"op": "delete_below", "index": 2}])
    assert result.get_chemical_symbols() == ["Li", "Be"]
    np.testing.assert_allclose(result.positions, [[1, 2, 3], [8, 8, 8]])
    with pytest.raises(OperationError, match="no atoms"):
        execute(atoms, [{"op": "delete_below", "index": 0}])


def test_rotation_right_hand_strict_radius_and_mic_membership():
    original = Atoms(
        "HHeLiBe",
        positions=[[0.5, 1, 1], [9.5, 1, 1], [2.5, 1, 1], [1.5, 1, 1]],
        cell=[10, 10, 10],
        pbc=True,
    )
    result = execute(
        original,
        [
            {
                "op": "rotate_around",
                "index": 0,
                "radius": 2,
                "angle": 90,
                "axis": [0, 0, 2],
            }
        ],
    )
    # He is selected across PBC (distance 1), then original +9 Cartesian vector is
    # rotated, producing y=10 -> serialized y=0. Li at radius exactly 2 is excluded.
    np.testing.assert_allclose(
        result.positions,
        [[0.5, 1, 1], [0.5, 0, 1], [2.5, 1, 1], [0.5, 2, 1]],
        atol=1e-10,
    )


def test_supercell_cell_volume_species_and_positions(atoms):
    result = execute(atoms, [{"op": "super_cell", "size": [2, 1, 2]}])
    np.testing.assert_allclose(result.cell.lengths(), [20, 10, 20])
    assert len(result) == 16
    assert result.get_chemical_symbols().count("Li") == 4
    assert result.get_volume() == pytest.approx(4000)
    np.testing.assert_allclose(result.positions[4:8], atoms.positions + [0, 0, 10])


@pytest.mark.parametrize(
    "operation",
    [
        {"op": "remove", "index": -1},
        {"op": "remove", "index": 4},
        {"op": "remove", "index": True},
        {"op": "remove", "index": 1.0},
        {"op": "change", "index": 0, "symbol": "Fe2"},
        {"op": "change", "index": 0, "symbol": "X"},
        {"op": "move", "index": 0, "d_pos": [float("nan"), 0, 0]},
        {"op": "move", "index": 0, "d_pos": [float("inf"), 0, 0]},
        {"op": "move", "index": 0, "d_pos": [True, 0, 0]},
        {"op": "move", "index": 0, "d_pos": [0, 0]},
        {"op": "move", "index": 0, "d_pos": [0, 0, 1000001]},
        {"op": "remove", "index": 0, "target_cif": "hidden"},
        {"op": "move"},
        {"op": "__import__", "code": "1 + 1"},
        {"op": []},
        {
            "op": "insert_between",
            "index1": 0,
            "index2": 1,
            "symbol": "C",
            "distance": 3,
        },
        {"op": "move_towards", "index1": 0, "index2": 0, "distance": 1},
        {"op": "move_towards", "index1": 0, "index2": 1, "distance": -1},
        {
            "op": "rotate_around",
            "index": 0,
            "radius": 0,
            "angle": 90,
            "axis": [0, 0, 1],
        },
        {
            "op": "rotate_around",
            "index": 0,
            "radius": 2,
            "angle": 360,
            "axis": [0, 0, 1],
        },
        {
            "op": "rotate_around",
            "index": 0,
            "radius": 2,
            "angle": 90,
            "axis": [0, 0, 0],
        },
        {"op": "super_cell", "size": [True, 1, 1]},
        {"op": "super_cell", "size": [1, 0, 1]},
        {"op": "super_cell", "size": [16, 16, 16]},
        {"op": "super_cell", "size": [1.0, 1, 1]},
    ],
)
def test_invalid_operations_are_corrective_errors(atoms, operation):
    with pytest.raises(OperationError, match="operation 0:"):
        execute(atoms, [operation])


@pytest.mark.parametrize(
    "operations", [[], {}, None, ["remove"], [{"op": "remove", "index": 0}] * 65]
)
def test_invalid_plan_shapes(atoms, operations):
    with pytest.raises(OperationError):
        execute(atoms, operations)


def test_empty_structure_multi_frame_and_unparseable_input():
    single = Atoms("H", positions=[[1, 1, 1]], cell=[10, 10, 10], pbc=True)
    with pytest.raises(OperationError, match="1..10000 atoms"):
        execute(single, [{"op": "remove", "index": 0}])
    with pytest.raises(OperationError, match="exactly one"):
        execute_operations(
            cif(single) + cif(single).replace("data_image0", "data_image1"),
            [{"op": "remove", "index": 0}],
        )
    with pytest.raises(OperationError):
        execute_operations("not a CIF", [{"op": "remove", "index": 0}])


def test_fractional_occupancy_is_not_silently_reinterpreted(atoms):
    text = cif(atoms).replace("H1         1.0", "H1         0.5")
    # ASE writer places occupancy at the end of its atom rows; use token positions
    # so this fixture is robust to ordinary writer whitespace changes.
    rows = text.splitlines()
    for i, row in enumerate(rows):
        if row.split()[:2] == ["H", "H1"]:
            words = row.split()
            words[-1] = "0.5"
            rows[i] = " ".join(words)
    with pytest.raises(OperationError, match="partially occupied"):
        execute_operations("\n".join(rows), [{"op": "remove", "index": 0}])


def test_all_ten_actions_match_released_upstream_executor(atoms, monkeypatch):
    """Optional local parity check; normal standalone installation has no upstream dependency."""
    from tasks.atomworld.core.data import DEFAULT_UPSTREAM

    upstream = DEFAULT_UPSTREAM / "src"
    if not upstream.exists():
        pytest.skip("optional upstream local checkout unavailable")
    monkeypatch.syspath_prepend(str(upstream))
    actions = importlib.import_module("atom_world.actions")
    cells = importlib.import_module("atom_world.cell_actions")
    cases = [
        (
            {"op": "add", "symbol": "C", "position": [2, 3, 4]},
            "AddAtomAction",
            {"symbol": "C", "position": np.array([2, 3, 4])},
        ),
        (
            {"op": "change", "index": 0, "symbol": "C"},
            "ChangeAtomAction",
            {"index": 0, "symbol": "C"},
        ),
        ({"op": "remove", "index": 0}, "RemoveAtomAction", {"index": 0}),
        (
            {"op": "swap", "index1": 0, "index2": 1},
            "SwapAtomsAction",
            {"index1": 0, "index2": 1},
        ),
        (
            {"op": "move", "index": 0, "d_pos": [1, 0, 0]},
            "MoveAtomAction",
            {"index": 0, "d_pos": np.array([1, 0, 0])},
        ),
        (
            {"op": "move_towards", "index1": 0, "index2": 1, "distance": 0.5},
            "MoveTowardsAtomAction",
            {"index1": 0, "index2": 1, "distance": 0.5},
        ),
        (
            {
                "op": "insert_between",
                "index1": 0,
                "index2": 1,
                "symbol": "C",
                "distance": 0.5,
            },
            "InsertBetweenAtomsAction",
            {"index1": 0, "index2": 1, "symbol": "C", "distance_ratio": 0.25},
        ),
        ({"op": "delete_below", "index": 2}, "DeleteBelowAtomAction", {"index": 2}),
        (
            {
                "op": "rotate_around",
                "index": 0,
                "radius": 3,
                "angle": 90,
                "axis": [0, 0, 1],
            },
            "RotateAroundAtomAction",
            {"index": 0, "radius": 3, "angle": 90, "axis": np.array([0, 0, 1])},
        ),
    ]
    for operation, name, kwargs in cases:
        expected = getattr(actions, name)(atoms.copy(), **kwargs).execute()
        actual = execute(atoms, [operation])
        assert actual.get_chemical_symbols() == expected.get_chemical_symbols(), name
        np.testing.assert_allclose(
            actual.get_scaled_positions(),
            expected.get_scaled_positions(),
            atol=1e-12,
            err_msg=name,
        )
    expected = cells.SuperCellAction([2, 1, 2]).execute(atoms.copy())
    actual = execute(atoms, [{"op": "super_cell", "size": [2, 1, 2]}])
    np.testing.assert_allclose(actual.cell, expected.cell)
    np.testing.assert_allclose(actual.positions, expected.positions)
