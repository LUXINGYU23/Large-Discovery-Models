"""Bounded ASE geometry execution, reproducing the upstream action semantics.

This module does not import the benchmark, inspect prompts, access files, run code,
or accept a target CIF. The caller provides only the public input and an explicit
structured plan. Validation errors are safe to return to the proposing model.
"""

from io import BytesIO, StringIO
import math

import numpy as np
from ase import Atom
from ase.data import chemical_symbols
from ase.io import read, write

from .schema import MAX_ATOMS, MAX_CIF_BYTES, MAX_MAGNITUDE, MAX_OPERATIONS, OPERATION_FIELDS


class OperationError(ValueError):
    """A public-input parse/validation/execution failure, never a judge result."""


def _number(value, name, *, minimum=-MAX_MAGNITUDE, maximum=MAX_MAGNITUDE):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OperationError(f"{name} must be a finite number, not {type(value).__name__}")
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise OperationError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return float(value)


def _vector(value, name):
    if not isinstance(value, list) or len(value) != 3:
        raise OperationError(f"{name} must be a JSON array of three finite numbers")
    return np.array([_number(v, name) for v in value])


def _index(value, name, count):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < count:
        raise OperationError(f"{name} must be an integer atom index in [0, {count - 1}]")
    return value


def _validate_fields(operation, count):
    if not isinstance(operation, dict):
        raise OperationError("each operation must be a JSON object")
    name = operation.get("op")
    if not isinstance(name, str) or name not in OPERATION_FIELDS:
        raise OperationError(f"op must be one of {', '.join(OPERATION_FIELDS)}")
    required = set(OPERATION_FIELDS[name]) | {"op"}
    missing, extra = required - operation.keys(), operation.keys() - required
    if missing or extra:
        raise OperationError(f"{name}: missing fields={sorted(missing)}; extra fields={sorted(map(str, extra))}")
    args = {key: value for key, value in operation.items() if key != "op"}
    for key in ("index", "index1", "index2"):
        if key in args:
            args[key] = _index(args[key], key, count)
    if "symbol" in args and (not isinstance(args["symbol"], str) or args["symbol"] not in chemical_symbols[1:]):
        raise OperationError("symbol must be one chemical element, e.g. Fe (not a formula or isotope)")
    for key in ("position", "d_pos", "axis"):
        if key in args:
            args[key] = _vector(args[key], key)
    for key in ("distance", "radius", "angle"):
        if key in args:
            args[key] = _number(args[key], key, minimum=0)
    if "radius" in args and args["radius"] <= 0:
        raise OperationError("radius must be positive")
    if "angle" in args and args["angle"] >= 360:
        raise OperationError("angle must be in [0, 360) degrees")
    if "axis" in args and np.linalg.norm(args["axis"]) == 0:
        raise OperationError("axis must be nonzero")
    if "size" in args:
        size = args["size"]
        if (not isinstance(size, list) or len(size) != 3
                or any(isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 16 for v in size)):
            raise OperationError("size must contain three integer repeat factors in [1, 16]")
        if math.prod(size) > 512 or count * math.prod(size) > MAX_ATOMS:
            raise OperationError("supercell exceeds the 512-cell or 10000-atom limit")
    return name, args


def _check_structure(atoms):
    if not 1 <= len(atoms) <= MAX_ATOMS:
        raise OperationError(f"structure must contain 1..{MAX_ATOMS} atoms")
    if not np.isfinite(atoms.positions).all() or np.abs(atoms.positions).max() > MAX_MAGNITUDE:
        raise OperationError("resulting Cartesian coordinates must be finite and within +/-1000000 angstroms")
    cell = np.asarray(atoms.cell)
    if (not np.isfinite(cell).all() or np.abs(cell).max() > MAX_MAGNITUDE
            or atoms.cell.rank != 3 or abs(np.linalg.det(cell)) <= 1e-12):
        raise OperationError("CIF requires a finite nonsingular three-dimensional cell")
    occupancy = atoms.info.get("occupancy", {})
    if any(len(species) != 1 or not np.isclose(next(iter(species.values())), 1)
           for species in occupancy.values()):
        raise OperationError("disordered or partially occupied sites are unsupported")


def _execute(atoms, name, a):
    if name == "add":
        atoms.append(Atom(a["symbol"], a["position"]))
    elif name == "change":
        atoms[a["index"]].symbol = a["symbol"]
    elif name == "remove":
        del atoms[a["index"]]
    elif name == "swap":
        i, j = a["index1"], a["index2"]
        atoms[i].symbol, atoms[j].symbol = atoms[j].symbol, atoms[i].symbol
    elif name == "move":
        atoms.positions[a["index"]] += a["d_pos"]
    elif name in ("move_towards", "insert_between"):
        i, j = a["index1"], a["index2"]
        delta = atoms.positions[j] - atoms.positions[i]
        separation = np.linalg.norm(delta)
        if separation == 0:
            raise OperationError("selected atoms must have distinct Cartesian positions")
        if name == "insert_between" and a["distance"] > separation + 1e-12:
            raise OperationError("insert_between distance exceeds the direct interatomic separation")
        position = atoms.positions[i] + delta * (a["distance"] / separation)
        if name == "move_towards":
            atoms.positions[i] = position
        else:
            atoms.append(Atom(a["symbol"], position))
    elif name == "delete_below":
        remove = atoms.positions[:, 2] < atoms.positions[a["index"], 2]
        if not remove.any():
            raise OperationError("no atoms have strictly lower Cartesian z than the reference")
        atoms = atoms[~remove]
    elif name == "rotate_around":
        i = a["index"]
        distances = atoms.get_distances(i, range(len(atoms)), mic=True)
        selected = np.flatnonzero((distances < a["radius"]) & (np.arange(len(atoms)) != i))
        if len(selected):
            subset = atoms[selected]
            subset.rotate(a["angle"], a["axis"], center=atoms.positions[i].copy())
            atoms.positions[selected] = subset.positions
    elif name == "super_cell":
        atoms = atoms.repeat(a["size"])
    return atoms


def execute_operations(input_cif: str, operations: list[dict]) -> str:
    """Apply a structured plan to public input, returning a bare P1 CIF.

    Raises OperationError with corrective public-input information. Operations are
    sequential and atomic from the caller's perspective: no partial CIF is returned.
    Input atom order follows ASE's CIF reader, as in the upstream generator. Output
    serialization wraps fractional coordinates, as in ASE's upstream CIF writer.
    """
    if not isinstance(input_cif, str) or not input_cif.strip():
        raise OperationError("input_cif must be nonempty CIF text")
    if len(input_cif.encode("utf-8")) > MAX_CIF_BYTES:
        raise OperationError("input_cif exceeds the 2000000-byte limit")
    if not isinstance(operations, list) or not 1 <= len(operations) <= MAX_OPERATIONS:
        raise OperationError(f"operations must be a JSON array containing 1..{MAX_OPERATIONS} operations")
    try:
        frames = read(StringIO(input_cif), format="cif", index=":")
        if len(frames) != 1:
            raise OperationError("input_cif must contain exactly one crystal structure")
        atoms = frames[0]
        _check_structure(atoms)
        # These CIF provenance arrays describe the original species. Keeping them
        # after a change/swap can make ASE restore stale occupancy in the output.
        atoms.info.pop("occupancy", None)
        atoms.arrays.pop("spacegroup_kinds", None)
    except OperationError:
        raise
    except Exception as error:
        raise OperationError(f"cannot parse input CIF ({type(error).__name__}); provide a valid ordered CIF") from error
    for step, operation in enumerate(operations):
        try:
            name, args = _validate_fields(operation, len(atoms))
            atoms = _execute(atoms, name, args)
            _check_structure(atoms)
        except OperationError as error:
            raise OperationError(f"operation {step}: {error}") from error
        except Exception as error:
            raise OperationError(f"operation {step}: geometry execution failed ({type(error).__name__})") from error
    try:
        output = BytesIO()
        write(output, atoms, format="cif")
        return output.getvalue().decode("latin-1")
    except Exception as error:
        raise OperationError(f"cannot serialize resulting structure ({type(error).__name__})") from error
