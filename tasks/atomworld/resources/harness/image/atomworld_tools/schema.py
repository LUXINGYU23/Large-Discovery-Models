"""JSON contract consumed by the LDM proposal adapter."""

from ase.data import chemical_symbols

MAX_OPERATIONS = 64
MAX_ATOMS = 10000
MAX_CIF_BYTES = 2000000
MAX_MAGNITUDE = 1000000

NUMBER = {"type": "number", "minimum": -MAX_MAGNITUDE, "maximum": MAX_MAGNITUDE}
VECTOR = {"type": "array", "items": NUMBER, "minItems": 3, "maxItems": 3}
INDEX = {"type": "integer", "minimum": 0, "maximum": MAX_ATOMS - 1}
SYMBOL = {"type": "string", "enum": chemical_symbols[1:]}
DISTANCE = {"type": "number", "minimum": 0, "maximum": MAX_MAGNITUDE}

OPERATION_FIELDS = {
    "add": {"symbol": SYMBOL, "position": VECTOR},
    "change": {"index": INDEX, "symbol": SYMBOL},
    "remove": {"index": INDEX},
    "swap": {"index1": INDEX, "index2": INDEX},
    "move": {"index": INDEX, "d_pos": VECTOR},
    "move_towards": {"index1": INDEX, "index2": INDEX, "distance": DISTANCE},
    "insert_between": {"index1": INDEX, "index2": INDEX, "symbol": SYMBOL, "distance": DISTANCE},
    "delete_below": {"index": INDEX},
    "rotate_around": {
        "index": INDEX,
        "radius": {"type": "number", "exclusiveMinimum": 0, "maximum": MAX_MAGNITUDE},
        "angle": {"type": "number", "minimum": 0, "exclusiveMaximum": 360},
        "axis": VECTOR,
    },
    "super_cell": {"size": {"type": "array", "minItems": 3, "maxItems": 3,
                            "items": {"type": "integer", "minimum": 1, "maximum": 16}}},
}

OPERATIONS_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "AtomWorld target-blind operations",
    "type": "array",
    "minItems": 1,
    "maxItems": MAX_OPERATIONS,
    "items": {"oneOf": [
        {"type": "object", "additionalProperties": False,
         "required": ["op", *fields],
         "properties": {"op": {"const": name}, **fields}}
        for name, fields in OPERATION_FIELDS.items()
    ]},
}

OPERATION_INSTRUCTIONS = """Return a JSON array of operations implementing the supplied action.
Use only the following objects (all listed fields are required; no extra fields):
{"op":"add","symbol":"Fe","position":[1,2,3]}
{"op":"change","index":0,"symbol":"Fe"}
{"op":"remove","index":0}
{"op":"swap","index1":0,"index2":1}
{"op":"move","index":0,"d_pos":[1,0,0]}
{"op":"move_towards","index1":0,"index2":1,"distance":0.5}
{"op":"insert_between","index1":0,"index2":1,"symbol":"Fe","distance":0.5}
{"op":"delete_below","index":0}
{"op":"rotate_around","index":0,"radius":2,"angle":90,"axis":[0,0,1]}
{"op":"super_cell","size":[2,1,1]}
These examples describe separate alternatives, not a plan to execute together.
Indices are zero based in the ASE-parsed input atom order; subsequent operations use
the current order. Coordinates/displacements/distances/radii are Cartesian angstroms,
angles are degrees with the right-hand rule. move_towards and insert_between use the
direct Cartesian vector, not the nearest periodic image. insert_between distance is
measured from index1 and cannot exceed the direct separation. rotate_around selects
neighbors using periodic minimum-image distance strictly less than radius, excludes
the center, then rotates their original Cartesian positions about that center.
swap exchanges species at fixed sites, matching the upstream generator and yielding
the same one-step physical structure as swapping the two positions. delete_below uses
strictly lower Cartesian z, preserving the reference and atoms at equal z. super_cell
repeats the cell along a,b,c. Numbers must be finite; axis must be nonzero. At most 64
operations, 10000 resulting atoms, and supercell factors 1..16 with product <=512.
An execution error may be corrected using the public input and this contract.
"""
