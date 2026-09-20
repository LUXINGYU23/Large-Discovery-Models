#!/bin/sh
set -eu
python - <<'PYTHON'
from atomworld_tools import execute_operations, OPERATIONS_SCHEMA
import ase
import numpy
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter, CifParser
source = str(CifWriter(Structure(Lattice.cubic(5), ['Na','Cl'], [[0,0,0],[0.5,0.5,0.5]])))
result = execute_operations(source, [{'op':'move','index':0,'d_pos':[1,0,0]}])
assert len(CifParser.from_str(result).parse_structures(primitive=False)[0]) == 2
assert len(OPERATIONS_SCHEMA['items']['oneOf']) == 10
PYTHON
