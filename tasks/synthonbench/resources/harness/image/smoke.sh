#!/bin/sh
set -eu

mkdir -p /workspace/research
python - <<'PY'
from rdkit import Chem
from rdkit.Chem import Descriptors

mol = Chem.MolFromSmiles("CCO")
assert mol is not None
with open("/workspace/research/proof.txt", "w", encoding="utf-8") as stream:
    stream.write(f"ethanol_mol_wt={Descriptors.MolWt(mol):.3f}\n")
PY
test -s /workspace/research/proof.txt
