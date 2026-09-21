#!/bin/sh
set -eu
python -c 'import numpy; from rdkit import Chem; assert Chem.MolFromSmiles("CCO") is not None'
mkdir -p /workspace/research
printf '%s\n' 'reasyn: NumPy and RDKit imported; valid molecule parsing passed' > /workspace/research/proof.txt
