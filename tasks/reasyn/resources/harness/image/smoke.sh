#!/bin/sh
set -eu
python -c 'import numpy; from rdkit import Chem; assert Chem.MolFromSmiles("CCO") is not None'
