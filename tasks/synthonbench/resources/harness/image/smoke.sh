#!/bin/sh
set -eu

mkdir -p /workspace/research
python - <<'PY'
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pyDOE3 import ff2n
from sklearn.linear_model import Ridge
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

mol = Chem.MolFromSmiles("CCO")
assert mol is not None
assert rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol).GetNumBits() == 2048
design = pd.DataFrame(ff2n(2), columns=["slot_a", "slot_b"])
design["score"] = 1 + design.slot_a + 2 * design.slot_b
assert np.allclose(smf.ols("score ~ slot_a + slot_b", data=design).fit().predict(design), design.score)
assert np.isfinite(Ridge().fit(design[["slot_a", "slot_b"]], design.score).coef_).all()
sns.scatterplot(data=design, x="slot_a", y="score")
plt.savefig("/workspace/research/analysis.png")
plt.close()
with open("/workspace/research/proof.txt", "w", encoding="utf-8") as stream:
    stream.write(f"ethanol_mol_wt={Descriptors.MolWt(mol):.3f}\n")
PY
test -s /workspace/research/proof.txt
