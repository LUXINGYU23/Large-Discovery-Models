#!/bin/sh
set -eu

mkdir -p /workspace/research
python - <<'PY'
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pyDOE3 import ff2n
from sklearn.linear_model import Ridge
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pypdf import PdfReader, PdfWriter

assert np.isfinite(np.asarray([1.0, 2.0])).all()
design = pd.DataFrame(ff2n(2), columns=["factor_a", "factor_b"])
design["score"] = 1 + design.factor_a + 2 * design.factor_b
fit = smf.ols("score ~ factor_a + factor_b", data=design).fit()
assert np.allclose(fit.predict(design), design.score)
assert np.isfinite(Ridge().fit(design[["factor_a", "factor_b"]], design.score).coef_).all()
sns.scatterplot(data=design, x="factor_a", y="score")
plt.savefig("/workspace/research/analysis.png")
plt.close()
path = "/workspace/research/smoke.pdf"
writer = PdfWriter()
writer.add_blank_page(width=72, height=72)
with open(path, "wb") as stream:
    writer.write(stream)
assert len(PdfReader(path).pages) == 1
PY
pdftotext /workspace/research/smoke.pdf - >/dev/null
printf 'pdf_and_cli_ok\n' > /workspace/research/proof.txt
test -s /workspace/research/proof.txt
