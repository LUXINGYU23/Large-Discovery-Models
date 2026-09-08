#!/bin/sh
set -eu

mkdir -p /workspace/research
python - <<'PY'
from pathlib import Path

import RNA
import logomaker
import matplotlib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from Bio import motifs
from Bio.Seq import Seq
from pyfaidx import Fasta
from pyDOE3 import ff2n
from pypdf import PdfReader, PdfWriter
from scipy.spatial.distance import hamming
from sklearn.linear_model import Ridge

matplotlib.use("Agg")
from matplotlib import pyplot as plt

root = Path("/workspace/research")
fasta_path = root / "smoke.fa"
fasta_path.write_text(">reference\nACGTACGT\n", encoding="utf-8")
fasta = Fasta(fasta_path.as_posix(), as_raw=True)
assert fasta["reference"][:] == "ACGTACGT"

sequence = Seq("ACGTACGT")
encoded = np.asarray(["ACGT".index(base) for base in sequence], dtype=float)
assert sequence.reverse_complement() == sequence
assert hamming(encoded, encoded.copy()) == 0.0
structure, energy = RNA.fold("GCGCUUCGCC")
assert len(structure) == 10 and np.isfinite(energy)

frame = pd.DataFrame({"mutation_count": [1, 2, 3], "utility": [0.1, 0.4, 0.8]})
model = Ridge(alpha=1.0).fit(frame[["mutation_count"]], frame["utility"])
assert np.isfinite(model.predict(pd.DataFrame({"mutation_count": [4]}))).all()

motif = motifs.create([Seq("ACGT"), Seq("ACGT"), Seq("AGGT")])
assert motif.counts["A"][0] == 3
pssm = motif.counts.normalize(pseudocounts=0.5).log_odds()
for probe in ("ACGT", "ACGTACGT"):
    for matrix in (pssm, pssm.reverse_complement()):
        scores = np.atleast_1d(matrix.calculate(Seq(probe)))
        assert scores.shape == (len(probe) - pssm.length + 1,)
        assert np.isfinite(scores).all() and scores[0] > 0.0

design = pd.DataFrame(ff2n(3), columns=["module_a", "module_b", "background"])
design["utility"] = (
    1 + 2 * design["module_a"] - design["module_b"]
    + 0.5 * design["module_a"] * design["module_b"]
    + 0.1 * design["background"]
)
fit = smf.ols("utility ~ module_a * module_b", data=design).fit()
assert np.isclose(fit.params["module_a:module_b"], 0.5)
assert np.isfinite(fit.get_influence().cooks_distance[0]).all()
assert np.isfinite(fit.get_robustcov_results(cov_type="HC3").bse).all()
matrix = logomaker.alignment_to_matrix(["ACGT", "ACGT", "AGGT"])
figure, axis = plt.subplots(figsize=(4, 2))
logomaker.Logo(matrix, ax=axis)
figure.savefig(root / "sequence_logo.png")
plt.close(figure)

pdf_path = root / "smoke.pdf"
writer = PdfWriter()
writer.add_blank_page(width=72, height=72)
with pdf_path.open("wb") as stream:
    writer.write(stream)
assert len(PdfReader(pdf_path).pages) == 1
PY

seqkit seq -n /workspace/research/smoke.fa | grep -qx reference
samtools faidx /workspace/research/smoke.fa reference:1-4 >/workspace/research/window.fa
printf 'chr1\t4\t8\nchr1\t0\t4\n' | bedtools sort >/workspace/research/windows.bed
printf '>a\nACGT\n>b\nAGGT\n' >/workspace/research/alignment.fa
mafft --quiet /workspace/research/alignment.fa >/workspace/research/alignment.out
jq -n -e '{status: "ok"} | .status == "ok"' >/dev/null
pdftotext /workspace/research/smoke.pdf - >/dev/null

printf 'sequence_research_stack_ok\n' > /workspace/research/proof.txt
test -s /workspace/research/proof.txt
