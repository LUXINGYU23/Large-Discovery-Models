---
name: biopython
description: Reconstruct and compare DNA candidates, manipulate FASTA, scan both strands with motif matrices, and verify sequence changes before interpreting an experiment. Use when writing or checking sequence-analysis code for this campaign.
license: MIT
metadata:
  runtime: Python 3.11 with Biopython 1.88 and NumPy is preinstalled in the research guest.
  source: K-Dense-AI/scientific-agent-skills
  source-commit: 1e5eeffbdad3749125afe7ab48a39694e27f181c
---

# Sequence Analysis with Biopython

Use the installed library for biological operations. Start with the supplied
sequence context and measured history; save reusable analysis in the workspace.
The task's mutation and submission contracts remain authoritative.

## Reconstruct Before Comparing

Patches use zero-based coordinates relative to the original paired start.
Fetch a measured parent with `get_sequence_window(candidate_id=..., start=...,
end_exclusive=...)`; omit `candidate_id` for the original start. The returned
`bases` and `bases_sha256` identify that exact window. Save the bases and verify
their ASCII SHA-256 before computing changes, rather than manually transcribing
a long mutation list. For a 200-base Malinois sequence, retrieve the complete
window. For longer cases, retain each window's absolute coordinate offset.

Derive every submitted patch against the original start, not the measured parent.
For complete-sequence windows saved as `reference_window` and `parent_window`:

```python
from Bio.Seq import Seq
from hashlib import sha256

reference = reference_window["bases"]
parent = parent_window["bases"]
for window in (reference_window, parent_window):
    assert sha256(window["bases"].encode("ascii")).hexdigest() == window["bases_sha256"]
# Apply the intended substitutions to a copy of parent before deriving the patch.
sequence = parent
assert len(sequence) == len(reference)
assert set(sequence) <= set("ACGT")
patch = [
    {"position": i, "base": new}
    for i, (old, new) in enumerate(zip(reference, sequence))
    if old != new
]
reverse = str(Seq(sequence).reverse_complement())
```

For equal-length substitution-only candidates, compare original coordinates
directly. A gapped alignment is not the mutation identity. Check editable
positions against the task context, and reject accidental insertions/deletions
from string replacement or overlapping module writes.

## Motifs and Composition

- Parse a sourced motif matrix with `Bio.motifs.read` or `Bio.motifs.parse`.
  Retain its accession, version, source URL, species, and experimental context.
- Use `counts.normalize(pseudocounts=...)` and `pwm.log_odds(background)` to
  construct a PSSM. Record the pseudocount, background, and score threshold.
- Scan both strands. A consensus-string match is a proxy, not measured binding
  or expression; a toy matrix is not a verified biological motif.
- Compare the complete reconstructed candidate and its named control, including
  native sites, reverse-strand sites, overlaps, spacing, and composition. Do not
  infer that a motif was removed just because the inserted module was changed.

For forward-coordinate hits on both strands:

```python
import numpy as np

# pssm is built from a verified matrix; sequence is the complete DNA sequence.
hits = []
if len(sequence) >= pssm.length:
    for strand, matrix in ((1, pssm), (-1, pssm.reverse_complement())):
        scores = np.atleast_1d(matrix.calculate(Seq(sequence)))
        hits.extend(
            (start, start + pssm.length, strand, float(score))
            for start, score in enumerate(scores) if score >= threshold
        )
```

Use `Bio.SeqUtils.gc_fraction` for GC fraction and NumPy/SciPy for summaries.
Composition differences or large Hamming distance do not by themselves establish
different regulatory mechanisms. Local motif or composition scores remain
research features, not oracle measurements.

## Files and Tools

Use `Bio.SeqIO.parse/read/write` with `SeqRecord` for FASTA. For gzip files,
explicitly open a text stream with `gzip.open(path, "rt")` and pass that stream
to `SeqIO.parse`; the filename suffix does not decompress the file.

SeqKit, pyfaidx, SAMtools, BEDTools, and MAFFT are installed for sequence files,
indexing, intervals, and alignments. Use the registered web tools for public
literature and Context7 for library documentation. No NCBI credentials, BLAST
databases, or external prediction service are assumed to exist.

Write Python source with the native `write` tool, then execute it with `bash`.
Do not include shell heredoc delimiters or shell commands inside a `.py` file.
Check a small example before generating the full panel. The submission file
must contain exactly the fields and candidate count specified by the turn.
