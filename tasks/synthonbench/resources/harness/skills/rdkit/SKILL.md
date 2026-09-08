---
name: rdkit
description: Compute properties, fingerprints, and structural comparisons of public SynthonSpace structures. Use when a chemical hypothesis or property-risk claim needs a reproducible calculation.
license: MIT
metadata:
  source: K-Dense-AI/scientific-agent-skills
  source-commit: 1e5eeffbdad3749125afe7ab48a39694e27f181c
---

# RDKit for Synthon Research

Use public SMILES returned by the task tools. Parse with `Chem.MolFromSmiles`
and check for `None` before calculation; do not silently drop an invalid
structure or repair it into an unrelated molecule. Record any transformation,
including treatment of dummy atoms, attachment labels, salt fragments, charge,
or stereochemistry, and distinguish the transformed proxy from its source.

The guest provides RDKit. For example:

```python
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

mol = Chem.MolFromSmiles("CCO")
assert mol is not None
properties = {
    "molecular_weight": Descriptors.MolWt(mol),
    "logp": Descriptors.MolLogP(mol),
    "tpsa": Descriptors.TPSA(mol),
}
generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fingerprint = generator.GetFingerprint(mol)
```

For a matched change, calculate properties on both exact structures using the
same preprocessing. Use `DataStructs.TanimotoSimilarity` on consistently generated
fingerprints for a structural comparison; similarity does not imply equivalent
binding or selectivity. Separate scaffold alternatives from superficial
substitutions of one dominant family.

Source-synthon descriptors are not assembled-product descriptors or activity
measurements. Do not add fragment logP or TPSA and report the sum as an official
product property. Any hypothetical assembly is a labeled research proxy,
not an oracle-evaluated product. Submission remains an exact official reaction
ID and ordered synthon IDs, never a canonical SMILES string.

Use calculated properties to test a stated risk or contrast, not as undeclared
hard filters or additional objectives. Join measurements by exact candidate ID
and preserve scripts and feature definitions in the workspace. Never retrieve
the hidden evaluator or call it from research code.

API reference: [RDKit Getting Started](https://www.rdkit.org/docs/GettingStartedInPython.html).
Use Context7 when another API is needed rather than guessing signatures.
