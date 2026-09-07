"""Lazy chemistry boundaries; mock canonicalization is deliberately a fixed fixture."""

from __future__ import annotations
import hashlib

MOCK_SMILES = (
    "CCO",
    "CCN",
    "CCC",
    "CCCO",
    "CCCN",
    "CCCC",
    "CC(=O)O",
    "c1ccccc1",
    "CCOC",
    "CCNC",
    "COC",
    "CC(C)O",
)


def canonicalize(smiles, *, mock=False, stereo=True):
    if not isinstance(smiles, str) or not smiles.strip() or len(smiles) > 2000:
        raise ValueError(
            "SMILES must be a nonempty string with at most 2000 characters"
        )
    if mock:
        if smiles not in MOCK_SMILES:
            raise ValueError(
                "mock SMILES must belong to the fixed qualification fixture"
            )
        return smiles
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("invalid SMILES")
    return Chem.MolToSmiles(mol, isomericSmiles=stereo)


def similarity(left, right, *, mock=False):
    if mock:
        return (
            1.0
            if left == right
            else 0.25
            + int(hashlib.sha256((left + "|" + right).encode()).hexdigest()[:4], 16)
            / 131070
        )
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    # Exact upstream settings: Morgan radius 2, 4096 bits, no chirality flag.
    a = AllChem.GetMorganGenerator(radius=2, fpSize=4096).GetFingerprint(
        Chem.MolFromSmiles(left)
    )
    b = AllChem.GetMorganGenerator(radius=2, fpSize=4096).GetFingerprint(
        Chem.MolFromSmiles(right)
    )
    return float(DataStructs.TanimotoSimilarity(a, b))
