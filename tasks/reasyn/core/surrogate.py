"""Shared GP adapter over molecular fingerprints; no optional import for mocks."""

from ldm_tts.contracts import SurrogateSpaceSpec
from ldm_tts.optimization.records import SurrogateVector
import hashlib


class MoleculeEncoder:
    def __init__(self, *, mock=False):
        self.mock = mock
        self.version = "fixture_hash32_v1" if mock else "morgan_r2_2048_v1"

    def describe(self):
        return SurrogateSpaceSpec(
            kind="vector",
            representation="Synthetic hash features"
            if self.mock
            else "Morgan radius 2 molecular fingerprint",
            dimension_policy="fixed",
            dimension=32 if self.mock else 2048,
            encoder="tasks.reasyn.core.surrogate:MoleculeEncoder",
            version=self.version,
        )

    def encode(self, candidate):
        smi = candidate.payload.get("smiles", candidate.payload.get("target_smiles"))
        if self.mock:
            values = tuple(v / 255 for v in hashlib.sha256(smi.encode()).digest())
        else:
            from rdkit import Chem
            from rdkit.Chem import rdFingerprintGenerator

            values = tuple(
                float(x)
                for x in rdFingerprintGenerator.GetMorganGenerator(
                    radius=2, fpSize=2048
                ).GetFingerprint(Chem.MolFromSmiles(smi))
            )
        return SurrogateVector(values, self.version, candidate.candidate_id)
