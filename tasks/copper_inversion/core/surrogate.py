"""Code-only representation and minimization adapter for the shared GP/acquisition."""

import ast
from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np

from ldm_tts.contracts import AcquisitionSpec, SurrogateSpaceSpec
from ldm_tts.optimization.acquisition import make_acquisition
from ldm_tts.optimization.gp import RBFGPSurrogate
from ldm_tts.optimization.records import BOSelectionResult, SurrogateVector
from .candidate import validate_source


FEATURE_VERSION = "copper-ast-sha256-256-v1"
DIMENSION = 256
GP_PARAMETERS = {
    "lengthscale": 1.5,
    "noise": 1e-4,
    "min_training_observations": 1,
    "feature_scale_floor": 1.0,
    "target_scale_floor": 1.0,
}


class CodeEncoder:
    """Signed feature hashing of AST nodes and literal/name fields, L2 normalized.

    No execution, result, parent, rationale or research lineage is inspected.
    SHA-256 fixes bucket/sign assignment across processes (unlike Python hash()).
    Comments and formatting do not change the vector; distinct programs may collide.
    """

    def __init__(self, root):
        self.root = Path(root).resolve()

    def describe(self):
        return SurrogateSpaceSpec(
            kind="vector", representation="Signed SHA256 AST feature counts, L2 normalized",
            dimension_policy="fixed", dimension=DIMENSION, version=FEATURE_VERSION,
        )

    def encode(self, candidate):
        path = self.root / candidate.payload["source_path"]
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError("source escaped sealed candidate root")
        source = path.read_bytes()
        if validate_source(source) != candidate.canonical_key:
            raise ValueError("source digest changed before encoding")
        vector = np.zeros(DIMENSION, dtype=np.float64)

        def add(feature):
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "big") % DIMENSION
            vector[bucket] += 1 if digest[8] & 1 else -1

        for node in ast.walk(ast.parse(source.decode("utf-8"))):
            kind = type(node).__name__
            add(kind)
            for name, value in ast.iter_fields(node):
                if not isinstance(value, (ast.AST, list)):
                    add(f"{kind}.{name}={value!r}")
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("invalid code representation")
        return SurrogateVector(
            tuple(float(v) for v in vector / norm), FEATURE_VERSION,
            source_id=candidate.candidate_id,
        )


class LossSelector:
    """Fit positive loss directly; shared LCB scores are larger-is-better -LCB.

    The measured weak baseline is mandatory. The shared GP's sparse-history or
    numerical fallback is not accepted as a fitted model for this task.
    """

    def __init__(self):
        self.surrogate = None
        self.acquisition = make_acquisition("lcb", minimize=(True,), beta=1.0)

    def describe(self):
        return AcquisitionSpec(
            name="lcb", objective_names=("loss_total",), score_direction="maximize",
            selection_rule="Largest shared negative lower confidence bound; SHA256 tie-break",
            parameters={"beta": 1.0, "feature_version": FEATURE_VERSION,
                        "dimension": DIMENSION, **GP_PARAMETERS},
        )

    @staticmethod
    def _validate_feature(feature, identity):
        if (feature is None or feature.version != FEATURE_VERSION
                or feature.source_id != identity
                or len(feature.values) != DIMENSION
                or not np.isfinite(feature.values).all()):
            raise ValueError("invalid or mismatched code feature")

    def fit(self, history):
        self.surrogate = None  # Never reuse an earlier model after a failed fit.
        if not history:
            raise ValueError("loss GP requires a successfully measured baseline")
        for observation in history:
            self._validate_feature(observation.feature, observation.candidate_id)
            if len(observation.objectives) != 1 or not np.isfinite(observation.scalar_score):
                raise ValueError("loss GP requires finite scalar observations")
        model = RBFGPSurrogate(history, feature_version=FEATURE_VERSION, **GP_PARAMETERS)
        if not model.ready or not np.isfinite(model.alpha).all():
            raise ValueError("loss GP fit failed; no random fallback")
        self.surrogate = model

    def select(self, candidates, representations, *, count=1, round_idx=0):
        if self.surrogate is None:
            raise ValueError("loss GP must be fitted before selection")
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= len(candidates):
            raise ValueError("invalid selection count")
        predictions = []
        for candidate in candidates:
            feature = representations.get(candidate.candidate_id)
            self._validate_feature(feature, candidate.candidate_id)
            prediction = self.surrogate.predict_record(candidate.candidate_id, feature.values)
            score = float(self.acquisition.score(prediction.scalar_mean, prediction.scalar_std))
            if not np.isfinite([prediction.scalar_mean, prediction.scalar_std, score]).all():
                raise ValueError("nonfinite GP prediction")
            predictions.append(replace(prediction, acquisition_score=score,
                                       metadata={**prediction.metadata, "acquisition": "lcb"}))
        ranked = sorted(predictions, key=lambda p: (-p.acquisition_score, p.candidate_id))
        return BOSelectionResult(
            tuple(p.candidate_id for p in ranked[:count]), tuple(predictions),
            metadata={"fit_status": self.surrogate.fit_status,
                      "history_size": len(self.surrogate.observations),
                      "best_observed_loss": min(o.scalar_score for o in self.surrogate.observations),
                      "round_index": round_idx, "parameters": self.describe().parameters},
        )
