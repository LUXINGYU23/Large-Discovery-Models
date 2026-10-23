"""Code-only representation and minimization adapter for the shared GP/acquisition."""

from dataclasses import replace

import numpy as np

from ldm_tts.contracts import AcquisitionSpec
from ldm_tts.optimization.acquisition import make_acquisition
from ldm_tts.optimization.gp import RBFGPSurrogate
from ldm_tts.optimization.records import BOSelectionResult


from .encoding import CodeEncoder, FEATURE_VERSION, DIMENSION, encoding_spec

GP_PARAMETERS = {
    "lengthscale": 1.5, "noise": 1e-4, "min_training_observations": 1,
    "feature_scale_floor": 1.0, "target_scale_floor": 1.0,
}


class LossSelector:
    """Fit positive loss directly; shared LCB scores are larger-is-better -LCB.

    The measured weak baseline is mandatory. The shared GP's sparse-history or
    numerical fallback is not accepted as a fitted model for this task.
    """

    def __init__(self, mode="ast"):
        spec = encoding_spec(mode)
        self.version, self.dimension = spec.version, spec.dimension
        self.surrogate = None
        self.acquisition = make_acquisition("lcb", minimize=(True,), beta=1.0)

    def describe(self):
        return AcquisitionSpec(
            name="lcb", objective_names=("loss_total",), score_direction="maximize",
            selection_rule="Largest shared negative lower confidence bound; SHA256 tie-break",
            parameters={"beta": 1.0, "feature_version": self.version,
                        "dimension": self.dimension, **GP_PARAMETERS},
        )

    def _validate_feature(self, feature, identity):
        if (feature is None or feature.version != self.version
                or feature.source_id != identity
                or len(feature.values) != self.dimension
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
        model = RBFGPSurrogate(history, feature_version=self.version, **GP_PARAMETERS)
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
