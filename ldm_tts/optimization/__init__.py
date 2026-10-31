"""Surrogate representation and acquisition-selection interface."""

from ldm_tts.optimization.records import (
    AcquisitionSelector,
    BOObservation,
    BOPrediction,
    BOSelectionResult,
    SurrogateEncoder,
    SurrogateVector,
)
from ldm_tts.optimization.warm_start import WarmStartAcquisitionSelector

__all__ = [
    "AcquisitionSelector",
    "BOObservation",
    "BOPrediction",
    "BOSelectionResult",
    "SurrogateEncoder",
    "SurrogateVector",
    "WarmStartAcquisitionSelector",
]
