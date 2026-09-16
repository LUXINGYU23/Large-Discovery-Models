"""Compiled priors change only the shared residual GP mean, not its covariance."""

import numpy as np
import pytest

from ldm_tts.optimization.gp import RBFGPSurrogate
from ldm_tts.optimization.records import BOObservation


def test_residual_gp_matches_frozen_linear_operator_and_single_measurement():
    history = [BOObservation.scalar(str(i), float(i % 2), [i, i / 2]) for i in range(3)]
    query = [[0.5, 0.0], [3, 1]]
    for n in (1, 3):
        observed = history[:n]
        baseline = RBFGPSurrogate(observed, residual_prior_mean=np.zeros(n))
        hp = np.arange(n) / 2 + 0.5
        qp = np.asarray([0.2, 1.0])
        compiled = RBFGPSurrogate(observed, residual_prior_mean=hp)
        assert baseline.fit_status == compiled.fit_status == "fitted"
        projection = baseline.posterior_projection(query)
        expected_delta = projection["location_scale"][1] * (qp - projection["weights"] @ hp)
        for i, vector in enumerate(query):
            before = baseline.predict_record(str(i), vector)
            after = compiled.predict_record(str(i), vector, query_prior_mean=qp[i])
            assert after.scalar_mean - before.scalar_mean == pytest.approx(expected_delta[i])
            assert after.scalar_std == pytest.approx(before.scalar_std)
            assert projection["std"][i] == pytest.approx(before.scalar_std)


def test_legacy_sparse_gp_and_invalid_residual_priors():
    history = [BOObservation.scalar("a", 1, [0, 1])]
    assert RBFGPSurrogate(history).fit_status == "fallback"
    for prior in ([], [np.nan], [0, 1]):
        with pytest.raises(ValueError, match="residual prior"):
            RBFGPSurrogate(history, residual_prior_mean=prior)
    with pytest.raises(ValueError, match="fitted residual GP"):
        RBFGPSurrogate(history).predict_record("q", [0, 2], query_prior_mean=1)
