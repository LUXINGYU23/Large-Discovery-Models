"""Expose the shared TK-ARD posterior to AntBO's native EI and restart logic."""
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
from scipy.linalg import solve_triangular
import torch

from ldm_tts.optimization.tk_ard import NOISE, gp_fit, train_tk


class AntBOGP:
    def __init__(self, train_x, train_y):
        self.x = train_x.detach().cpu().numpy()
        y = train_y.detach().cpu().numpy().astype(float)
        self.center, self.scale = y.mean(), max(y.std(), 0.01)
        standardized = (y - self.center) / self.scale
        mismatch = self.x[:, None, :] != self.x[None, :, :]
        self.position_weights, _ = train_tk(mismatch, standardized)
        gram = np.exp(-(mismatch @ self.position_weights))
        self.factor, self.alpha = gp_fit(gram, standardized)

    def __call__(self, x):
        query = x.detach().cpu().numpy()
        cross = np.exp(-((query[:, None, :] != self.x[None, :, :]) @ self.position_weights))
        mean = self.center + self.scale * (cross @ self.alpha)
        solved = solve_triangular(self.factor, cross.T, lower=True)
        variance = self.scale ** 2 * np.maximum(1 - (solved ** 2).sum(axis=0), 1e-9)
        return torch.distributions.Normal(torch.as_tensor(mean, device=x.device),
                                          torch.as_tensor(np.sqrt(variance), device=x.device))

    def likelihood(self, distribution):
        variance = distribution.variance + (self.scale ** 2 * NOISE)
        return torch.distributions.Normal(distribution.mean, variance.sqrt())

    def state_dict(self):
        # Single-point EI fits once per measured state and ignores this return.
        return {}

    def eval(self):
        return self


@contextmanager
def install_training_hook(receipts):
    """The pinned source has no GP factory. Redirect only its two fit entry points."""
    from bo import gp, localbo_cat, optimizer

    def train(train_x, train_y, **settings):
        if settings["kern"] != "transformed_overlap":
            raise RuntimeError("AntBO requested a kernel outside the paired experiment")
        if len(train_y) > 1 and not np.isfinite(train_y.detach().cpu().numpy()).all():
            raise RuntimeError("AntBO received invalid GP targets")
        model = AntBOGP(train_x, train_y)
        receipts.append({"observations": len(train_y), "position_weights": model.position_weights.tolist()})
        return model

    # AntBO will still standardize local targets. Skip the rank transform so
    # those targets preserve the same cardinal energy differences as LDM.
    with patch.object(gp, "train_gp", train), patch.object(localbo_cat, "train_gp", train), \
            patch.object(optimizer, "copula_standardize", lambda x: x):
        yield
