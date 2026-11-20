"""Fixed-amplitude transformed-overlap covariance and its position weight fit."""
import numpy as np
from scipy.linalg import cho_solve, cholesky
from scipy.optimize import minimize
from scipy.special import softmax

TK_ARD_VERSION = "normalized-transformed-overlap-ard-v1"
TK_PENALTY = 0.5
NOISE = 1e-4 + 1e-8


def gp_fit(gram, y):
    factor = cholesky(gram + NOISE * np.eye(len(y)), lower=True)
    return factor, cho_solve((factor, True), y)


def train_tk(mismatches, y):
    """Fit simplex position weights using the preregistered penalized marginal likelihood."""
    mismatches = np.asarray(mismatches, dtype=float)
    def loss(logits):
        weights = softmax(logits)
        kernel = np.exp(-(mismatches @ weights))
        factor, alpha = gp_fit(kernel, y)
        inverse = cho_solve((factor, True), np.eye(len(y)))
        derivative = -0.5 * (inverse - np.outer(alpha, alpha)) * kernel
        grad_w = np.einsum("ij,ijk->k", derivative, mismatches)
        grad_logits = weights * (grad_w - weights @ grad_w) + TK_PENALTY * logits
        loss_value = 0.5 * y @ alpha + np.log(np.diag(factor)).sum() + 0.5 * TK_PENALTY * (logits @ logits)
        return loss_value, grad_logits

    result = minimize(loss, np.zeros(mismatches.shape[2]), jac=True, method="L-BFGS-B",
                      options={"maxiter": 100, "ftol": 1e-9})
    if not result.success:
        raise RuntimeError(f"TK optimization failed: {result.message}")
    return softmax(result.x), result.nit
