from __future__ import annotations

from typing import Optional, Any, Literal

import numpy as np
import torch
from botorch.fit import fit_gpytorch_model
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel, CosineKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP

from tasks.antibody.core import CategoricalOverlap, TransformedCategorical, OrdinalKernel, FastStringKernel
from tasks.antibody.core.kernels import BERTWarpRBF, BERTWarpCosine
SEARCH_STRATS = Literal['glocal', 'local', 'local-no-hamming', 'batch_local', 'global']


def identity(x: Any) -> Any:
    return x


class GP(ExactGP):
    def __init__(self, train_x: torch.tensor, train_y: torch.tensor, likelihood: Likelihood,
                 outputscale_constraint=None, ard_dims=None, kern=None, mean_mode=ConstantMean, cat_dims=None,
                 batch_shape=torch.Size(), transform_inputs=None):
        if transform_inputs is None:
            transform_inputs = identity
        self.transform_inputs = transform_inputs
        if transform_inputs:
            train_x = transform_inputs(train_x)
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.dim = train_x.shape[1]
        self.ard_dims = ard_dims
        self.cat_dims = cat_dims
        self.mean_module = mean_mode(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(kern, outputscale_constraint=outputscale_constraint, batch_shape=batch_shape)

    def forward(self, x: torch.tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, *args, **kwargs) -> MultivariateNormal:
        return super().__call__(*[self.transform_inputs(input_point) for input_point in args], **kwargs)

def train_gp(train_x: torch.tensor, train_y: torch.tensor, use_ard: bool, num_steps: int,
             kern: str = 'transformed_overlap', hypers: Optional[dict] = None, noise_variance: float = None,
             cat_configs=None, antigen: str = None, search_strategy: SEARCH_STRATS = 'local', **params):
    """
    Fit a GP model where train_x is in [0, 1]^d and train_y is standardized.
    （train_x, train_y）: pairs of x and y (trained)

    noise_variance: if provided, this value will be used as the noise variance for the GP model. Otherwise, the noise
        variance will be inferred from the model.
    """
    assert train_x.ndim == 2
    assert train_y.ndim == 1
    assert train_x.shape[0] == train_y.shape[0]

    if hypers is None:
        hypers = {}

    device = train_x.device
    # Create hyper parameter bounds
    if noise_variance is None:
        noise_variance = 0.005
        noise_constraint = Interval(1e-6, 0.1)
    else:
        if np.abs(noise_variance) < 1e-6:
            noise_variance = 0.05
            noise_constraint = Interval(1e-6, 0.1)
        else:
            noise_constraint = Interval(0.99 * noise_variance, 1.01 * noise_variance)
    if use_ard:
        lengthscale_constraint = Interval(0.01, 0.5)
    else:
        lengthscale_constraint = Interval(0.01, 2.5)  # [0.005, sqrt(dim)]

    outputscale_constraint = Interval(0.5, 5.)

    likelihood = GaussianLikelihood(noise_constraint=noise_constraint).to(device=train_x.device, dtype=train_y.dtype)

    ard_dims = train_x.shape[1] if use_ard else None
    transform_inputs = None
    if kern == 'overlap':
        kernel = CategoricalOverlap(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims, )
    elif kern == 'transformed_overlap':
        kernel = TransformedCategorical(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims)
    elif kern == 'ordinal':
        kernel = OrdinalKernel(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims, config=cat_configs)
    elif kern == 'ssk':
        kernel = FastStringKernel(seq_length=train_x.shape[1], alphabet_size=params['alphabet_size'],
                                  device=train_x.device)
    elif kern in ['rbfBERT', 'rbf-pca-BERT', 'cosine-BERT', 'cosine-pca-BERT']:
        from tasks.antibody.core.utils import BERTFeatures, batch_iterator
        bert = BERTFeatures(params['BERT_model'], params['BERT_tokeniser'])
        nm_samples = train_x.shape[0]
        if nm_samples > params['BERT_batchsize']:
            reprsn1 = []
            for x in batch_iterator(train_x, params['BERT_batchsize']):
                features1 = bert.compute_features(x)
                reprsn1.append(features1)
            reprsn1 = torch.cat(reprsn1, 0)
        else:
            reprsn1 = bert.compute_features(train_x)
        if kern in ['rbf-pca-BERT', 'cosine-pca-BERT']:
            from joblib import load
            pca = load(f"./{antigen}_pca.joblib")  # /nfs/aiml/asif/CDRdata/pca
            scaler = load(f"./{antigen}_scaler.joblib")  # /nfs/aiml/asif/CDRdata/pca
            reprsn1 = torch.from_numpy(pca.transform(scaler.transform(reprsn1.cpu().numpy())))
        train_x = reprsn1.clone().to(device=device)
        del reprsn1, bert
        ard_dims = train_x.shape[1] if use_ard else None
        if kern in ['rbfBERT', 'rbf-pca-BERT']:
            kernel = BERTWarpRBF(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims)
        else:
            kernel = BERTWarpCosine(lengthscale_constraint=lengthscale_constraint, ard_num_dims=None)
        if kern in ['rbfBERT', "cosine-BERT"]:
            min_x = train_x.min(0)[0]
            max_x = train_x.max(0)[0]

            def transform_inputs(input_x: torch.tensor) -> torch.tensor:
                return (input_x - min_x.to(input_x)) / (max_x.to(input_x) - min_x.to(input_x) + 1e-8)
    elif kern == 'rbf':
        kernel = RBFKernel(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims)
    elif kern == "mat52":
        kernel = MaternKernel(nu=2.5, lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims)
    else:
        raise ValueError('Unknown kernel choice %s' % kern)

    model = GP(
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        kern=kernel,
        outputscale_constraint=outputscale_constraint,
        ard_dims=ard_dims,
        transform_inputs=transform_inputs,
    ).to(device=train_x.device, dtype=train_x.dtype)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    if search_strategy in ['glocal', 'global', 'batch_local']:
        if hypers:
            model.load_state_dict(hypers)
        fit_gpytorch_model(mll)
    else:
        # Initialize model hypers
        if hypers:
            model.load_state_dict(hypers)
        else:
            hypers = {}
            hypers["covar_module.outputscale"] = 1.0
            if not isinstance(kernel, (FastStringKernel, CosineKernel)):
                hypers["covar_module.base_kernel.lengthscale"] = np.sqrt(0.01 * 0.5)
            hypers["likelihood.noise"] = noise_variance if noise_variance is not None else 0.005
            model.initialize(**hypers)

        # Use the adam optimizer
        optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.03)

        for i in range(num_steps):
            optimizer.zero_grad()
            output = model(train_x, )
            loss = -mll(output, train_y).float()
            loss.backward()
            optimizer.step()

    # Switch to eval mode
    model.eval()
    return model
