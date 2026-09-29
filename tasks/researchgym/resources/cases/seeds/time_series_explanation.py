"""Seed attribution: the released integrated_gradients_base explainer as one entry point."""
import torch as th
from captum.attr import IntegratedGradients
from captum._utils.common import _run_forward


def attribute(classifier, test_loader, timesteps, device):
    explainer = IntegratedGradients(classifier.predict)
    results = []
    for x_batch, data_mask in test_loader:
        x_batch = x_batch.to(device)
        data_mask = data_mask.to(device)
        steps = timesteps[: x_batch.shape[0], :]
        with th.autograd.set_grad_enabled(False):
            logits = _run_forward(classifier, x_batch, additional_forward_args=(data_mask, steps, False))
        target = th.argmax(logits, -1)
        batch = explainer.attribute(
            x_batch,
            baselines=x_batch * 0,
            target=target,
            additional_forward_args=(data_mask, steps, False),
        )
        results.append(batch.detach().cpu())
    return th.cat(results, dim=0)
