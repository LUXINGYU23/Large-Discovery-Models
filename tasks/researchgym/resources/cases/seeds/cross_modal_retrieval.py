"""Seed adaptation: released Tent entropy minimization restated without importing baselines."""
import torch
import torch.nn as nn


class LDMTTA(nn.Module):
    def __init__(self, model, optimizer, steps=1):
        super().__init__()
        self.model, self.optimizer, self.steps = model, optimizer, steps

    def forward(self, x, device, args, metric_logger, if_adapt=True, counter=None, if_vis=False):
        if not if_adapt:
            with torch.no_grad():
                return self.model.module.forward_output(x, device, args)
        for _ in range(self.steps):
            outputs = self._adapt(x, device, args, metric_logger)
        return outputs

    @torch.enable_grad()
    def _adapt(self, x, device, args, metric_logger):
        outputs = self.model.module.forward_output(x, device, args)
        loss = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1).mean(0)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        metric_logger.update(loss_total=loss.item(), lr=self.optimizer.param_groups[0]["lr"])
        return outputs
