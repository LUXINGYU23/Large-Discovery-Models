#!/bin/sh
set -eu

mkdir -p /workspace/research
cd /workspace/research
# The capabilities promised by program_research/AGENTS.md: CPU PyTorch autograd,
# NumPy/SciPy/pandas/scikit-learn analysis, and compiling a candidate module.
cat > candidate.py <<'PY'
import torch
import torch.nn as nn


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity, device):
        self.device, self.obs = device, torch.zeros(capacity, obs_dim)

    def add(self, obs, act, rew, next_obs, done):
        return None

    def sample(self, batch_size):
        return self.obs[:batch_size].to(self.device)

    def __len__(self):
        return 0
PY
python -m py_compile candidate.py
python - <<'PY'
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import torch
from sklearn.linear_model import Ridge

model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
x = torch.randn(16, 4)
loss = -(model(x).softmax(1) * model(x).log_softmax(1)).sum(1).mean()
loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
frame = pd.DataFrame({"feature": [0.1, 0.4, 0.8], "score": [0.2, 0.5, 0.9]})
fit = Ridge(alpha=1.0).fit(frame[["feature"]], frame["score"])
assert np.isfinite(fit.predict(frame[["feature"]])).all()
assert np.isfinite(scipy.stats.ttest_ind([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]).statistic)
Path("candidates.json").write_text(json.dumps({"candidates": [{
    "program": Path("candidate.py").read_text(), "change_summary": "smoke", "rationale": "smoke"}]}))
PY
printf 'researchgym_research_stack_ok\n' > /workspace/research/proof.txt
test -s /workspace/research/proof.txt
