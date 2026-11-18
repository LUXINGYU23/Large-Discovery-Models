"""Native AntBO suggestion/observation adapter for the shared CDRH3 case."""
import os
from pathlib import Path
import random
import sys

import numpy as np

from tasks.antibody.core.engine_adapters import _sequence_violations
from tasks.antibody.core.ldm_light.ldm_acq import indices_to_seqs, seqs_to_indices


class NativeAntBO:
    def __init__(self, seed):
        import torch
        source = Path(os.environ.get("ANTBO_HOME", "")).resolve()
        if not (source / "bo/optimizer.py").is_file():
            raise ValueError("Set ANTBO_HOME to the upstream HEBO/AntBO directory")
        sys.path.insert(0, str(source))
        from bo.optimizer import Optimizer

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        self.optimizer = Optimizer(np.full(11, 20), min_cuda=10**9, n_init=1,
                                   cdr_constraints=True, kernel_type="transformed_overlap",
                                   search_strategy="local", alphabet_size=20, acq="ei", seed=seed)
        # Initialize upstream trust-region state. Replace its first random
        # point with the common measured initialization used by all methods.
        self.optimizer.suggest(1)
        # Upstream emits a (0, 1, 11) empty queue here. Its observe()
        # routine indexes a (remaining_points, 11) table.
        self.optimizer.x_init = self.optimizer.x_init.reshape(-1, 11)
        self.seen = 0

    def propose(self, history):
        import torch
        for row in history[self.seen:]:
            self.optimizer.observe(seqs_to_indices([row["sequence"]]), torch.tensor([row["energy"]]))
        self.seen = len(history)
        evaluated = {row["sequence"] for row in history}
        for _ in range(32):
            seq, = indices_to_seqs(self.optimizer.suggest(1))
            if seq not in evaluated and not _sequence_violations(seq, 11):
                return {"sequence": seq}
        raise RuntimeError("AntBO could not suggest an admissible new CDRH3 in 32 attempts")
