"""Subprocess bridge to released frozen ReaSyn models; never edits upstream.

This file intentionally has no LDM imports: it runs with the upstream Python
and working directory, loading a checkpoint pair once for its target batch.
"""

from __future__ import annotations
import argparse
import json
import pathlib
import pickle
import random
import re
import sys


def _install_sklearn_pickle_compat():
    """Restore the pre-1.4 name for float64 distance metrics.

    The released fingerprint index was serialized with a newer scikit-learn
    that records ``ManhattanDistance64``.  ReaSyn pins scikit-learn 1.2.2,
    where the same float64 Cython class is named ``ManhattanDistance``.
    """
    from sklearn.metrics import _dist_metrics

    if not hasattr(_dist_metrics, "ManhattanDistance64"):
        _dist_metrics.ManhattanDistance64 = _dist_metrics.ManhattanDistance


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--request", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    request = json.loads(args.request.read_text())
    sys.path.insert(0, str(pathlib.Path.cwd()))
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from reasyn.models.reasyn import ReaSyn
    from reasyn.chem.mol import Molecule
    from reasyn.chem.stack import Stack
    from reasyn.sampler.sampler import Sampler
    from reasyn.utils.sample_utils import TimeLimit

    models = []
    for checkpoint in request["model_paths"]:
        # Released Lightning checkpoints contain config metadata, requiring the
        # explicit trusted-checkpoint loader with torch >= 2.6.
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = OmegaConf.create(ckpt["hyper_parameters"]["config"])
        model = ReaSyn(config.model).to(request["device"])
        model.load_state_dict({k[6:]: v for k, v in ckpt["state_dict"].items()})
        models.append(model.eval())
    _install_sklearn_pickle_compat()
    with open(request["fpindex"], "rb") as handle:
        fpindex = pickle.load(handle)
    with open(request["rxn_matrix"], "rb") as handle:
        matrix = pickle.load(handle)
    if request.get("additional_fpindex"):
        with open(request["additional_fpindex"], "rb") as handle:
            extra = pickle.load(handle)
        fpindex._molecules += extra._molecules
        fpindex._smiles += extra._smiles
        fpindex._fp = np.vstack([fpindex._fp, extra._fp])
    stock = {mol.csmiles for mol in fpindex._molecules}

    def verified(row):
        stack = Stack()
        for token in row["synthesis"].split(";"):
            if re.fullmatch(r"R\d+", token):
                idx = int(token[1:])
                if idx >= len(matrix.reactions) or not stack.push_rxn(
                    matrix.reactions[idx], idx
                ):
                    return False
            else:
                mol = Molecule(token)
                if mol.csmiles not in stock:
                    return False
                stack.push_mol(mol, 0)
        return stack.get_stack_depth() == 1 and Molecule(row["smiles"]).csmiles in {
            m.csmiles for m in stack.get_top()
        }

    outputs = []
    settings = request["settings"]
    for index, target in enumerate(request["targets"]):
        seed = request["sampling_seed"] + index
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        sampler = Sampler(
            fpindex=fpindex,
            rxn_matrix=matrix,
            mol=Molecule(target),
            model=models,
            factor=settings["search_width"],
            max_active_states=settings["exhaustiveness"],
            exact_break=settings["exact_break"],
        )
        with torch.inference_mode():
            sampler.evolve(
                gpu_lock=None,
                time_limit=TimeLimit(settings["time_limit"]),
                num_cycles=settings["num_cycles"],
                max_evolve_steps=8,
                num_editflow_samples=settings["num_editflow_samples"],
                num_editflow_steps=100,
            )
        frame = sampler.get_dataframe()
        if frame.empty:
            continue
        for row in frame.head(settings["max_results"]).to_dict(orient="records"):
            if verified(row):
                outputs.append(
                    {
                        "target": target,
                        "smiles": str(row["smiles"]),
                        "synthesis": str(row["synthesis"]),
                        "num_steps": int(row["num_steps"]),
                        "projection_similarity": float(row["score"]),
                        "sampling_seed": seed,
                        "pathway_verified": True,
                    }
                )
    args.output.write_text(
        json.dumps(
            {
                "rows": outputs,
                "target_count": len(request["targets"]),
                "parameter_count": sum(
                    sum(p.numel() for p in m.parameters()) for m in models
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
