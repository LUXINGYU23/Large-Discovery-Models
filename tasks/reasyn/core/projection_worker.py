"""Subprocess bridge to released frozen ReaSyn models; never edits upstream.

This file intentionally has no LDM imports: it runs with the upstream Python
and working directory, loading a checkpoint pair once for its target batch.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import pathlib
import pickle
import random
import re
import sys
import tempfile


def request_digest(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()


def load_progress(output, request):
    """Read complete or interrupted output bound to the exact projection request.

    Results from older runs have no completion list and represent a full batch.
    New checkpoints identify occurrences by position, so repeated targets keep
    their independent seeds and an empty result still counts as completed work.
    """
    if not output.exists():
        return None
    data = json.loads(output.read_text())
    size = len(request["targets"])
    digest = request_digest(request)
    if data.get("target_count") != size or not isinstance(data.get("rows"), list):
        raise ValueError("projection checkpoint target count or rows mismatch")
    if "completed_target_indices" not in data:
        if data.get("complete") is False:
            raise ValueError("projection checkpoint omitted completed targets")
        completed = list(range(size))
        complete = True
    else:
        completed = data["completed_target_indices"]
        if (
            not isinstance(completed, list)
            or any(
                type(index) is not int or not 0 <= index < size
                for index in completed
            )
            or len(set(completed)) != len(completed)
        ):
            raise ValueError("projection checkpoint completed target indices invalid")
        complete = len(completed) == size
        if data.get("complete") is not complete:
            raise ValueError("projection checkpoint completion state mismatch")
        if data.get("request_sha256") != digest:
            raise ValueError("projection checkpoint request mismatch")
        for row in data["rows"]:
            index = row.get("target_index")
            if (
                type(index) is not int
                or index not in completed
                or row.get("target") != request["targets"][index]
                or row.get("sampling_seed") != request["sampling_seed"] + index
            ):
                raise ValueError("projection checkpoint occurrence identity mismatch")
    if data.get("request_sha256", digest) != digest:
        raise ValueError("projection checkpoint request mismatch")
    return {**data, "completed_target_indices": completed, "complete": complete}


def write_progress(output, request, rows, completed, **metadata):
    """Atomically checkpoint after every target, including targets with no rows."""
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        **metadata,
        "rows": rows,
        "target_count": len(request["targets"]),
        "completed_target_indices": sorted(completed),
        "complete": len(completed) == len(request["targets"]),
        "request_sha256": request_digest(request),
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=".projection-",
            delete=False,
        ) as handle:
            temporary = pathlib.Path(handle.name)
            json.dump(data, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _install_sklearn_pickle_compat():
    """Restore the pre-1.4 name for float64 distance metrics.

    The released fingerprint index was serialized with a newer scikit-learn
    that records ``ManhattanDistance64``.  ReaSyn pins scikit-learn 1.2.2,
    where the same float64 Cython class is named ``ManhattanDistance``.
    """
    from sklearn.metrics import _dist_metrics

    if not hasattr(_dist_metrics, "ManhattanDistance64"):
        _dist_metrics.ManhattanDistance64 = _dist_metrics.ManhattanDistance


class _AssetUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Only assets that actually contain a sklearn distance metric need this
        # optional dependency. Mock assets and the lightweight test environment do
        # not import sklearn merely because they exercise the real worker bridge.
        if (
            module == "sklearn.metrics._dist_metrics"
            and name == "ManhattanDistance64"
        ):
            _install_sklearn_pickle_compat()
        return super().find_class(module, name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--request", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    request = json.loads(args.request.read_text())
    progress = load_progress(args.output, request)
    if progress is not None and progress["complete"]:
        return 0
    outputs = list(progress["rows"]) if progress is not None else []
    completed = (
        set(progress["completed_target_indices"])
        if progress is not None
        else set()
    )
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
    with open(request["fpindex"], "rb") as handle:
        fpindex = _AssetUnpickler(handle).load()
    with open(request["rxn_matrix"], "rb") as handle:
        matrix = _AssetUnpickler(handle).load()
    if request.get("additional_fpindex"):
        with open(request["additional_fpindex"], "rb") as handle:
            extra = _AssetUnpickler(handle).load()
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

    parameter_count = sum(sum(p.numel() for p in m.parameters()) for m in models)
    settings = request["settings"]
    for index, target in enumerate(request["targets"]):
        if index in completed:
            continue
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
        rows = (
            []
            if frame.empty
            else frame.head(settings["max_results"]).to_dict(orient="records")
        )
        for row in rows:
            if verified(row):
                outputs.append(
                    {
                        "target": target,
                        "target_index": index,
                        "smiles": str(row["smiles"]),
                        "synthesis": str(row["synthesis"]),
                        "num_steps": int(row["num_steps"]),
                        "projection_similarity": float(row["score"]),
                        "sampling_seed": seed,
                        "pathway_verified": True,
                    }
                )
        completed.add(index)
        write_progress(
            args.output, request, outputs, completed, parameter_count=parameter_count
        )
    if not request["targets"]:
        write_progress(
            args.output, request, outputs, completed, parameter_count=parameter_count
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
