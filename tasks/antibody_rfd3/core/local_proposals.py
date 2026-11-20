"""Collect proposals visited by the native AntBO search for the HER2 comparison."""
from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from ldm_tts.async_runtime import Proposal, ResearchOutcome
from ldm_tts.contracts import CandidateRejection, RawProposal
from ldm_tts.engine.run_store import atomic_json_write
from tasks.antibody.core.engine_adapters import AntibodyCandidateDomain, _sequence_violations


class LocalProposalResearch:
    version = "1"
    name = "her2-antbo-local-proposals"

    def __init__(self, args, directory, panel):
        self.args, self.directory, self.panel = args, directory, panel
        self.domain = AntibodyCandidateDomain(11)

    def research(self, round_input, history):
        outcome = self.panel.research(round_input, history) if self.panel else ResearchOutcome()
        if outcome.status != "committed":
            return outcome
        count = self.args.local_proposal_count
        generation = round_input["generation"]
        root = self.directory / "local-proposals" / f"g{generation:06d}"
        root.mkdir(parents=True, exist_ok=True)
        keys = {o.canonical_key for o in history} | set(round_input["pending_keys"])
        keys.update(p.candidate.canonical_key for p in outcome.proposals)
        keys.update(row["canonical_key"] for row in round_input["policy"]["reservoir"])
        payload = {"seed": self.args.seed + generation - 1, "count": count, "excluded": sorted(keys),
                   "history": [[o.canonical_key, o.evaluation.metrics["absolut_energy"]]
                               for o in history if o.evaluation.succeeded]}
        input_path, result_path = root / "input.json", root / "result.json"
        if input_path.exists() and json.loads(input_path.read_text()) != payload:
            raise ValueError("Native proposal input changed after research was frozen")
        atomic_json_write(input_path, payload)
        if not result_path.exists():
            env = {key: os.environ[key] for key in ("PATH", "TMPDIR", "PYTHONPATH", "LD_LIBRARY_PATH")
                   if key in os.environ}
            env.update(ANTBO_HOME=str(self.args.root / "env/sources/HEBO-77636146/AntBO"),
                       OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
            python = self.args.root / "env/antbo_native/bin/python"
            with (root / "search.log").open("ab") as log:
                subprocess.run([str(python), "-m", __name__, str(input_path), str(result_path)],
                               env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300, check=True)
        result = json.loads(result_path.read_text())
        candidates = []
        for seq in result["proposed"]:
            candidate = self.domain.admit(RawProposal({"sequence": seq}, "antbo_local"))
            if isinstance(candidate, CandidateRejection) or candidate.canonical_key in keys:
                raise ValueError("Native local search returned an inadmissible or duplicate proposal")
            keys.add(candidate.canonical_key)
            candidates.append(Proposal(candidate, "antbo_local"))
        if len(candidates) != count or len(candidates) + len(outcome.proposals) != self.args.pool_size:
            raise RuntimeError("The merged proposal pool did not meet its source quotas")
        metadata = {**outcome.metadata, "pool_sources": {"agent": len(outcome.proposals), "antbo_local": count},
                    "local_search": result["statistics"]}
        return replace(outcome, proposals=(*outcome.proposals, *candidates), metadata=metadata)


def native_pool(payload):
    """Freeze the measured history, run native EI search, and retain its best eligible visits."""
    from unittest.mock import patch

    import numpy as np
    import torch

    from tasks.antibody.core.ldm_light.ldm_acq import indices_to_seqs, seqs_to_indices
    from .baselines import NativeAntBO

    native = NativeAntBO(payload["seed"]).optimizer
    sequences, energies = zip(*payload["history"], strict=True)
    native.observe(seqs_to_indices(list(sequences)), torch.tensor(energies))
    from bo import localbo_utils

    original = localbo_utils.local_search
    excluded = set(payload["excluded"])
    visited, eligible = set(), {}
    calls = 0

    def trace(x_center, f, *args, **kwargs):
        def acquisition(x):
            nonlocal calls
            value = f(x)
            calls += 1
            seq, = indices_to_seqs(np.asarray(x).reshape(1, -1))
            score = float(value.detach().item())
            if seq not in visited and seq not in excluded and not _sequence_violations(seq, 11) and math.isfinite(score):
                eligible[seq] = score
            visited.add(seq)
            return value

        return original(x_center, acquisition, *args, **kwargs)

    with patch.object(localbo_utils, "local_search", trace):
        for attempt in range(1, 5):
            native.suggest(1)
            if len(eligible) >= payload["count"]:
                break
    if len(eligible) < payload["count"]:
        raise RuntimeError(f"Native search found only {len(eligible)} eligible visits after {attempt} searches")
    ranked = sorted(eligible, key=eligible.__getitem__, reverse=True)
    return {"proposed": ranked[:payload["count"]],
            "statistics": {"acquisition_calls": calls, "unique_visits": len(visited),
                           "eligible_visits": len(eligible), "searches": attempt}}


if __name__ == "__main__":
    input_path, output_path = map(Path, sys.argv[1:])
    atomic_json_write(output_path, native_pool(json.loads(input_path.read_text())))
