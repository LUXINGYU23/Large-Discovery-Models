"""PoC proposal and selection adapters for the asynchronous research lane."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import tempfile
import numpy as np

from ldm_tts.async_runtime import PacingDecision, Proposal, ResearchOutcome, StreamingPacing
from ldm_tts.contracts import CandidateRejection, RawProposal
from tasks.antibody.core.engine_adapters import AntibodyCandidateDomain


class InitializationPacing(StreamingPacing):
    """Wait for the shared first measurement before a second research generation."""

    name = "antibody-streaming"

    def __init__(self, initial_evaluations=1):
        self.initial_evaluations = initial_evaluations

    def research(self, view):
        if self.initial_evaluations > 1 and view.head_ingest_seq < self.initial_evaluations:
            return PacingDecision("wait", "shared_initialization")
        if view.head_ingest_seq == 0 and view.outstanding:
            return PacingDecision("wait", "initial_measurement_pending")
        return super().research(view)


class GenerationQuotaPacing(InitializationPacing):
    """Reserve online evaluations for each completed research generation."""

    name = "antibody-streaming-quota"

    def __init__(self, initial_evaluations, per_generation):
        super().__init__(initial_evaluations)
        self.per_generation = per_generation

    def decide(self, view):
        limit = self.initial_evaluations + view.current_generation * self.per_generation
        if view.budget.upper >= limit and view.budget.free:
            return PacingDecision("wait", "generation_quota")
        return super().decide(view)

    def research(self, view):
        generation = view.current_generation
        if generation:
            dispatched = self.initial_evaluations + generation * self.per_generation
            previous = dispatched - self.per_generation
            if view.budget.upper < dispatched or view.head_ingest_seq <= previous:
                return PacingDecision("wait", "generation_feedback")
        return super().research(view)


class NativeResearch:
    version = "1"

    def __init__(self, args, worker):
        self.args, self.worker = args, worker
        self.name = f"antibody-{args.method}"
        self.native = None
        self.domain = AntibodyCandidateDomain(11)

    @staticmethod
    def _write_checkpoint(path, state):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except BaseException:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise

    @staticmethod
    def _read_checkpoint(path):
        from .baselines import prepare_antbo_source
        prepare_antbo_source()  # The serialized native optimizer is imported from this configured source tree.
        with path.open("rb") as stream:
            return pickle.load(stream)

    def antbo_proposal(self, generation, measured, pending):
        import torch
        from .baselines import NativeAntBO

        directory = self.worker.jobs.parent / "antbo_checkpoints"
        path = directory / f"g{generation:06d}.pkl"
        inputs = {"generation": generation, "history": measured, "pending": sorted(pending)}
        fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True, allow_nan=False).encode()).hexdigest()

        def restore(state):
            if state.get("status") != "succeeded":
                raise RuntimeError("AntBO generation has an unresolved checkpoint; inspect it before retrying")
            self.native = state["native"]
            np.random.set_state(state["numpy_rng"])
            random.setstate(state["python_rng"])
            torch.set_rng_state(state["torch_rng"])

        if path.exists():
            state = self._read_checkpoint(path)
            if state.get("input_sha256") != fingerprint:
                raise RuntimeError("AntBO generation checkpoint differs from its frozen inputs")
            restore(state)
            return state["proposal"]
        earlier = sorted(p for p in directory.glob("g*.pkl") if p.name < path.name) if directory.exists() else []
        if earlier:
            restore(self._read_checkpoint(earlier[-1]))
        else:
            self.native = NativeAntBO(self.args.seed)
        self._write_checkpoint(path, {"status": "started", "input_sha256": fingerprint})
        proposal = self.native.propose(measured, pending)
        self._write_checkpoint(path, {"status": "succeeded", "input_sha256": fingerprint, "proposal": proposal,
            "native": self.native, "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state()})
        return proposal

    def research(self, round_input, history):
        generation = round_input["generation"]
        self.worker.round = generation - 1
        pending = set(round_input.get("pending_keys", ()))
        self.worker.evaluated = {o.canonical_key for o in history} | pending
        metric = "mock_energy" if self.args.oracle == "mock" else "absolut_energy"
        measured = [dict(round=o.round_idx, sequence=o.candidate.payload["sequence"],
                         backbone_id=o.candidate.payload.get("backbone_id"),
                         energy=o.evaluation.metrics[metric], oracle=self.args.oracle)
                    for o in history if o.evaluation.succeeded]
        if measured and self.args.method == "antbo":
            rows = [self.antbo_proposal(generation, measured, pending)]
        else:
            jobs = [job for job in self.worker.records() if job["round"] == self.worker.round]
            if any(job["status"] != "succeeded" for job in jobs):
                raise RuntimeError("Generation has an unresolved or failed receipt; inspect it before retrying")
            rows = self.worker.eligible(row for job in jobs for row in job.get("candidates", []))[:1]
            while not rows and self.worker.remaining()["backbones"] > 0:
                count = min(self.args.samples_per_call, self.worker.remaining()["sequences"])
                if count < 1:
                    break
                generated = self.worker.execute("design_h3", count, self.args.temperature,
                    call_id=f"g{generation}-native-{len(jobs)}")
                jobs.append({"status": "succeeded"})
                rows = self.worker.eligible(generated)[:1]
            if not rows:
                raise RuntimeError("Fixed proposal budget exhausted without an admissible H3")
        proposals = []
        for row in rows:
            candidate = self.domain.admit(RawProposal(row, self.args.method))
            if not isinstance(candidate, CandidateRejection) and candidate.canonical_key not in self.worker.evaluated:
                proposals.append(Proposal(candidate, self.args.method))
        return ResearchOutcome(proposals=tuple(proposals), metadata={"history_count": len(measured),
                               "generation_remaining": self.worker.remaining()})
