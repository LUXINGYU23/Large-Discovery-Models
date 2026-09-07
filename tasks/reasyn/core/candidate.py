"""Admission separates model target suggestions from trusted projector outputs."""

from __future__ import annotations
import hashlib
import json
from ldm_tts.contracts import Candidate, CandidateRejection
from .chemistry import canonicalize


class ReaSynDomain:
    def __init__(self, benchmark, *, mock=False):
        self.benchmark, self.mock = benchmark, mock

    def admit(self, proposal):
        try:
            p = dict(proposal.payload)
            if self.benchmark == "reconstruction":
                if set(p) != {"target_smiles", "sampling_seed"}:
                    raise ValueError(
                        "query requires target_smiles and sampling_seed only"
                    )
                p["target_smiles"] = canonicalize(p["target_smiles"], mock=self.mock)
                if (
                    isinstance(p["sampling_seed"], bool)
                    or not isinstance(p["sampling_seed"], int)
                    or p["sampling_seed"] < 0
                ):
                    raise ValueError("sampling_seed must be a nonnegative integer")
                identity = json.dumps(p, sort_keys=True)
            else:
                if set(p) != {"smiles", "synthesis", "projection_artifact"}:
                    raise ValueError(
                        "product requires smiles, synthesis and projection_artifact"
                    )
                if proposal.source != "reasyn_projector" or not proposal.metadata.get(
                    "pathway_verified"
                ):
                    raise ValueError(
                        "only task-owned replay-verified projector products are admitted"
                    )
                p["smiles"] = canonicalize(p["smiles"], mock=self.mock)
                if not isinstance(p["synthesis"], str) or not p["synthesis"]:
                    raise ValueError("synthesis trace is required")
                identity = p[
                    "smiles"
                ]  # TDC preserves stereo exactly as upstream Oracle.
            key = hashlib.sha256(identity.encode()).hexdigest()
            return Candidate("reasyn-" + key[:16], p, key, proposal.source)
        except (ValueError, TypeError, KeyError) as exc:
            return CandidateRejection(
                "invalid_reasyn_candidate", str(exc), proposal.source
            )
