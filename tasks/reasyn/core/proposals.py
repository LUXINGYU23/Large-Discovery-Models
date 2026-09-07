"""Feedback-conditioned LDM proposal expansion; no second optimization loop."""

from __future__ import annotations
import json
from ldm_tts.contracts import RawProposal
from ldm_tts.data import make_complete_design_ir
from ldm_tts.engine.expansion import ExpansionResult
from ldm_tts.transport import ProposalRequest
from ldm_tts.transport.parsing import load_json_object
from .chemistry import MOCK_SMILES, canonicalize


def parse_targets(text, *, count, mock=False):
    data = load_json_object(text)
    if (
        set(data) != {"candidates"}
        or not isinstance(data["candidates"], list)
        or len(data["candidates"]) != count
    ):
        raise ValueError(f"response requires exactly {count} candidates")
    targets = []
    for item in data["candidates"]:
        if not isinstance(item, dict) or set(item) != {"target_smiles"}:
            raise ValueError("each candidate must contain only target_smiles")
        targets.append(canonicalize(item["target_smiles"], mock=mock))
    return targets


class ReaSynExpander:
    def __init__(self, args, projector, sink, *, target="", client=None):
        self.args = args
        self.projector = projector
        self.sink = sink
        self.target = target
        self.client = client
        self.before_request = None

    def expand(self, request):
        n = request.reservoir_size
        history = [
            {"candidate": o.candidate.payload, "metrics": dict(o.metrics)}
            for o in request.observations[-12:]
        ]
        goal = (
            f"Reconstruct original molecule {self.target} as closely as possible."
            if self.args.benchmark == "reconstruction"
            else f"Maximize the TDC oracle {self.args.oracle}."
        )
        # Keep provenance/artifact paths out of the model-visible history.
        history = [
            {
                "smiles": h["candidate"].get(
                    "smiles", h["candidate"].get("target_smiles")
                ),
                "metrics": h["metrics"],
            }
            for h in history
        ]
        attempts = ()
        if self.client:
            if self.before_request:
                self.before_request()
            response = self.client.propose(
                ProposalRequest(
                    messages=(
                        {
                            "role": "system",
                            "content": "You guide frozen ReaSyn synthesis projection using molecular reasoning and measured feedback. Suggest chemically valid molecular targets; the task verifies synthesis. Output JSON only.",
                        },
                        {
                            "role": "user",
                            "content": goal
                            + f" Propose {n} projection target SMILES, balancing close refinements and new structures. "
                            + 'Format: {"candidates":[{"target_smiles":"CCO"}]}. Do not claim oracle scores. '
                            + "Measured history: "
                            + json.dumps(history, sort_keys=True),
                        },
                    ),
                    metadata={"round_idx": request.round_idx},
                )
            )
            attempts = (response,)
            try:
                targets = parse_targets(response.text, count=n, mock=self.args.mock)
            except ValueError as exc:
                return ExpansionResult(
                    schema_update={},
                    attempts=attempts,
                    metadata={
                        "rejection": "invalid_projection_targets",
                        "message": str(exc),
                        "mode": self.args.proposal_mode,
                    },
                )
            self._collect(targets, history, request.round_idx, synthetic=self.args.mock)
        elif self.args.proposal_mode == "baseline":
            if self.args.benchmark == "reconstruction":
                targets = [self.target] * n
            else:
                raise ValueError(
                    "baseline mode is defined for reconstruction; use upstream Graph GA for TDC baseline"
                )
        else:
            targets = [
                MOCK_SMILES[(request.round_idx * n + i) % len(MOCK_SMILES)]
                for i in range(n)
            ]
            # Exercise exactly the production JSON parser and accepted-action IR.
            targets = parse_targets(
                json.dumps({"candidates": [{"target_smiles": s} for s in targets]}),
                count=n,
                mock=True,
            )
            self._collect(targets, history, request.round_idx, synthetic=True)
        seed = self.args.seed + request.round_idx * n
        if self.args.benchmark == "reconstruction":
            proposals = tuple(
                RawProposal(
                    {"target_smiles": s, "sampling_seed": seed + i},
                    "ldm_projection_query",
                )
                for i, s in enumerate(targets)
            )
        else:
            rows, artifact = self.projector.project(
                targets, sampling_seed=seed, identity=f"round-{request.round_idx:06d}"
            )
            # Match upstream Graph GA-ReaSyn projection: first (highest-similarity)
            # synthesizable product per target, then canonical product deduplication.
            chosen = {}
            for row in rows:
                chosen.setdefault(row["target"], row)
            proposals = tuple(
                RawProposal(
                    {
                        "smiles": r["smiles"],
                        "synthesis": r["synthesis"],
                        "projection_artifact": artifact,
                    },
                    "reasyn_projector",
                    {"pathway_verified": True},
                )
                for r in chosen.values()
            )
        if not proposals:
            return ExpansionResult(
                schema_update={}, attempts=attempts, metadata={"empty_projection": True}
            )
        return ExpansionResult(
            proposals=proposals,
            attempts=attempts,
            metadata={"targets_proposed": n, "mode": self.args.proposal_mode},
        )

    def _collect(self, targets, history, round_idx, *, synthetic):
        ir = make_complete_design_ir(
            task_id="reasyn",
            domain="ReaSyn projection target SMILES",
            task_description=(
                "Reconstruct " + self.target
                if self.target
                else "Maximize TDC " + self.args.oracle
            ),
            objectives=[
                {
                    "name": "similarity" if self.target else "oracle_score",
                    "direction": "maximize",
                }
            ],
            observations=history,
            candidates=[{"target_smiles": s} for s in targets],
            design_space_description="Chemically valid molecular targets passed to the frozen ReaSyn projector.",
            request_description="Propose molecular targets using measured feedback.",
            num_candidates=len(targets),
            reasoning_available=False,
        )
        self.sink.append(
            ir,
            provenance={
                "round_idx": round_idx,
                "synthetic_fixture": synthetic,
                "benchmark": self.args.benchmark,
            },
        )
