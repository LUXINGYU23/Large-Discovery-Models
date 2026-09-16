"""Independent proposal minibatches and bounded task-owned projection refill."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import random
from pathlib import Path

from ldm_tts.contracts import CandidateRejection, RawProposal
from ldm_tts.data import make_complete_design_ir
from ldm_tts.engine.expansion import (
    ExpansionResult,
    attach_proposal_attempt_receipt,
)
from ldm_tts.engine.run_store import atomic_json_write
from ldm_tts.transport import ProposalRequest, ProposalResponse
from ldm_tts.transport.parsing import load_json_object
from .candidate import ReaSynDomain
from .chemistry import MOCK_SMILES, canonicalize
from .sampling import attach_empirical_base_measure


DEFAULT_PROPOSAL_RECOVERY_SEED_SPAN = 1_000_000


def parse_targets(text, *, count, mock=False):
    data = load_json_object(text)
    if (set(data) != {"candidates"} or not isinstance(data["candidates"], list)
            or len(data["candidates"]) != count):
        raise ValueError(f"response requires exactly {count} candidates")
    targets = []
    for item in data["candidates"]:
        if not isinstance(item, dict) or set(item) != {"target_smiles"}:
            raise ValueError("each candidate must contain only target_smiles")
        targets.append(canonicalize(item["target_smiles"], mock=mock))
    return targets


class ProposalExhausted(RuntimeError):
    """A scientific round could not obtain its configured legal sampling quota."""

    def __init__(self, metadata, *, attempts=()):
        self.metadata = dict(metadata)
        self.attempts = tuple(attempts)
        super().__init__(
            "ReaSyn proposal replenishment exhausted: "
            f"{metadata['valid_occurrences']}/{metadata['required_occurrences']} occurrences, "
            f"{metadata['unique_candidates']}/{metadata['required_unique_candidates']} unique candidates"
        )


class ReaSynExpander:
    def __init__(self, args, projector, sink, *, target="", client=None):
        self.args, self.projector, self.sink = args, projector, sink
        self.target, self.client = target, client
        self.before_request = None
        self.recovery_pass = 0

    def expand(self, request):
        sample_count = request.reservoir_size
        batch_size = min(getattr(self.args, "proposal_batch_size", 2), sample_count)
        refill_limit = getattr(self.args, "max_replenishment_batches", 4)
        required_unique = min(getattr(self.args, "evaluations_per_round", 1), sample_count)
        if batch_size < 1 or refill_limit < 0:
            raise ValueError("proposal batch size must be positive and refill limit nonnegative")
        initial_batches = (sample_count + batch_size - 1) // batch_size
        max_batches = initial_batches + refill_limit
        full_history = self._history(request)
        recent_history = full_history[-12:]
        domain = ReaSynDomain(self.args.benchmark, mock=self.args.mock)
        evaluated_keys = {o.canonical_key for o in request.observations}
        excluded_products = sorted({h["smiles"] for h in full_history}) if self.args.benchmark == "tdc" else []
        proposals, attempts, feedback = [], [], []
        unique_keys = set()
        rejection_counts = Counter()
        targets_requested = 0
        batches_attempted = 0
        # The seed stride covers every bounded refill slot, including rejections.
        stride = sample_count + refill_limit * batch_size
        recovery_pass = self._recovery_pass()
        round_span = self._recovery_seed_span(request)
        round_seed_index = recovery_pass * round_span + request.round_idx
        for minibatch_index in range(max_batches):
            if len(proposals) >= sample_count and len(unique_keys) >= required_unique:
                break
            if minibatch_index < initial_batches:
                n = min(batch_size, sample_count - minibatch_index * batch_size)
            else:
                n = min(batch_size, max(sample_count - len(proposals), required_unique - len(unique_keys)))
            offset = targets_requested
            targets_requested += n
            batches_attempted += 1
            seed = self.args.seed + round_seed_index * stride + offset
            lineage = []
            proposal_request = self._request(
                request, count=n, minibatch_index=minibatch_index,
                history=full_history, excluded_products=excluded_products, feedback=feedback,
                recovery_pass=recovery_pass, recovery_seed_span=round_span,
            )
            if self.client:
                response = self._propose(proposal_request)
                attempts.append(response)
                lineage = response.metadata.get("harness_lineage", [])
                if lineage and len(lineage) != n:
                    raise ValueError("Harness lineage must align with target occurrences")
                try:
                    targets = parse_targets(response.text, count=n, mock=self.args.mock)
                except (ValueError, TypeError) as exc:
                    rejection_counts["invalid_projection_targets"] += n
                    feedback.append({"reason": "invalid_projection_targets", "message": str(exc)})
                    continue
                self._collect(targets, recent_history, request.round_idx, synthetic=self.args.mock)
            elif getattr(self.args, "search_method", "") == "bo":
                pool = list(getattr(self.args, "bo_targets", MOCK_SMILES if self.args.mock else ()))
                if not pool:
                    raise ValueError("BO requires a score-blind projection target pool")
                random.Random(self.args.seed + round_seed_index).shuffle(pool)
                targets = [pool[(offset + i) % len(pool)] for i in range(n)]
            elif self.args.proposal_mode == "baseline":
                if self.args.benchmark != "reconstruction":
                    raise ValueError("baseline mode is only defined for reconstruction")
                targets = [self.target] * n
            else:
                targets = [MOCK_SMILES[(round_seed_index * sample_count + offset + i) % len(MOCK_SMILES)] for i in range(n)]
                targets = parse_targets(json.dumps({"candidates": [{"target_smiles": s} for s in targets]}), count=n, mock=True)
                self._collect(targets, recent_history, request.round_idx, synthetic=True)
            if self.args.benchmark == "reconstruction":
                proposed = tuple(
                    RawProposal({"target_smiles": s, "sampling_seed": seed + i}, "ldm_projection_query",
                                {"minibatch_index": minibatch_index, "proposal_index": offset + i,
                                 **({"proposal_recovery_pass": recovery_pass} if recovery_pass else {}),
                                 **({"harness_lineage": lineage[i]} if lineage else {})})
                    for i, s in enumerate(targets)
                )
            else:
                identity = self._batch_identity(request.round_idx, minibatch_index, recovery_pass)
                rows, artifact = self.projector.project(
                    targets, sampling_seed=seed,
                    identity=identity,
                )
                chosen = self._first_per_occurrence(rows, targets)
                proposed = tuple(
                    RawProposal(
                        {"smiles": row["smiles"], "synthesis": row["synthesis"], "projection_artifact": artifact},
                        "reasyn_projector",
                        {"pathway_verified": True, "minibatch_index": minibatch_index,
                         "proposal_index": offset + index, "projection_target": targets[index],
                         "sampling_seed": seed + index,
                         **({"proposal_recovery_pass": recovery_pass} if recovery_pass else {}),
                         **({"harness_lineage": lineage[index]} if lineage else {})},
                    )
                    for index, row in chosen.items()
                )
                missing = n - len(chosen)
                if missing:
                    rejection_counts["no_synthesizable_product"] += missing
                    feedback.append({"reason": "no_synthesizable_product", "count": missing})
            for proposal in proposed:
                admitted = domain.admit(proposal)
                if isinstance(admitted, CandidateRejection):
                    rejection_counts[admitted.reason] += 1
                    feedback.append({"reason": admitted.reason, "message": admitted.message})
                elif admitted.canonical_key in evaluated_keys:
                    rejection_counts["already_evaluated_product" if self.args.benchmark == "tdc" else "already_evaluated_query_seed"] += 1
                    feedback.append({"reason": "already_evaluated", "smiles": admitted.payload.get("smiles", admitted.payload.get("target_smiles"))})
                else:
                    # Same-round repetitions remain legitimate empirical occurrences.
                    proposals.append(proposal)
                    unique_keys.add(admitted.canonical_key)
        metadata = {
            "round_idx": request.round_idx,
            "mode": self.args.proposal_mode,
            "sampling_mode": "independent_minibatch_requests",
            "minibatch_count": batches_attempted,
            "request_count": len(attempts), "targets_proposed": targets_requested,
            "proposal_batch_size": batch_size, "required_occurrences": sample_count,
            "valid_occurrences": len(proposals), "unique_candidates": len(unique_keys),
            "required_unique_candidates": required_unique,
            "max_replenishment_batches": refill_limit,
            "replenishment_batches": max(0, batches_attempted - initial_batches),
            "proposal_recovery_pass": recovery_pass,
            "proposal_recovery_seed_span": round_span,
            "projection_target_limit": stride if self.args.benchmark == "tdc" else 0,
            "history_exclusion_count": len(evaluated_keys),
            "rejection_counts": dict(rejection_counts), "rejection_feedback": feedback,
            "q0_identity_space": "canonical_query" if self.args.benchmark == "reconstruction" else "canonical_product",
            "evaluation_identity_space": "canonical_query_and_sampling_seed" if self.args.benchmark == "reconstruction" else "canonical_product",
        }
        if len(proposals) < sample_count or len(unique_keys) < required_unique:
            metadata["stop_reason"] = "proposal_replenishment_budget_exhausted"
            self._write_diagnostics(request.round_idx, metadata)
            raise ProposalExhausted(metadata, attempts=attempts)
        annotated = attach_empirical_base_measure(
            tuple(proposals), benchmark=self.args.benchmark, mock=self.args.mock,
            observations=request.observations,
        )
        metadata["stop_reason"] = "sampling_quota_satisfied"
        self._write_diagnostics(request.round_idx, metadata)
        return ExpansionResult(proposals=annotated, attempts=tuple(attempts), metadata=metadata)

    @staticmethod
    def _first_per_occurrence(rows, targets):
        chosen = {}
        for row in rows:
            index = row.get("target_index")
            if index is None:
                matches = [i for i, target in enumerate(targets) if target == row["target"]]
                if len(matches) != 1:
                    raise ValueError("projector must return target_index for repeated target occurrences")
                index = matches[0]
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(targets) or row["target"] != targets[index]:
                raise ValueError("projector target occurrence index does not match request")
            chosen.setdefault(index, row)
        return chosen

    @staticmethod
    def _history(request):
        return [
            {"candidate_id": o.candidate.candidate_id,
             "round_idx": getattr(o, "round_idx", None),
             "smiles": o.candidate.payload.get("smiles", o.candidate.payload.get("target_smiles")),
             **({"sampling_seed": o.candidate.payload["sampling_seed"]} if "sampling_seed" in o.candidate.payload else {}),
             **({"research_annotations": o.candidate.metadata["research_annotations"]}
                if o.candidate.metadata.get("research_annotations") else {}),
             "metrics": dict(o.metrics)}
            for o in request.observations
        ]

    def _request(
        self,
        request,
        *,
        count,
        minibatch_index,
        history,
        excluded_products,
        feedback,
        recovery_pass,
        recovery_seed_span,
    ):
        goal = (f"Reconstruct original molecule {self.target} as closely as possible."
                if self.args.benchmark == "reconstruction" else f"Maximize the TDC oracle {self.args.oracle}.")
        metadata = {
            "round_idx": request.round_idx, "count": count,
            "minibatch_index": minibatch_index, "history": history,
            "excluded_products": excluded_products, "rejection_feedback": feedback.copy(),
            "benchmark": self.args.benchmark, "original_target": self.target,
            "sampling_mode": "independent_minibatch_requests",
        }
        if recovery_pass:
            metadata["proposal_recovery_pass"] = recovery_pass
            metadata["proposal_recovery_seed_span"] = recovery_seed_span
        recovery_note = (
            f" This is recovery pass {recovery_pass} for the same scientific round; "
            "use the rejection feedback to explore fresh projection targets."
            if recovery_pass
            else ""
        )
        return ProposalRequest(
            messages=(
                {"role": "system", "content": "Guide frozen ReaSyn synthesis projection using molecular reasoning and measured feedback. Suggest chemically valid targets; the task verifies synthesis. Output JSON only."},
                {"role": "user", "content": goal + f" Propose exactly {count} projection target SMILES. "
                 + 'Format: {"candidates":[{"target_smiles":"CCO"}]}. Do not claim oracle scores. '
                 + "Within-round repeated targets are allowed and allocate empirical proposal probability; each occurrence has an independent projection seed. "
                 + recovery_note
                 + "Recent measured history: " + json.dumps(history[-12:], sort_keys=True)
                 + " Complete measured product exclusion list (do not reproduce these projected products): " + json.dumps(excluded_products)
                 + " Rejections requiring repair: " + json.dumps(feedback, sort_keys=True)},
            ), metadata=metadata,
        )

    def _propose(self, request):
        # Cache each completed independent turn before projection. A restart must
        # not regenerate earlier targets and invalidate already-paid projections.
        root = getattr(self.projector, "run_dir", None)
        cache = None
        digest = hashlib.sha256(json.dumps({"messages": request.messages, "metadata": request.metadata}, sort_keys=True).encode()).hexdigest()
        if root is not None:
            cache = self._batch_cache_path(
                Path(root),
                request.metadata["round_idx"],
                request.metadata["minibatch_index"],
                int(request.metadata.get("proposal_recovery_pass", 0)),
            )
            receipt = cache.relative_to(Path(root)).as_posix()
            if cache.exists():
                record = json.loads(cache.read_text())
                if record["request_sha256"] != digest:
                    raise ValueError("proposal minibatch resume request mismatch")
                data = record["response"]
                data["tool_calls"] = tuple(data.get("tool_calls", ()))
                return attach_proposal_attempt_receipt(ProposalResponse(**data), receipt)
        if self.before_request:
            self.before_request()
        response = self.client.propose(request)
        if cache is not None:
            response = attach_proposal_attempt_receipt(response, receipt)
            atomic_json_write(cache, {"request_sha256": digest, "response": response.to_dict()})
        return response

    def _write_diagnostics(self, round_idx, metadata):
        root = getattr(self.projector, "run_dir", None)
        if root is not None:
            recovery_pass = int(metadata.get("proposal_recovery_pass", 0))
            atomic_json_write(
                self._batch_cache_path(Path(root), round_idx, 0, recovery_pass).parent / "diagnostics.json",
                metadata,
            )

    @staticmethod
    def _batch_identity(round_idx, minibatch_index, recovery_pass=0):
        suffix = f"round-{round_idx:06d}/batch-{minibatch_index:06d}"
        if recovery_pass:
            return f"recovery-{recovery_pass:06d}/{suffix}"
        return suffix

    @staticmethod
    def _batch_cache_path(root, round_idx, minibatch_index, recovery_pass=0):
        parent = Path(root) / "proposal_batches"
        if recovery_pass:
            parent = parent / f"recovery-{recovery_pass:06d}"
        return parent / f"round-{round_idx:06d}" / f"batch-{minibatch_index:06d}.json"

    def _recovery_pass(self):
        value = getattr(self, "recovery_pass", 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("proposal recovery pass must be a nonnegative integer")
        return value

    def _recovery_seed_span(self, request):
        value = getattr(self.args, "proposal_recovery_seed_span", None)
        if value is None:
            value = max(
                DEFAULT_PROPOSAL_RECOVERY_SEED_SPAN,
                request.round_idx + 1,
                int(getattr(self.args, "max_oracle_calls", 0) or 0),
            )
        if isinstance(value, bool) or not isinstance(value, int) or value <= request.round_idx:
            raise ValueError("proposal recovery seed span must exceed the active round index")
        return value

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
            observations=[{"smiles": row["smiles"], "metrics": row["metrics"]} for row in history],
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
