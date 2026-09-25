"""Candidate-program proposals: strict validation, direct requests, and the expander.

Every accepted submission contains exactly the requested number of programs.
Invalid entries are repaired in the same logical request or session; a round
that still cannot be filled pauses resumably instead of shrinking. Only
authoritative measured candidates are excluded; programs proposed earlier but
never evaluated remain eligible.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ldm_tts.contracts import RawProposal
from ldm_tts.data import make_complete_design_ir
from ldm_tts.engine.expansion import ExpansionResult, attach_proposal_attempt_receipt
from ldm_tts.engine.run_store import atomic_json_write
from ldm_tts.transport import ProposalRequest, ProposalResponse
from ldm_tts.transport.openai import EndpointRequestError
from ldm_tts.transport.parsing import load_json_object

from .cases import CaseSpec
from .clock import WallClockExhausted
from .history import best_records, compact, measured_records
from .sampling import attach_empirical_base_measure
from .usage import ApiPricing, account_tokens, normalize_tokens

ROW_FIELDS = {"program", "change_summary", "rationale"}
OPTIONAL_FIELDS = {"comparison_candidate_ids"}
LDM_METHODS = ("ldm", "ldm_harness", "ldm_harness_compiled")


class ProposalExhausted(RuntimeError):
    """A round could not obtain its exact number of valid proposal occurrences."""

    def __init__(self, metadata: dict[str, Any], *, attempts=()) -> None:
        self.metadata, self.attempts = dict(metadata), tuple(attempts)
        super().__init__(f"proposal repair budget exhausted: {metadata.get('reason', '')}")


def validate_rows(rows: Any, *, case: CaseSpec, count: int, measured_keys: set[str],
                  measured_ids: set[str]) -> tuple[list[dict], list[dict]]:
    """Return (normalized rows, errors) using the Campaign admission rules."""
    if not isinstance(rows, list):
        return [], [_error("/candidates", "candidates_not_array", "candidates must be an array",
                           f"Provide exactly {count} candidate objects.")]
    errors = []
    if len(rows) != count:
        errors.append(_error("/candidates", "wrong_candidate_count", f"{len(rows)} candidates submitted; exactly "
                             f"{count} are required", f"Submit exactly {count} distinct programs."))
    normalized, seen = [], {}
    for index, row in enumerate(rows):
        pointer = f"/candidates/{index}"
        if not isinstance(row, dict) or not ROW_FIELDS <= set(row) or set(row) - ROW_FIELDS - OPTIONAL_FIELDS:
            errors.append(_error(pointer, "invalid_candidate_fields", "each candidate needs program, change_summary "
                                 "and rationale (optional comparison_candidate_ids)", "Use only the documented fields."))
            continue
        row_errors = []
        for field, limit in (("change_summary", 400), ("rationale", 2000)):
            if not isinstance(row[field], str) or not row[field].strip() or len(row[field]) > limit:
                row_errors.append(_error(f"{pointer}/{field}", f"invalid_{field}",
                                         f"{field} must be a nonempty string of at most {limit} characters",
                                         "Write a short research note."))
        comparisons = row.get("comparison_candidate_ids", [])
        if not isinstance(comparisons, list) or any(c not in measured_ids for c in comparisons):
            row_errors.append(_error(f"{pointer}/comparison_candidate_ids", "unknown_comparison_candidate",
                                     "comparison_candidate_ids must name measured candidates",
                                     "Use candidate IDs returned by the measured history."))
        for problem in case.check_program(row["program"]):
            row_errors.append(_error(f"{pointer}/program", problem["code"], problem["message"], problem["hint"]))
        if not row_errors:
            program = row["program"].strip() + "\n"
            key = case.canonical_key(program)
            if key in measured_keys:
                row_errors.append(_error(f"{pointer}/program", "already_measured", "this program was already "
                                         "evaluated", "Inspect its measured result and propose a different method."))
            elif key in seen:
                row_errors.append(_error(f"{pointer}/program", "duplicate_in_submission",
                                         f"same canonical program as candidates[{seen[key]}]",
                                         "Entries of one submission must be distinct methods."))
            else:
                seen[key] = index
                normalized.append({"program": program, "change_summary": row["change_summary"].strip(),
                                   "rationale": row["rationale"].strip(), "comparison_candidate_ids": comparisons})
        errors.extend(row_errors)
    return (normalized if not errors else []), errors


def _error(pointer, code, message, hint):
    return {"path": pointer, "code": code, "message": message, "hint": hint}


@dataclass
class ProposalBatch:
    rows: list[dict]
    attempts: list[ProposalResponse]
    diagnostics: dict[str, Any]


class DirectProgramSource:
    """Independent direct requests; each minibatch is repaired within its own exchange."""

    def __init__(self, *, args, case: CaseSpec, client, run_dir: Path, pricing: ApiPricing, sink=None) -> None:
        self.args, self.case, self.client, self.pricing, self.sink = args, case, client, pricing, sink
        self.root = Path(run_dir) / "proposal_requests"
        self.runtime = None
        self.clock = None
        # A synthetic mock client is a local proposal source: it exercises parsing,
        # repair and caching but is not a model request and is never charged.
        self.synthetic = bool(getattr(args, "proposal_mode", "") == "mock")

    def collect(self, *, round_idx, count, batch_size, records, measured_keys, recovery_pass,
                refill=0, avoid=()) -> ProposalBatch:
        rows, attempts, minibatches = [], [], []
        index = 0
        while len(rows) < count:
            size = min(batch_size, count - len(rows))
            batch_rows, batch_attempts, info = self._minibatch(round_idx, index, size, records, measured_keys,
                                                               recovery_pass, refill, list(avoid))
            rows.extend(batch_rows)
            attempts.extend(batch_attempts)
            minibatches.append(info)
            index += 1
        if self.synthetic:
            return ProposalBatch(rows, [], {"minibatches": minibatches, "synthetic_requests": len(attempts)})
        return ProposalBatch(rows, attempts, {"minibatches": minibatches, "requests": len(attempts)})

    def _minibatch(self, round_idx, index, size, records, measured_keys, recovery_pass, refill=0, avoid=()):
        measured_ids = {r["candidate_id"] for r in records}
        messages = list(self._messages(size, records, avoid))
        attempts, last_errors = [], []
        for repair in range(self.args.max_repair_requests + 1):
            refill_part = f"-f{refill:02d}" if refill else ""
            request_id = f"p{recovery_pass:03d}-r{round_idx:06d}{refill_part}-b{index:04d}-a{repair:02d}"
            response = self._request(request_id, ProposalRequest(messages=tuple(messages), metadata={
                "round_idx": round_idx, "minibatch_index": index, "repair": repair, "count": size,
                "refill": refill}))
            attempts.append(response)
            try:
                data = load_json_object(response.text)
                if set(data) != {"candidates"}:
                    raise ValueError("response object must contain only candidates")
                rows, last_errors = validate_rows(data["candidates"], case=self.case, count=size,
                                                  measured_keys=measured_keys, measured_ids=measured_ids)
            except ValueError as exc:
                rows, last_errors = [], [_error("", "invalid_json", str(exc), "Return one JSON object only.")]
            if rows:
                note = {"source": "direct_request", "request_id": request_id, "repair_requests": repair}
                self._collect(rows, records, round_idx)
                return ([{**row, "note": {**note, "item_index": i}} for i, row in enumerate(rows)], attempts,
                        {"minibatch_index": index, "requests": repair + 1, "count": size})
            messages += [{"role": "assistant", "content": response.text},
                         {"role": "user", "content": "The submission was rejected. Errors:\n"
                          + json.dumps(last_errors[:24], indent=1)
                          + f"\nReturn the complete JSON object again with exactly {size} candidates: keep valid "
                            "entries and replace or repair the listed ones."}]
        raise ProposalExhausted({"reason": "direct_repair_requests_exhausted", "round_idx": round_idx,
                                 "minibatch_index": index, "errors": last_errors[:24]},
                                attempts=() if self.synthetic else attempts)

    def _request(self, request_id: str, request: ProposalRequest) -> ProposalResponse:
        cache = self.root / f"{request_id}.json"
        digest = hashlib.sha256(json.dumps({"messages": request.messages, "metadata": request.metadata},
                                           sort_keys=True).encode()).hexdigest()
        if cache.exists():
            record = json.loads(cache.read_text())
            if record["request_sha256"] != digest:
                raise ValueError(f"direct proposal request {request_id} changed on resume")
            data = record["response"]
            data["tool_calls"] = tuple(data.get("tool_calls", ()))
            return ProposalResponse(**data)
        # Each transport attempt has a durable identity. An attempt left pending by
        # a stopped process was already charged; it is recorded as failed with
        # unknown usage and never reused, so resumed requests are always new ones.
        attempt, fresh = 0, 0
        while True:
            key = f"direct:{request_id}:t{attempt}"
            marker = self.root / f"{request_id}.t{attempt}.pending"
            if marker.exists():
                if self.runtime is not None:
                    self.runtime.consume_many({"llm_failed_requests": 1, "llm_usage_unknown": 1}, usage_key=key)
                attempt += 1
                continue
            if self.clock is not None and self.clock.remaining() <= 0:
                raise WallClockExhausted("campaign wall-clock allowance is exhausted before a proposal request")
            if self.runtime is not None and not self.synthetic:
                self.pricing.check(self.runtime)
                self.runtime.consume_many({"llm_requests": 1, "proposal_request_attempts": 1}, usage_key=key)
            atomic_json_write(marker, {"request_sha256": digest, "started_at_unix": time.time()})
            try:
                response = self.client.propose(request)
            except EndpointRequestError:
                if self.runtime is not None:
                    self.runtime.consume_many({"llm_failed_requests": 1, "llm_usage_unknown": 1}, usage_key=key)
                attempt, fresh = attempt + 1, fresh + 1
                if fresh > self.args.llm_transport_retries:
                    raise
                time.sleep(min(60.0, 2.0 ** fresh))
                continue
            if self.runtime is not None and not self.synthetic:
                account_tokens(self.runtime, self.pricing, "llm", normalize_tokens(response.usage), usage_key=key)
            response = attach_proposal_attempt_receipt(response, f"direct:{request_id}")
            atomic_json_write(cache, {"request_sha256": digest, "transport_attempt": attempt,
                                      "response": response.to_dict()})
            marker.unlink()
            return response

    def _messages(self, count: int, records: list[dict], avoid=()):
        contract = self.case.public_contract()
        best = best_records(records, 2)
        failures = [r for r in records if r["status"] != "succeeded"][-3:]
        system = (
            "You design research methods for one fixed ResearchGym case. Reply with one JSON object only: "
            '{"candidates": [{"program": "<complete Python module>", "change_summary": "<what changed>", '
            '"rationale": "<why it should score better>"}]}. Each program replaces the case\'s candidate file '
            "and must satisfy the interface and static rules exactly. Emit program source as a JSON string "
            "with escaped newlines, never markdown fences.")
        user = {
            "case": contract,
            "interface_reference_program": self.case.seed_program(),
            "requested_candidates": count,
            "submission_rules": [
                f"Return exactly {count} candidates; entries in this reply must be distinct programs.",
                "Programs already evaluated in the measured history are rejected; unevaluated earlier ideas remain allowed.",
                "Independent requests may propose the same program; agreement is recorded as proposal frequency.",
            ],
            "measured_history_index": [compact(r) for r in records][-64:],
            "measured_history_total": len(records),
            "best_measured_programs": [{**compact(r), "metrics": r["metrics"], "program": r["program"]} for r in best],
            "recent_failures": [{**compact(r), "error": r["error"]} for r in failures],
        }
        if avoid:
            user["replenishment"] = {
                "reason": "this round's independent proposals contain too few distinct programs for the evaluation batch",
                "already_proposed_this_round": list(avoid)[-32:],
                "request": "propose programs that differ from these"}
        return ({"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, indent=1)})

    def _collect(self, rows, records, round_idx):
        if self.sink is None:
            return
        contract = self.case.public_contract()
        ir = make_complete_design_ir(
            task_id="researchgym", domain=f"ResearchGym {self.case.case_id} method program",
            task_description=contract["interface"] + " " + contract["scoring"],
            objectives=[{"name": self.case.metric["name"], "direction": "maximize"}],
            design_space_description="A complete Python module implementing the case entry point.",
            observations=[{"program": r["program"], "metrics": {"objective": r["objective"]}, "status": r["status"]}
                          for r in records[-8:]],
            candidates=[{"program": row["program"], "change_summary": row["change_summary"]} for row in rows],
            request_description="Propose distinct method programs using the measured results.",
            num_candidates=len(rows), round_idx=round_idx, num_evaluated=len(records), reasoning_available=False,
        )
        self.sink.append(ir, provenance={"case_id": self.case.case_id, "round_idx": round_idx,
                                         "synthetic_fixture": bool(self.args.mock)})


class MockProposalClient:
    """Deterministic synthetic proposal text; exercises the real parser and repair path."""

    def __init__(self, case: CaseSpec, *, seed: int) -> None:
        self.case, self.seed = case, seed
        self.variants = ("softmax", "topk", "normalize", "clone", "dropout", "zero_grad", "median", "entropy")

    def propose(self, request: ProposalRequest) -> ProposalResponse:
        meta = request.metadata
        flagged = set()
        if meta["repair"]:
            flagged = {int(i) for i in re.findall(r"/candidates/(\d+)", request.messages[-1]["content"])}
        rows = []
        for i in range(meta["count"]):
            # Overlapping indices across minibatches create cross-request agreement;
            # a repair replaces only the flagged entries, as a model reading the errors would.
            shift = meta["repair"] * 5 if i in flagged else 0
            k = (self.seed + meta["round_idx"] * 3 + meta["minibatch_index"] + i + shift
                 + meta.get("refill", 0) * 11) % 37
            name = self.variants[k % len(self.variants)]
            program = self.case.seed_program().rstrip() + (
                f"\n\n\ndef _ldm_mock_variant_{k}(values):\n    scale = {k + 1}\n"
                + "".join(f"    values = values.{self.variants[(k + j) % len(self.variants)]}()\n" for j in range(k % 4))
                + f"    return values if scale else values.{name}()\n")
            rows.append({"program": program, "change_summary": f"synthetic variant {k}",
                         "rationale": "synthetic fixture; not a scientific proposal"})
        if meta["repair"] == 0 and meta["round_idx"] == 0 and meta["minibatch_index"] == 0:
            rows[0] = {**rows[0], "program": "def unrelated():\n    return 1\n"}
        return ProposalResponse(text=json.dumps({"candidates": rows}), usage={"prompt_tokens": 100, "completion_tokens": 50})


class ProgramExpander:
    """Task ReservoirExpander for llm, ldm, harness, ldm_harness and ldm_harness_compiled."""

    def __init__(self, *, args, case: CaseSpec, domain, source, run_dir: Path) -> None:
        self.args, self.case, self.domain, self.source = args, case, domain, source
        self.run_dir = Path(run_dir)
        self.recovery_pass = 0

    def expand(self, request):
        ldm = self.args.search_method in LDM_METHODS
        count = request.reservoir_size if ldm else self.args.evaluations_per_round
        records = measured_records(request.observations, self.objective_name)
        measured_keys = {o.candidate.canonical_key for o in request.observations}
        rows, attempts, diagnostics = [], [], {"collections": []}
        refill = 0
        try:
            batch = self.source.collect(round_idx=request.round_idx, count=count,
                                        batch_size=self.args.proposal_batch_size if ldm else count,
                                        records=records, measured_keys=measured_keys,
                                        recovery_pass=self.recovery_pass)
            self._extend(rows, attempts, diagnostics, batch, count)
            # Agreement between independent requests can leave fewer distinct programs
            # than the evaluation batch. Ask for more occurrences instead of silently
            # evaluating fewer candidates; the extra occurrences also enter q0.
            while ldm and self._distinct(rows) < self.args.evaluations_per_round:
                if refill >= self.args.max_refill_collections:
                    raise ProposalExhausted({"reason": "distinct_programs_below_evaluation_batch",
                                             "distinct_programs": self._distinct(rows),
                                             "evaluation_batch": self.args.evaluations_per_round}, attempts=attempts)
                refill += 1
                missing = self.args.evaluations_per_round - self._distinct(rows)
                batch = self.source.collect(round_idx=request.round_idx, count=missing, batch_size=missing,
                                            records=records, measured_keys=measured_keys,
                                            recovery_pass=self.recovery_pass, refill=refill,
                                            avoid=[row["change_summary"] for row in rows])
                self._extend(rows, attempts, diagnostics, batch, missing)
        except ProposalExhausted as exc:
            exc.attempts = tuple(attempts) + tuple(a for a in exc.attempts if a not in attempts)
            exc.metadata.update(round_idx=request.round_idx, recovery_pass=self.recovery_pass)
            atomic_json_write(self.run_dir / "proposal_diagnostics" / f"round-{request.round_idx:06d}.json", exc.metadata)
            raise
        batch = ProposalBatch(rows, attempts, {**diagnostics, "replenishment_collections": refill})
        proposals = tuple(RawProposal(
            {"program": row["program"]}, "researchgym_" + self.args.search_method,
            metadata={"round_idx": request.round_idx, "proposal_index": i,
                      "research_note": {**row["note"], "change_summary": row["change_summary"],
                                        "rationale": row["rationale"],
                                        "comparison_candidate_ids": row["comparison_candidate_ids"]}},
        ) for i, row in enumerate(batch.rows))
        if ldm:
            proposals = attach_empirical_base_measure(proposals, self.domain, measured_keys=measured_keys)
        else:
            proposals = tuple(RawProposal(p.payload, p.source, metadata={
                **{k: v for k, v in p.metadata.items() if k != "research_note"},
                "research_annotations": [p.metadata["research_note"]]}) for p in proposals)
        metadata = {"round_idx": request.round_idx, "search_method": self.args.search_method,
                    "required_occurrences": count + sum(c["count"] for c in diagnostics["collections"][1:]),
                    "valid_occurrences": len(proposals),
                    "distinct_programs": len({self.case.canonical_key(p.payload["program"]) for p in proposals}),
                    "history_exclusion_count": len(measured_keys), "recovery_pass": self.recovery_pass,
                    "occurrence_rule": "distinct within one request or session; cross-request repeats count toward q0",
                    **batch.diagnostics}
        atomic_json_write(self.run_dir / "proposal_diagnostics" / f"round-{request.round_idx:06d}.json", metadata)
        return ExpansionResult(proposals=proposals, attempts=tuple(batch.attempts), metadata=metadata,
                               selection_mode="acquisition" if ldm else "reservoir_order")

    def _distinct(self, rows) -> int:
        return len({self.case.canonical_key(row["program"]) for row in rows})

    @staticmethod
    def _extend(rows, attempts, diagnostics, batch, count) -> None:
        if len(batch.rows) != count:
            raise ProposalExhausted({"reason": "source returned the wrong number of occurrences",
                                     "expected": count, "received": len(batch.rows)}, attempts=batch.attempts)
        rows.extend(batch.rows)
        attempts.extend(batch.attempts)
        diagnostics["collections"].append({"count": count, **batch.diagnostics})

    @property
    def objective_name(self) -> str:
        return "mock_score" if self.args.mock else self.case.metric["name"]


def seed_proposals(case: CaseSpec) -> tuple[RawProposal, ...]:
    """The released baseline restated for the slot; round 0 of shared_start."""
    return (RawProposal({"program": case.seed_program().strip() + "\n"}, "campaign_initialization",
                        metadata={"research_annotations": [{"source": "released_baseline_seed"}]}),)
