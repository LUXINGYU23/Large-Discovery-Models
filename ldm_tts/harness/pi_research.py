"""Pi research source for the asynchronous research lane (design 5.5, 5.8; ``ResearchSource`` protocol).

One generation = one ``run_turn`` of the harness pool (every profile). Each turn continues its session's
cursor chain: ``history_from_seq`` is the previous turn's ``history_to_seq`` (from the frozen round input,
``sessions``), ``history_to_seq`` is the round baseline ``watermarks.base_history_seq``; this replaces the
numeric example's ``history_from_seq = 0`` (design 13). The prompt carries only the fixed-size header
``turn_header(base, last_visible)`` plus bounded task messages, never the history: the agent queries the shared
history tool. Everything in a turn is a pure function of the frozen round input and the immutable log prefix,
so a recovered generation resubmits byte-identical turns.

The submission is a delta of the lineage's resident proposals: new candidates and re-proposals (renewals) in
``candidates``, optional withdrawals. The validator is ``collision_errors(view(history_seq), ...)`` plus the
task's legality check: a pure function of the submission and the recorded watermark. Accepted proposals reach
the reservoir only when the generation is ready (5.5 rule 5), through the lane's ``generation_end`` transaction.

Abandoned generations (``round_input["unsettled"]``): the controller cannot know whether the sidecar committed
their turns, and a committed turn moved the sidecar's cursor. Before its own turn, a generation settles each
session: the newest abandoned turn of the profile for which the host answered a terminal validation (recorded
before the answer was sent; without it the sidecar cannot have committed) is replayed verbatim from the host's
record of sent turns. The sidecar returns its commit record (or completes the commit from the saved submission);
the session continues from that turn. A turn without a terminal answer is never replayed, so an abandoned round
is not resumed against the operator's decision. Its proposals stay discarded.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from ldm_tts.async_runtime.history_bridge import ingest_watermark
from ldm_tts.async_runtime.lane import ResearchOutcome
from ldm_tts.async_runtime.proposals import Proposal
from ldm_tts.contracts import Candidate
from ldm_tts.harness.history import DUPLICATE_IN_SUBMISSION, INVALID, HistoryLog, collision_errors
from ldm_tts.harness.history_host import HistoryService
from ldm_tts.harness.protocol import (HarnessError, HarnessSubmissionContract, HarnessSubmissionError,
                                      HarnessSubmissionRequest, HarnessSubmissionValidation, HarnessTurn)


@dataclass(frozen=True)
class ProposalContract:
    """Task declaration of the proposal submission: the strict tool schema, the candidate legality check
    (``admit(raw) -> Candidate | None``), the count range and the optional withdrawal list."""

    submission: HarnessSubmissionContract
    admit: Callable[[Any], Candidate | None]
    count: tuple[int, int] = (1, 8)
    candidates_field: str = "candidates"
    withdraw_field: str | None = None

    def items(self, submission: Mapping[str, Any], field: str | None) -> list:
        value = submission.get(field) if field else None
        return list(value) if isinstance(value, list) else []


def proposal_validator(contract: ProposalContract, log: HistoryLog, *, generation: int
                       ) -> Callable[[HarnessSubmissionRequest], HarnessSubmissionValidation]:
    """Pure function of ``(submission, history_seq)``: recovery reruns it with the recorded watermark."""

    def validate(request: HarnessSubmissionRequest) -> HarnessSubmissionValidation:
        if request.history_seq is None:
            raise HarnessError("proposal validation requires the recorded history watermark")
        candidates = contract.items(request.submission, contract.candidates_field)
        keys = [None if c is None else c.canonical_key for c in map(contract.admit, candidates)]
        errors = list(collision_errors(log.view(request.history_seq), keys, pointer=f"/{contract.candidates_field}",
                                       count=contract.count))
        withdrawals = contract.items(request.submission, contract.withdraw_field)
        for index, item in enumerate(withdrawals):
            admitted = contract.admit(item)
            path = f"/{contract.withdraw_field}/{index}"
            if admitted is None:
                errors.append(HarnessSubmissionError(path, INVALID, "The candidate is not legal for this task."))
            elif admitted.canonical_key in keys:
                errors.append(HarnessSubmissionError(path, DUPLICATE_IN_SUBMISSION,
                                                     "The candidate is both proposed and withdrawn."))
        if errors:
            return HarnessSubmissionValidation("retry", tuple(errors))
        receipt = f"{len(keys)} proposals recorded for generation {generation}"
        return HarnessSubmissionValidation(receipt=receipt + (f", {len(withdrawals)} withdrawn." if withdrawals else "."))

    return validate


def _summary(round_input: Mapping[str, Any]) -> dict[str, Any]:
    """Fixed-size campaign context for the turn header (budget, the previous generation's fate)."""
    budget = round_input.get("budget") or {}
    previous = round_input.get("previous") or None
    return {"generation": round_input["generation"],
            "budget": {k: budget[k] for k in ("limit", "free", "consumed", "held") if k in budget},
            "previous_generation": None if previous is None else {
                k: previous.get(k) for k in ("generation", "state", "proposals", "admitted", "reason")}}


class PiResearchSource:
    """``ResearchSource`` over a history-enabled ``HarnessClient`` and the campaign's ``HistoryService``."""

    version = "1"

    def __init__(self, client, service: HistoryService, contract: ProposalContract, *, instructions: str,
                 name: str = "pi-research", recovery_timeout_seconds: float = 0.0,
                 lineage_of: Callable[[str], str] | None = None):
        if client.history is not service:
            raise ValueError("the research client must answer history requests from the campaign's service")
        self.client, self.service, self.contract = client, service, contract
        self.instructions, self.name = instructions.strip(), name
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.lineage_of = lineage_of or service.lineage_of
        self.profiles = tuple(profile.profile_id for profile in client.config.profiles)

    # ---- the frozen turn -------------------------------------------------------------------------------

    def turns(self, round_input: Mapping[str, Any], sessions: Mapping[str, Any] | None = None
              ) -> tuple[HarnessTurn, ...]:
        generation = int(round_input["generation"])
        base = round_input["watermarks"].get("base_history_seq")
        if base is None:
            raise ValueError("the round input has no base_history_seq (is the history bridge attached?)")
        sessions = (round_input.get("sessions") or {}) if sessions is None else sessions
        turns = []
        for profile in self.profiles:
            state = sessions.get(profile) or {}
            from_seq = int(state.get("history_to_seq", 0))
            last_visible = min(int(state.get("visible_seq", from_seq)), base)
            start, to_seq, digest = self.service.turn_window(from_seq, base)
            header = self.service.engine.turn_header(to_seq, last_visible, extra=_summary(round_input))
            turns.append(HarnessTurn(profile, f"g{generation:06d}-{profile}", generation, start, to_seq, digest,
                                     self.message(round_input, header)))
        return tuple(turns)

    def message(self, round_input: Mapping[str, Any], header: Mapping[str, Any]) -> str:
        tool = self.service.config.tool_name
        lines = [self.instructions, "",
                 f"Turn header (fixed size; the experiment history is not listed here, query it with {tool}):",
                 json.dumps(header, sort_keys=True, separators=(", ", ": "))]
        if round_input.get("note"):
            lines += ["", f"Note from the campaign: {round_input['note']}"]
        messages = round_input.get("messages") or ()
        if messages:
            lines += ["", "Task messages (hints from the task, not measurements):"]
            lines += [f"- [{m['source']}] {m['text']}" for m in messages]
        withdraw = self.contract.withdraw_field
        lines += ["", f"Submit with {self.contract.submission.tool_name}. A proposal is not an experiment: the "
                      "campaign's selection decides whether and when any proposal is measured. Proposing again a "
                      "candidate of yours that is still in the reservoir renews it"
                      + (f"; list candidates to withdraw in {withdraw}" if withdraw else "") + "."]
        return "\n".join(lines)

    # ---- abandoned generations -------------------------------------------------------------------------

    def settle(self, round_input: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """``(sessions, settled)``: the session states this generation continues from, after replaying the newest
        abandoned turn per profile that may have committed on the sidecar (see the module docstring)."""
        sessions = {profile: dict(state) for profile, state in (round_input.get("sessions") or {}).items()}
        replays = []
        for profile in self.profiles:
            for generation in sorted(round_input.get("unsettled") or (), reverse=True):
                turn_id = f"g{int(generation):06d}-{profile}"
                sent = self.service.sent_turn(profile, turn_id)
                if sent is not None and self.service.terminal_decision(profile, turn_id) is not None:
                    replays.append(HarnessTurn(profile, turn_id, int(sent["roundIndex"]), int(sent["historyFromSeq"]),
                                               int(sent["historyToSeq"]), sent["historyDigest"], sent["message"],
                                               tuple(sent.get("forbiddenQueryTerms") or ())))
                    break
        if not replays:
            return sessions, {}
        generations = {turn.turn_id: turn.round_index for turn in replays}

        def validate(request):  # only reached if the sidecar never saved the accepted submission (it resumes)
            return proposal_validator(self.contract, self.service.log, generation=generations[request.turn_id])(request)

        settled = {}
        for result in self.client.run_turn(tuple(replays), submission_validator=validate,
                                           recovery_timeout_seconds=self.recovery_timeout_seconds):
            sessions[result.profile_id] = {"generation": result.round_index, "turn_id": result.turn_id,
                                           "history_to_seq": result.history_to_seq, "visible_seq": result.visible_seq}
            settled[result.profile_id] = {**sessions[result.profile_id], "replayed": result.replayed,
                                          "submission_status": result.submission_status}
        return sessions, settled

    # ---- one generation --------------------------------------------------------------------------------

    def research(self, round_input: Mapping[str, Any], history) -> ResearchOutcome:
        generation = int(round_input["generation"])
        log = self.service.log
        sessions, settled = self.settle(round_input)
        turns = self.turns(round_input, sessions)
        for turn in turns:
            self.service.turn_sent(turn)
        results = self.client.run_turn(
            turns, submission_validator=proposal_validator(self.contract, log, generation=generation),
            recovery_timeout_seconds=self.recovery_timeout_seconds)
        proposals, withdrawals, sessions, delivered = [], [], {}, None
        for result in results:
            history_seq = self.service.accepted_history_seq(result.profile_id, result.turn_id)
            sessions[result.profile_id] = {
                "turn_id": result.turn_id, "history_from_seq": result.history_from_seq,
                "history_to_seq": result.history_to_seq, "visible_seq": result.visible_seq,
                "history_seq": history_seq, "submission_status": result.submission_status,
                "replayed": result.replayed, "usage": dict(result.usage)}
            delivered = set(result.task_messages) if delivered is None else delivered & set(result.task_messages)
            if result.submission_status != "accepted":
                continue
            lineage = self.lineage_of(result.profile_id)
            seq = result.visible_seq if history_seq is None else history_seq
            watermark = ingest_watermark(log, min(seq, log.head))
            for item in self.contract.items(result.submission, self.contract.candidates_field):
                candidate = self.contract.admit(item)
                if candidate is not None:
                    tagged = replace(candidate, metadata={**candidate.metadata, "generation": generation})
                    proposals.append(Proposal(tagged, lineage, watermark))
            for item in self.contract.items(result.submission, self.contract.withdraw_field):
                candidate = self.contract.admit(item)
                if candidate is not None:
                    withdrawals.append((lineage, candidate.canonical_key))
        accepted = any(r.submission_status == "accepted" for r in results)
        metadata = {"sessions": sessions, "task_messages": sorted(delivered or ()),
                    "base_history_seq": round_input["watermarks"]["base_history_seq"]}
        if settled:
            metadata["settled"] = settled
        reason = "" if accepted else "every submission was rejected: " + ", ".join(sorted(
            {error.code for r in results for error in r.validation_errors}))
        return ResearchOutcome("committed" if accepted else "rejected", tuple(proposals), tuple(withdrawals),
                               reason, min(ingest_watermark(log, r.visible_seq) for r in results), metadata)


__all__ = ["PiResearchSource", "ProposalContract", "proposal_validator"]
