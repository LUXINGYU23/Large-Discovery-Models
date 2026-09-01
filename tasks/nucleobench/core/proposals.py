"""Task-local proposal utilities shared by BO and LDM methods."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

from ldm_tts.contracts import RawProposal
from ldm_tts.engine.expansion import ExpansionRequest, ExpansionResult
from ldm_tts.transport import ProposalClient, ProposalRequest, ProposalResponse
from ldm_tts.transport.openai import (
    WIRE_APIS,
    EndpointCircuitBreaker,
    OpenAICompatibleProposalClient,
)
from tasks.nucleobench.core.candidate import (
    CandidatePayloadError,
    MutationContext,
    NucleoBenchCandidateDomain,
    prepare_candidate_payload,
)
from tasks.nucleobench.core.constants import NUCLEOBENCH_Q0_METADATA_KEY
from tasks.nucleobench.core.prompting import (
    build_mutation_prompt_messages,
    prompt_sha256,
)

DIRECT_LDM_REQUEST_COUNT = 4
DEFAULT_PROPOSAL_MAX_WORKERS = 4
DEFAULT_PROPOSAL_REQUEST_WAVES = 4
PROPOSAL_SOURCE = "nucleobench_direct_model"
TRANSIENT_MAX_RETRIES = 3
TRANSIENT_RETRY_BACKOFF_SECONDS = 10.0
TRANSIENT_CIRCUIT_FAILURE_THRESHOLD = 32


class ProposalResponseError(ValueError):
    """A strict response error with a stable refill reason."""

    def __init__(
        self,
        reason: str,
        message: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.metadata = dict(metadata or {})


class ProposalGenerationError(RuntimeError):
    """Raised after the configured request waves cannot fill the reservoir."""

    def __init__(
        self,
        message: str,
        *,
        attempts: Sequence[ProposalResponse],
        metadata: Mapping[str, Any],
    ) -> None:
        super().__init__(message)
        self.attempts = tuple(attempts)
        self.metadata = dict(metadata)


@dataclass(frozen=True)
class ParsedMutationResponse:
    proposals: tuple[tuple[int, dict[str, Any]], ...]
    errors: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class _RequestSpec:
    lineage_index: int
    candidate_count: int
    wave_index: int
    request_id: str


class DirectMutationProposalExpander:
    """Generate fixed direct-LLM or oversampled LDM mutation occurrences."""

    def __init__(
        self,
        client: ProposalClient,
        domain: NucleoBenchCandidateDomain,
        *,
        search_method: str,
        evaluations_per_round: int,
        seed: int = 0,
        max_workers: int = DEFAULT_PROPOSAL_MAX_WORKERS,
        max_request_waves: int = DEFAULT_PROPOSAL_REQUEST_WAVES,
        before_requests: Callable[[int], None] | None = None,
    ) -> None:
        if search_method not in {"ldm", "llm"}:
            raise ValueError(
                "direct mutation proposals require search_method='ldm' or 'llm'"
            )
        if evaluations_per_round < 1:
            raise ValueError("evaluations_per_round must be positive")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        if max_workers < 1 or max_request_waves < 1:
            raise ValueError("proposal workers and request waves must be positive")
        self.client = client
        self.domain = domain
        self.search_method = search_method
        self.evaluations_per_round = evaluations_per_round
        self.seed = seed
        self.max_workers = max_workers
        self.max_request_waves = max_request_waves
        self.before_requests = before_requests

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        targets = self._lineage_targets()
        target_count = sum(targets)
        if request.reservoir_size != target_count:
            raise ValueError(
                f"{self.search_method} requires reservoir_size={target_count}, "
                f"got {request.reservoir_size}"
            )

        evaluated = {item.canonical_key for item in request.observations}
        accepted: list[list[RawProposal]] = [[] for _ in targets]
        accepted_direct_keys: set[str] = set()
        attempts: list[ProposalResponse] = []
        errors: list[dict[str, Any]] = []
        request_count = 0

        for wave_index in range(self.max_request_waves):
            specs = tuple(
                self._request_spec(
                    request, lineage, target - len(accepted[lineage]), wave_index
                )
                for lineage, target in enumerate(targets)
                if len(accepted[lineage]) < target
            )
            if not specs:
                break
            proposal_requests = tuple(
                self._proposal_request(
                    request,
                    spec,
                    additional_exclusions=(
                        tuple(
                            item.payload
                            for lineage_items in accepted
                            for item in lineage_items
                        )
                        if self.search_method == "llm"
                        else ()
                    ),
                )
                for spec in specs
            )
            if self.before_requests is not None:
                self.before_requests(len(proposal_requests))
            responses = self._propose_all(proposal_requests)
            attempts.extend(responses)
            request_count += len(responses)

            for spec, proposal_request, response in zip(
                specs, proposal_requests, responses, strict=True
            ):
                try:
                    parsed = parse_mutation_response(
                        response.text,
                        expected_count=spec.candidate_count,
                    )
                except ProposalResponseError as exc:
                    errors.append(
                        _rejection(exc.reason, str(exc), spec, **exc.metadata)
                    )
                    continue
                errors.extend(
                    {
                        **item,
                        "request_id": spec.request_id,
                        "lineage_index": spec.lineage_index,
                        "wave_index": spec.wave_index,
                    }
                    for item in parsed.errors
                )
                for candidate_index, payload in parsed.proposals:
                    try:
                        prepared = prepare_candidate_payload(
                            payload, self.domain.context
                        )
                    except CandidatePayloadError as exc:
                        errors.append(
                            _rejection(
                                exc.reason,
                                str(exc),
                                spec,
                                candidate_index=candidate_index,
                                **exc.metadata,
                            )
                        )
                        continue
                    if prepared.canonical_key in evaluated:
                        errors.append(
                            _rejection(
                                "historical_duplicate",
                                "Candidate is already present in evaluated history.",
                                spec,
                                candidate_index=candidate_index,
                                canonical_key=prepared.canonical_key,
                            )
                        )
                        continue
                    if (
                        self.search_method == "llm"
                        and prepared.canonical_key in accepted_direct_keys
                    ):
                        errors.append(
                            _rejection(
                                "same_round_duplicate",
                                "Direct evaluation already accepted this candidate this round.",
                                spec,
                                candidate_index=candidate_index,
                                canonical_key=prepared.canonical_key,
                            )
                        )
                        continue
                    accepted[spec.lineage_index].append(
                        RawProposal(
                            prepared.payload,
                            PROPOSAL_SOURCE,
                            {
                                "collectable": True,
                                "round_idx": request.round_idx,
                                "request_id": spec.request_id,
                                "lineage_index": spec.lineage_index,
                                "seed_lineage": f"{self.seed}:{spec.lineage_index}",
                                "wave_index": spec.wave_index,
                                "candidate_index": candidate_index,
                                "prompt_sha256": proposal_request.metadata[
                                    "prompt_sha256"
                                ],
                            },
                        )
                    )
                    if self.search_method == "llm":
                        accepted_direct_keys.add(prepared.canonical_key)

        missing = sum(
            target - len(accepted[index]) for index, target in enumerate(targets)
        )
        metadata = _expansion_metadata(
            request,
            search_method=self.search_method,
            target_count=target_count,
            request_count=request_count,
            max_workers=self.max_workers,
            max_request_waves=self.max_request_waves,
            errors=errors,
            missing_count=missing,
        )
        if missing:
            raise ProposalGenerationError(
                f"Direct proposal generation exhausted {self.max_request_waves} request "
                f"waves with {missing} of {target_count} occurrences still missing.",
                attempts=attempts,
                metadata=metadata,
            )

        proposals = tuple(
            replace(
                item,
                metadata={**item.metadata, "proposal_index": proposal_index},
            )
            for proposal_index, item in enumerate(
                item for lineage_items in accepted for item in lineage_items
            )
        )
        if self.search_method == "ldm":
            proposals = attach_empirical_base_measure(proposals, request, self.domain)
        return ExpansionResult(
            proposals=proposals,
            attempts=tuple(attempts),
            metadata=metadata,
            selection_mode=(
                "reservoir_order" if self.search_method == "llm" else "acquisition"
            ),
        )

    def _lineage_targets(self) -> tuple[int, ...]:
        if self.search_method == "ldm":
            return (self.evaluations_per_round,) * DIRECT_LDM_REQUEST_COUNT
        return (1,) * self.evaluations_per_round

    def _request_spec(
        self,
        request: ExpansionRequest,
        lineage_index: int,
        candidate_count: int,
        wave_index: int,
    ) -> _RequestSpec:
        request_id = (
            f"nucleobench-r{request.round_idx:04d}-s{self.seed:04d}-"
            f"l{lineage_index:03d}-w{wave_index:02d}"
        )
        return _RequestSpec(lineage_index, candidate_count, wave_index, request_id)

    def _proposal_request(
        self,
        request: ExpansionRequest,
        spec: _RequestSpec,
        *,
        additional_exclusions: Sequence[Mapping[str, Any]],
    ) -> ProposalRequest:
        messages = build_mutation_prompt_messages(
            request,
            self.domain.context,
            candidate_count=spec.candidate_count,
            request_id=spec.request_id,
            lineage_index=self.seed * 10_000 + spec.lineage_index,
            wave_index=spec.wave_index,
            same_round_agreement_allowed=self.search_method == "ldm",
            additional_exclusions=additional_exclusions,
        )
        return ProposalRequest(
            messages=messages,
            metadata={
                "round_idx": request.round_idx,
                "request_id": spec.request_id,
                "lineage_index": spec.lineage_index,
                "seed_lineage": f"{self.seed}:{spec.lineage_index}",
                "wave_index": spec.wave_index,
                "candidate_count": spec.candidate_count,
                "sampling_mode": (
                    "independent_minibatch_requests"
                    if self.search_method == "ldm"
                    else "independent_single_candidate_requests"
                ),
                "prompt_sha256": prompt_sha256(messages),
            },
        )

    def _propose_all(
        self,
        requests: Sequence[ProposalRequest],
    ) -> tuple[ProposalResponse, ...]:
        if len(requests) == 1:
            return tuple(self.client.propose(item) for item in requests)
        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(requests))
        ) as executor:
            futures = [executor.submit(self.client.propose, item) for item in requests]
            return tuple(future.result() for future in futures)


def parse_mutation_response(
    text: str,
    *,
    expected_count: int,
) -> ParsedMutationResponse:
    """Parse one complete strict JSON response while preserving valid batch peers."""

    if expected_count < 1:
        raise ValueError("expected_count must be positive")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProposalResponseError(
            "malformed_json",
            f"Response is not one complete JSON value: {exc.msg}.",
        ) from exc
    if not isinstance(payload, Mapping):
        raise ProposalResponseError(
            "invalid_response_root",
            "Response root must be a JSON object.",
        )
    if expected_count == 1:
        _require_response_fields(payload, {"mutations"}, "candidate")
        return ParsedMutationResponse(((0, {"mutations": payload["mutations"]}),))

    _require_response_fields(payload, {"candidates"}, "response")
    candidates = payload["candidates"]
    if not isinstance(candidates, list) or len(candidates) != expected_count:
        received = len(candidates) if isinstance(candidates, list) else None
        raise ProposalResponseError(
            "wrong_cardinality",
            f"Response must contain exactly {expected_count} candidates.",
            metadata={"expected_count": expected_count, "received_count": received},
        )

    parsed: dict[int, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    for response_index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            errors.append(
                {
                    "reason": "invalid_candidate_object",
                    "message": "Batch candidate must be a JSON object.",
                    "candidate_index": response_index,
                }
            )
            continue
        try:
            _require_response_fields(
                candidate,
                {"proposal_index", "mutations"},
                "batch candidate",
            )
        except ProposalResponseError as exc:
            errors.append(
                {
                    "reason": exc.reason,
                    "message": str(exc),
                    "candidate_index": response_index,
                    **exc.metadata,
                }
            )
            continue
        proposal_index = candidate["proposal_index"]
        if (
            isinstance(proposal_index, bool)
            or not isinstance(proposal_index, int)
            or not 0 <= proposal_index < expected_count
        ):
            errors.append(
                {
                    "reason": "invalid_proposal_index",
                    "message": "proposal_index is outside the requested batch.",
                    "candidate_index": response_index,
                    "proposal_index": proposal_index,
                }
            )
            continue
        if proposal_index in parsed:
            errors.append(
                {
                    "reason": "duplicate_proposal_index",
                    "message": "proposal_index appears more than once in the response.",
                    "candidate_index": response_index,
                    "proposal_index": proposal_index,
                }
            )
            continue
        parsed[proposal_index] = {"mutations": candidate["mutations"]}
    return ParsedMutationResponse(tuple(sorted(parsed.items())), tuple(errors))


def build_openai_mutation_client(
    *,
    base_url: str,
    model: str,
    api_key: str,
    wire_api: str,
    timeout_seconds: float,
    max_tokens: int,
    temperature: float,
    reasoning: str = "off",
    json_mode: bool = True,
    extra_body: Mapping[str, Any] | None = None,
) -> OpenAICompatibleProposalClient:
    """Build the shared configurable OpenAI-compatible proposal transport."""

    if wire_api not in WIRE_APIS:
        raise ValueError(f"wire_api must be one of {WIRE_APIS}")
    body = dict(extra_body or {})
    if reasoning != "off":
        _set_request_option(
            body,
            "reasoning" if wire_api == "responses" else "reasoning_effort",
            {"effort": reasoning} if wire_api == "responses" else reasoning,
        )
    if json_mode:
        _set_request_option(
            body,
            "text" if wire_api == "responses" else "response_format",
            (
                {"format": {"type": "json_object"}}
                if wire_api == "responses"
                else {"type": "json_object"}
            ),
        )
    return OpenAICompatibleProposalClient(
        url=base_url,
        model=model,
        api_key=api_key,
        wire_api=wire_api,
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
        temperature=temperature,
        max_retries=TRANSIENT_MAX_RETRIES,
        retry_backoff_seconds=TRANSIENT_RETRY_BACKOFF_SECONDS,
        extra_body=body,
        require_models_preflight=False,
        breaker=EndpointCircuitBreaker(
            failure_threshold=TRANSIENT_CIRCUIT_FAILURE_THRESHOLD,
        ),
    )


def _set_request_option(body: dict[str, Any], name: str, value: Any) -> None:
    if name in body:
        raise ValueError(f"{name} is configured both explicitly and in extra_body")
    body[name] = value


def _require_response_fields(
    payload: Mapping[Any, Any],
    expected: set[str],
    label: str,
) -> None:
    unexpected = sorted(str(item) for item in set(payload).difference(expected))
    if unexpected:
        raise ProposalResponseError(
            "unexpected_response_fields",
            f"{label} contains unexpected field(s): {', '.join(unexpected)}.",
        )
    missing = sorted(expected.difference(payload))
    if missing:
        raise ProposalResponseError(
            "missing_response_fields",
            f"{label} is missing required field(s): {', '.join(missing)}.",
        )


def _rejection(
    reason: str,
    message: str,
    spec: _RequestSpec,
    **metadata: Any,
) -> dict[str, Any]:
    return {
        "reason": reason,
        "message": message,
        "request_id": spec.request_id,
        "lineage_index": spec.lineage_index,
        "wave_index": spec.wave_index,
        **metadata,
    }


def _expansion_metadata(
    request: ExpansionRequest,
    *,
    search_method: str,
    target_count: int,
    request_count: int,
    max_workers: int,
    max_request_waves: int,
    errors: Sequence[Mapping[str, Any]],
    missing_count: int,
) -> dict[str, Any]:
    reason_counts = Counter(str(item["reason"]) for item in errors)
    return {
        "mode": "direct_model_proposals",
        "search_method": search_method,
        "sampling_mode": (
            "independent_minibatch_requests"
            if search_method == "ldm"
            else "independent_single_candidate_requests"
        ),
        "round_idx": request.round_idx,
        "target_occurrence_count": target_count,
        "accepted_occurrence_count": target_count - missing_count,
        "missing_occurrence_count": missing_count,
        "request_count": request_count,
        "max_workers": max_workers,
        "max_request_waves": max_request_waves,
        "rejection_counts": dict(reason_counts),
        "rejections": [dict(item) for item in errors],
    }


def attach_empirical_base_measure(
    proposals: tuple[RawProposal, ...],
    request: ExpansionRequest,
    domain: NucleoBenchCandidateDomain,
) -> tuple[RawProposal, ...]:
    """Attach q0 before shared reservoir deduplication."""

    evaluated = {item.canonical_key for item in request.observations}
    keys: dict[int, str] = {}
    counts: Counter[str] = Counter()
    for index, proposal in enumerate(proposals):
        try:
            prepared = prepare_candidate_payload(proposal.payload, domain.context)
        except CandidatePayloadError:
            continue
        if prepared.canonical_key in evaluated:
            continue
        keys[index] = prepared.canonical_key
        counts[prepared.canonical_key] += 1
    total = sum(counts.values())
    return tuple(
        _annotate_q0(proposal, keys.get(index), counts, total)
        for index, proposal in enumerate(proposals)
    )


class ScoreBlindMutationPoolExpander:
    """Generate a deterministic finite BO pool around start and incumbent."""

    def __init__(
        self,
        context: MutationContext,
        *,
        seed: int,
        distance_steps: tuple[int, ...] = (1, 2, 4, 8),
    ) -> None:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.context = context
        self.seed = seed
        self.distance_steps = tuple(
            dict.fromkeys(
                min(step, len(context.editable_positions))
                for step in distance_steps
                if step > 0
            )
        )
        if not self.distance_steps:
            raise ValueError("distance_steps must contain a positive distance")

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        parents = [_editable_bases(self.context, {"mutations": []})]
        if request.parent is not None:
            incumbent = _editable_bases(self.context, request.parent.payload)
            if incumbent != parents[0]:
                parents.append(incumbent)
        evaluated = {item.canonical_key for item in request.observations}
        rng = random.Random(_pool_seed(self.seed, request, evaluated))
        proposals: list[RawProposal] = []
        seen: set[str] = set()
        max_attempts = max(256, request.reservoir_size * 128)
        for attempt in range(max_attempts):
            parent = parents[attempt % len(parents)]
            distance = self.distance_steps[
                (attempt // len(parents)) % len(self.distance_steps)
            ]
            child = list(parent)
            for offset in rng.sample(range(len(child)), distance):
                child[offset] = rng.choice(
                    tuple(base for base in "ACGT" if base != child[offset])
                )
            payload = _patch_from_editable_bases(self.context, child)
            if not payload["mutations"]:
                continue
            prepared = prepare_candidate_payload(payload, self.context)
            if prepared.canonical_key in evaluated or prepared.canonical_key in seen:
                continue
            seen.add(prepared.canonical_key)
            proposals.append(
                RawProposal(
                    prepared.payload,
                    "nucleobench_score_blind_bo",
                    {"round_idx": request.round_idx},
                )
            )
            if len(proposals) == request.reservoir_size:
                break
        if len(proposals) != request.reservoir_size:
            raise ValueError(
                "score-blind mutation search could not fill the requested unseen pool"
            )
        return ExpansionResult(
            tuple(proposals),
            metadata={
                "mode": "score_blind_mutation_pool",
                "distance_steps": list(self.distance_steps),
                "parent_count": len(parents),
            },
        )


def _annotate_q0(
    proposal: RawProposal,
    canonical_key: str | None,
    counts: Mapping[str, int],
    total: int,
) -> RawProposal:
    if canonical_key is None:
        return proposal
    occurrence_count = counts[canonical_key]
    return replace(
        proposal,
        metadata={
            **proposal.metadata,
            NUCLEOBENCH_Q0_METADATA_KEY: {
                "occurrence_count": occurrence_count,
                "valid_occurrence_count": total,
                "probability": occurrence_count / total,
            },
        },
    )


def _editable_bases(context: MutationContext, payload: object) -> tuple[str, ...]:
    prepared = prepare_candidate_payload(payload, context, allow_empty=True)
    mutations = {
        int(item["position"]): str(item["base"])
        for item in prepared.payload["mutations"]
    }
    return tuple(
        mutations.get(position, context.start_sequence[position])
        for position in context.editable_positions
    )


def _patch_from_editable_bases(
    context: MutationContext,
    bases: list[str],
) -> dict[str, list[dict[str, object]]]:
    return {
        "mutations": [
            {"position": position, "base": base}
            for position, base in zip(context.editable_positions, bases, strict=True)
            if base != context.start_sequence[position]
        ]
    }


def _pool_seed(seed: int, request: ExpansionRequest, evaluated: set[str]) -> int:
    payload = json.dumps(
        {
            "seed": seed,
            "round_idx": request.round_idx,
            "evaluated": sorted(evaluated),
            "parent": None if request.parent is None else request.parent.canonical_key,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


__all__ = [
    "DEFAULT_PROPOSAL_MAX_WORKERS",
    "DEFAULT_PROPOSAL_REQUEST_WAVES",
    "DIRECT_LDM_REQUEST_COUNT",
    "DirectMutationProposalExpander",
    "ParsedMutationResponse",
    "ProposalGenerationError",
    "ProposalResponseError",
    "ScoreBlindMutationPoolExpander",
    "attach_empirical_base_measure",
    "build_openai_mutation_client",
    "parse_mutation_response",
]
