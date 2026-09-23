"""Host-owned accounting for AlphaBench research Harness turns."""

import hashlib
import json
from pathlib import Path

from ldm_tts.contracts import Candidate, RawProposal
from ldm_tts.contracts.evaluation import EvaluationPaused
from ldm_tts.data.ir import make_complete_design_ir
from ldm_tts.engine.expansion import ExpansionResult
from ldm_tts.harness import (HarnessArtifactRule, HarnessError, HarnessProviderAuthorizationRequest,
    HarnessSubmissionContract, HarnessSubmissionError, HarnessSubmissionValidation,
    HarnessTurn, canonical_sha256)
from .candidate import FactorDomain
from .protocol import digest
from .receipts import Receipts
from .harness_tools import check_position, public_observation
from .grammar import REGISTRY


class HarnessMeter:
    def __init__(self, runtime, host):
        self.runtime = runtime
        self.host = host
        self.receipts = Receipts(runtime.run_dir / "private" / "harness" / "provider")

    def reserve_turns(self, turn_ids):
        groups = {"engine:proposal_attempt:" + turn_id: {"harness_turns": 1, "proposal_attempts": 1}
                  for turn_id in turn_ids}
        if len(groups) != len(turn_ids):
            raise ValueError("Harness turn IDs must be unique")
        self.host.call(lambda: self.runtime.consume_groups(groups))

    def authorize(self, request: HarnessProviderAuthorizationRequest):
        if request.campaign_id != self.runtime.run_id:
            raise ValueError("Harness provider request belongs to a different campaign")
        identity = {"campaign_id": request.campaign_id, "profile_id": request.profile_id,
                    "turn_id": request.turn_id, "provider_request_id": request.provider_request_id}
        record = {**identity, "request_digest": request.request_digest, "authorized": True}
        key = "task:harness:provider:" + digest(identity)

        def reserve():
            previous = self.receipts.load(identity)
            if previous is not None:
                if previous != record:
                    raise ValueError("Harness provider request identity was reused with a different digest")
                raise EvaluationPaused("Harness provider request ID was already authorized", status="paused_provider")
            if key in self.runtime.budget.metadata.get("cumulative_usage", {}):
                raise EvaluationPaused("Harness provider authorization requires reconciliation", status="paused_provider")
            self.runtime.consume_many({"model_requests": 1}, usage_key=key)
            self.receipts.accept(identity, record)
            return True

        return self.host.call(reserve)

    def reconcile(self, result):
        def verify():
            count = 0
            for path in self.receipts.root.glob("*.json"):
                record = json.loads(path.read_text(encoding="utf-8"))
                identity = {key: record[key] for key in ("campaign_id", "profile_id", "turn_id", "provider_request_id")}
                if (path != self.receipts.path(identity) or record.get("authorized") is not True
                        or not isinstance(record.get("request_digest"), str) or len(record["request_digest"]) != 64
                        or record.get("campaign_id") != self.runtime.run_id):
                    raise EvaluationPaused("Harness provider receipt integrity failure", status="paused_provider")
                if record["turn_id"] == result.turn_id and record["profile_id"] == result.profile_id:
                    count += 1
            used = result.usage.get("providerCalls")
            if type(used) is not int or used < 0 or used > count:
                raise EvaluationPaused("Harness provider usage exceeds Host authorizations", status="paused_provider")
            return {"host_authorizations": count, "sidecar_provider_calls": used}

        return self.host.call(verify)


def submission_contract(max_candidates):
    if max_candidates < 1:
        raise ValueError("Harness candidate count must be positive")
    return HarnessSubmissionContract(
        contract_id="alphabench_factor_batch", tool_name="submit_candidates",
        payload_schema={"type": "object", "properties": {"artifact_path": {
            "type": "string", "const": "candidates.json",
            "description": "Workspace-relative JSON file with the exact candidate_count from the current turn; each candidate has name and expression.",
        }}, "required": ["artifact_path"], "additionalProperties": False},
        artifact_rules=(HarnessArtifactRule("/artifact_path", (".json",), max_candidates * 8192),),
    )


def _candidate_file(submission, artifacts, artifact_root, count):
    if dict(submission) != {"artifact_path": "candidates.json"} or len(artifacts) != 1:
        raise ValueError("submit the candidates.json artifact path")
    artifact = artifacts[0]
    if artifact.path_pointer != "/artifact_path" or artifact.relative_path != "candidates.json":
        raise ValueError("submission must snapshot candidates.json")
    root = Path(artifact_root).resolve()
    path = (root / artifact.snapshot_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("candidate snapshot is unavailable inside the artifact root")
    body = path.read_bytes()
    if len(body) != artifact.size_bytes or hashlib.sha256(body).hexdigest() != artifact.sha256:
        raise ValueError("candidate snapshot differs from its recorded hash or size")
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"candidates"} or not isinstance(payload["candidates"], list):
        raise ValueError("candidates.json must contain only a candidates array")
    if len(payload["candidates"]) != count:
        raise ValueError(f"expected exactly {count} candidates, found {len(payload['candidates'])}")
    return payload["candidates"]


class HarnessExpander:
    def __init__(self, client, protocol, gateway, artifact_root, *, tools=None, collection=None):
        self.client, self.protocol, self.gateway = client, protocol, gateway
        self.artifact_root = Path(artifact_root).resolve()
        self.domain = FactorDomain(protocol.backend, protocol.grammar_depth)
        self.meter = HarnessMeter(gateway.runtime, gateway.host)
        self.accepted = Receipts(gateway.runtime.run_dir / "private" / "harness" / "accepted")
        self.tools = tools
        self.collection = collection
        self.profiles = client.config.profiles
        expected = 1 if protocol.method == "harness" else protocol.sessions
        if len(self.profiles) != expected:
            raise ValueError("Harness session count differs from the frozen protocol")

    def expand(self, request):
        snapshot = self.tools.prepare(request) if self.tools else None
        try:
            return self._expand(request, snapshot)
        finally:
            if self.tools:
                self.tools.clear()

    def _expand(self, request, snapshot):
        direct = self.protocol.method == "harness"
        count = request.context["evaluation_budget"]["effective"] if direct else self.protocol.candidates_per_session
        expected = self.protocol.batch_size if direct else len(self.profiles) * count
        if request.reservoir_size != expected or count < 1:
            raise ValueError("Harness reservoir size differs from the frozen method")
        history = [public_observation(item) for item in request.observations]
        delta = history if request.round_idx == 0 else [item for item in history if item["round_idx"] == request.round_idx - 1]
        from_seq = len(history) - len(delta)
        history_digest = canonical_sha256(delta)
        evaluated = {item.canonical_key for item in request.observations}
        turns = tuple(HarnessTurn(
            profile_id=profile.profile_id,
            turn_id=f"round_{request.round_idx:04d}_{profile.profile_id}_" + canonical_sha256({
                "run": self.gateway.runtime.run_id, "profile_set": self.client.config.profile_set_sha256,
                "profile": profile.profile_id, "round": request.round_idx, "history": history_digest,
            })[:16],
            round_index=request.round_idx, history_from_seq=from_seq,
            history_to_seq=len(history), history_digest=history_digest,
            message=json.dumps({"backend": self.protocol.backend, "market": self.protocol.market,
                "objective": self.protocol.objective, "candidate_count": count,
                "history_delta": delta, "history_digest": history_digest,
                "evaluated_count": len(history), "filter_profile": self.protocol.filter_profile,
                **({"research_snapshot": snapshot} if snapshot else {})}, sort_keys=True),
            forbidden_query_terms=tuple(sorted({item.candidate_id for item in request.observations})),
        ) for profile in self.profiles)
        self.meter.reserve_turns([turn.turn_id for turn in turns])
        try:
            results = self.client.run_turn(turns,
                submission_validator=lambda submission: self._validate(submission, count, evaluated, request.round_idx),
                provider_authorizer=self.meter.authorize,
                recovery_timeout_seconds=2 * self.client.config.limits.wall_time_seconds)
        except HarnessError as exc:
            if self.tools and self.tools.failure:
                raise self.tools.failure from exc
            raise EvaluationPaused("Harness turn requires recovery: " + str(exc), status="paused_harness") from exc
        if self.tools and self.tools.failure:
            raise self.tools.failure
        by_profile = {result.profile_id: result for result in results}
        if len(by_profile) != len(self.profiles) or set(by_profile) != {profile.profile_id for profile in self.profiles}:
            raise EvaluationPaused("Harness strict session barrier is incomplete", status="paused_harness")
        occurrences = []
        summaries = []
        accepted_actions = []
        for profile in self.profiles:
            result = by_profile[profile.profile_id]
            accounting = self.meter.reconcile(result)
            if result.submission_status != "accepted":
                raise EvaluationPaused("Harness session did not submit an accepted batch", status="paused_harness")
            candidates = _candidate_file(result.submission, result.submitted_artifacts, self.artifact_root, count)
            admitted = [self.domain.admit(RawProposal(item, "harness")) for item in candidates]
            if not all(isinstance(item, Candidate) for item in admitted):
                raise EvaluationPaused("Harness committed candidate no longer satisfies the frozen grammar", status="paused_harness")
            keys = [item.canonical_key for item in admitted]
            receipt = self.accepted.load([result.turn_id, result.submission_digest])
            if receipt != {"canonical_keys": keys}:
                raise EvaluationPaused("Harness accepted submission lacks its Host validation receipt", status="paused_harness")
            occurrences.extend((item, key, result, index) for index, (item, key) in enumerate(zip(candidates, keys)))
            accepted_actions.append((result, candidates))
            summaries.append({"profile_id": result.profile_id, "turn_id": result.turn_id,
                "submission_id": result.submission_id, "usage": result.usage,
                "tool_budget": getattr(result, "tool_budget", {}),
                "artifacts": getattr(result, "artifacts", {}), **accounting})
        if self.collection:
            for result, candidates in accepted_actions:
                ir = make_complete_design_ir(task_id="alphabench", domain="financial factor expressions",
                    task_description="Generate causal factors using only public contract and search observations.",
                    objectives=[{"name": self.protocol.objective, "direction": "maximize"}],
                    design_space_description=json.dumps(REGISTRY), observations=history,
                    candidates=candidates, request_description=next(turn.message for turn in turns
                        if turn.profile_id == result.profile_id), num_candidates=len(candidates),
                    allows_new_parameters=False, reasoning_available=False)
                action_id = digest([result.turn_id, result.submission_digest])
                self.gateway.host.call(lambda action_id=action_id, ir=ir, result=result: self.collection.accept(
                    action_id, ir, {"run": self.gateway.runtime.run_id, "protocol": self.protocol.identity,
                        "profile_id": result.profile_id, "turn_id": result.turn_id,
                        "submission_digest": result.submission_digest,
                        "native_artifacts": getattr(result, "artifacts", {})}))
        groups = {}
        for item, key, result, index in occurrences:
            group = groups.setdefault(key, {"candidate": item, "lineage": []})
            group["lineage"].append({"profile_id": result.profile_id, "turn_id": result.turn_id, "index": index})
        proposals = tuple(RawProposal(group["candidate"], "harness", {
            "q0": len(group["lineage"]) / len(occurrences),
            "attempt_position": [request.round_idx, index], "harness_lineage": group["lineage"],
        }) for index, group in enumerate(groups.values()))
        return ExpansionResult(proposals=proposals,
            selection_mode="reservoir_order" if direct else "acquisition",
            metadata={"sampling_mode": "persistent_direct_session" if direct else "persistent_parallel_sessions",
                "occurrences": len(occurrences), "unique": len(proposals), "turns": summaries})

    def _validate(self, submission, count, evaluated, round_idx):
        try:
            candidates = _candidate_file(submission.submission, submission.artifacts, self.artifact_root, count)
        except (OSError, ValueError) as exc:
            return HarnessSubmissionValidation("retry", (HarnessSubmissionError(
                "/artifact_path", "invalid_candidate_file", str(exc), "Repair candidates.json and submit it again."),))
        errors, keys, seen = [], [], {}
        for index, item in enumerate(candidates):
            pointer = f"/candidates/{index}"
            if (not isinstance(item, dict) or set(item) - {"name", "expression", "research_note"}
                    or not isinstance(item.get("name"), str) or not item["name"].strip()
                    or len(item["name"]) > 128 or not isinstance(item.get("expression"), str)
                    or ("research_note" in item and (not isinstance(item["research_note"], str)
                        or len(item["research_note"]) > 2000))):
                errors.append(HarnessSubmissionError(pointer, "invalid_candidate",
                    "candidate requires name and expression and permits only a short research_note"))
                continue
            candidate = self.domain.admit(RawProposal(item, "harness"))
            if not isinstance(candidate, Candidate):
                errors.append(HarnessSubmissionError(pointer + "/expression", "invalid_expression", candidate.message))
                continue
            key = candidate.canonical_key
            if key in evaluated:
                errors.append(HarnessSubmissionError(pointer, "historical_duplicate", "factor was already evaluated"))
                continue
            if key in seen:
                errors.append(HarnessSubmissionError(pointer, "same_session_duplicate", f"factor duplicates index {seen[key]}"))
                continue
            seen[key] = index
            checked = self.gateway.evaluate(candidate, phase="check", position=check_position(round_idx, candidate))
            if not self.protocol.check_passed(checked):
                errors.append(HarnessSubmissionError(pointer + "/expression", "dynamic_check_failed", "factor failed the frozen check"))
                continue
            keys.append(key)
        if errors:
            return HarnessSubmissionValidation("retry", tuple(errors))
        self.gateway.host.call(lambda: self.accepted.accept([submission.turn_id, submission.digest], {"canonical_keys": keys}))
        return HarnessSubmissionValidation()
