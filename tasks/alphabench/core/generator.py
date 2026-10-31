"""Bounded full-response generation, canonical admission, and paid dynamic repair."""

from collections import Counter
import json

from ldm_tts.contracts import Candidate, RawProposal
from ldm_tts.contracts.evaluation import EvaluationPaused
from ldm_tts.data.ir import make_complete_design_ir
from ldm_tts.engine.expansion import ExpansionResult, attach_proposal_attempt_receipt
from ldm_tts.engine.run_store import atomic_json_write
from ldm_tts.transport import ProposalRequest, ProposalResponse
from ldm_tts.transport.openai import EndpointRequestError
from .candidate import FactorDomain
from .grammar import REGISTRY
from .protocol import LDM_METHODS, digest
from .receipts import Receipts


class Generator:
    def __init__(self, protocol, runtime, client, gateway, collection):
        self.protocol, self.runtime, self.client = protocol, runtime, client
        self.gateway, self.collection = gateway, collection
        self.domain = FactorDomain(protocol.backend, protocol.grammar_depth)
        self.receipts = Receipts(runtime.run_dir / "private" / "model")

    def request(self, identity, messages, *, proposal=True):
        key = digest({"run": self.runtime.run_id, "identity": identity, "protocol": self.protocol.identity})
        request = {"messages": messages, "protocol": self.protocol.identity, "logical": identity}

        def reserve():
            amounts = {"model_requests": 1, **({"proposal_attempts": 1} if proposal else {})}
            self.runtime.consume_many(amounts, usage_key=("engine:proposal_attempt:" if proposal else "task:model:") + key)

        try:
            previous = self.receipts.load(key)
            marker = self.runtime.run_dir / "private/model_recovery" / self.receipts.path(key).name
            if previous and previous["state"] == "dispatch_intent" and marker.exists():
                raw = self._retry_unknown(key, request, previous, marker, proposal)
            else:
                raw = self.receipts.execute(key, request, reserve=reserve,
                    operation=lambda: self.client.propose(ProposalRequest(tuple(messages))).to_dict(),
                    owner=self.gateway.host, authorize=self.gateway.before_dispatch)
        except EndpointRequestError as exc:
            raise EvaluationPaused("model request failed: " + str(exc) + "; inspect the durable receipt before retrying",
                                   status="paused_provider") from exc
        response = ProposalResponse(**raw)
        if not self.gateway.mock and response.metadata.get("model") != self.protocol.model:
            raise EvaluationPaused("provider returned a different model", status="paused_provider")
        self.gateway.host.call(lambda: self.runtime.consume_many({"model_tokens": response.usage.get("total_tokens", 0)}, usage_key="task:model-usage:" + key))
        return attach_proposal_attempt_receipt(response, key)

    def _retry_unknown(self, key, request, previous, marker, proposal):
        if (previous["identity"] != key or previous["request"] != request or
                previous["request_digest"] != digest(request)):
            raise ValueError("unknown model receipt differs from the resumed request")
        resolution = json.loads(marker.read_text(encoding="utf-8"))
        if resolution != {"action": "discard_unknown_response_and_retry_once",
                          "original_receipt_digest": digest(previous)}:
            raise ValueError("model recovery authorization differs from the unknown receipt")
        retries = Receipts(self.runtime.run_dir / "private/model_retries")
        retry_key = digest({"retry_of": key, "attempt": 1})

        def reserve():
            amounts = {"model_requests": 1, **({"proposal_attempts": 1} if proposal else {})}
            self.runtime.consume_many(amounts, usage_key=("engine:proposal_attempt:" if proposal else "task:model:") + retry_key)

        raw = retries.execute(retry_key, {**request, "retry_of": key}, reserve=reserve,
            operation=lambda: self.client.propose(ProposalRequest(tuple(request["messages"]))).to_dict(),
            owner=self.gateway.host, authorize=self.gateway.before_dispatch)
        previous.update(state="completed", response=raw, response_digest=digest(raw),
                        recovery={"discarded_unknown_receipt_digest": resolution["original_receipt_digest"],
                                  "retry_receipt": retry_key})
        self.gateway.host.call(lambda: atomic_json_write(self.receipts.path(key), previous))
        return raw

    def generate(self, *, identity, count, instruction, history=(), partial=False):
        accepted, attempts, raw_occurrences = {}, [], []
        format_failures = 0
        excluded = {item.canonical_key for item in history}
        visible = [{"expression": item.candidate.payload["expression"], "status": item.evaluation.status,
                    "metrics": item.evaluation.metrics} for item in history]
        messages = [{"role": "system", "content": "Propose causal financial factor expressions. Return a JSON object with a candidates array. Each candidate has name and expression. Never mix dialects. Complete operator signatures: " + json.dumps({k: REGISTRY[k] for k in (["qlib"] if self.protocol.backend == "qlib" else ["qlib", "assay"])})},
                    {"role": "user", "content": json.dumps({"instruction": instruction, "required": count,
                         "search_history": visible, "max_depth": self.protocol.grammar_depth})}]
        original_messages = list(messages)
        for attempt in range(5):
            response = self.request([identity, attempt], messages)
            attempts.append(response)
            errors = []
            try:
                payload = json.loads(response.text)
                candidates = payload["candidates"]
                if not isinstance(candidates, list):
                    raise ValueError("candidates must be a list")
            except (ValueError, KeyError, TypeError):
                candidates = []
                format_failures += 1
                errors.append({"code": "invalid_json", "message": "Return {candidates: [...]} with name and expression."})
            for index, payload in enumerate(candidates):
                occurrence = {"attempt": attempt, "index": index, "payload": payload, "status": "unprocessed"}
                raw_occurrences.append(occurrence)
                if len(accepted) == count:
                    continue
                candidate = self.domain.admit(RawProposal(payload, "model"))
                if not isinstance(candidate, Candidate):
                    occurrence.update(status="static_rejected", error=candidate.reason)
                    errors.append({"index": index, "code": "static_rejected", "message": candidate.reason})
                    continue
                if candidate.canonical_key in excluded or candidate.candidate_id in accepted:
                    occurrence["status"] = "duplicate"
                    errors.append({"index": index, "code": "duplicate"})
                    continue
                checked = self.gateway.evaluate(candidate, phase="check", position=[identity, attempt, index])
                occurrence["check_receipt"] = self.gateway.identity("check", [identity, attempt, index], candidate)
                valid = self.protocol.check_passed(checked)
                if not valid:
                    occurrence.update(status=self.protocol.check_kind + "_rejected", error=checked.get("error", "factor check rejected"))
                    errors.append({"index": index, "code": occurrence["status"], "message": occurrence["error"]})
                    continue
                occurrence["status"] = "accepted"
                accepted[candidate.candidate_id] = candidate.payload
            if len(accepted) == count:
                break
            messages.append({"role": "user", "content": json.dumps({"remaining": count-len(accepted),
                            "accepted": list(accepted.values()), "errors": errors})})
        minimum = max(1, count // 2) if partial else count
        result = {"candidates": list(accepted.values()), "occurrences": raw_occurrences,
                  "requested": count, "attempt_count": len(attempts), "format_failures": format_failures,
                  "complete": len(accepted) >= minimum}
        self.gateway.host.call(lambda: Receipts(self.runtime.run_dir / "generation").accept(identity, result))
        if not result["complete"]:
            raise EvaluationPaused("generation exhausted its five repair attempts", status="paused_generation")
        ir = make_complete_design_ir(task_id="alphabench", domain="financial factor expressions",
            task_description="Generate causal factors using only public contract and search observations.",
            objectives=[{"name": self.protocol.objective, "direction": "maximize"}],
            design_space_description=json.dumps(REGISTRY), observations=visible,
            candidates=result["candidates"], request_description=json.dumps(original_messages),
            num_candidates=len(accepted), allows_new_parameters=False, reasoning_available=False)
        self.gateway.host.call(lambda: self.collection.accept(identity, ir, {"run": self.runtime.run_id, "protocol": self.protocol.identity}))
        return result, tuple(attempts)


class DirectExpander:
    def __init__(self, generator):
        self.generator = generator

    def expand(self, request):
        protocol = self.generator.protocol
        count = protocol.candidates_per_session if protocol.method in LDM_METHODS else request.context["evaluation_budget"]["effective"]
        occurrences, attempts = [], []
        for session in range(protocol.sessions if protocol.method in LDM_METHODS else 1):
            result, responses = self.generator.generate(identity=["round", request.round_idx, "session", session], count=count,
                instruction="Improve signed search " + protocol.objective + f". Independent sample {session}.", history=request.observations)
            occurrences.extend(result["candidates"])
            attempts.extend(responses)
        unique = {item["expression"]: item for item in occurrences}
        counts = Counter(item["expression"] for item in occurrences)
        proposals = tuple(RawProposal(item, protocol.method, metadata={"attempt_position": [request.round_idx, index],
                             "q0": counts[expression] / len(occurrences)}) for index, (expression, item) in enumerate(unique.items()))
        return ExpansionResult(proposals=proposals, attempts=tuple(attempts),
                               selection_mode="acquisition" if protocol.method in LDM_METHODS else "reservoir_order")
