"""Single-writer dispatch receipts; unknown physical outcomes never auto-retry."""

import json
from pathlib import Path

from ldm_tts.contracts.evaluation import EvaluationPaused
from ldm_tts.engine.run_store import atomic_json_write
from .protocol import digest


class Receipts:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, identity):
        return self.root / (digest(identity) + ".json")

    def load(self, identity):
        path = self.path(identity)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def execute(self, identity, request, *, reserve, operation, reconcile=None, owner=None):
        request_digest = digest(request)
        invoke = owner.call if owner else lambda callback: callback()

        def prepare():
            record = self.load(identity)
            if record is not None:
                if record["request_digest"] != request_digest:
                    raise ValueError("receipt identity reused for a different request")
                if record["state"] == "completed":
                    if digest(record["response"]) != record["response_digest"]:
                        raise ValueError("receipt response integrity failure")
                    return record
                if record["state"] == "dispatch_intent":
                    response = reconcile() if reconcile else None
                    if response is not None:
                        record.update(state="completed", response=response, response_digest=digest(response))
                        atomic_json_write(self.path(identity), record)
                        return record
                    raise EvaluationPaused(f"physical request requires reconciliation: {digest(identity)}")
            reserve()
            record = {"identity": identity, "request_digest": request_digest, "request": request,
                      "state": "reserved"}
            atomic_json_write(self.path(identity), record)
            record["state"] = "dispatch_intent"
            atomic_json_write(self.path(identity), record)
            return record

        record = invoke(prepare)
        if record["state"] == "completed":
            return record["response"]
        response = operation()

        def complete():
            record.update(state="completed", response=response, response_digest=digest(response))
            atomic_json_write(self.path(identity), record)

        invoke(complete)
        return response

    def accept(self, identity, payload):
        record = self.load(identity)
        if record is not None:
            if record != payload:
                raise ValueError("accepted action identity conflict")
        else:
            atomic_json_write(self.path(identity), payload)
        return payload
