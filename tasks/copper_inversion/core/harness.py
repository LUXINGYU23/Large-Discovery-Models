"""One persistent researcher, shared Harness snapshots and task-specific admission."""

import hashlib
import json
import time
from pathlib import Path
from ldm_tts.contracts import RawProposal
from ldm_tts.engine.expansion import ExpansionResult
from ldm_tts.harness import (
    HarnessTurn,
    HarnessSubmissionContract,
    HarnessArtifactRule,
    HarnessSubmissionValidation,
    HarnessSubmissionError,
    HarnessError,
)
from ldm_tts.harness.research_runtime import usage_counts, proposal_turn_summary
from ldm_tts.engine.run_store import atomic_json_write
from ldm_tts.harness.protocol import canonical_sha256
from ldm_tts.transport import ProposalResponse
from ldm_tts.engine.expansion import attach_proposal_attempt_receipt
from .candidate import seal, validate_source, MAX_SOURCE_BYTES


def submission_contract():
    fields = {
        "artifact_path": {"type": "string", "const": "candidates/0/solver.py"},
        "parent_sha256": {"type": ["string", "null"]},
        "change_summary": {"type": "string", "minLength": 1, "maxLength": 1000},
        "rationale": {"type": "string", "minLength": 1, "maxLength": 2000},
    }
    return HarnessSubmissionContract(
        "copper_solver",
        "submit_solver",
        {
            "type": "object",
            "properties": {"candidates": {"type": "array", "minItems": 1, "maxItems": 1,
                "items": {"type": "object", "properties": fields,
                          "required": list(fields), "additionalProperties": False}}},
            "required": ["candidates"],
            "additionalProperties": False,
        },
        (HarnessArtifactRule("/candidates/0/artifact_path", (".py",), MAX_SOURCE_BYTES),),
        max_validation_attempts=3,
    )


def parse_snapshot(submission, artifacts, artifact_root, known):
    if not isinstance(submission, dict) or set(submission) != {"candidates"}:
        raise ValueError("submit a candidates array")
    rows = submission["candidates"]
    if not isinstance(rows, list) or len(rows) != 1 or len(artifacts) != 1:
        raise ValueError("submit one solver per researcher")
    row = rows[0]
    if not isinstance(row, dict) or set(row) != {
        "artifact_path",
        "parent_sha256",
        "change_summary",
        "rationale",
    }:
        raise ValueError("invalid submission fields")
    for name, maximum in (("change_summary", 1000), ("rationale", 2000)):
        if not isinstance(row[name], str) or not 1 <= len(row[name]) <= maximum:
            raise ValueError("invalid " + name)
    if row["artifact_path"] != "candidates/0/solver.py":
        raise ValueError("submit one solver")
    a = artifacts[0]
    root = Path(artifact_root).resolve()
    relative = Path(a.snapshot_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("snapshot path mismatch")
    p = root / relative
    if any((root.joinpath(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts) + 1)):
        raise ValueError("snapshot symlink forbidden")
    p = p.resolve()
    if (
        not p.is_relative_to(root)
        or a.relative_path != row["artifact_path"]
        or a.path_pointer != "/candidates/0/artifact_path"
    ):
        raise ValueError("snapshot path mismatch")
    source = p.read_bytes()
    if len(source) != a.size_bytes or hashlib.sha256(source).hexdigest() != a.sha256:
        raise ValueError("snapshot digest mismatch")
    sha = validate_source(source)
    if sha in known:
        raise ValueError("source already evaluated; submit a changed algorithm")
    if (
        row["parent_sha256"] is not None
        and (not isinstance(row["parent_sha256"], str) or row["parent_sha256"] not in known)
    ):
        raise ValueError("unknown parent digest")
    return source


def serialize_history(root, observations):
    return [
        {"candidate_id": o.canonical_key, "round_index": o.round_idx,
         "status": o.evaluation.status, "metrics": o.metrics,
         "resource_usage": o.evaluation.resource_usage,
         "model_slices": json.loads((Path(root) / o.evaluation.artifacts["model_slices"]).read_text())
             if o.evaluation.succeeded and "model_slices" in o.evaluation.artifacts else None,
         "research_utility": o.metrics.get("research_utility") if o.evaluation.succeeded else None,
         "error": o.evaluation.error,
         "source": (Path(root) / o.candidate.payload["source_path"]).read_text(),
         "annotations": o.candidate.metadata}
        for o in observations
    ]


def write_history(root, artifact_root, observations):
    history = serialize_history(root, observations)
    atomic_json_write(Path(artifact_root) / "public/history.json", history)
    atomic_json_write(Path(artifact_root) / "measured_history/observations.json", history)
    return history


class Expander:
    def __init__(self, root, client, checks, account, *, artifact_root=None,
                 profiles=("researcher",), inherited_history_counts=None,
                 selection_mode="reservoir_order"):
        self.root = Path(root)
        self.artifact_root = Path(artifact_root) if artifact_root is not None else self.root / "harness"
        self.client, self.checks, self.account = client, checks, account
        self.profiles = tuple(profiles)
        if not self.profiles or len(set(self.profiles)) != len(self.profiles):
            raise ValueError("distinct researcher identities required")
        self.inherited_history_counts = inherited_history_counts
        self.selection_mode = selection_mode

    def expand(self, request):
        if request.reservoir_size != len(self.profiles):
            raise ValueError("one solver per configured researcher is required")
        history = write_history(self.root, self.artifact_root, request.observations)
        known = {o.canonical_key for o in request.observations}
        turns = []
        for profile in self.profiles:
            cursor = (self.inherited_history_counts[profile]
                      if self.inherited_history_counts is not None else max(0, len(history) - 1))
            if not 0 <= cursor <= len(history):
                raise ValueError("invalid inherited history cursor")
            turns.append(HarnessTurn(
                profile_id=profile, turn_id=f"{profile}_round_{request.round_idx}",
                round_index=request.round_idx, history_from_seq=cursor,
                history_to_seq=len(history), history_digest=canonical_sha256(history[cursor:]),
                message=json.dumps({
                    "instruction": "Read get_inversion_context, inspect measured feedback and improve the algorithm. Submit one complete solver. Only selected candidates are scored; an unmeasured proposal is not a failure.",
                    "round": request.round_idx, "profile_id": profile,
                    "research_wall_time_seconds": self.client.config.limits.wall_time_seconds,
                    "known_sources": sorted(known),
                    "new_measurements": history[cursor:],
                }),
            ))
        expected = {t.profile_id: t for t in turns}
        # Admission receipts reserve distinct sources before acknowledging acceptance.
        # They survive a partial native Harness barrier; they do not publish a reservoir.
        admitted = {}
        for turn in turns:
            receipt = self.artifact_root / "admissions" / (turn.input_digest + ".json")
            if receipt.exists():
                record = json.loads(receipt.read_text())
                if record["input_digest"] != turn.input_digest or record["profile_id"] != turn.profile_id:
                    raise ValueError("admission receipt identity changed")
                if record["sha256"] in admitted.values():
                    raise ValueError("duplicate admitted source in batch")
                admitted[turn.profile_id] = record["sha256"]

        def validate(s):
            try:
                turn = expected[s.profile_id]
                if s.turn_id != turn.turn_id:
                    raise ValueError("submission does not match requested turn")
                body = parse_snapshot(s.submission, s.artifacts, self.artifact_root, known)
                payload = seal(self.root, body)
                sha = payload["sha256"]
                if any(owner != s.profile_id and value == sha for owner, value in admitted.items()):
                    raise ValueError("source duplicates another branch in this batch; change the algorithm")
                if s.profile_id in admitted and admitted[s.profile_id] != sha:
                    raise ValueError("accepted source changed during replay")
                self.checks(payload)
                receipt = self.artifact_root / "admissions" / (turn.input_digest + ".json")
                record = {"input_digest": turn.input_digest, "profile_id": s.profile_id,
                          "sha256": sha, "submission_digest": s.digest}
                if receipt.exists() and json.loads(receipt.read_text()) != record:
                    raise ValueError("accepted submission changed during replay")
                atomic_json_write(receipt, record)
                admitted[s.profile_id] = sha
                return HarnessSubmissionValidation("accept")
            except (ValueError, OSError, SyntaxError, UnicodeError) as exc:
                return HarnessSubmissionValidation("retry", (
                    HarnessSubmissionError("/candidates/0/artifact_path", "invalid_solver", str(exc)),))

        def validate_batch(requests):
            # Hold well-formed snapshots until every branch is present. Invalid
            # peers may repair in the same turn without spending interface jobs.
            errors, seen = {}, dict(admitted)
            for request in requests:
                try:
                    if request.profile_id not in expected or request.turn_id != expected[request.profile_id].turn_id:
                        raise ValueError("submission does not match requested turn")
                    body = parse_snapshot(request.submission, request.artifacts, self.artifact_root, known)
                    sha = hashlib.sha256(body).hexdigest()
                    if any(owner != request.profile_id and value == sha for owner, value in seen.items()):
                        raise ValueError("source duplicates another branch in this batch; change the algorithm")
                    if request.profile_id in admitted and admitted[request.profile_id] != sha:
                        raise ValueError("accepted source changed during replay")
                    seen[request.profile_id] = sha
                except (ValueError, OSError, SyntaxError, UnicodeError) as exc:
                    errors[request.profile_id] = HarnessSubmissionValidation("retry", (
                        HarnessSubmissionError("/candidates/0/artifact_path", "invalid_solver", str(exc)),))
            if errors:
                return errors
            if set(seen) != set(expected):
                return {}
            return {request.profile_id: validate(request) for request in requests}

        for turn in turns:
            self.account({"harness_turns": 1}, usage_key="copper:turn:" + turn.input_digest)
        started, usage = time.monotonic(), {}
        try:
            results = self.client.run_turn(tuple(turns),
                **({"submission_batch_validator": validate_batch} if len(turns) > 1 else {"submission_validator": validate}),
                recovery_timeout_seconds=2 * self.client.config.limits.wall_time_seconds)
            usage = {r.profile_id: r.usage for r in results}
        except HarnessError as exc:
            usage = exc.turn_usage
            raise
        finally:
            for turn in turns:
                self.account(usage_counts(usage.get(turn.profile_id, {})),
                             usage_key="copper:usage:" + turn.input_digest)
            self.account({"harness_wall_time_seconds": time.monotonic() - started})
        if len(results) != len(turns) or {r.profile_id for r in results} != set(expected):
            raise ValueError("incomplete researcher batch")
        proposals, attempts, sessions = [], [], []
        for result in sorted(results, key=lambda r: self.profiles.index(r.profile_id)):
            if result.submission_status != "accepted":
                raise ValueError("research turn not accepted")
            body = parse_snapshot(result.submission, result.submitted_artifacts, self.artifact_root, known)
            payload = seal(self.root, body)
            if admitted.get(result.profile_id) != payload["sha256"]:
                raise ValueError("native turn has no matching controller admission")
            self.checks(payload)
            lineage = {"profile_id": result.profile_id, "session_id": result.session_id,
                       "turn_id": result.turn_id, "submission_id": result.submission_id}
            proposals.append(RawProposal(payload, "persistent_harness", metadata={
                **{k: result.submission["candidates"][0][k] for k in ("parent_sha256", "change_summary", "rationale")},
                "harness_lineage": lineage,
            }))
            attempts.append(attach_proposal_attempt_receipt(
                ProposalResponse(text="accepted solver", metadata={}), result.submission_id))
            sessions.append(proposal_turn_summary(result, 1))
        return ExpansionResult(proposals=tuple(proposals), attempts=tuple(attempts),
                               selection_mode=self.selection_mode, metadata={"sessions": sessions})
