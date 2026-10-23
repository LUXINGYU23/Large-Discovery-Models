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


def submission_contract(count=1):
    fields = {
        "artifact_path": {"type": "string", "pattern": r"^candidates/[0-9]+/solver\.py$"},
        "parent_sha256": {"type": ["string", "null"]},
        "change_summary": {"type": "string", "minLength": 1, "maxLength": 1000},
        "rationale": {"type": "string", "minLength": 1, "maxLength": 2000},
    }
    return HarnessSubmissionContract(
        "copper_solver",
        "submit_solver",
        {
            "type": "object",
            "properties": {"candidates": {"type": "array", "minItems": count, "maxItems": count,
                "items": {"type": "object", "properties": fields,
                          "required": list(fields), "additionalProperties": False}}},
            "required": ["candidates"],
            "additionalProperties": False,
        },
        tuple(HarnessArtifactRule(f"/candidates/{i}/artifact_path", (".py",), MAX_SOURCE_BYTES) for i in range(count)),
        max_validation_attempts=3 * count,
    )


class SolverSubmissionError(ValueError):
    def __init__(self, index, message):
        self.path = f"/candidates/{index}/artifact_path"
        super().__init__(f"candidate {index}: {message}")


def submission_error(exc):
    return HarnessSubmissionError(getattr(exc, "path", "/candidates"), "invalid_solver", str(exc))


def parse_snapshot(submission, artifacts, artifact_root, known, count=1):
    if not isinstance(submission, dict) or set(submission) != {"candidates"}:
        raise ValueError("submit a candidates array")
    rows = submission["candidates"]
    if not isinstance(rows, list) or len(rows) != count or len(artifacts) != count:
        raise ValueError(f"submit exactly {count} solver artifacts per researcher")
    by_pointer = {a.path_pointer: a for a in artifacts}
    if len(by_pointer) != count:
        raise ValueError("duplicate snapshot pointer")
    root = Path(artifact_root).resolve()
    sources, seen = [], set()
    for i, row in enumerate(rows):
        try:
            if not isinstance(row, dict) or set(row) != {
                "artifact_path", "parent_sha256", "change_summary", "rationale",
            }:
                raise ValueError("invalid submission fields")
            for name, maximum in (("change_summary", 1000), ("rationale", 2000)):
                if not isinstance(row[name], str) or not 1 <= len(row[name]) <= maximum:
                    raise ValueError("invalid " + name)
            pointer = f"/candidates/{i}/artifact_path"
            if row["artifact_path"] != f"candidates/{i}/solver.py" or pointer not in by_pointer:
                raise ValueError("snapshot path mismatch")
            a = by_pointer[pointer]
            relative = Path(a.snapshot_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("snapshot path mismatch")
            if any(root.joinpath(*relative.parts[:j]).is_symlink() for j in range(1, len(relative.parts) + 1)):
                raise ValueError("snapshot symlink forbidden")
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or a.relative_path != row["artifact_path"]:
                raise ValueError("snapshot path mismatch")
            source = path.read_bytes()
            if len(source) != a.size_bytes or hashlib.sha256(source).hexdigest() != a.sha256:
                raise ValueError("snapshot digest mismatch")
            sha = validate_source(source)
            if sha in known:
                raise ValueError("source already evaluated; submit a changed algorithm")
            if sha in seen:
                raise ValueError("source duplicates another candidate in this submission")
            if row["parent_sha256"] is not None and (
                not isinstance(row["parent_sha256"], str) or row["parent_sha256"] not in known
            ):
                raise ValueError("unknown parent digest")
            seen.add(sha)
            sources.append(source)
        except (ValueError, OSError, SyntaxError, UnicodeError) as exc:
            raise SolverSubmissionError(i, str(exc)) from exc
    return tuple(sources)


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
                 selection_mode="reservoir_order", candidates_per_researcher=1):
        self.root = Path(root)
        self.artifact_root = Path(artifact_root) if artifact_root is not None else self.root / "harness"
        self.client, self.checks, self.account = client, checks, account
        self.profiles = tuple(profiles)
        if not self.profiles or len(set(self.profiles)) != len(self.profiles):
            raise ValueError("distinct researcher identities required")
        self.inherited_history_counts = inherited_history_counts
        self.selection_mode = selection_mode
        self.candidates_per_researcher = candidates_per_researcher

    def expand(self, request):
        if request.reservoir_size != len(self.profiles) * self.candidates_per_researcher:
            raise ValueError("reservoir must match configured researcher candidate count")
        history = write_history(self.root, self.artifact_root, request.observations)
        known = {o.canonical_key for o in request.observations}
        turns = []
        for profile in self.profiles:
            cursor_file = self.artifact_root / "history-cursors" / (profile + ".json")
            cursor = (self.inherited_history_counts[profile]
                      if self.inherited_history_counts is not None else json.loads(cursor_file.read_text()) if cursor_file.exists() else 0)
            if not 0 <= cursor <= len(history):
                raise ValueError("invalid inherited history cursor")
            turns.append(HarnessTurn(
                profile_id=profile, turn_id=f"{profile}_round_{request.round_idx}",
                round_index=request.round_idx, history_from_seq=cursor,
                history_to_seq=len(history), history_digest=canonical_sha256(history[cursor:]),
                message=json.dumps({
                    "instruction": "Read get_inversion_context, inspect measured feedback and improve the algorithm. Submit the requested number of distinct complete solvers. Only selected candidates are scored; an unmeasured proposal is not a failure.",
                    "round": request.round_idx, "profile_id": profile,
                    "candidates_required": self.candidates_per_researcher,
                    "research_wall_time_seconds": self.client.config.limits.wall_time_seconds,
                    "known_sources": sorted(known),
                    "new_measurements": [{k: v for k, v in row.items() if k not in ("source", "model_slices", "annotations")}
                                         for row in history[cursor:]],
                    "full_measured_history": "Call get_inversion_context; inspect selected entries in exported history.json for full code, slices and annotations. Do not dump the entire growing history into context.",
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
                if set(record["sha256s"]) & {sha for values in admitted.values() for sha in values}:
                    raise ValueError("duplicate admitted source in batch")
                admitted[turn.profile_id] = record["sha256s"]

        def validate(s):
            try:
                turn = expected[s.profile_id]
                if s.turn_id != turn.turn_id:
                    raise ValueError("submission does not match requested turn")
                bodies = parse_snapshot(s.submission, s.artifacts, self.artifact_root, known, self.candidates_per_researcher)
                payloads = [seal(self.root, body) for body in bodies]
                shas = [p["sha256"] for p in payloads]
                if any(owner != s.profile_id and set(values) & set(shas) for owner, values in admitted.items()):
                    raise ValueError("source duplicates another branch in this batch; change the algorithm")
                if s.profile_id in admitted and admitted[s.profile_id] != shas:
                    raise ValueError("accepted source changed during replay")
                for index, payload in enumerate(payloads):
                    try:
                        self.checks(payload)
                    except (ValueError, OSError) as exc:
                        raise SolverSubmissionError(index, str(exc)) from exc
                receipt = self.artifact_root / "admissions" / (turn.input_digest + ".json")
                record = {"input_digest": turn.input_digest, "profile_id": s.profile_id,
                          "sha256s": shas, "submission_digest": s.digest}
                if receipt.exists() and json.loads(receipt.read_text()) != record:
                    raise ValueError("accepted submission changed during replay")
                atomic_json_write(receipt, record)
                admitted[s.profile_id] = shas
                return HarnessSubmissionValidation("accept")
            except (ValueError, OSError, SyntaxError, UnicodeError) as exc:
                return HarnessSubmissionValidation("retry", (
                    submission_error(exc),))

        def validate_batch(requests):
            # Hold well-formed snapshots until every branch is present. Invalid
            # peers may repair in the same turn without spending interface jobs.
            errors, seen = {}, dict(admitted)
            for request in requests:
                try:
                    if request.profile_id not in expected or request.turn_id != expected[request.profile_id].turn_id:
                        raise ValueError("submission does not match requested turn")
                    bodies = parse_snapshot(request.submission, request.artifacts, self.artifact_root, known, self.candidates_per_researcher)
                    shas = [hashlib.sha256(body).hexdigest() for body in bodies]
                    if any(owner != request.profile_id and set(values) & set(shas) for owner, values in seen.items()):
                        raise ValueError("source duplicates another branch in this batch; change the algorithm")
                    if request.profile_id in admitted and admitted[request.profile_id] != shas:
                        raise ValueError("accepted source changed during replay")
                    seen[request.profile_id] = shas
                except (ValueError, OSError, SyntaxError, UnicodeError) as exc:
                    errors[request.profile_id] = HarnessSubmissionValidation("retry", (
                        submission_error(exc),))
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
                raise ValueError(f"research turn {result.turn_id} rejected: {result.validation_errors}")
            bodies = parse_snapshot(result.submission, result.submitted_artifacts, self.artifact_root, known, self.candidates_per_researcher)
            payloads = [seal(self.root, body) for body in bodies]
            if admitted.get(result.profile_id) != [p["sha256"] for p in payloads]:
                raise ValueError("native turn has no matching controller admission")
            for index, payload in enumerate(payloads):
                self.checks(payload)
                lineage = {"profile_id": result.profile_id, "session_id": result.session_id,
                           "turn_id": result.turn_id, "submission_id": result.submission_id,
                           "candidate_index": index}
                proposals.append(RawProposal(payload, "persistent_harness", metadata={
                    **{k: result.submission["candidates"][index][k] for k in ("parent_sha256", "change_summary", "rationale")},
                    "harness_lineage": lineage,
                }))
            atomic_json_write(self.artifact_root / "history-cursors" / (result.profile_id + ".json"), len(history))
            attempts.append(attach_proposal_attempt_receipt(
                ProposalResponse(text="accepted solver", metadata={}), result.submission_id))
            sessions.append(proposal_turn_summary(result, self.candidates_per_researcher))
        return ExpansionResult(proposals=tuple(proposals), attempts=tuple(attempts),
                               selection_mode=self.selection_mode, metadata={"sessions": sessions})
