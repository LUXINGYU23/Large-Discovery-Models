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
from ldm_tts.harness.research_runtime import usage_counts
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
            "properties": fields,
            "required": list(fields),
            "additionalProperties": False,
        },
        (HarnessArtifactRule("/artifact_path", (".py",), MAX_SOURCE_BYTES),),
        max_validation_attempts=3,
    )


def parse_snapshot(submission, artifacts, artifact_root, known):
    if set(submission) != {
        "artifact_path",
        "parent_sha256",
        "change_summary",
        "rationale",
    }:
        raise ValueError("invalid submission fields")
    if submission["artifact_path"] != "candidates/0/solver.py" or len(artifacts) != 1:
        raise ValueError("submit one solver")
    a = artifacts[0]
    root = Path(artifact_root).resolve()
    p = (root / a.snapshot_path).resolve()
    if (
        not p.is_relative_to(root)
        or a.relative_path != submission["artifact_path"]
        or a.path_pointer != "/artifact_path"
    ):
        raise ValueError("snapshot path mismatch")
    source = p.read_bytes()
    if len(source) != a.size_bytes or hashlib.sha256(source).hexdigest() != a.sha256:
        raise ValueError("snapshot digest mismatch")
    sha = validate_source(source)
    if sha in known:
        raise ValueError("source already evaluated; submit a changed algorithm")
    if (
        submission["parent_sha256"] is not None
        and submission["parent_sha256"] not in known
    ):
        raise ValueError("unknown parent digest")
    return source


class Expander:
    def __init__(self, root, client, checks, account):
        self.root = Path(root)
        self.client = client
        self.checks = checks
        self.account = account

    def expand(self, request):
        history = []
        for o in request.observations:
            history.append(
                {
                    "candidate_id": o.canonical_key,
                    "round_index": o.round_idx,
                    "status": o.evaluation.status,
                    "metrics": o.metrics,
                    "error": o.evaluation.error,
                    "source": (
                        self.root / o.candidate.payload["source_path"]
                    ).read_text(),
                    "annotations": o.candidate.metadata,
                }
            )
        public = self.root / "harness/public"
        public.mkdir(parents=True, exist_ok=True)
        (public / "history.json").write_text(json.dumps(history, allow_nan=False))
        known = {o.canonical_key for o in request.observations}

        def validate(s):
            try:
                body = parse_snapshot(
                    s.submission, s.artifacts, self.root / "harness", known
                )
                payload = seal(self.root, body)
                self.checks(payload)
                return HarnessSubmissionValidation("accept")
            except (ValueError, OSError, SyntaxError, UnicodeError) as exc:
                return HarnessSubmissionValidation(
                    "retry",
                    (
                        HarnessSubmissionError(
                            "/artifact_path", "invalid_solver", str(exc)
                        ),
                    ),
                )

        turn = HarnessTurn(
            profile_id="researcher",
            turn_id=f"round_{request.round_idx}",
            round_index=request.round_idx,
            history_from_seq=max(0, len(history) - 1),
            history_to_seq=len(history),
            history_digest=canonical_sha256(history[-1:]),
            message=json.dumps(
                {
                    "instruction": "Read get_inversion_context, inspect measured feedback and improve the algorithm. Submit one complete solver.",
                    "round": request.round_idx,
                    "known_sources": sorted(known),
                }
            ),
        )
        self.account({"harness_turns": 1}, usage_key="copper:turn:" + turn.turn_id)
        started = time.monotonic()
        usage = {}
        try:
            (result,) = self.client.run_turn(
                (turn,),
                submission_validator=validate,
                recovery_timeout_seconds=2
                * self.client.config.limits.wall_time_seconds,
            )
            usage = result.usage
        except HarnessError as exc:
            usage = exc.turn_usage.get("researcher", {})
            raise
        finally:
            self.account(usage_counts(usage), usage_key="copper:usage:" + turn.turn_id)
            self.account({"harness_wall_time_seconds": time.monotonic() - started})
        if result.submission_status != "accepted":
            raise ValueError("research turn not accepted")
        # Revalidate immutable bytes on replay; admission jobs reuse durable receipts.
        body = parse_snapshot(
            result.submission, result.submitted_artifacts, self.root / "harness", known
        )
        payload = seal(self.root, body)
        self.checks(payload)
        proposal = RawProposal(
            payload,
            "persistent_harness",
            metadata={
                k: result.submission[k]
                for k in ("parent_sha256", "change_summary", "rationale")
            },
        )
        attempt = attach_proposal_attempt_receipt(
            ProposalResponse(text="accepted solver", metadata={}), result.submission_id
        )
        return ExpansionResult(
            proposals=(proposal,), attempts=(attempt,), selection_mode="reservoir_order"
        )
