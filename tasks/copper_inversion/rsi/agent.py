"""One isolated Pi session and workspace per branch, renewed at each reset."""

import hashlib
import json
import time
from pathlib import Path

from ldm_tts.harness import (HarnessArtifactRule, HarnessClient, HarnessError, HarnessLimits,
                            HarnessNetworkPolicy, HarnessProfile, HarnessSubmissionContract,
                            HarnessSubmissionError, HarnessSubmissionValidation, HarnessToolExtension,
                            HarnessTurn)
from ldm_tts.harness.container import pi_sidecar_command
from ldm_tts.harness.pi import PiHarnessConfig, load_pi_guest_runtime
from ldm_tts.harness.protocol import canonical_sha256, file_sha256
from ..core.candidate import MAX_SOURCE_BYTES, validate_source
from ..v2.research import PublicTrainingExport
from .controller import _validate_submission
from .render import render, write_sources
from .usage import collect


RESOURCES = Path(__file__).resolve().parents[1] / "resources"


def submission_contract(count):
    fields = {
        "artifact_path": {"type": "string", "pattern": r"^candidates/[0-9]+/solver\.py$"},
        "parent_sha256": {"type": ["string", "null"]},
        "change_summary": {"type": "string", "minLength": 1, "maxLength": 1000},
        "rationale": {"type": "string", "minLength": 1, "maxLength": 2000},
        "predicted_risk": {"type": "number", "minimum": 0, "maximum": 1},
        "interval80_low": {"type": "number", "minimum": 0, "maximum": 1},
        "interval80_high": {"type": "number", "minimum": 0, "maximum": 1},
    }
    return HarnessSubmissionContract(
        "rsi_solver", "submit_solver",
        {"type": "object", "additionalProperties": False,
         "properties": {"candidates": {"type": "array", "minItems": count, "maxItems": count,
            "items": {"type": "object", "properties": fields, "required": list(fields),
                      "additionalProperties": False}}}, "required": ["candidates"]},
        tuple(HarnessArtifactRule(f"/candidates/{i}/artifact_path", (".py",), MAX_SOURCE_BYTES)
              for i in range(count)), max_validation_attempts=3 * count)


def parse_submission(submission, artifacts, root, count, known, visible):
    if not isinstance(submission, dict) or set(submission) != {"candidates"}:
        raise ValueError("invalid submission")
    values = submission["candidates"]
    if not isinstance(values, list) or len(values) != count or len(artifacts) != count:
        raise ValueError("invalid submission count")
    by_pointer = {a.path_pointer: a for a in artifacts}
    if len(by_pointer) != count:
        raise ValueError("invalid artifact pointers")
    root = Path(root).resolve()
    used, result = set(known), []
    for index, row in enumerate(values):
        if not isinstance(row, dict) or set(row) != {"artifact_path", "parent_sha256", "change_summary",
                "rationale", "predicted_risk", "interval80_low", "interval80_high"}:
            raise ValueError("invalid proposal fields")
        pointer = f"/candidates/{index}/artifact_path"
        if row["artifact_path"] != f"candidates/{index}/solver.py" or pointer not in by_pointer:
            raise ValueError("invalid snapshot path")
        artifact = by_pointer[pointer]
        relative = Path(artifact.snapshot_path)
        if (relative.is_absolute() or ".." in relative.parts
                or any(root.joinpath(*relative.parts[:j]).is_symlink() for j in range(1, len(relative.parts) + 1))):
            raise ValueError("invalid snapshot path")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or artifact.relative_path != row["artifact_path"]:
            raise ValueError("invalid snapshot path")
        body = path.read_bytes()
        if len(body) != artifact.size_bytes or hashlib.sha256(body).hexdigest() != artifact.sha256:
            raise ValueError("snapshot changed")
        sha = validate_source(body)
        if sha in used:
            raise ValueError("source rejected: not novel")
        if row["parent_sha256"] is not None and row["parent_sha256"] not in visible:
            raise ValueError("parent is not visible to this branch")
        proposal = {**{k: v for k, v in row.items() if k != "artifact_path"}, "source": body.decode("utf-8")}
        _validate_submission(proposal)
        used.add(sha)
        result.append(proposal)
    return result


class PiAgent:
    def __init__(self, *, public_catalog, api_key, sidecar_image, cache_dir, arm, reset_period):
        if not api_key:
            raise ValueError("protected model API key required")
        self.export = PublicTrainingExport(public_catalog)
        self.key, self.image, self.cache = api_key, sidecar_image, Path(cache_dir)
        self.arm, self.reset_period = arm, reset_period

    def propose(self, *, branch, round_index, epoch, count, session_round, message, view,
                turn_root, candidate_root, known, seed, harness_seconds):
        turn_root = Path(turn_root)
        turn_root.mkdir(parents=True, exist_ok=True)
        self.export(turn_root, None)
        public = turn_root / "public"
        (public / "TASK.md").write_bytes((RESOURCES / "rsi/TASK.md").read_bytes())
        evidence = public / "evidence"
        evidence.mkdir(exist_ok=True)
        (evidence / "history.txt").write_text(render(view), encoding="utf-8")
        write_sources(evidence / "sources", view, candidate_root)
        profile = HarnessProfile("researcher", Path("/resources/rsi/AGENTS.md"),
                                 agents_sha256=file_sha256(RESOURCES / "rsi/AGENTS.md"))
        config = PiHarnessConfig(
            artifact_root=Path("/artifacts"), profiles=(profile,),
            campaign_id=f"rsi-{Path(candidate_root).name}-e{epoch}-b{branch}",
            task_id="copper_inversion", case_id="public-training", seed=seed + branch,
            submission_contract=submission_contract(count), base_url="https://api.deepseek.com",
            model="deepseek-flash", thinking="low", force_first_tool_call=False,
            guest_runtime=load_pi_guest_runtime("copper_inversion", RESOURCES / "harness/image"),
            tool_extensions=(HarnessToolExtension(Path("/resources/rsi/rsi.mjs"),
                             file_sha256(RESOURCES / "rsi/rsi.mjs"), ("get_inversion_context",)),),
            context7_enabled=False, network_policy=HarnessNetworkPolicy(denied_hosts=("*",)),
            limits=HarnessLimits(wall_time_seconds=harness_seconds,
                                 tool_call_budgets={"bash": 8, "web_search": 0, "web_fetch": 0}))
        turn = HarnessTurn("researcher", f"round_{round_index:03d}", round_index,
                           session_round, session_round + 1, canonical_sha256([message]), message)
        client = HarnessClient(pi_sidecar_command(turn_root, self.image, cache_root=self.cache,
                               mounts=((RESOURCES, "/resources", True),)),
                               api_key=self.key, config=config, response_timeout_seconds=harness_seconds + 120)
        visible = {r["program_id"] for r in view}

        def validate(request):
            try:
                parse_submission(request.submission, request.artifacts, turn_root, count, known, visible)
                return HarnessSubmissionValidation("accept")
            except (ValueError, SyntaxError, UnicodeError, OSError):
                # Deliberately avoid identifying whether another branch or the
                # current branch supplied a duplicate.
                return HarnessSubmissionValidation("retry", (HarnessSubmissionError(
                    "/candidates", "invalid_solver", "source rejected: not novel or invalid submission"),))

        started = time.monotonic()
        with client:
            (result,) = client.run_turn((turn,), submission_validator=validate,
                                        recovery_timeout_seconds=2 * harness_seconds)
        if result.submission_status != "accepted":
            raise HarnessError("researcher did not produce an accepted submission")
        rows = parse_submission(result.submission, result.submitted_artifacts, turn_root, count, known, visible)
        usage = collect(turn_root / result.artifacts["turn"])
        provider_calls = result.usage.get("providerCalls")
        if type(provider_calls) is not int or provider_calls < 1:
            usage["responses_missing_usage"] += 1
        elif provider_calls != usage["requests"]:
            usage["responses_missing_usage"] += abs(provider_calls - usage["requests"])
            usage["requests"] = max(usage["requests"], provider_calls)
        usage["session_id"] = result.session_id
        usage["wall_seconds"] = time.monotonic() - started
        return rows, usage
