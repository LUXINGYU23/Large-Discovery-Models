"""One isolated Pi sidecar per branch and artifact root, kept open across that root's rounds.

The root is ``harness/epoch_kk/branch_b`` (``controller.turn_root``). Clearing arms get a new
root, hence a fresh session, workspace and client, per epoch (``every_k``) or per round
(``every_round``); ``clear=never`` keeps one root and one client for the whole run. The old
client is closed, and its named container gone, before the branch's next client starts. A
restarted process reopens the same root and the sidecar session store continues it
(``harnesses/pi/src/session-store.ts`` pins the initialization identity per root).
"""

import hashlib
import json
import re
import subprocess
import threading
import time
import weakref
from pathlib import Path

from ldm_tts.harness import (HarnessArtifactRule, HarnessClient, HarnessError, HarnessLimits,
                            HarnessNetworkPolicy, HarnessProfile, HarnessSubmissionContract,
                            HarnessSubmissionError, HarnessSubmissionValidation, HarnessToolExtension,
                            HarnessTurn)
from ldm_tts.harness.container import pi_sidecar_command
from ldm_tts.harness.pi import PiHarnessConfig, load_pi_guest_runtime
from ldm_tts.harness.protocol import canonical_sha256, file_sha256
from ..core.candidate import MAX_SOURCE_BYTES, validate_source
from ..core.encoding import MAX_NORMALIZED_BYTES, normalized_code
from ..v2.research import PublicTrainingExport
from .case_isolation import check_source_case_ids, resolve_case_isolation
from .controller import _validate_submission, profile_name, NoSubmission
from .render import write_view
from .usage import collect


RESOURCES = Path(__file__).resolve().parents[1] / "resources"
# The only task resources the sidecar sees (gap audit §3.9): not resources/v2 or resources/harness.
# The guest image is resolved from the cache by its recipe digest, so harness/image needs no mount.
TOOL_RESOURCES, GUEST_RESOURCES = RESOURCES / "rsi", Path("/resources/rsi")


def sidecar_mounts():
    return ((TOOL_RESOURCES, str(GUEST_RESOURCES), True),)


def container_name(turn_root, profile, run, epoch):
    """One name per artifact root: a second sidecar on the same root fails to start."""
    digest = hashlib.sha256(str(Path(turn_root).resolve()).encode()).hexdigest()[:10]
    return re.sub(r"[^A-Za-z0-9_.-]", "-", f"rsi-{run}-e{epoch:02d}-{profile}-{digest}")


def _await_container_gone(name, timeout=120.):
    deadline = time.monotonic() + timeout
    while subprocess.run(["docker", "container", "inspect", name], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL).returncode == 0:
        if time.monotonic() >= deadline:
            raise HarnessError(f"sidecar container {name} is still present; stop it before reusing its root")
        time.sleep(1)


def fork_sources(turn_root):
    """Pi fork snapshots placed by a conversation-mode continuation; pinned on every turn of that root."""
    try:
        from .continuation import fork_sources as found
    except ModuleNotFoundError as exc:  # before the continuation module exists
        if exc.name != __package__ + ".continuation":
            raise
        return {}
    return found(turn_root)


def _close_all(clients, lock):
    with lock:
        held = list(clients.values())
        clients.clear()
    for _, client in held:
        client.close()


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


class NotNovel(ValueError):
    """A submitted source is already known; the reply must not say whose (contracts §5)."""

    def __init__(self):
        super().__init__("source rejected: not novel")


def parse_submission(submission, artifacts, root, count, known, visible, *, forbidden_case_tokens=()):
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
            raise ValueError(f"candidates[{index}].artifact_path must be candidates/{index}/solver.py "
                             "(entry i points to directory i, counting from 0 in submission order)")
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
        check_source_case_ids(body, forbidden_case_tokens)
        size = len(normalized_code(body).encode("utf-8"))
        if size > MAX_NORMALIZED_BYTES:
            raise ValueError(f"candidates[{index}] is {size} bytes after normalization (comments removed, code "
                             f"re-formatted); the limit is {MAX_NORMALIZED_BYTES} bytes")
        if sha in used:
            raise NotNovel()
        if row["parent_sha256"] is not None and row["parent_sha256"] not in visible:
            raise ValueError("parent is not visible to this branch")
        proposal = {**{k: v for k, v in row.items() if k != "artifact_path"}, "source": body.decode("utf-8")}
        _validate_submission(proposal)
        used.add(sha)
        result.append(proposal)
    return result


EXHAUSTED = "Submission recovery exhausted: no accepted submission after two continuation prompts."


def _exhausted_submission(error, turn_dir, profile):
    """The sidecar exposes a generic code; require its exact durable recovery state too."""
    if (error.retryable or error.code != "sidecar_error" or str(error) not in (EXHAUSTED, "Error: " + EXHAUSTED)
            or profile not in error.turn_usage or type(error.turn_usage[profile].get("providerCalls")) is not int):
        return False
    state_path = turn_dir / "history-recovery.json"
    if not state_path.is_file():
        return False
    try:
        state = json.loads(state_path.read_text())
    except (ValueError, OSError):
        return False
    return (isinstance(state, dict) and state.get("continuations") == 2
            and isinstance(state.get("failure"), dict) and state["failure"].get("message") == EXHAUSTED)


class PiAgent:
    """Real-model researcher; thread-safe per the contract in ``controller.py``."""

    def __init__(self, *, public_catalog, api_key, sidecar_image, cache_dir, arm, reset_period,
                 base_url, model, thinking, bash_calls, case_isolation=None):
        if not api_key:
            raise ValueError("protected model API key required")
        if type(bash_calls) is not int or bash_calls < 0:
            raise ValueError("invalid bash call budget")
        self.case_isolation = resolve_case_isolation(case_isolation)
        try:
            self.export = PublicTrainingExport(public_catalog,
                                               anonymize_cases=self.case_isolation["anonymize_cases"],
                                               canonical_origin=self.case_isolation["canonical_origin"])
        except TypeError:
            self.export = PublicTrainingExport(public_catalog)
        self.forbidden_case_tokens = (getattr(self.export, "forbidden_tokens", frozenset())
                                      if self.case_isolation["refuse_case_ids"] else frozenset())
        self.key, self.image, self.cache = api_key, sidecar_image, Path(cache_dir)
        self.arm, self.reset_period = arm, reset_period
        self.base_url, self.model, self.thinking, self.bash_calls = base_url, model, thinking, bash_calls
        self.guest_runtime = load_pi_guest_runtime("copper_inversion", RESOURCES / "harness/image")
        self.agents_sha256 = file_sha256(TOOL_RESOURCES / "AGENTS.md")
        self.tool_sha256 = file_sha256(TOOL_RESOURCES / "rsi.mjs")
        self.config(profile="researcher_0", run="check", epoch=0, seed=0, count=1, harness_seconds=1)
        self._clients, self._lock = {}, threading.Lock()  # profile -> (artifact root, open client)
        self._finalizer = weakref.finalize(self, _close_all, self._clients, self._lock)

    def config(self, *, profile, run, epoch, seed, count, harness_seconds, fork_sources=None):
        """Constant for one (run, epoch, profile): the sidecar refuses a changed identity on a reused root."""
        return PiHarnessConfig(
            artifact_root=Path("/artifacts"),
            profiles=(HarnessProfile(profile, GUEST_RESOURCES / "AGENTS.md", agents_sha256=self.agents_sha256),),
            campaign_id=f"rsi-{run}-e{epoch:02d}-{profile}", task_id="copper_inversion",
            case_id="public-training", seed=seed, submission_contract=submission_contract(count),
            fork_sources=dict(fork_sources or {}),
            base_url=self.base_url, model=self.model, thinking=self.thinking, force_first_tool_call=False,
            guest_runtime=self.guest_runtime,
            tool_extensions=(HarnessToolExtension(GUEST_RESOURCES / "rsi.mjs", self.tool_sha256,
                                                  ("get_inversion_context",)),),
            context7_enabled=False, network_policy=HarnessNetworkPolicy(denied_hosts=("*",)),
            limits=HarnessLimits(wall_time_seconds=harness_seconds,
                                 tool_call_budgets={"bash": self.bash_calls, "web_search": 0, "web_fetch": 0}))

    def sidecar_command(self, turn_root, name):
        command = pi_sidecar_command(Path(turn_root), self.image, cache_root=self.cache, mounts=sidecar_mounts())
        at = command.index("run") + 1
        return command[:at] + ["--name", name] + command[at:]

    def _new_client(self, turn_root, config, name):
        _await_container_gone(name)
        return HarnessClient(self.sidecar_command(turn_root, name), api_key=self.key, config=config,
                             response_timeout_seconds=config.limits.wall_time_seconds + 120)

    def _client(self, profile, turn_root, config, name):
        with self._lock:
            held = self._clients.get(profile)
        if held is not None and held[0] == turn_root and held[1].config == config:
            return held[1]
        self.release(profile)  # the previous root's sidecar stops completely first
        client = self._new_client(turn_root, config, name)
        client.start()
        with self._lock:
            self._clients[profile] = (turn_root, client)
        return client

    def release(self, profile):
        with self._lock:
            held = self._clients.pop(profile, None)
        if held is not None:
            held[1].close()

    def close(self):
        _close_all(self._clients, self._lock)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def propose(self, *, branch, round_index, epoch, count, session_round, message, view,
                turn_root, candidate_root, known, seed, harness_seconds, profile=None, turn_index=0):
        if turn_index != 0:
            raise ValueError("several turns per round are not supported (extension arm)")
        profile = profile_name(branch) if profile is None else profile
        turn_root = Path(turn_root).resolve()
        turn_root.mkdir(parents=True, exist_ok=True)
        self.export(turn_root, None)
        public = turn_root / "public"
        (public / "TASK.md").write_bytes((TOOL_RESOURCES / "TASK.md").read_bytes())
        write_view(public / "evidence", view, candidate_root)  # the controller wrote the same view
        run = Path(candidate_root).resolve().name
        config = self.config(profile=profile, run=run, epoch=epoch, seed=seed + branch, count=count,
                             harness_seconds=harness_seconds, fork_sources=fork_sources(turn_root))
        turn = HarnessTurn(profile, f"round_{round_index:03d}", round_index,
                           session_round, session_round + 1, canonical_sha256([message]), message)
        visible = {r["program_id"] for r in view}

        def taken():
            return known.excluding(branch) if hasattr(known, "excluding") else known

        def validate(request):
            try:
                rows = parse_submission(request.submission, request.artifacts, turn_root, count, taken(), visible,
                                        forbidden_case_tokens=self.forbidden_case_tokens)
                if hasattr(known, "claim") and not known.claim(
                        branch, [validate_source(r["source"].encode("utf-8")) for r in rows]):
                    raise NotNovel()  # another branch submitted it this round
                return HarnessSubmissionValidation("accept")
            except NotNovel:
                # Deliberately avoid identifying whether another branch or the
                # current branch supplied a duplicate.
                reason = "source rejected: not novel"
            except SyntaxError as exc:
                reason = f"invalid submission: a solver does not parse (line {exc.lineno}: {exc.msg})"
            except UnicodeError:
                reason = "invalid submission: a solver is not UTF-8 text"
            except OSError:
                reason = "invalid submission: a submitted artifact could not be read"  # no host paths
            except ValueError as exc:
                # A structural defect the researcher can repair; the rehearsal lost a run to a
                # path/index mismatch hidden behind the uniform novelty reply.
                reason = "invalid submission: " + str(exc)
            return HarnessSubmissionValidation("retry", (HarnessSubmissionError(
                "/candidates", "invalid_solver", reason),))

        started = time.monotonic()
        client = self._client(profile, turn_root, config, container_name(turn_root, profile, run, epoch))
        try:
            (result,) = client.run_turn((turn,), submission_validator=validate,
                                        recovery_timeout_seconds=2 * harness_seconds)
        except HarnessError as exc:
            turn_dir = turn_root / "sessions" / profile / "turns" / turn.turn_id
            if _exhausted_submission(exc, turn_dir, profile):
                usage = collect(turn_dir, exc.turn_usage[profile]["providerCalls"])
                usage.update(turn=str(turn_dir.relative_to(turn_root)), wall_seconds=time.monotonic() - started)
                raise NoSubmission(usage, cursor_advanced=False) from exc
            self.release(profile)  # infrastructure failure keeps its pause semantics
            raise
        except BaseException:
            self.release(profile)
            raise
        turn_dir = (turn_root / result.artifacts["turn"]).resolve()
        if not turn_dir.is_relative_to(turn_root):
            raise HarnessError("sidecar turn directory escapes the artifact root")
        usage = collect(turn_dir, result.usage.get("providerCalls"))
        usage.update(session_id=result.session_id, turn=result.artifacts["turn"],
                     wall_seconds=time.monotonic() - started)
        if result.submission_status == "rejected":
            raise NoSubmission(usage, cursor_advanced=True)
        if result.submission_status != "accepted":
            raise HarnessError("unexpected submission status")
        rows = parse_submission(result.submission, result.submitted_artifacts, turn_root, count, taken(), visible,
                                forbidden_case_tokens=self.forbidden_case_tokens)
        return rows, usage
