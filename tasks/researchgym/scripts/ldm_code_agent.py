#!/usr/bin/env python3
"""A small, auditable code-writing agent for the full ResearchGym tasks.

Unlike the four-case program-search adapter, this agent owns a complete
ResearchGym task workspace.  It can inspect files, write code, and execute
experiments.  Its API credential is never passed to commands run by the agent.

The model is asked for a compact JSON action protocol rather than being asked
to emit one method snippet.  This makes the actual code-writing trajectory and
its command logs reviewable after a run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from tasks.researchgym.table3_protocol import PROFILES, TASKS, ProtocolError


MAX_TOOL_OUTPUT_CHARS = 16_000
MAX_FILE_READ_CHARS = 32_000
# Full task inspection commonly needs a short batch of reads plus one command.
# Eight actions caused otherwise valid DeepSeek trajectories to terminate before
# they reached an experiment; the bounded value still limits one response while
# avoiding needless paid retries.
MAX_ACTIONS_PER_TURN = 16
NETWORK_COMMAND = re.compile(
    r"https?://|\b(?:curl|wget|aria2c)\b|\bgit\s+clone\b|"
    r"\b(?:urllib|requests)\.(?:request|urlopen|get|post)\b|\bpip\s+(?:install|download)\b|"
    r"\bpretrained\s*=\s*True\b",
    re.IGNORECASE,
)


class AgentError(RuntimeError):
    """A launch configuration or model action was invalid."""


class BudgetExhausted(AgentError):
    """A new model request would violate the conservative budget reserve."""


@dataclass(frozen=True)
class Pricing:
    """Provider-specific USD price card supplied by the experiment operator."""

    input_usd_per_mtok: float
    cached_input_usd_per_mtok: float
    output_usd_per_mtok: float
    call_reserve_usd: float

    @classmethod
    def from_env(cls) -> "Pricing":
        values: dict[str, float] = {}
        names = {
            "input_usd_per_mtok": "LDM_INPUT_USD_PER_MTOK",
            "cached_input_usd_per_mtok": "LDM_CACHED_INPUT_USD_PER_MTOK",
            "output_usd_per_mtok": "LDM_OUTPUT_USD_PER_MTOK",
            "call_reserve_usd": "LDM_PROVIDER_CALL_RESERVE_USD",
        }
        for key, env_name in names.items():
            raw = os.environ.get(env_name)
            try:
                value = float(raw) if raw is not None else math.nan
            except ValueError:
                value = math.nan
            if not math.isfinite(value) or value <= 0:
                raise AgentError(f"{env_name} must be a finite positive USD value before a budgeted run")
            values[key] = value
        return cls(**values)

    def cost(self, usage: Mapping[str, Any]) -> float:
        prompt = _as_nonnegative_int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
        completion = _as_nonnegative_int(usage.get("completion_tokens", usage.get("output_tokens", 0)))
        details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
        cached = _as_nonnegative_int(details.get("cached_tokens", details.get("cache_read_input_tokens", 0)))
        # OpenAI-compatible endpoints differ on whether reasoning tokens are
        # included in completion_tokens.  Count an explicitly separate field
        # only when the endpoint exposes it outside completion details.
        completion_details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
        reasoning = _as_nonnegative_int(completion_details.get("reasoning_tokens", 0))
        billed_output = max(completion, reasoning)
        new_input = max(0, prompt - cached)
        return ((new_input * self.input_usd_per_mtok + cached * self.cached_input_usd_per_mtok
                 + billed_output * self.output_usd_per_mtok) / 1_000_000.0)


def _as_nonnegative_int(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, number)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(root: Path, raw: Any) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise AgentError("tool path must be a non-empty relative string")
    candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise AgentError("tool path escapes the task workspace") from exc
    return candidate


def _clip(value: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n...[truncated {len(value) - limit} chars]"


def parse_actions(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Extract a single JSON action object from a model response."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, count=1)
        stripped = re.sub(r"\s*```$", "", stripped, count=1)
    start = stripped.find("{")
    if start < 0:
        raise AgentError("model response did not contain a JSON action object")
    action_text = stripped[start:]
    try:
        payload, _ = json.JSONDecoder().raw_decode(action_text)
    except json.JSONDecodeError as first_error:
        # DeepSeek Flash sometimes emits a literal newline, or a regex-style
        # backslash, inside an otherwise complete JSON string.  Escape only
        # those string-literal defects; a missing delimiter/truncated object
        # remains an error and is never guessed into an executable command.
        repaired = _repair_json_string_literals(action_text)
        if repaired == action_text:
            raise AgentError(f"model returned malformed action JSON: {first_error}") from first_error
        try:
            payload, _ = json.JSONDecoder().raw_decode(repaired)
        except json.JSONDecodeError as exc:
            raise AgentError(f"model returned malformed action JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise AgentError("model action payload must be an object")
    actions = payload.get("actions")
    if not isinstance(actions, list) or not actions:
        raise AgentError("model action payload needs a non-empty actions list")
    if len(actions) > MAX_ACTIONS_PER_TURN:
        raise AgentError(f"model emitted more than {MAX_ACTIONS_PER_TURN} actions")
    if not all(isinstance(action, dict) and isinstance(action.get("tool"), str) for action in actions):
        raise AgentError("each action must be an object with a string tool")
    return str(payload.get("analysis", ""))[:4_000], actions


def _repair_json_string_literals(value: str) -> str:
    """Escape controls and invalid escapes inside JSON string literals only."""
    output: list[str] = []
    in_string = False
    escaped = False
    valid_escapes = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}
    for char in value:
        if not in_string:
            output.append(char)
            if char == '"':
                in_string = True
            continue
        if escaped:
            if char not in valid_escapes:
                # The existing backslash plus this one decodes to one literal
                # backslash; retain the model's following character verbatim.
                output.append("\\")
            output.append(char)
            escaped = False
            continue
        if char == "\\":
            output.append(char)
            escaped = True
        elif char == '"':
            output.append(char)
            in_string = False
        elif char == "\n":
            output.append("\\n")
        elif char == "\r":
            output.append("\\r")
        elif char == "\t":
            output.append("\\t")
        else:
            output.append(char)
    return "".join(output)


class WorkspaceTools:
    """Filesystem and command tools exposed to the LDM agent."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()

    def execute(self, action: Mapping[str, Any]) -> dict[str, Any]:
        tool = action["tool"]
        try:
            if tool == "list":
                return self._list(action)
            if tool == "read":
                return self._read(action)
            if tool == "search":
                return self._search(action)
            if tool == "write":
                return self._write(action)
            if tool == "replace":
                return self._replace(action)
            if tool == "bash":
                return self._bash(action)
            if tool == "submit":
                return self._submit(action)
            raise AgentError(f"unknown tool {tool!r}")
        except Exception as exc:
            return {"tool": tool, "ok": False, "error": str(exc)}

    def _list(self, action: Mapping[str, Any]) -> dict[str, Any]:
        target = _safe_relative(self.workspace, action.get("path", "."))
        if not target.exists():
            raise AgentError(f"not found: {target.relative_to(self.workspace)}")
        if target.is_file():
            names = [target.relative_to(self.workspace).as_posix()]
        else:
            max_entries = min(1000, max(1, int(action.get("max_entries", 200))))
            names = [path.relative_to(self.workspace).as_posix() for path in sorted(target.rglob("*"))[:max_entries]]
        return {"tool": "list", "ok": True, "paths": names}

    def _read(self, action: Mapping[str, Any]) -> dict[str, Any]:
        target = _safe_relative(self.workspace, action.get("path"))
        if not target.is_file():
            raise AgentError(f"not a file: {target.relative_to(self.workspace)}")
        offset = max(0, int(action.get("offset", 0)))
        limit = min(MAX_FILE_READ_CHARS, max(1, int(action.get("limit", 12_000))))
        data = target.read_text(encoding="utf-8", errors="replace")
        return {"tool": "read", "ok": True, "path": target.relative_to(self.workspace).as_posix(),
                "offset": offset, "content": data[offset:offset + limit], "truncated": offset + limit < len(data)}

    def _search(self, action: Mapping[str, Any]) -> dict[str, Any]:
        """Find exact text in one file, returning line numbers and read offsets."""
        target = _safe_relative(self.workspace, action.get("path"))
        query = action.get("query")
        if not target.is_file() or not isinstance(query, str) or not 1 <= len(query) <= 200:
            raise AgentError("search needs an existing file and a 1–200 character query")
        if target.stat().st_size > 2_000_000:
            raise AgentError("search file exceeds 2 MB")
        matches, offset = [], 0
        for number, line in enumerate(target.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True), 1):
            if query.lower() in line.lower():
                matches.append({"line": number, "offset": offset, "text": line.strip()[:350]})
                if len(matches) >= 30:
                    break
            offset += len(line)
        return {"tool": "search", "ok": True, "path": target.relative_to(self.workspace).as_posix(),
                "query": query, "matches": matches}

    def _write(self, action: Mapping[str, Any]) -> dict[str, Any]:
        target = _safe_relative(self.workspace, action.get("path"))
        content = action.get("content")
        if not isinstance(content, str):
            raise AgentError("write content must be a string")
        if len(content) > 2_000_000:
            raise AgentError("write content exceeds 2 MB")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"tool": "write", "ok": True, "path": target.relative_to(self.workspace).as_posix(),
                "bytes": len(content.encode("utf-8")), "sha256": _sha256(target)}

    def _replace(self, action: Mapping[str, Any]) -> dict[str, Any]:
        """Edit one exact source span without re-emitting a large complete file."""
        target = _safe_relative(self.workspace, action.get("path"))
        old, new = action.get("old"), action.get("new")
        if not target.is_file() or not isinstance(old, str) or not old or not isinstance(new, str):
            raise AgentError("replace needs an existing file and non-empty old/new string fields")
        if len(old) > 100_000 or len(new) > 100_000:
            raise AgentError("replace span exceeds 100 KB")
        source = target.read_text(encoding="utf-8")
        if source.count(old) != 1:
            raise AgentError("replace old span must occur exactly once; read the file and use a precise snippet")
        target.write_text(source.replace(old, new, 1), encoding="utf-8")
        return {"tool": "replace", "ok": True, "path": target.relative_to(self.workspace).as_posix(),
                "sha256": _sha256(target)}

    def _bash(self, action: Mapping[str, Any]) -> dict[str, Any]:
        command = action.get("command")
        if not isinstance(command, str) or not command.strip():
            raise AgentError("bash command must be a non-empty string")
        # The manifest used by this harness has external_access=disabled.
        # Block common direct fetch and package-install forms before execution;
        # required data, models, and task interpreters are supplied as explicit
        # read-only mounts by the launcher.
        if NETWORK_COMMAND.search(command):
            raise AgentError("network-capable shell command is forbidden: use only mounted task inputs")
        timeout = min(3_600, max(1, int(action.get("timeout_seconds", 600))))
        # The model's credential must never be inherited by an experiment,
        # shell, child Python interpreter, or task code.
        child_env = {key: value for key, value in os.environ.items()
                     if not any(token in key.upper() for token in ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH"))}
        result = subprocess.run(["bash", "-lc", command], cwd=self.workspace, env=child_env,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=timeout, check=False)
        return {"tool": "bash", "ok": result.returncode == 0, "returncode": result.returncode,
                "output": _clip(result.stdout), "timeout_seconds": timeout}

    def _submit(self, action: Mapping[str, Any]) -> dict[str, Any]:
        summary = action.get("summary", "")
        if not isinstance(summary, str):
            raise AgentError("submit summary must be a string")
        return {"tool": "submit", "ok": True, "summary": summary[:8_000]}


class ChatCompletionsClient:
    """Minimal OpenAI-compatible chat client with explicit usage accounting."""

    def __init__(self, *, api_url: str, api_key: str, model: str, pricing: Pricing,
                 budget_usd: float, request_timeout: int, initial_cost_usd: float = 0.0) -> None:
        if not api_url.startswith(("https://", "http://")):
            raise AgentError("--api-url must be an http(s) URL")
        if not api_key:
            raise AgentError("LLM_API_KEY is required")
        self.api_url, self.api_key, self.model = api_url, api_key, model
        if not math.isfinite(initial_cost_usd) or initial_cost_usd < 0:
            raise AgentError("initial API cost must be a finite non-negative USD value")
        if initial_cost_usd > budget_usd + 1e-9:
            raise AgentError("prior API cost already exceeds the fixed experiment cap")
        self.pricing, self.budget_usd, self.request_timeout = pricing, budget_usd, request_timeout
        self.prior_cost_usd = initial_cost_usd
        self.total_cost = initial_cost_usd
        self.usage_events: list[dict[str, Any]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        if self.total_cost + self.pricing.call_reserve_usd > self.budget_usd + 1e-9:
            raise BudgetExhausted(
                f"conservative call reserve would exceed ${self.budget_usd:.2f} "
                f"(spent ${self.total_cost:.6f}, reserve ${self.pricing.call_reserve_usd:.6f})"
            )
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "chat_template_kwargs": {"reasoning": True},
            "reasoning_content": True,
            "max_tokens": 4096,
        }
        request = urllib.request.Request(self.api_url, data=json.dumps(payload).encode("utf-8"), method="POST",
                                         headers={"Authorization": f"Bearer {self.api_key}",
                                                  "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # Do not include an HTTP body: a proxy could reflect sensitive data.
            raise AgentError(f"provider returned HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise AgentError(f"provider request failed: {type(exc).__name__}") from exc
        # A successful HTTP response may still have empty assistant content.
        # It is a paid call, so record usage before validating the content.
        usage = data.get("usage") if isinstance(data, Mapping) else None
        if not isinstance(usage, Mapping):
            self.total_cost += self.pricing.call_reserve_usd
            self.usage_events.append({
                "at": _now(), "usage": None, "unknown_usage": True,
                "cost_usd": self.pricing.call_reserve_usd,
                "cumulative_cost_usd": self.total_cost,
            })
            raise AgentError("provider response omitted token usage; refusing unaccounted model call")
        cost = self.pricing.cost(usage)
        self.total_cost += cost
        event = {"at": _now(), "usage": dict(usage), "cost_usd": cost, "cumulative_cost_usd": self.total_cost}
        self.usage_events.append(event)
        if self.total_cost > self.budget_usd + 1e-9:
            raise BudgetExhausted(f"provider response exceeded the strict ${self.budget_usd:.2f} cap")
        choices = data.get("choices") or []
        message = choices[0].get("message", {}) if choices and isinstance(choices[0], Mapping) else {}
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str) or not content.strip():
            raise AgentError("provider response had no assistant content")
        return content


def _system_prompt() -> str:
    return """You are LDM-CodeAgent operating a complete ResearchGym research task.
Your work must be real: inspect the workspace, edit the task's actual code, run
experiments, and run the supplied official grader before submission.  Do not
invent metrics.  You have no web/search access and must work from the provided
task, code, data, and logs.  API credentials are not available to shell commands.

On every turn respond with exactly one JSON object:
{"analysis":"short plan", "actions":[...]}
Each action must be one of:
{"tool":"list","path":"."}
{"tool":"read","path":"relative/file.py","offset":0,"limit":12000}
{"tool":"search","path":"relative/file.py","query":"function or keyword"}
{"tool":"write","path":"relative/file.py","content":"complete file content"}
{"tool":"replace","path":"relative/file.py","old":"exact existing snippet","new":"replacement code"}
{"tool":"bash","command":"command executed in the task workspace","timeout_seconds":600}
{"tool":"submit","summary":"what changed, commands run, and official-grade file"}

Use relative paths.  Work incrementally, read before overwriting, and use bash
for tests/training/grading.  A submit action ends the trajectory; submit only
after you have invoked the official grader and inspected its result.  JSON
strings must escape literal newlines as \\n, and emit no more than 16 actions in
one response.

Work toward a scored result, not an exhaustive repository survey.  Search a
large file for a function or keyword instead of reading it in tiny sequential
chunks; search returns character offsets you can pass to read.  In your
first 10 turns, inspect the task entry point and grader, then choose a method
and WRITE its code with write or replace.  Use replace for a small edit to a
large existing file.  If you keep reading or listing after
turn 10 without writing a candidate, the trajectory will be stopped as invalid.
Run a small smoke test by turn 20, a complete evaluation
by turn 35, and the official grader as soon as an evaluation log exists.  Do
not repeatedly read files already inspected.  If an existing candidate or
checkpoint is in the workspace, build on it.  Every tool action consumes the
limited API and wall-clock budget."""


def _task_prompt(task_id: str, workspace: Path, seed: int, profile_name: str) -> str:
    task = TASKS[task_id]
    description = workspace / "task_description.md"
    text = description.read_text(encoding="utf-8", errors="replace") if description.exists() else "(not found)"
    task_python = os.environ.get("RESEARCHGYM_TASK_PYTHON", "").strip()
    runtime = (f"Use {task_python} for task commands; it is the prebuilt task interpreter.\n"
               if task_python else "Use the task's available Python interpreter for task commands.\n")
    return (
        f"Task={task.display_name} ({task_id}); independent seed={seed}; profile={profile_name}.\n"
        f"Official Table 3 metrics: {', '.join(task.metrics)}. Primary selection: {task.primary_note}.\n"
        "The independent trajectory seed is not necessarily the task's pretrained-model or data seed. "
        "Inspect available checkpoint filenames and use the benchmark's actual required seed; "
        "never rename or fabricate missing checkpoints.\n"
        f"Workspace: {workspace}.\n{runtime}\nTask description (possibly truncated):\n{_clip(text, 24_000)}"
    )


def resolve_api_url(raw: str) -> str:
    """Accept either a full chat endpoint or the common OpenAI-compatible base."""
    value = raw.strip().rstrip("/")
    if not value:
        return ""
    if value.endswith("/chat/completions"):
        return value
    return value + "/chat/completions"


def load_prior_receipt(path: Path, *, task_id: str, seed: int,
                       profile_name: str, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Load one failed predecessor while retaining the per-seed budget cap.

    A harness correction can justify a clean retry, but it must not make the
    original paid calls disappear from cost or wall-clock accounting.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AgentError("prior agent receipt must be a JSON object")
    expected = {
        "task_id": task_id,
        "seed": seed,
        "profile": profile_name,
        "protocol": manifest.get("protocol"),
        "source_commit": manifest.get("source_commit"),
        "model": manifest.get("model"),
        "harness": manifest.get("harness"),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise AgentError(f"prior receipt {field} does not match this trajectory")
    if payload.get("status") == "submitted":
        raise AgentError("a submitted trajectory cannot be retried")
    for field in ("api_cost_usd", "wall_clock_hours"):
        value = payload.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise AgentError(f"prior receipt has invalid {field}")
    return payload


def _compact_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    # Keep immutable instructions and the latest interaction context.  Tool
    # observations themselves are persisted on disk, so compaction is auditable.
    if len(messages) <= 8:
        return messages
    return messages[:2] + messages[-6:]


def _code_snapshot(workspace: Path) -> dict[str, str]:
    """Evidence that the agent changed source, regardless of edit tool."""
    result = {}
    for path in workspace.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".sh"}:
            result[path.relative_to(workspace).as_posix()] = _sha256(path)
    return result


def run_agent(*, task_id: str, workspace: Path, seed: int, profile_name: str,
              manifest: Mapping[str, Any], output_dir: Path, client: ChatCompletionsClient,
              max_turns: int, prior_receipt: Mapping[str, Any] | None = None) -> dict[str, Any]:
    profile = PROFILES[profile_name]
    if manifest.get("profile", {}).get("name") != profile_name:
        raise AgentError("manifest/profile mismatch")
    if manifest.get("protocol") != "table3_agent_trajectory_v1":
        raise AgentError("manifest is not a Table 3 agent protocol")
    if manifest.get("source_commit") is None:
        raise AgentError("manifest lacks source commit")
    if task_id not in TASKS or seed not in manifest.get("seeds", []):
        raise AgentError("task or seed is not in the immutable manifest")
    if not workspace.is_dir():
        raise AgentError(f"workspace is not a directory: {workspace}")
    prior_wall_clock_hours = float(prior_receipt.get("wall_clock_hours", 0.0)) if prior_receipt else 0.0
    if prior_wall_clock_hours > profile.wall_clock_hours + 1e-9:
        raise AgentError("prior trajectory already exceeds the fixed wall-clock cap")
    if client.total_cost > profile.api_budget_usd + 1e-9:
        raise AgentError("prior trajectory already exceeds the fixed API cap")
    output_dir.mkdir(parents=True, exist_ok=False)
    started_monotonic = time.monotonic()
    started_at = _now()
    deadline = started_monotonic + max(0.0, profile.wall_clock_hours - prior_wall_clock_hours) * 3600
    tools = WorkspaceTools(workspace)
    transcript: list[dict[str, Any]] = []
    messages = [{"role": "system", "content": _system_prompt()},
                {"role": "user", "content": _task_prompt(task_id, workspace, seed, profile_name)}]
    status, submitted = "turn_exhausted", None
    consecutive_invalid_actions = 0
    candidate_written = False
    initial_code = _code_snapshot(workspace)
    active_action: dict[str, str] | None = None
    checkpoint_lock = threading.Lock()
    heartbeat_stop = threading.Event()

    def checkpoint() -> None:
        # Persist progress while a trajectory is still running.  A launcher
        # crash must not hide paid calls or leave monitoring dependent on a
        # final receipt that has not yet been written.
        with checkpoint_lock:
            progress = {
                "status": "running", "task_id": task_id, "seed": seed,
                "candidate_written": candidate_written, "active_action": active_action,
                "turns_completed": len(transcript), "api_calls": len(client.usage_events),
                "api_cost_usd": client.total_cost,
                "wall_clock_hours": prior_wall_clock_hours + (time.monotonic() - started_monotonic) / 3600.0,
                "at": _now(),
            }
            tmp = output_dir / "progress.json.tmp"
            tmp.write_text(json.dumps(progress, sort_keys=True) + "\n", encoding="utf-8")
            tmp.replace(output_dir / "progress.json")

    def heartbeat() -> None:
        while not heartbeat_stop.wait(30.0):
            checkpoint()

    checkpoint()
    heartbeat_thread = threading.Thread(target=heartbeat, name="researchgym-agent-progress", daemon=True)
    heartbeat_thread.start()
    for turn in range(max_turns):
        if turn >= 24 and not candidate_written:
            status = "no_candidate"
            break
        if time.monotonic() >= deadline:
            status = "time_exhausted"
            break
        try:
            response = ""
            request_messages = _compact_messages(messages)
            urgency = (" No source code has changed. Your next response MUST include a write or replace "
                       "action editing a .py or .sh method. A simple runnable baseline is better than more "
                       "inspection; you can revise it after a smoke test."
                       if turn >= 10 and not candidate_written else "")
            request_messages.append({
                "role": "user",
                "content": (
                    f"Progress: turn {turn + 1}/{max_turns}; API spent "
                    f"${client.total_cost:.4f}/${profile.api_budget_usd:.2f}. "
                    "Prioritize writing, evaluating, and grading a concrete method now." + urgency
                ),
            })
            response = client.complete(request_messages)
            checkpoint()
            analysis, actions = parse_actions(response)
        except BudgetExhausted as exc:
            status = "budget_exhausted"
            transcript.append({"turn": turn, "at": _now(), "error": str(exc)})
            checkpoint()
            break
        except AgentError as exc:
            # A provider can occasionally return a nearly-valid action object
            # containing an unescaped newline in a shell command.  No tool has
            # run at this point, so ask for a corrected action rather than
            # charging a whole trajectory to a formatting transient.  Bound
            # retries to avoid an endless paid correction loop.
            consecutive_invalid_actions += 1
            transcript.append({
                "turn": turn,
                "at": _now(),
                "error": str(exc),
                "response_sha256": hashlib.sha256(response.encode()).hexdigest()
                if "response" in locals() else None,
            })
            checkpoint()
            if consecutive_invalid_actions >= 3:
                status = "agent_error"
                break
            messages.append({
                "role": "user",
                "content": (
                    "Your preceding action response was rejected and no tool was run: "
                    f"{exc}. Return a replacement as one strict JSON object only. "
                    "Escape newlines inside JSON strings as \\n and use at most 16 actions."
                ),
            })
            continue
        consecutive_invalid_actions = 0
        observations = []
        if turn >= 16 and not candidate_written and not any(
                action.get("tool") in {"write", "replace"} for action in actions):
            observations.append({"tool": "source_change_guard", "ok": False,
                                 "error": "Inspection-only turn refused: edit a .py or .sh source file now."})
        else:
            for action in actions:
                active_action = {"tool": str(action.get("tool", ""))}
                if isinstance(action.get("path"), str):
                    active_action["path"] = action["path"][:300]
                checkpoint()
                observation = tools.execute(action)
                active_action = None
                observations.append(observation)
                if observation.get("tool") == "submit" and observation.get("ok"):
                    submitted = observation
                    break
        transcript.append({"turn": turn, "at": _now(), "analysis": analysis, "actions": actions,
                           "observations": observations, "response_sha256": hashlib.sha256(response.encode()).hexdigest()})
        candidate_written = candidate_written or _code_snapshot(workspace) != initial_code
        checkpoint()
        # Keep the response only as a digest in transcript; complete action
        # requests and tool results are enough to reproduce the workspace edits.
        messages.append({"role": "assistant", "content": json.dumps({"analysis": analysis, "actions": actions})})
        messages.append({"role": "user", "content": "Tool results:\n" + _clip(json.dumps(observations, ensure_ascii=False), 20_000)})
        if submitted is not None:
            status = "submitted"
            break
    elapsed = time.monotonic() - started_monotonic
    heartbeat_stop.set()
    heartbeat_thread.join(timeout=1)
    (output_dir / "transcript.json").write_text(json.dumps(transcript, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "usage.json").write_text(json.dumps(client.usage_events, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    receipt = {
        "run_id": uuid.uuid4().hex,
        "status": status,
        "submitted": submitted,
        "task_id": task_id,
        "seed": seed,
        "profile": profile_name,
        "protocol": manifest["protocol"],
        "source_commit": manifest["source_commit"],
        "model": manifest["model"],
        "harness": manifest["harness"],
        "started_at": started_at,
        "wall_clock_hours": prior_wall_clock_hours + elapsed / 3600.0,
        "api_cost_usd": client.total_cost,
        "api_calls": int(prior_receipt.get("api_calls", 0)) + len(client.usage_events)
        if prior_receipt else len(client.usage_events),
        "workspace": str(workspace),
        "transcript_sha256": _sha256(output_dir / "transcript.json"),
        "usage_sha256": _sha256(output_dir / "usage.json"),
    }
    if prior_receipt:
        receipt["prior_agent_receipt"] = {
            "run_id": prior_receipt.get("run_id"),
            "status": prior_receipt.get("status"),
            "api_cost_usd": prior_receipt["api_cost_usd"],
            "wall_clock_hours": prior_receipt["wall_clock_hours"],
            "transcript_sha256": prior_receipt.get("transcript_sha256"),
            "usage_sha256": prior_receipt.get("usage_sha256"),
        }
    (output_dir / "agent_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "progress.json").write_text(
        json.dumps({"status": status, "task_id": task_id, "seed": seed,
                    "candidate_written": candidate_written, "active_action": None,
                    "turns_completed": len(transcript), "api_calls": receipt["api_calls"],
                    "api_cost_usd": receipt["api_cost_usd"],
                    "wall_clock_hours": receipt["wall_clock_hours"], "at": _now()}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-url", default=os.environ.get("LLM_API_URL") or os.environ.get("LLM_BASE_URL", ""))
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL_NAME", ""))
    parser.add_argument("--max-turns", type=int, default=240)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--prior-agent-receipt", type=Path,
                        help="Failed predecessor receipt; its cost and elapsed time remain charged to this seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.max_turns <= 0 or args.request_timeout <= 0:
            raise AgentError("--max-turns and --request-timeout must be positive")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        pricing = Pricing.from_env()
        profile = PROFILES[args.profile]
        prior_receipt = (load_prior_receipt(args.prior_agent_receipt, task_id=args.task, seed=args.seed,
                                            profile_name=args.profile, manifest=manifest)
                         if args.prior_agent_receipt else None)
        client = ChatCompletionsClient(api_url=resolve_api_url(args.api_url), api_key=os.environ.get("LLM_API_KEY", ""),
                                       model=args.model, pricing=pricing, budget_usd=profile.api_budget_usd,
                                       request_timeout=args.request_timeout,
                                       initial_cost_usd=float(prior_receipt["api_cost_usd"]) if prior_receipt else 0.0)
        receipt = run_agent(task_id=args.task, workspace=args.workspace, seed=args.seed,
                            profile_name=args.profile, manifest=manifest, output_dir=args.output_dir,
                            client=client, max_turns=args.max_turns, prior_receipt=prior_receipt)
    except (OSError, ValueError, json.JSONDecodeError, ProtocolError, AgentError) as exc:
        print(f"LDM-CodeAgent launch failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: receipt[key] for key in ("run_id", "status", "task_id", "seed", "api_cost_usd")}, sort_keys=True))
    return 0 if receipt["status"] == "submitted" else 3


if __name__ == "__main__":
    raise SystemExit(main())
