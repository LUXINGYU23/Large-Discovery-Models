"""LLM and ReaSyn source adapters for tilted case2 methods."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import multiprocessing
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import traceback
import time
from typing import Any, Callable

from tasks.small_molecule.core.ldm_tilted_case2.canonicalize import RawCandidate


@dataclass
class ParsedLLMResult:
    parsed: Any
    raw_text: str
    parsed_json: dict[str, Any]
    attempts: list[dict[str, Any]] = field(default_factory=list)


def call_llm_json(
    llm,
    system: str,
    user: str,
    parser: Callable[[str], Any],
    *,
    max_retries: int,
    retry_wait_seconds: float = 0.0,
    stage: str = "",
    source_id: str = "",
) -> ParsedLLMResult:
    attempts: list[dict[str, Any]] = []
    prompt = user
    last_error: Exception | None = None
    for attempt_idx in range(max_retries + 1):
        t0 = time.monotonic()
        try:
            raw_text = _chat_with_hard_timeout(llm, system, prompt)
        except Exception as exc:
            last_error = exc
            error = f"{type(exc).__name__}: {exc}"
            attempts.append(_attempt_record(
                attempt_idx, stage, source_id, system, prompt, "",
                error=error,
                duration_ms=(time.monotonic() - t0) * 1000.0,
            ))
            prompt = user + f"\nPrevious LLM call error: {exc}. Return valid JSON only."
            _sleep_before_retry(attempt_idx, max_retries, retry_wait_seconds)
            continue
        try:
            parsed = parser(raw_text)
            parsed_json = parsed.to_dict() if hasattr(parsed, "to_dict") else {}
            attempts.append(_attempt_record(
                attempt_idx, stage, source_id, system, prompt, raw_text,
                parsed_json=parsed_json,
                duration_ms=(time.monotonic() - t0) * 1000.0,
            ))
            return ParsedLLMResult(parsed, raw_text, parsed_json, attempts)
        except Exception as exc:
            last_error = exc
            attempts.append(_attempt_record(
                attempt_idx, stage, source_id, system, prompt, raw_text,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=(time.monotonic() - t0) * 1000.0,
            ))
            prompt = user + f"\nPrevious JSON parse error: {exc}. Return corrected JSON only."
            _sleep_before_retry(attempt_idx, max_retries, retry_wait_seconds)
    raise ValueError(f"LLM JSON call failed after {len(attempts)} attempts: {last_error}")


def _sleep_before_retry(attempt_idx: int, max_retries: int, retry_wait_seconds: float) -> None:
    if attempt_idx >= max_retries:
        return
    wait_seconds = float(retry_wait_seconds)
    if wait_seconds <= 0.0:
        return
    time.sleep(wait_seconds)


def _attempt_record(
    attempt_idx: int,
    stage: str,
    source_id: str,
    system: str,
    user: str,
    raw_text: str,
    *,
    error: str | None = None,
    parsed_json: dict[str, Any] | None = None,
    duration_ms: float = 0.0,
) -> dict[str, Any]:
    return {
        "attempt": attempt_idx + 1,
        "stage": stage,
        "source_id": source_id,
        "system_prompt": system,
        "user_prompt": user,
        "raw_text": raw_text,
        "raw_output": raw_text,
        "parsed_json": parsed_json or {},
        "error": error,
        "duration_ms": duration_ms,
        "json_mode": True,
    }


def _chat_with_hard_timeout(llm, system: str, user: str) -> str:
    timeout = getattr(llm, "timeout", None)
    seconds = _positive_timeout(timeout)
    if seconds is not None and _is_openai_chat_client(llm):
        return _chat_openai_subprocess(llm, system, user, seconds)
    if seconds is not None and _can_use_process_timeout():
        return _chat_in_process_with_timeout(llm, system, user, seconds)
    with _hard_timeout(timeout):
        return llm.chat(system, user, json_mode=True)


def _is_openai_chat_client(llm) -> bool:
    return llm.__class__.__name__ == "OpenAIChatClient" and hasattr(llm, "config")


def _chat_openai_subprocess(llm, system: str, user: str, seconds: float) -> str:
    payload = {"system": system, "user": user}
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as fh:
        json.dump(payload, fh)
        payload_path = fh.name
    env = dict(os.environ)
    workspace_root = str(Path(__file__).resolve().parents[4])
    inherited_python_path = [
        entry
        for entry in env.get("PYTHONPATH", "").split(os.pathsep)
        if entry and entry != workspace_root
    ]
    env["PYTHONPATH"] = os.pathsep.join([workspace_root, *inherited_python_path])
    env.update({
        "LLM_API_KEY": str(llm.config.api_key),
        "LLM_BASE_URL": str(llm.config.base_url),
        "LDM_SUBPROCESS_LLM_MODEL": str(llm.config.model),
        "LDM_SUBPROCESS_LLM_TIMEOUT": str(seconds),
        "LDM_SUBPROCESS_LLM_REQUEST_TIMEOUT": str(seconds),
        "LDM_SUBPROCESS_LLM_TEMPERATURE": str(getattr(llm, "temperature", 0.2)),
        "LDM_SUBPROCESS_LLM_MAX_TOKENS": _optional_int_env(getattr(llm, "max_tokens", None)),
        "LDM_SUBPROCESS_LLM_TOP_P": _optional_float_env(getattr(llm, "top_p", None)),
        "LDM_SUBPROCESS_LLM_PRESENCE_PENALTY": _optional_float_env(
            getattr(llm, "presence_penalty", None)
        ),
        "LDM_SUBPROCESS_LLM_EXTRA_BODY": _optional_json_env(getattr(llm, "extra_body", None)),
    })
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _OPENAI_CHAT_SUBPROCESS_SCRIPT, payload_path],
            text=True,
            capture_output=True,
            timeout=seconds + 5.0,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"LLM subprocess exceeded hard timeout of {seconds:.1f}s") from exc
    finally:
        try:
            os.unlink(payload_path)
        except FileNotFoundError:
            pass
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(
            f"LLM subprocess failed with exit code {proc.returncode}: {stderr[-2000:]}"
        )
    return proc.stdout


_OPENAI_CHAT_SUBPROCESS_SCRIPT = r"""
import json
import os
import sys

from tasks.small_molecule.core.llm_advisor.client import OpenAIChatClient
from tasks.small_molecule.core.llm_advisor.config import LLMClientConfig

payload = json.load(open(sys.argv[1], encoding="utf-8"))
client = OpenAIChatClient(
    LLMClientConfig(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
        model=os.environ["LDM_SUBPROCESS_LLM_MODEL"],
    ),
    temperature=float(os.environ["LDM_SUBPROCESS_LLM_TEMPERATURE"]),
    timeout=float(os.environ["LDM_SUBPROCESS_LLM_REQUEST_TIMEOUT"]),
    max_tokens=int(os.environ["LDM_SUBPROCESS_LLM_MAX_TOKENS"])
    if os.environ.get("LDM_SUBPROCESS_LLM_MAX_TOKENS")
    else None,
    top_p=float(os.environ["LDM_SUBPROCESS_LLM_TOP_P"])
    if os.environ.get("LDM_SUBPROCESS_LLM_TOP_P")
    else None,
    presence_penalty=float(os.environ["LDM_SUBPROCESS_LLM_PRESENCE_PENALTY"])
    if os.environ.get("LDM_SUBPROCESS_LLM_PRESENCE_PENALTY")
    else None,
    extra_body=json.loads(os.environ["LDM_SUBPROCESS_LLM_EXTRA_BODY"])
    if os.environ.get("LDM_SUBPROCESS_LLM_EXTRA_BODY")
    else None,
)
sys.stdout.write(client.chat(payload["system"], payload["user"], json_mode=True))
"""


def _optional_int_env(value: Any) -> str:
    if value is None:
        return ""
    return str(int(value))


def _optional_float_env(value: Any) -> str:
    if value is None:
        return ""
    return str(float(value))


def _optional_json_env(value: Any) -> str:
    if not value:
        return ""
    return json.dumps(value)


def _chat_in_process_with_timeout(llm, system: str, user: str, seconds: float) -> str:
    ctx = multiprocessing.get_context("fork")
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=_llm_chat_worker, args=(llm, system, user, result_queue))
    process.start()
    process.join(seconds)
    try:
        if process.is_alive():
            process.terminate()
            process.join(5.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(1.0)
            raise TimeoutError(f"LLM call exceeded hard timeout of {seconds:.1f}s")

        try:
            status, payload = result_queue.get_nowait()
        except queue.Empty as exc:
            raise RuntimeError(
                f"LLM worker exited without a result; exitcode={process.exitcode}"
            ) from exc
        if status == "ok":
            return str(payload)
        raise RuntimeError(str(payload))
    finally:
        result_queue.close()
        result_queue.join_thread()


def _llm_chat_worker(llm, system: str, user: str, result_queue) -> None:
    try:
        result_queue.put(("ok", llm.chat(system, user, json_mode=True)))
    except BaseException as exc:
        result_queue.put((
            "error",
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        ))


@contextmanager
def _hard_timeout(timeout: Any):
    seconds = _positive_timeout(timeout)
    if seconds is None or not _can_use_signal_timeout():
        yield
        return

    def _raise_timeout(_signum, _frame) -> None:
        raise TimeoutError(f"LLM call exceeded hard timeout of {seconds:.1f}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def _positive_timeout(timeout: Any) -> float | None:
    try:
        seconds = float(timeout)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    return seconds


def _can_use_signal_timeout() -> bool:
    return (
        os.name == "posix"
        and hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
    )


def _can_use_process_timeout() -> bool:
    return os.name == "posix" and "fork" in multiprocessing.get_all_start_methods()


def call_reasyn_source(
    analog_fn,
    seeds: list[str],
    *,
    source_id: str,
    budget: int,
) -> list[RawCandidate]:
    try:
        raw = list(analog_fn(list(seeds)))
    except Exception as exc:
        call_reasyn_source.last_error = str(exc)
        return []
    out: list[RawCandidate] = []
    for smiles in raw[:budget]:
        out.append(
            RawCandidate(
                str(smiles),
                source_id,
                metadata={"seed_smiles": seeds[0] if seeds else None},
            )
        )
    call_reasyn_source.last_error = None
    return out


call_reasyn_source.last_error = None
