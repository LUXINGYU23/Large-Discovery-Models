"""LLM4AD external-arm integration (EoH, ReEvo, FunSearch, (1+1)-EPS) for RSI copper inversion.

Integrates ``llm4ad==1.20.0`` (pinned upstream commit ``f53042ed6300dab5a64a02b6c2ec6b09126dd917``,
own venv ``$R/env/llm4ad``) under the benchmark fairness contract:
- One ``GeoEvaluation(Evaluation)`` wrapping ``BudgetGate`` -> ``RiskClient`` (comment stripping,
  32 kB normalized limit, case isolation, aggregate-only scalar ``-risk`` fitness, exact 321-slot gate).
- One ``DeepSeekLLM(LLM)`` for ``deepseek-flash`` (``reasoning_effort="high"``, ``max_tokens=65536``,
  ``thinking={"type": "enabled"}``, bounded LLM call cap, preserved credential-free request/response logs).
- Four methods enabled from the same adapter with their documented defaults adapted only where required
  by the 321-evaluation budget: ``eoh``, ``reevo``, ``funsearch``, and ``eps`` (``HillClimb`` = (1+1)-EPS).
- Compatible with ``rsi/audit.py``, ``experiments/rsi/freeze.py`` (top-5 training finalists -> validation
  selection), ``experiments/rsi/unseal.py`` (one-shot sealed evaluation under ``LLM4AD``), and
  ``experiments/rsi/export_outcomes.py`` (per-seed outcome rows).
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import json
import os
import pwd
import random
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ldm_tts.engine.run_store import atomic_json_write
from tasks.copper_inversion.core.encoding import MAX_NORMALIZED_BYTES
from tasks.copper_inversion.core.execution import digest
from tasks.copper_inversion.experiments.rsi import freeze
from tasks.copper_inversion.experiments.rsi.budget_gate import (
    DEFAULT_SLOTS,
    BudgetGate,
    prepare_submission,
)
from tasks.copper_inversion.rsi import ledger as ledgers, protocol as protocols
from tasks.copper_inversion.rsi.client import RiskClient
from tasks.copper_inversion.rsi.controller import INITIAL_SOURCE
from tasks.copper_inversion.rsi.integrity import shared_identity
from tasks.copper_inversion.v2.public_catalog import catalog_digest, load_catalog

TASK_BRIEF_PATH = Path(__file__).resolve().parents[2] / "resources/rsi/TASK.md"
RUNTIME_MODULE = "tasks.copper_inversion.experiments.v2.parallel_runtime"
DEFAULT_API_KEY_FILE = Path("/Aidata/shared/ldm/async-ldm-core/secrets/deepseek.key")
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_THINKING = "high"
DEFAULT_MAX_TOKENS = 65536
LLM4AD_PINNED_VERSION = "1.20.0"
LLM4AD_PINNED_COMMIT = "f53042ed6300dab5a64a02b6c2ec6b09126dd917"
LLM4AD_LICENSE = "BSD-2-Clause (FunSearch database: Apache-2.0)"

METHODS = ("eoh", "reevo", "funsearch", "eps")
METHOD_ARMS = {
    "eoh": "llm4ad_eoh",
    "reevo": "llm4ad_reevo",
    "funsearch": "llm4ad_funsearch",
    "eps": "llm4ad_eps",
}
PROXY_ENV_KEYS = (
    "http_proxy", "https_proxy", "all_proxy",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
)

_FENCE_BLOCK_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)[ \t]*\n(.*?)```", re.DOTALL)
_THOUGHT_BRACE_RE = re.compile(r"\{([^{}]+)\}", re.DOTALL)


def clear_proxy_env() -> None:
    """Remove broken localhost proxy variables inherited from interactive shells on H20."""
    for key in PROXY_ENV_KEYS:
        os.environ.pop(key, None)


def build_template_program() -> str:
    """Return starting solver A0 with a public-contract docstring on solve() (strips to exact A0)."""
    return (
        "from tasks.copper_inversion.v2.baselines import initialize\n\n\n"
        "def solve(problem, budget, seed):\n"
        '    """Predict 3D orebody occupancy from sparse borehole and outcrop observations.\n\n'
        "    Args:\n"
        "        problem: dict with public keys:\n"
        "            'shape': [nx, ny, nz] int grid dimensions;\n"
        "            'origin': [ox, oy, oz] float physical grid origin;\n"
        "            'spacing': [dx, dy, dz] float positive cell sizes;\n"
        "            'domain_mask': bool ndarray (nx, ny, nz), True inside active domain;\n"
        "            'forbidden_mask': bool ndarray (nx, ny, nz), True where ore (>0) is forbidden;\n"
        "            'open_boundary_faces': list of open faces in {'x-','x+','y-','y+','z-','z+'};\n"
        "            'observations': list of modality dicts ('drill', optional 'drill_outcrop') with\n"
        "                'points' (N, 3) float xyz at cell centers origin + (idx + 0.5)*spacing,\n"
        "                'labels' (N,) in {0, 0.5, 1} (1=positive, 0=negative, 0.5=contact),\n"
        "                'groups' (N,) int hole/group ids, 'weight', 'transition_scale',\n"
        "                'contact_sigma', 'label_error';\n"
        "            'priors': dict with 'area_weight', 'volume_weight', 'volume_bounds',\n"
        "                'fragment_weight', 'fragment_volume', 'hard_volume_bounds', 'max_components'.\n"
        "        budget: dict with 'wall_seconds' (30s on tier S), 'cpus', 'memory_mb'.\n"
        "        seed: nonnegative int execution seed.\n\n"
        "    Returns:\n"
        "        Finite float32/float64 ndarray of shape tuple(problem['shape']); positive values (>0)\n"
        "        predict ore and must satisfy forbidden_mask, closed faces, hard_volume_bounds, and\n"
        "        max_components (e.g. tasks.copper_inversion.v2.baselines.constrained_initial(problem, mask, rank)).\n"
        "        Evaluator maximizes scalar fitness -risk in [-1.0, 0.0] (higher is better).\n"
        "        Put all imports (numpy, scipy, skfmm) and helper functions INSIDE solve.\n"
        '    """\n'
        "    return initialize(problem)\n"
    )


def build_task_description(task_brief_path: str | Path = TASK_BRIEF_PATH) -> str:
    """Build aggregate-only single-function task description from TASK.md."""
    raw = Path(task_brief_path).read_text(encoding="utf-8")
    marker = "Write the requested complete programs to"
    core = raw.split(marker, 1)[0].strip() if marker in raw else raw.strip()
    footer = (
        "Implement a single top-level Python function matching the requested header `(problem, budget, seed)`.\n"
        "Define helper functions inside `solve`; put all `import` statements inside `solve` "
        "(top-level statements outside the function body are discarded).\n"
        f"Each solver must stay within {MAX_NORMALIZED_BYTES:,} bytes once comments and docstrings are removed.\n"
        "The evaluator returns only scalar fitness `-risk` in [-1.0, 0.0] (higher is better; maximize `-risk`; "
        "invalid or failed candidates return None). No per-case scores, tracebacks, external files, or network access."
    )
    return core + "\n\n" + footer


def _inline_top_level_imports(code_text: str) -> str:
    """Move top-level import lines appearing above a single `def` into the function body.

    LLM4AD's ``SampleTrimmer.trim_preface_of_function`` drops everything above ``def ...:``.
    Inlining stray top-level imports into the function body preserves LLM intent without
    altering the template preface.
    """
    try:
        tree = ast.parse(code_text)
    except SyntaxError:
        return code_text
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    imports = [
        n for n in tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom))
        and not (isinstance(n, ast.ImportFrom) and n.module == "__future__")
    ]
    if len(funcs) != 1 or not imports:
        return code_text
    func = funcs[0]
    pre_imports = [n for n in imports if n.end_lineno is not None and n.end_lineno < func.lineno]
    if not pre_imports:
        return code_text
    lines = code_text.splitlines()
    import_stmts = [ast.unparse(n) for n in pre_imports]
    body_first_line = func.body[0].lineno - 1
    body_indent_match = re.match(r"^([ \t]*)", lines[body_first_line]) if body_first_line < len(lines) else None
    indent = body_indent_match.group(1) if body_indent_match and body_indent_match.group(1) else "    "
    inlined = [indent + stmt for stmt in import_stmts]
    new_lines = lines[func.lineno - 1:body_first_line] + inlined + lines[body_first_line:]
    return "\n".join(new_lines) + "\n"


def normalize_llm_reply(text: str) -> str:
    """Unwrap markdown code fences while preserving EoH `{thought}` headers and completion indents."""
    if not text:
        return ""
    fenced = list(_FENCE_BLOCK_RE.finditer(text))
    if not fenced:
        def_pos = re.search(r"(?m)^def[ \t]+", text)
        if def_pos is not None:
            prefix = _THOUGHT_BRACE_RE.sub(
                lambda m: "{" + " ".join(m.group(1).split()) + "}",
                text[:def_pos.start()],
                count=1,
            )
            body = _inline_top_level_imports(text[def_pos.start():])
            return prefix + body
        return text

    chosen = None
    for match in fenced:
        if re.search(r"(?m)^def[ \t]+", match.group(1)):
            chosen = match
    if chosen is None:
        chosen = max(fenced, key=lambda m: len(m.group(1)))

    prefix = text[:chosen.start()]
    thought_match = _THOUGHT_BRACE_RE.search(prefix)
    code_block = chosen.group(1).strip("\r\n")
    if re.search(r"(?m)^def[ \t]+", code_block):
        code_block = _inline_top_level_imports(code_block)
    else:
        code_block = code_block + "\n"

    if thought_match is not None:
        clean_thought = "{" + " ".join(thought_match.group(1).split()) + "}"
        return clean_thought + "\n" + code_block
    return code_block


def run_id_for(method: str, seed: int) -> str:
    if method not in METHOD_ARMS:
        raise ValueError(f"unknown LLM4AD method {method!r}; expected one of {METHODS}")
    return f"{METHOD_ARMS[method]}-s{int(seed)}"


def method_defaults(method: str, slots: int = DEFAULT_SLOTS) -> dict[str, Any]:
    """Documented method hyperparameters adapted only where required by ``slots``."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    smoke = slots < 30
    common = {
        "max_sample_nums": max(slots + 4, 2 * slots),
        "max_llm_calls": max(slots + 6, 2 * slots),
    }
    if method == "eoh":
        return {
            **common,
            "pop_size": min(4, max(2, slots // 2)) if smoke else 10,
            "selection_num": 2,
            "use_e2_operator": True,
            "use_m1_operator": True,
            "use_m2_operator": True,
            "max_generations": None,
            "num_samplers": 2 if smoke else 4,
            "num_evaluators": 2 if smoke else 4,
        }
    if method == "reevo":
        return {
            **common,
            # ReEvo issues 1 short-term reflection per crossover and 1 long-term reflection per pop_size.
            "max_llm_calls": max(slots + 10, 3 * slots),
            "pop_size": min(4, max(2, slots // 3)) if smoke else 20,
            "mutation_rate": 0.5,
            "num_samplers": 1 if smoke else 2,
            "num_evaluators": 1 if smoke else 2,
        }
    if method == "funsearch":
        return {
            **common,
            "num_islands": 3 if smoke else 5,
            "functions_per_prompt": 2,
            # ~45 min (~1/4 of measured ~3h run time at ~140s/call across 4 samplers for 321 slots).
            "reset_period": 30 if smoke else 2700,
            "cluster_sampling_temperature_init": 0.1,
            "cluster_sampling_temperature_period": 30000,
            "samples_per_prompt": 2 if smoke else 4,
            "num_samplers": 2 if smoke else 4,
            "num_evaluators": 2 if smoke else 4,
        }
    # method == "eps" -> HillClimb ((1+1)-EPS, Zhang et al., PPSN 2024)
    return {
        **common,
        "num_samplers": 2 if smoke else 4,
        "num_evaluators": 2 if smoke else 4,
    }


def _build_llm4ad_classes():
    """Import LLM4AD base classes lazily so unit tests without llm4ad installed can import helpers."""
    from llm4ad.base import Evaluation, LLM
    from llm4ad.method import _runtime

    class DeepSeekLLM(LLM):
        """OpenAI-compatible DeepSeek client with high reasoning effort, large max_tokens, and call cap."""

        def __init__(
            self,
            *,
            api_key: str,
            base_url: str = DEFAULT_BASE_URL,
            model: str = DEFAULT_MODEL,
            max_tokens: int = DEFAULT_MAX_TOKENS,
            reasoning_effort: str | None = DEFAULT_THINKING,
            enable_thinking: bool = True,
            timeout: float = 600.0,
            max_retries: int = 3,
            max_llm_calls: int | None = None,
            gate: BudgetGate | None = None,
            stop_callback: Callable[[], None] | None = None,
            requests_dir: str | Path | None = None,
            calls_log_path: str | Path | None = None,
            client_factory: Callable[..., Any] | None = None,
        ):
            super().__init__(do_auto_trim=True, debug_mode=False)
            if not api_key or not str(api_key).strip():
                raise ValueError("non-empty API key is required")
            clear_proxy_env()
            self._host = base_url
            self._model = model
            self._timeout = float(timeout)
            self._max_tokens = int(max_tokens)
            self._reasoning_effort = reasoning_effort if reasoning_effort and reasoning_effort != "off" else None
            self._enable_thinking = bool(enable_thinking and self._reasoning_effort is not None)
            self._max_retries = int(max_retries)
            self._max_llm_calls = int(max_llm_calls) if max_llm_calls is not None else None
            self._gate = gate
            self._stop_callback = stop_callback
            self._requests_dir = Path(requests_dir) if requests_dir is not None else None
            if self._requests_dir is not None:
                self._requests_dir.mkdir(parents=True, exist_ok=True)
            self._calls_log_path = Path(calls_log_path) if calls_log_path is not None else None
            self._lock = threading.Lock()
            self._reserved_calls = 0
            self.llm_cap_reached = False
            self._stats = {
                "calls": 0,
                "reserved_calls": 0,
                "failed_api_calls": 0,
                "length_truncated_calls": 0,
                "empty_content_calls": 0,
                "prompt_tokens": 0,
                "cached_prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "total_tokens": 0,
                "missing_usage_calls": 0,
            }
            if client_factory is not None:
                self._client = client_factory(
                    api_key=str(api_key).strip(),
                    base_url=base_url,
                    timeout=self._timeout,
                    max_retries=self._max_retries,
                )
            else:
                import openai
                self._client = openai.OpenAI(
                    api_key=str(api_key).strip(),
                    base_url=base_url,
                    timeout=self._timeout,
                    max_retries=self._max_retries,
                )

        def set_stop_callback(self, callback: Callable[[], None] | None) -> None:
            with self._lock:
                self._stop_callback = callback

        def usage_summary(self) -> dict[str, Any]:
            with self._lock:
                return {**self._stats, "llm_cap_reached": self.llm_cap_reached}

        def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
            if self._gate is not None and self._gate.exhausted:
                if self._stop_callback is not None:
                    self._stop_callback()
                raise _runtime._SearchStopped()

            with self._lock:
                if self._max_llm_calls is not None and self._reserved_calls >= self._max_llm_calls:
                    self.llm_cap_reached = True
                    cb = self._stop_callback
                else:
                    self._reserved_calls += 1
                    self._stats["reserved_calls"] = self._reserved_calls
                    call_idx = self._reserved_calls
                    cb = None
            if cb is not None:
                cb()
                raise _runtime._SearchStopped()
            if self._max_llm_calls is not None and self._reserved_calls > self._max_llm_calls:
                raise _runtime._SearchStopped()

            messages = [{"role": "user", "content": prompt.strip()}] if isinstance(prompt, str) else list(prompt)
            req_kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "max_tokens": self._max_tokens,
                "stream": False,
            }
            if self._reasoning_effort is not None:
                req_kwargs["reasoning_effort"] = self._reasoning_effort
            if self._enable_thinking:
                req_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

            if self._requests_dir is not None:
                req_payload = json.dumps(req_kwargs, sort_keys=True, ensure_ascii=False).encode("utf-8")
                (self._requests_dir / f"{call_idx:04d}.request.bin").write_bytes(req_payload)

            t0 = time.monotonic()
            try:
                response = self._client.chat.completions.create(**req_kwargs)
            except Exception:
                with self._lock:
                    self._stats["failed_api_calls"] += 1
                raise
            elapsed = time.monotonic() - t0

            choice = response.choices[0]
            raw_content = getattr(choice.message, "content", None) or ""
            finish_reason = getattr(choice, "finish_reason", None)
            usage = getattr(response, "usage", None)

            p_tok = c_tok = t_tok = r_tok = cached_tok = 0
            has_usage = usage is not None
            if has_usage:
                p_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
                c_tok = int(getattr(usage, "completion_tokens", 0) or 0)
                t_tok = int(getattr(usage, "total_tokens", 0) or (p_tok + c_tok))
                c_details = getattr(usage, "completion_tokens_details", None)
                if c_details is not None:
                    r_tok = int(getattr(c_details, "reasoning_tokens", 0) or 0)
                p_details = getattr(usage, "prompt_tokens_details", None)
                if p_details is not None:
                    cached_tok = int(getattr(p_details, "cached_tokens", 0) or 0)
                if not cached_tok:
                    cached_tok = int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)

            with self._lock:
                self._stats["calls"] += 1
                if finish_reason == "length":
                    self._stats["length_truncated_calls"] += 1
                if not raw_content.strip():
                    self._stats["empty_content_calls"] += 1
                if has_usage:
                    self._stats["prompt_tokens"] += p_tok
                    self._stats["cached_prompt_tokens"] += cached_tok
                    self._stats["completion_tokens"] += c_tok
                    self._stats["reasoning_tokens"] += r_tok
                    self._stats["total_tokens"] += t_tok
                else:
                    self._stats["missing_usage_calls"] += 1
                if self._calls_log_path is not None:
                    row = {
                        "call": call_idx,
                        "elapsed_seconds": round(elapsed, 3),
                        "finish_reason": finish_reason,
                        "prompt_tokens": p_tok,
                        "cached_prompt_tokens": cached_tok,
                        "completion_tokens": c_tok,
                        "reasoning_tokens": r_tok,
                        "total_tokens": t_tok,
                        "content_chars": len(raw_content),
                    }
                    self._calls_log_path.parent.mkdir(parents=True, exist_ok=True)
                    with self._calls_log_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, sort_keys=True) + "\n")

            if self._requests_dir is not None:
                atomic_json_write(self._requests_dir / f"{call_idx:04d}.response.json", {
                    "call": call_idx,
                    "id": getattr(response, "id", None),
                    "model": getattr(response, "model", self._model),
                    "finish_reason": finish_reason,
                    "content": raw_content,
                    "usage": {
                        "prompt_tokens": p_tok,
                        "cached_prompt_tokens": cached_tok,
                        "completion_tokens": c_tok,
                        "reasoning_tokens": r_tok,
                        "total_tokens": t_tok,
                    },
                })

            return normalize_llm_reply(raw_content)

        def close(self):
            close_fn = getattr(self._client, "close", None)
            if callable(close_fn):
                close_fn()

    class GeoEvaluation(Evaluation):
        """Single-function LLM4AD evaluator submitting candidates through BudgetGate -> RiskClient."""

        def __init__(
            self,
            gate: BudgetGate,
            *,
            task_brief_path: str | Path = TASK_BRIEF_PATH,
            stop_callback: Callable[[], None] | None = None,
        ):
            super().__init__(
                template_program=build_template_program(),
                task_description=build_task_description(task_brief_path),
                use_numba_accelerate=False,
                use_protected_div=False,
                random_seed=None,
                timeout_seconds=None,
                exec_code=False,
                safe_evaluate=False,
            )
            self._gate = gate
            self._stop_callback = stop_callback

        def set_stop_callback(self, callback: Callable[[], None] | None) -> None:
            self._stop_callback = callback

        def evaluate_program(self, program_str: str, callable_func: Callable | None = None, **kwargs) -> float | None:
            risk, _ = self._gate.submit(program_str)
            if self._gate.exhausted and self._stop_callback is not None:
                self._stop_callback()
            if risk is None:
                return None
            return -float(risk)

    return DeepSeekLLM, GeoEvaluation, _runtime


def _seed_initial_population(method_name: str, method_obj, evaluation, profiler) -> float:
    """Ensure starting solver A0 is evaluated at seq=0 and seeded in population-based methods."""
    from llm4ad.base import TextFunctionProgramConverter

    template_str = evaluation.template_program
    score = evaluation.evaluate_program(template_str, None)
    if score is None:
        raise RuntimeError("initial template solver A0 failed or was rejected on training evaluation")
    if method_name in ("eoh", "reevo"):
        func = TextFunctionProgramConverter.text_to_function(template_str)
        func.score = float(score)
        func.evaluate_time = 0.0
        func.sample_time = 0.0
        func.operator = "initial_seed_A0"
        func.algorithm = "{Initial nearest-observation Euclidean baseline solver returning initialize(problem).}"
        if profiler is not None:
            profiler.register_function(func, program=str(template_str))
        method_obj._population.register_function(func)
    return float(score)


def _install_reevo_post_init_guard(method_obj, _runtime) -> None:
    """Prevent ReEvo from entering _iteratively_ga_evolve with <2 individuals if stopped during init."""
    orig_init = method_obj._iteratively_init_population

    def guarded_init(*args, **kwargs):
        try:
            return orig_init(*args, **kwargs)
        finally:
            pop = method_obj._population
            with pop._lock:
                if len(pop._population) == 0 and len(pop._next_gen_pop) > 0:
                    pop.survival()
                if len(pop._population) < 2:
                    _runtime._state(method_obj).stopped.set()

    method_obj._iteratively_init_population = guarded_init


def build_method(
    method_name: str,
    llm,
    evaluation,
    *,
    profiler_dir: str | Path,
    params: dict[str, Any],
):
    """Instantiate one of EoH, ReEvo, FunSearch, or HillClimb ((1+1)-EPS) with pinned settings."""
    profiler_dir = str(Path(profiler_dir).resolve())
    if method_name == "eoh":
        from llm4ad.method.eoh import EoH, EoHProfiler

        profiler = EoHProfiler(log_dir=profiler_dir, log_style="simple", create_random_path=False)
        method = EoH(
            llm=llm,
            evaluation=evaluation,
            profiler=profiler,
            max_generations=params["max_generations"],
            max_sample_nums=params["max_sample_nums"],
            pop_size=params["pop_size"],
            selection_num=params["selection_num"],
            use_e2_operator=params["use_e2_operator"],
            use_m1_operator=params["use_m1_operator"],
            use_m2_operator=params["use_m2_operator"],
            num_samplers=params["num_samplers"],
            num_evaluators=params["num_evaluators"],
            multi_thread_or_process_eval="thread",
        )
        return method, profiler

    if method_name == "reevo":
        from llm4ad.method.reevo import ReEvo, ReEvoProfiler

        profiler = ReEvoProfiler(log_dir=profiler_dir, log_style="simple", create_random_path=False)
        method = ReEvo(
            llm=llm,
            evaluation=evaluation,
            profiler=profiler,
            max_sample_nums=params["max_sample_nums"],
            pop_size=params["pop_size"],
            mutation_rate=params["mutation_rate"],
            num_samplers=params["num_samplers"],
            num_evaluators=params["num_evaluators"],
            multi_thread_or_process_eval="thread",
        )
        return method, profiler

    if method_name == "funsearch":
        from llm4ad.method.funsearch import FunSearch
        from llm4ad.method.funsearch import programs_database
        from llm4ad.method.funsearch.config import ProgramsDatabaseConfig
        from llm4ad.method.funsearch.profiler import FunSearchProfiler

        profiler = FunSearchProfiler(log_dir=profiler_dir, log_style="simple", create_random_path=False)
        method = FunSearch(
            llm=llm,
            evaluation=evaluation,
            profiler=profiler,
            num_samplers=params["num_samplers"],
            num_evaluators=params["num_evaluators"],
            samples_per_prompt=params["samples_per_prompt"],
            max_sample_nums=params["max_sample_nums"],
            multi_thread_or_process_eval="thread",
        )
        db_cfg = ProgramsDatabaseConfig(
            functions_per_prompt=params["functions_per_prompt"],
            num_islands=params["num_islands"],
            reset_period=params["reset_period"],
            cluster_sampling_temperature_init=params["cluster_sampling_temperature_init"],
            cluster_sampling_temperature_period=params["cluster_sampling_temperature_period"],
        )
        method.db_config = db_cfg
        method._database = programs_database.ProgramsDatabase(
            db_cfg, method._template_program, method._function_to_evolve_name
        )
        return method, profiler

    if method_name == "eps":
        from llm4ad.method.hillclimb import HillClimb, HillClimbProfiler

        profiler = HillClimbProfiler(log_dir=profiler_dir, log_style="simple", create_random_path=False)
        method = HillClimb(
            llm=llm,
            evaluation=evaluation,
            profiler=profiler,
            max_sample_nums=params["max_sample_nums"],
            num_samplers=params["num_samplers"],
            num_evaluators=params["num_evaluators"],
            multi_thread_or_process_eval="thread",
        )
        return method, profiler

    raise ValueError(f"unknown method {method_name!r}; expected one of {METHODS}")


def resolve_method_params(args) -> dict[str, Any]:
    slots = int(args.slots)
    params = method_defaults(args.method, slots=slots)
    if args.max_sample_nums is not None:
        params["max_sample_nums"] = int(args.max_sample_nums)
    if args.max_llm_calls is not None:
        params["max_llm_calls"] = int(args.max_llm_calls)
    if args.num_samplers is not None:
        params["num_samplers"] = int(args.num_samplers)
    if args.num_evaluators is not None:
        params["num_evaluators"] = int(args.num_evaluators)
    if args.pop_size is not None and args.method in ("eoh", "reevo"):
        params["pop_size"] = int(args.pop_size)
    if args.num_islands is not None and args.method == "funsearch":
        params["num_islands"] = int(args.num_islands)
    if args.reset_period is not None and args.method == "funsearch":
        params["reset_period"] = int(args.reset_period)
    if args.samples_per_prompt is not None and args.method == "funsearch":
        params["samples_per_prompt"] = int(args.samples_per_prompt)
    for worker_key in ("num_samplers", "num_evaluators"):
        if not 1 <= int(params[worker_key]) <= 16:
            raise ValueError(f"{worker_key} must be in 1..16 (got {params[worker_key]})")
    return params


def execute_llm4ad_search(
    *,
    method_name: str,
    seed: int,
    out_dir: Path,
    evaluator,
    api_key: str,
    slots: int = DEFAULT_SLOTS,
    case_feedback: str = "none",
    base_url: str = DEFAULT_BASE_URL,
    model_name: str = DEFAULT_MODEL,
    thinking: str = DEFAULT_THINKING,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    llm_timeout: float = 600.0,
    llm_retries: int = 3,
    params: dict[str, Any] | None = None,
    task_brief_path: Path = TASK_BRIEF_PATH,
    client_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Core runner shared by live RiskClient runs and in-process mock unit tests."""
    random.seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32 - 1))

    resolved_params = dict(params or method_defaults(method_name, slots=slots))
    DeepSeekLLM, GeoEvaluation, _runtime = _build_llm4ad_classes()

    llm4ad_dir = out_dir / "llm4ad"
    llm4ad_dir.mkdir(parents=True, exist_ok=True)
    requests_dir = out_dir / "harness" / "requests"
    requests_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "task_brief.txt").write_text(
        Path(task_brief_path).read_text(encoding="utf-8"), encoding="utf-8"
    )

    gate = BudgetGate(
        out_dir,
        evaluator,
        slots=int(slots),
        case_feedback=case_feedback,
        submissions_filename="llm4ad-submissions.jsonl",
    )
    llm = DeepSeekLLM(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        max_tokens=int(max_tokens),
        reasoning_effort=thinking,
        enable_thinking=(thinking != "off"),
        timeout=float(llm_timeout),
        max_retries=int(llm_retries),
        max_llm_calls=int(resolved_params["max_llm_calls"]),
        gate=gate,
        requests_dir=requests_dir,
        calls_log_path=llm4ad_dir / "llm-calls.jsonl",
        client_factory=client_factory,
    )
    geo_eval = GeoEvaluation(gate, task_brief_path=task_brief_path)
    method_obj, profiler = build_method(
        method_name,
        llm,
        geo_eval,
        profiler_dir=llm4ad_dir / "profiler",
        params=resolved_params,
    )

    def stop_search():
        _runtime._state(method_obj).stopped.set()

    gate.set_on_exhausted(stop_search)
    llm.set_stop_callback(stop_search)
    geo_eval.set_stop_callback(stop_search)
    if method_name == "reevo":
        _install_reevo_post_init_guard(method_obj, _runtime)

    atomic_json_write(llm4ad_dir / "config-sanitized.json", {
        "method": method_name,
        "arm": METHOD_ARMS[method_name],
        "seed": int(seed),
        "slots": int(slots),
        "model": model_name,
        "base_url": base_url,
        "reasoning_effort": thinking,
        "enable_thinking": bool(thinking != "off"),
        "max_tokens": int(max_tokens),
        "method_params": resolved_params,
    })

    started = time.monotonic()
    # Evaluate A0 at seq=0 first (FunSearch & HillClimb re-evaluate template_program on entry, hitting 0-cost cache).
    _seed_initial_population(method_name, method_obj, geo_eval, profiler if method_name in ("eoh", "reevo") else None)
    if not gate.exhausted:
        method_obj.run()
    wall_seconds = time.monotonic() - started

    usage = llm.usage_summary()
    gate_summary = gate.summary()
    atomic_json_write(out_dir / "usage.json", {**usage, "gate": gate_summary})
    return {
        "usage": usage,
        "gate_summary": gate_summary,
        "wall_seconds": round(wall_seconds, 2),
        "method_params": resolved_params,
    }


def run_llm4ad_arm(args) -> dict[str, Any]:
    """Execute one LLM4AD run against a live RiskOnlyService and seal training."""
    clear_proxy_env()
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    pid = args.protocol_id or protocols.protocol_id(protocol)
    tier_spec = protocol["tiers"][args.tier]
    judge_spec = protocol["judges"][args.judge]
    isolation = protocols.case_isolation(protocol)
    if isolation["case_feedback"] != "none":
        raise ValueError("LLM4AD external arm requires case_isolation.case_feedback == 'none'")
    service_slots = int(tier_spec["slots"])
    slots = int(args.slots) if args.slots is not None else service_slots
    if slots < 1 or slots > service_slots:
        raise ValueError(f"slots must lie in 1..{service_slots} (got {slots})")
    args.slots = slots
    solver_seconds = int(tier_spec["solver_seconds"])
    finalists_count = int((protocol.get("loop") or {}).get("finalists", 5))
    catalog = load_catalog(args.public_catalog)
    public_digest = catalog_digest(args.public_catalog)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Verify A0 template stripping invariance before touching the service.
    a0_bytes, a0_sha = prepare_submission(INITIAL_SOURCE)
    tpl_bytes, tpl_sha = prepare_submission(build_template_program())
    if (a0_bytes, a0_sha) != (tpl_bytes, tpl_sha):
        raise RuntimeError("template_program stripped source differs from INITIAL_SOURCE (A0)")

    client = RiskClient(
        out_dir,
        args.evaluation_socket,
        args.evaluation_contract,
        service_uid=args.evaluation_service_uid,
    )
    info = client.verify_contract(
        public_digest,
        solver_seconds,
        service_slots,
        objective=judge_spec.get("objective", "confirmation"),
        tier=args.tier,
        import_ledger_prefix_sha256=None,
        case_isolation=isolation,
    )
    if info["used_slots"] != 0:
        raise ValueError("a new LLM4AD run requires an unused private evaluator")

    api_key = Path(args.api_key_file).read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError("protected model API key file is empty")

    model_cfg = protocol.get("model") or {}
    base_url = args.base_url or model_cfg.get("base_url") or DEFAULT_BASE_URL
    model_name = args.model or model_cfg.get("name") or DEFAULT_MODEL
    thinking = args.thinking if args.thinking is not None else model_cfg.get("thinking", DEFAULT_THINKING)
    params = resolve_method_params(args)

    import importlib.metadata
    try:
        llm4ad_version = importlib.metadata.version("llm4ad")
    except Exception:
        llm4ad_version = LLM4AD_PINNED_VERSION

    arm_name = METHOD_ARMS[args.method]
    manifest = {
        "schema": "rsi-run-manifest-v1",
        "external_system": "llm4ad",
        "llm4ad_method": args.method,
        "llm4ad_version": llm4ad_version,
        "llm4ad_commit": LLM4AD_PINNED_COMMIT,
        "llm4ad_license": LLM4AD_LICENSE,
        "code_commit": protocols.code_commit(),
        "run_id": out_dir.name,
        "experiment": args.experiment,
        "protocol_id": pid,
        "protocol_sha256": protocols.protocol_sha256(protocol),
        "arm": arm_name,
        "tier": args.tier,
        "judge": args.judge,
        "seed": int(args.seed),
        "slots": slots,
        "solver_seconds": solver_seconds,
        "finalists": finalists_count,
        "model": {
            "base_url": base_url,
            "name": model_name,
            "thinking": thinking,
            "max_tokens": int(args.llm_max_tokens),
        },
        "method_params": params,
        "case_isolation": isolation,
        "public_digest": public_digest,
        "shared_implementation_sha256": shared_identity(),
        "stage": catalog["stage"],
        "synthetic_test": bool(args.fixture_only),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json_write(out_dir / "manifest.json", manifest)

    search_out = execute_llm4ad_search(
        method_name=args.method,
        seed=int(args.seed),
        out_dir=out_dir,
        evaluator=client,
        api_key=api_key,
        slots=slots,
        case_feedback=isolation["case_feedback"],
        base_url=base_url,
        model_name=model_name,
        thinking=thinking,
        max_tokens=int(args.llm_max_tokens),
        llm_timeout=float(args.llm_timeout),
        llm_retries=int(args.llm_retries),
        params=params,
    )

    records = ledgers.read(out_dir / "ledger.jsonl")
    valid_records = [r for r in records if r["status"] == "valid"]
    finalists = sorted(
        valid_records,
        key=lambda r: (r["risk"], -r["legal_rate"], r["seconds_per_solve"], r["program_id"]),
    )[:finalists_count]
    training_finalists = [r["program_id"] for r in finalists]

    sealed = client.seal_training()
    if sealed["used_slots"] != len(records) or sealed["source_sha256s"] != training_finalists:
        raise ValueError(
            f"public and private training records or finalists differ: "
            f"sealed_slots={sealed['used_slots']} ledger={len(records)} "
            f"sealed_finalists={sealed['source_sha256s']} ledger_finalists={training_finalists}"
        )

    stop_reason = (
        "completed"
        if len(records) >= slots
        else ("llm_call_cap" if search_out["usage"].get("llm_cap_reached") else "sample_cap")
    )
    result = {
        "status": "completed",
        "stop_reason": stop_reason,
        "external_system": "llm4ad",
        "llm4ad_method": args.method,
        "llm4ad_version": llm4ad_version,
        "llm4ad_commit": LLM4AD_PINNED_COMMIT,
        "arm": arm_name,
        "seed": int(args.seed),
        "charged_records": len(records),
        "valid_records": len(valid_records),
        "best_risk": finalists[0]["risk"] if finalists else 1.0,
        "training_finalists": training_finalists,
        "gate_summary": search_out["gate_summary"],
        "token_usage": search_out["usage"],
        "wall_seconds": search_out["wall_seconds"],
        "scope": "dry run / smoke fixture only; not a scientific result"
                 if args.fixture_only else "training feedback only; validation selection and sealed evaluation not run",
    }
    atomic_json_write(out_dir / "result.json", result)
    atomic_json_write(out_dir / "run-status.json", {"status": "completed", "stop_reason": stop_reason})
    return result


# Trusted-operator batch orchestration (service + client + audit + optional freeze) ------------

def _account_ids(user: str) -> tuple[int, int, list[int]]:
    entry = pwd.getpwnam(user)
    return entry.pw_uid, entry.pw_gid, os.getgrouplist(user, entry.pw_gid)


def _readable_tree(root: Path) -> None:
    for path in [Path(root), *Path(root).rglob("*")]:
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | (0o555 if path.is_dir() else 0o444))


def _thread_capped_env(code_dir: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "LLM_API_KEY" and k not in PROXY_ENV_KEYS}
    env.update(
        PYTHONPATH=str(code_dir),
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    return env


_SERVICE_SETUP_LOCK = threading.Lock()


def _start_isolated_service(
    *,
    root: Path,
    pid: str,
    experiment: str,
    run_name: str,
    protocol: dict,
    release: Path,
    policy_path: Path,
    code_dir: Path,
    service_python: Path,
    service_user: str,
    client_user: str,
    image: str,
    tier: str,
    judge: str,
    slots: int,
) -> tuple[subprocess.Popen, dict[str, Any], Path]:
    paths = freeze.layout(root, pid, experiment, run_name)
    socket_dir = root / "rpc" / digest([pid, experiment, run_name])[:12]
    private = paths["service"]
    uid, gid, groups = _account_ids(service_user)
    client_uid, client_gid, _ = _account_ids(client_user)
    if uid == client_uid:
        raise ValueError("research and evaluator accounts must differ")
    socket_path = socket_dir / "e.sock"
    if len(os.fsencode(socket_path)) >= 104:
        raise ValueError(f"Unix socket path is too long: {socket_path}")
    config_path = private / "service-config.json"
    with _SERVICE_SETUP_LOCK:
        if not config_path.exists():
            for folder in (root / "private", root / "private" / pid, private.parent.parent, private.parent):
                if not folder.exists():
                    folder.mkdir(parents=True, mode=0o755 if folder == root / "private" else 0o700, exist_ok=True)
                    if folder != root / "private":
                        os.chown(folder, uid, gid)
            private.mkdir(parents=True, mode=0o700, exist_ok=True)
            os.chown(private, uid, gid)
            socket_dir.mkdir(parents=True, mode=0o750, exist_ok=True)
            os.chown(socket_dir, uid, client_gid)
            tier_spec = protocol["tiers"][tier]
            judge_spec = protocol["judges"][judge]
            train_manifest = release / judge_spec.get("train_manifest", tier_spec["train_manifest"])
            svc_cfg: dict[str, Any] = {
                "manifest": str(train_manifest.resolve()),
                "root": str(private),
                "image": image,
                "seconds": int(tier_spec["solver_seconds"]),
                "slots": int(slots),
                "caller_uid": client_uid,
                "socket": str(socket_path),
                "objective": judge_spec["objective"],
                "tier": tier,
            }
            if "resample" in judge_spec:
                svc_cfg["resample"] = dict(judge_spec["resample"])
            if "confirmation_packs" in judge_spec:
                packs = judge_spec["confirmation_packs"]
                svc_cfg["confirmation_packs"] = list(packs) if isinstance(packs, list) else packs
            if protocol["loop"]["finalists"] != 3:
                svc_cfg["finalists"] = int(protocol["loop"]["finalists"])
            if "case_isolation" in protocol:
                svc_cfg["case_isolation"] = protocols.case_isolation(protocol)
            config_path.write_text(json.dumps(svc_cfg, sort_keys=True), encoding="utf-8")
            os.chown(config_path, uid, gid)
            config_path.chmod(0o600)
            for folder in (private / "tmp", private / "tmp" / "docker"):
                folder.mkdir(mode=0o700, exist_ok=True)
                os.chown(folder, uid, gid)
        elif socket_path.exists():
            socket_path.unlink()

    env = {
        **_thread_capped_env(code_dir),
        "TMPDIR": str(private / "tmp"),
        "DOCKER_CONFIG": str(private / "tmp" / "docker"),
    }
    log_path = private / "service.log"
    argv = [
        "nice", "-n", "10", str(service_python), "-m", RUNTIME_MODULE,
        "--policy", str(policy_path.resolve()), "--tier", tier,
        "rsi-service", "--config", str(config_path),
    ]
    with log_path.open("ab") as log_stream:
        proc = subprocess.Popen(
            argv, cwd=str(code_dir), env=env, stdin=subprocess.DEVNULL,
            stdout=log_stream, stderr=subprocess.STDOUT,
            user=uid, group=gid, extra_groups=groups, start_new_session=True,
        )
    os.chown(log_path, uid, gid)
    deadline = time.monotonic() + 120.0
    ready = None
    while ready is None:
        if socket_path.exists() and log_path.exists():
            for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("{") and '"listening"' in line:
                    row = json.loads(line)
                    if row.get("pid") == proc.pid:
                        ready = row
        if ready is None:
            if proc.poll() is not None or time.monotonic() >= deadline:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:] if log_path.is_file() else ""
                raise RuntimeError(f"evaluator service did not start for {run_name}:\n{tail}")
            time.sleep(0.2)
    os.chown(socket_path, uid, client_gid)
    socket_path.chmod(0o660)
    deployment = {
        "socket": str(socket_path),
        "contract_digest": ready["contract_digest"],
        "service_uid": uid,
        "pid": proc.pid,
        "case_workers": ready.get("case_workers"),
    }
    (socket_dir / "deployment.json").write_text(json.dumps(deployment, sort_keys=True) + "\n", encoding="utf-8")
    return proc, deployment, private


def _stop_process_group(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def launch_batch(args) -> dict[str, Any]:
    """Run (method, seed) pairs with isolated services, CPU/thread caps, budget/blind audit, and optional freeze."""
    if os.geteuid() != 0:
        raise PermissionError("llm4ad_arm launch must run as root (sudo -n) to isolate evaluator/client UIDs")
    clear_proxy_env()
    root = Path(args.root).resolve()
    protocol_path = Path(args.protocol).resolve()
    protocol = protocols.load(protocol_path)
    pid = args.protocol_id or protocols.protocol_id(protocol)
    release = Path(args.release).resolve()
    policy_path = Path(args.policy).resolve()
    code_dir = Path(args.code_dir).resolve()
    # Preserve venv symlink paths (Path.resolve() would dereference bin/python -> base cpython).
    service_python = Path(os.path.abspath(args.service_python))
    client_python = Path(os.path.abspath(args.client_python))
    public_catalog = Path(args.public_catalog).resolve()
    api_key_file = Path(args.api_key_file).resolve()
    image = args.image or protocol["hashes"]["evaluator_image"]
    slots = int(args.slots) if args.slots is not None else int(protocol["tiers"][args.tier]["slots"])

    methods = list(METHODS) if args.method == "all" else [m.strip() for m in args.method.split(",") if m.strip()]
    for m in methods:
        if m not in METHODS:
            raise ValueError(f"unknown method {m!r}; expected {METHODS} or 'all'")
    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")

    plan = [(m, s, run_id_for(m, s)) for m in methods for s in seeds]
    client_uid, client_gid, client_groups = _account_ids(args.client_user)
    svc_uid, svc_gid, svc_groups = _account_ids(args.service_user)

    runs_parent = freeze.layout(root, pid, args.experiment)["runs"]
    runs_parent.mkdir(parents=True, exist_ok=True)
    for folder in [runs_parent, *runs_parent.parents]:
        if folder == root or folder == root.parent:
            break
        if folder.stat().st_uid == 0:
            os.chown(folder, client_uid, client_gid)

    def run_one(item: tuple[str, int, str]) -> dict[str, Any]:
        method_name, seed, rname = item
        paths = freeze.layout(root, pid, args.experiment, rname)
        if paths["run"].exists():
            raise FileExistsError(f"run directory already exists: {paths['run']}")
        svc_proc, dep, private_dir = _start_isolated_service(
            root=root, pid=pid, experiment=args.experiment, run_name=rname,
            protocol=protocol, release=release, policy_path=policy_path,
            code_dir=code_dir, service_python=service_python,
            service_user=args.service_user, client_user=args.client_user,
            image=image, tier=args.tier, judge=args.judge,
            slots=int(protocol["tiers"][args.tier]["slots"]),
        )
        try:
            cmd = [
                "nice", "-n", "10", str(client_python),
                "-m", "tasks.copper_inversion.experiments.rsi.llm4ad_arm", "run",
                "--protocol", str(protocol_path),
                "--protocol-id", pid,
                "--experiment", args.experiment,
                "--method", method_name,
                "--tier", args.tier,
                "--judge", args.judge,
                "--seed", str(seed),
                "--slots", str(slots),
                "--out-dir", str(paths["run"]),
                "--public-catalog", str(public_catalog),
                "--evaluation-socket", dep["socket"],
                "--evaluation-contract", dep["contract_digest"],
                "--evaluation-service-uid", str(dep["service_uid"]),
                "--api-key-file", str(api_key_file),
                "--llm-max-tokens", str(args.llm_max_tokens),
                "--llm-timeout", str(args.llm_timeout),
            ]
            if args.thinking is not None:
                cmd += ["--thinking", args.thinking]
            if args.num_samplers is not None:
                cmd += ["--num-samplers", str(args.num_samplers)]
            if args.num_evaluators is not None:
                cmd += ["--num-evaluators", str(args.num_evaluators)]
            if args.pop_size is not None:
                cmd += ["--pop-size", str(args.pop_size)]
            if args.max_llm_calls is not None:
                cmd += ["--max-llm-calls", str(args.max_llm_calls)]
            if args.fixture_only:
                cmd.append("--fixture-only")

            client_log = runs_parent / f"{rname}.client.log"
            with client_log.open("ab") as log_stream:
                cproc = subprocess.Popen(
                    cmd, cwd=str(code_dir), env=_thread_capped_env(code_dir),
                    stdin=subprocess.DEVNULL, stdout=log_stream, stderr=subprocess.STDOUT,
                    user=client_uid, group=client_gid, extra_groups=client_groups,
                    start_new_session=True,
                )
            os.chown(client_log, client_uid, client_gid)
            rc = cproc.wait()
            if rc != 0:
                tail = client_log.read_text(encoding="utf-8", errors="replace")[-2000:]
                raise RuntimeError(f"client failed for {rname} (exit {rc}):\n{tail}")
            _readable_tree(paths["run"])

            from tasks.copper_inversion.rsi.audit import reconcile_budget, scan_blind
            budget_check = reconcile_budget(paths["run"], private_dir)
            blind_check = scan_blind(
                paths["run"], private_dir, key_file=api_key_file, require_requests=True
            )
            res = json.loads((paths["run"] / "result.json").read_text(encoding="utf-8"))
            finalist_check = freeze.finalists(
                paths["run"], private_dir, count=int(protocol["loop"]["finalists"])
            )
            summary_row: dict[str, Any] = {
                "run_id": rname,
                "method": method_name,
                "seed": seed,
                "charged_records": res["charged_records"],
                "valid_records": res["valid_records"],
                "best_risk": res["best_risk"],
                "finalists_count": len(finalist_check["source_sha256s"]),
                "llm_calls": res["token_usage"]["calls"],
                "total_tokens": res["token_usage"]["total_tokens"],
                "wall_seconds": res["wall_seconds"],
                "budget_audit": budget_check["status"],
                "blind_audit": blind_check["status"],
                "blind_leaks": blind_check["leaks"],
                "blind_review_required": blind_check["review_required"],
            }
        finally:
            _stop_process_group(svc_proc)

        if args.freeze:
            freeze_log = private_dir / "freeze.log"
            freeze_argv = [
                "nice", "-n", "10", str(service_python), "-m", RUNTIME_MODULE,
                "--policy", str(policy_path), "--tier", args.tier, "rsi-freeze",
                "--root", str(root), "--protocol-id", pid, "--experiment", args.experiment,
                "--run", rname, "--release", str(release), "--protocol", str(protocol_path),
            ]
            if args.no_validation:
                freeze_argv.append("--no-validation")
            with freeze_log.open("ab") as log_stream:
                fproc = subprocess.Popen(
                    freeze_argv, cwd=str(code_dir), env=_thread_capped_env(code_dir),
                    stdin=subprocess.DEVNULL, stdout=log_stream, stderr=subprocess.STDOUT,
                    user=svc_uid, group=svc_gid, extra_groups=svc_groups,
                    start_new_session=True,
                )
            os.chown(freeze_log, svc_uid, svc_gid)
            frc = fproc.wait()
            if frc != 0:
                tail = freeze_log.read_text(encoding="utf-8", errors="replace")[-2000:]
                raise RuntimeError(f"freeze failed for {rname} (exit {frc}):\n{tail}")
            summary_row["freeze_status"] = "frozen" if args.no_validation else "selected"
        return summary_row

    max_parallel = max(1, min(int(args.max_parallel_runs), len(plan), 8))
    results = []
    if max_parallel == 1:
        for item in plan:
            results.append(run_one(item))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futs = [pool.submit(run_one, item) for item in plan]
            for fut in futs:
                results.append(fut.result())

    frozen_list_path = None
    if args.freeze:
        exp_paths = freeze.layout(root, pid, args.experiment)
        if exp_paths["selection"].is_dir():
            solvers = [
                {"run_id": run_p.name, "prefix": pref_p.name}
                for run_p in sorted(p for p in exp_paths["selection"].iterdir() if p.is_dir())
                for pref_p in sorted(p for p in run_p.iterdir() if (p / "prefix.json").is_file())
            ]
            target = root / "private" / pid / "frozen-lists" / f"{args.experiment}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_json_write(target, {
                "experiment": args.experiment,
                "protocol_id": pid,
                "solvers": solvers,
            })
            os.chown(target.parent, svc_uid, svc_gid)
            os.chown(target, svc_uid, svc_gid)
            frozen_list_path = str(target)

    return {
        "status": "completed",
        "experiment": args.experiment,
        "protocol_id": pid,
        "runs": results,
        "frozen_list": frozen_list_path,
    }


def _add_common_llm_and_method_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--thinking", default=None,
                        help="Reasoning effort override (defaults to protocol.model.thinking or 'high')")
    parser.add_argument("--slots", type=int, default=None, help="Override slot limit (defaults to tier slots = 321)")
    parser.add_argument("--llm-timeout", type=float, default=900.0)
    parser.add_argument("--llm-max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--llm-retries", type=int, default=3)
    parser.add_argument("--max-sample-nums", type=int, default=None)
    parser.add_argument("--max-llm-calls", type=int, default=None)
    parser.add_argument("--num-samplers", type=int, default=None)
    parser.add_argument("--num-evaluators", type=int, default=None)
    parser.add_argument("--pop-size", type=int, default=None)
    parser.add_argument("--num-islands", type=int, default=None)
    parser.add_argument("--reset-period", type=int, default=None)
    parser.add_argument("--samples-per-prompt", type=int, default=None)
    parser.add_argument("--fixture-only", action="store_true")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run one LLM4AD (method, seed) client against a live RiskOnlyService.")
    run_p.add_argument("--protocol", required=True, type=Path)
    run_p.add_argument("--protocol-id", default=None)
    run_p.add_argument("--experiment", default="LLM4AD", choices=protocols.EXPERIMENTS)
    run_p.add_argument("--method", required=True, choices=METHODS)
    run_p.add_argument("--tier", default="S", choices=("S", "M", "F"))
    run_p.add_argument("--judge", default="standard")
    run_p.add_argument("--seed", required=True, type=int)
    run_p.add_argument("--out-dir", required=True, type=Path)
    run_p.add_argument("--public-catalog", required=True, type=Path)
    run_p.add_argument("--evaluation-socket", required=True, type=Path)
    run_p.add_argument("--evaluation-contract", required=True)
    run_p.add_argument("--evaluation-service-uid", required=True, type=int)
    run_p.add_argument("--api-key-file", type=Path, default=DEFAULT_API_KEY_FILE)
    run_p.add_argument("--base-url", default=None)
    run_p.add_argument("--model", default=None)
    _add_common_llm_and_method_args(run_p)

    launch_p = sub.add_parser(
        "launch",
        help="Trusted-operator batch launcher: isolated RiskOnlyService + LLM4AD client + audit + optional freeze.",
    )
    launch_p.add_argument("--root", required=True, type=Path)
    launch_p.add_argument("--protocol", required=True, type=Path)
    launch_p.add_argument("--protocol-id", default=None)
    launch_p.add_argument("--release", required=True, type=Path)
    launch_p.add_argument("--policy", required=True, type=Path)
    launch_p.add_argument("--code-dir", type=Path, default=Path(__file__).resolve().parents[4])
    launch_p.add_argument("--service-python", type=Path,
                          default=Path("/Aidata/shared/xingyulu/rsi4geo/env/python/bin/python"))
    launch_p.add_argument("--client-python", type=Path,
                          default=Path("/Aidata/shared/xingyulu/rsi4geo/env/llm4ad/bin/python"))
    launch_p.add_argument("--service-user", default="ldm-copper-eval")
    launch_p.add_argument("--client-user", default="zsgpu")
    launch_p.add_argument("--image", default=None)
    launch_p.add_argument("--public-catalog", required=True, type=Path)
    launch_p.add_argument("--api-key-file", type=Path, default=DEFAULT_API_KEY_FILE)
    launch_p.add_argument("--experiment", default="LLM4AD", choices=protocols.EXPERIMENTS)
    launch_p.add_argument("--method", required=True, help="eoh, reevo, funsearch, eps, comma-list, or 'all'")
    launch_p.add_argument("--seeds", default="401,402,403,404", help="Comma-separated integer seeds")
    launch_p.add_argument("--tier", default="S", choices=("S", "M", "F"))
    launch_p.add_argument("--judge", default="standard")
    launch_p.add_argument("--max-parallel-runs", type=int, default=4)
    launch_p.add_argument("--freeze", action="store_true",
                          help="Freeze top-5 finalists and run validation selection after each run completes")
    launch_p.add_argument("--no-validation", action="store_true",
                          help="With --freeze, stop after freezing finalists without evaluating validation")
    _add_common_llm_and_method_args(launch_p)

    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "run":
        result = run_llm4ad_arm(args)
        print(json.dumps({"run_dir": str(Path(args.out_dir).resolve()), **result}, sort_keys=True), flush=True)
        return 0
    result = launch_batch(args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
