"""core/ldm/llm/function.py — LLMFunction: unified retry/fallback for LLM calls.

Subclasses implement construct_prompt, parse_response, and fallback.
__call__ handles the retry loop with automatic error feedback injection
via self.previous_attempts.
"""
from __future__ import annotations

from tasks.antibody.core.ldm.config import DSLConfig
from tasks.antibody.core.ldm.dsl.bias import BiasAtom
from tasks.antibody.core.ldm.dsl.exceptions import DSLSyntaxError
from tasks.antibody.core.ldm.dsl.sandbox import safe_exec_dsl
from tasks.antibody.core.ldm.dsl.search_space import SearchSpaceAtom
from tasks.antibody.core.ldm.dsl.validator import validate_bias_atom, validate_search_atom
from tasks.antibody.core.ldm.llm.client import LLMClient
from tasks.antibody.core.ldm.llm.response_parser import parse_response
from tasks.antibody.core.ldm.orchestrator.prompts import build_system_prompt, build_user_prompt
from tasks.antibody.core.ldm.orchestrator.status import OrchestratorStatus


# ------------------------------------------------------------------ #
#  Base class
# ------------------------------------------------------------------ #

class LLMFunction:
    """Base class for retryable LLM calls.

    Subclass implements:
      construct_prompt(*args, **kwargs) -> str
      parse_response(raw, *args, **kwargs) -> result
      fallback(*args, **kwargs) -> result

    __call__ orchestrates: prompt → call → parse → retry/fallback.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        max_retries: int = 3,
        temperature: float = 0.25,
        timeout_s: int = 30,
    ):
        self.llm = llm_client
        self.max_retries = max_retries
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.previous_attempts: list[tuple[str | None, str | None]] = []
        self.fallback_used: bool = False

    def construct_prompt(self, *args, **kwargs) -> str:
        raise NotImplementedError

    def parse_response(self, raw: str, *args, **kwargs):
        raise NotImplementedError

    def fallback(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} has no fallback")

    @property
    def last_error(self) -> str | None:
        if not self.previous_attempts:
            return None
        return self.previous_attempts[-1][1]

    def __call__(self, *args, **kwargs):
        self.previous_attempts = []
        self.fallback_used = False
        for _ in range(self.max_retries):
            prompt = self.construct_prompt(*args, **kwargs)
            try:
                raw = self.llm.call(
                    prompt,
                    temperature=self.temperature,
                    timeout_s=self.timeout_s,
                )
            except Exception as e:
                self.previous_attempts.append((None, str(e)))
                continue
            try:
                return self.parse_response(raw, *args, **kwargs)
            except Exception as e:
                self.previous_attempts.append((raw, str(e)))
                continue
        self.fallback_used = True
        return self.fallback(*args, **kwargs)


# ------------------------------------------------------------------ #
#  SearchPlanFunction — orchestrator.step()
# ------------------------------------------------------------------ #

class SearchPlanFunction(LLMFunction):
    """LLM call for producing search plan (trust region + bias).

    Returns (search_dsl, bias_dsl, rationale).
    On fallback returns (None, None, None).
    """

    def __init__(self, llm_client: LLMClient, config: DSLConfig,
                 status: OrchestratorStatus):
        super().__init__(
            llm_client,
            max_retries=config.max_retries,
            temperature=config.llm_temperature,
            timeout_s=config.llm_call_timeout_s,
        )
        self.config = config
        self.status = status

    def construct_prompt(self) -> str:
        sys_prompt = build_system_prompt(self.config)
        user_prompt = build_user_prompt(
            self.status, self.config,
            last_rejection_reason=self.last_error,
        )
        return sys_prompt + "\n\n" + user_prompt

    def parse_response(self, raw: str):
        update = parse_response(raw)

        search_dsl = None
        bias_dsl = None
        errors: list[str] = []

        if update.update_trust_region is not None:
            try:
                atom = safe_exec_dsl(
                    update.update_trust_region,
                    whitelist=self.config.atoms_whitelist,
                    expect_kind=SearchSpaceAtom,
                )
            except DSLSyntaxError as e:
                raise ValueError(f"trust_region exec failed: {e}") from None

            errs = validate_search_atom(
                atom,
                max_depth=self.config.max_nesting_depth,
                sample_timeout_s=self.config.sample_timeout_s,
            )
            atom_budget = getattr(atom, "budget", 0)
            if atom_budget > self.config.acq_search_budget:
                errs.append(
                    f"Proposed budget {atom_budget} exceeds total budget "
                    f"{self.config.acq_search_budget}. Reduce restart×steps "
                    f"or sampling budget."
                )
            if errs:
                raise ValueError("; ".join(errs))
            search_dsl = atom

        if update.update_bias is not None:
            try:
                atom = safe_exec_dsl(
                    update.update_bias,
                    whitelist=self.config.atoms_whitelist,
                    expect_kind=BiasAtom,
                )
            except DSLSyntaxError as e:
                raise ValueError(f"bias exec failed: {e}") from None

            errs = validate_bias_atom(atom)
            if errs:
                raise ValueError("; ".join(errs))
            bias_dsl = atom

        # any_rejected: if a field was attempted but didn't produce an atom
        tr_attempted = update.update_trust_region is not None
        bias_attempted = update.update_bias is not None
        tr_rejected = tr_attempted and search_dsl is None
        bias_rejected = bias_attempted and bias_dsl is None
        if tr_rejected or bias_rejected:
            parts = ["Some updates were rejected:"]
            if tr_attempted:
                parts.append(
                    f"- update_trust_region: {'APPLIED' if search_dsl else 'REJECTED'}"
                )
            if bias_attempted:
                parts.append(
                    f"- update_bias: {'APPLIED' if bias_dsl else 'REJECTED'}"
                )
            raise ValueError("\n".join(parts))

        if search_dsl is None and bias_dsl is None and not update.is_noop:
            raise ValueError("No atoms produced")

        return (search_dsl, bias_dsl, update.rationale)

    def fallback(self):
        return (None, None, None)
