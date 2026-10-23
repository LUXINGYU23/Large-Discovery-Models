"""core/ldm/orchestrator/loop.py — Orchestrator (the public API)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tasks.antibody.core.ldm.config import DSLConfig
from tasks.antibody.core.ldm.dsl.bias import BiasAtom
from tasks.antibody.core.ldm.dsl.search_space import SearchSpaceAtom
from tasks.antibody.core.ldm.llm.client import LLMClient
from tasks.antibody.core.ldm.llm.function import SearchPlanFunction
from tasks.antibody.core.ldm.llm.response_parser import ParsedUpdate
from tasks.antibody.core.ldm.orchestrator.decision_log import DecisionLog
from tasks.antibody.core.ldm.orchestrator.status import OrchestratorStatus


@dataclass
class OrchestratorDecision:
    search_dsl: Optional[SearchSpaceAtom]
    bias_dsl: Optional[BiasAtom]
    applied: bool
    fallback_used: bool
    rejection_reason: Optional[str]
    source: str
    rationale: Optional[str] = None
    search_updated: bool = False
    bias_updated: bool = False


class Orchestrator:
    """Public LDM orchestrator."""

    def __init__(
        self,
        config: DSLConfig,
        llm_client: LLMClient,
        decision_log_path: str | Path | None = None,
    ) -> None:
        self.config = config
        self.llm = llm_client
        self._cache: dict[str, ParsedUpdate] = {}
        self._current_search_dsl: Optional[SearchSpaceAtom] = None
        self._current_bias_dsl: Optional[BiasAtom] = None
        self._current_rationale: Optional[str] = None
        self._n_evals_in_current_tr: int = 0
        self.log: DecisionLog | None = None
        if decision_log_path is not None:
            self.log = DecisionLog(decision_log_path)
            self.log.update_config_snapshot({
                "llm_temperature": config.llm_temperature,
                "acq_search_budget": config.acq_search_budget,
                "num_llm_review": config.num_llm_review,
            })

    @property
    def current_search_dsl(self) -> Optional[SearchSpaceAtom]:
        return self._current_search_dsl

    @property
    def current_bias_dsl(self) -> Optional[BiasAtom]:
        return self._current_bias_dsl

    def _hash_status(self, status: OrchestratorStatus) -> str:
        payload = {
            "iter": status.iteration,
            "antigen": status.antigen_id,
            "antigen_seed": status.antigen_seed,
            "iter_seed": status.iter_seed,
            "best_value": round(float(status.best_value), 4),
            "n_iters_without_improvement": status.n_iters_without_improvement,
            "n_evals": status.n_evals,
            "history_len": len(status.full_history),
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()[:16]

    def step(self, status: OrchestratorStatus) -> OrchestratorDecision:
        """Compute the next DSL state. Called once per BO iteration."""
        status.n_evals_in_current_tr = self._n_evals_in_current_tr
        h = self._hash_status(status)
        if h in self._cache:
            cached = self._cache[h]
            self._apply_cached(cached)
            return self._make_decision(
                source="cache", applied=True, rationale=cached.rationale,
            )

        func = SearchPlanFunction(self.llm, self.config, status)
        search_dsl, bias_dsl, rationale = func()

        if func.fallback_used:
            if self.log is not None and func.previous_attempts:
                self.log.append(self._build_entry(
                    status, h,
                    func.previous_attempts[-1][0] if func.previous_attempts else "",
                    None,
                    {"any_applied": False,
                     "all_errors": [e for _, e in func.previous_attempts]},
                    retry_count=len(func.previous_attempts),
                    source="fallback",
                ))
            return OrchestratorDecision(
                search_dsl=self._current_search_dsl,
                bias_dsl=self._current_bias_dsl,
                applied=False,
                fallback_used=True,
                rejection_reason=func.last_error or "all retries failed",
                source="fallback",
            )

        bs = self.config.batch_size
        if search_dsl is not None:
            self._current_search_dsl = search_dsl
            self._n_evals_in_current_tr = bs
        else:
            self._n_evals_in_current_tr += bs
        if bias_dsl is not None:
            self._current_bias_dsl = bias_dsl
        self._current_rationale = rationale

        # Cache the raw update for replay
        self._cache[h] = ParsedUpdate(
            update_trust_region=repr(search_dsl) if search_dsl else None,
            update_bias=repr(bias_dsl) if bias_dsl else None,
            rationale=rationale,
        )

        if self.log is not None:
            self.log.append(self._build_entry(
                status, h, "", self._cache[h],
                {"any_applied": True, "all_errors": []},
                retry_count=len(func.previous_attempts),
                source="llm",
            ))

        return self._make_decision(
            source="llm", applied=True, rationale=rationale,
            search_updated=search_dsl is not None,
            bias_updated=bias_dsl is not None,
        )

    def _apply_cached(self, cached: ParsedUpdate) -> None:
        """Apply a cached update by re-executing the DSL sources."""
        from tasks.antibody.core.ldm.dsl.sandbox import safe_exec_dsl
        from tasks.antibody.core.ldm.dsl.validator import validate_search_atom, validate_bias_atom

        if cached.update_trust_region is not None:
            try:
                atom = safe_exec_dsl(
                    cached.update_trust_region,
                    whitelist=self.config.atoms_whitelist,
                    expect_kind=SearchSpaceAtom,
                )
                errs = validate_search_atom(
                    atom,
                    max_depth=self.config.max_nesting_depth,
                    sample_timeout_s=self.config.sample_timeout_s,
                )
                if not errs:
                    self._current_search_dsl = atom
            except Exception:
                pass
        if cached.update_bias is not None:
            try:
                atom = safe_exec_dsl(
                    cached.update_bias,
                    whitelist=self.config.atoms_whitelist,
                    expect_kind=BiasAtom,
                )
                errs = validate_bias_atom(atom)
                if not errs:
                    self._current_bias_dsl = atom
            except Exception:
                pass

    def _make_decision(self, source: str, applied: bool,
                       rationale: str | None = None,
                       search_updated: bool = False,
                       bias_updated: bool = False) -> OrchestratorDecision:
        return OrchestratorDecision(
            search_dsl=self._current_search_dsl,
            bias_dsl=self._current_bias_dsl,
            applied=applied,
            fallback_used=(source == "fallback"),
            rejection_reason=None,
            source=source,
            rationale=rationale,
            search_updated=search_updated,
            bias_updated=bias_updated,
        )

    def _build_entry(self, status, status_hash, raw_response, update,
                     field_outcomes, retry_count, source) -> dict:
        return {
            "timestamp": None,
            "antigen_id": status.antigen_id,
            "seed": status.antigen_seed,
            "iteration": status.iteration,
            "status_hash": status_hash,
            "n_evals": status.n_evals,
            "best_value": status.best_value,
            "best_sequence": status.best_sequence,
            "n_iters_without_improvement": status.n_iters_without_improvement,
            "llm_response_raw": raw_response,
            "rationale": update.rationale if update else None,
            "llm_response_parsed": {
                "update_trust_region": update.update_trust_region if update else None,
                "update_bias": update.update_bias if update else None,
            },
            "field_outcomes": field_outcomes,
            "outcome": "applied" if field_outcomes.get("any_applied") else "fallback",
            "fallback_used": source == "fallback",
            "retry_count": retry_count,
            "source": source,
            "state_after": {
                "search_dsl_source": repr(self._current_search_dsl) if self._current_search_dsl else None,
                "bias_dsl_source": repr(self._current_bias_dsl) if self._current_bias_dsl else None,
            },
        }
