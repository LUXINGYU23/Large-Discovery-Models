"""Frozen scientific and execution identities; no inferred data defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path

METHODS = ("alphabench_cot", "alphabench_tot", "alphabench_ea", "llm", "ldm", "harness", "ldm_harness", "ldm_harness_compiled")
LDM_METHODS = ("ldm", "ldm_harness", "ldm_harness_compiled")
BACKENDS = {"qlib": ("csi300", "csi500", "csi1000", "sp500"),
            "assay": ("csi300", "csi500", "csi1000", "sp500", "nasdaq100")}
PROFILES = ("upstream_searcher_v1", "upstream_benchmark_v1", "ldm_matched_v1")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class T3Protocol:
    backend: str = "qlib"
    market: str = "csi300"
    method: str = "llm"
    profile: str = "ldm_matched_v1"
    filter_profile: str = "paper_filter_v1"
    init_mode: str = "cold"
    data_digest: str = ""
    environment_digest: str = ""
    model: str = "deepseek-flash"
    endpoint: str = "https://api.deepseek.com/responses"
    wire_api: str = "responses"
    reasoning_effort: str = "max"
    temperature: float = .7
    native_parameters: dict = field(default_factory=dict)
    forward_n: int = 1
    label: str = "close_return"
    objective: str = "rank_ic"
    validation_metric: str = "rank_ic"
    direction: int = 1
    random_seed: int = 42
    rounds: int = 2
    evaluations: int = 4
    batch_size: int = 2
    sessions: int = 2
    candidates_per_session: int = 4
    cold_seed_count: int = 30
    alpha158_groups: tuple[str, ...] = ("kbar", "rolling")
    factor_select_n: int = 50
    stock_topk: int = 50
    stock_n_drop: int = 5
    worker_timeout: int = 120
    request_timeout: int = 180
    max_model_tokens: int = 32768
    assay_portfolio: dict = field(default_factory=dict)
    budgets: dict = field(default_factory=lambda: {
        "model_requests": 120, "proposal_attempts": 100, "dynamic_checks": 400, "lint_checks": 400,
        "initialization_evaluations": 160, "validation_evaluations": 164,
        "test_evaluations": 50, "analysis_jobs": 4, "quality_checks": 400,
        "oracle_job_slots": 2000, "benchmark_jobs": 2000, "policy_turns": 4,
        "harness_turns": 20,
    })

    def __post_init__(self):
        if self.backend not in BACKENDS or self.market not in BACKENDS[self.backend]:
            raise ValueError("unsupported backend/market pair")
        if self.method not in METHODS or self.profile not in PROFILES:
            raise ValueError("unknown method or protocol profile")
        if self.init_mode not in {"cold", "alpha158", "file", "import_pool"}:
            raise ValueError("unknown initialization mode")
        if self.filter_profile not in {"paper_filter_v1", "qlib_code_filter_v1", "assay_code_filter_v1"}:
            raise ValueError("unknown filter profile")
        if self.backend == "qlib" and self.filter_profile == "assay_code_filter_v1":
            raise ValueError("Assay lint cannot validate a Qlib campaign")
        if self.backend == "assay" and self.filter_profile == "qlib_code_filter_v1":
            raise ValueError("Qlib code filtering requires the Qlib backend")
        if self.label not in ({"close_return"} if self.backend == "qlib" else {"close_return", "open_return"}):
            raise ValueError("label is not supported by the selected backend")
        if not isinstance(self.assay_portfolio, dict) or self.backend == "qlib" and self.assay_portfolio:
            raise ValueError("Assay portfolio settings belong only to the Assay backend")
        if self.objective not in {"ic", "rank_ic"} or self.direction not in {-1, 1}:
            raise ValueError("invalid frozen objective/direction")
        if self.validation_metric not in {"ic", "rank_ic", "icir", "rank_icir"}:
            raise ValueError("invalid frozen validation metric")
        if (self.model, self.wire_api, self.reasoning_effort) != ("deepseek-flash", "responses", "max"):
            raise ValueError("this campaign contract requires deepseek-flash / Responses / max")
        if self.endpoint != "https://api.deepseek.com/responses":
            raise ValueError("provider endpoint differs from the approved contract")
        if type(self.temperature) not in {int, float} or not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be finite and between zero and two")
        if not isinstance(self.native_parameters, dict) or self.native_parameters and not self.method.startswith("alphabench_"):
            raise ValueError("native_parameters belong only to a native method")
        if type(self.cold_seed_count) is not int or self.cold_seed_count < 0:
            raise ValueError("cold_seed_count must be a nonnegative integer")
        if not self.alpha158_groups or len(set(self.alpha158_groups)) != len(self.alpha158_groups) or set(self.alpha158_groups) - {"kbar", "price", "rolling"}:
            raise ValueError("alpha158_groups must name distinct source collections")
        for name in ("forward_n", "rounds", "evaluations", "batch_size", "sessions", "candidates_per_session",
                     "factor_select_n", "stock_topk", "worker_timeout", "request_timeout", "max_model_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.method in LDM_METHODS and self.batch_size > self.candidates_per_session:
            raise ValueError("LDM batch size must not exceed per-session K")
        if not 0 <= self.stock_n_drop < self.stock_topk:
            raise ValueError("drop count must be below portfolio size")
        if self.request_timeout <= self.worker_timeout:
            raise ValueError("request timeout must exceed worker timeout")
        required = {"model_requests", "proposal_attempts", "dynamic_checks", "lint_checks", "initialization_evaluations",
                    "validation_evaluations", "test_evaluations", "analysis_jobs", "quality_checks",
                    "oracle_job_slots", "benchmark_jobs", "policy_turns", "harness_turns"}
        if set(self.budgets) != required or any(type(v) is not int or v < 0 for v in self.budgets.values()):
            raise ValueError("all stage budgets must be explicit finite nonnegative integers")

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))

    @property
    def identity(self):
        return digest(self.to_dict())

    @property
    def grammar_depth(self):
        return {"paper_filter_v1": 5, "qlib_code_filter_v1": 6, "assay_code_filter_v1": None}[self.filter_profile]

    @property
    def check_kind(self):
        return "lint" if self.filter_profile == "assay_code_filter_v1" else "dynamic"

    def check_passed(self, raw, *, paper=False):
        if (raw.get("check_kind") != self.check_kind or type(raw.get("success")) is not bool
                or any(raw.get(key) for key in ("metrics", "daily", "scores", "portfolio"))):
            raise ValueError("check response differs from the frozen kind or exposes evaluation results")
        elapsed = raw.get("elapsed_seconds")
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("check response lacks valid execution time")
        ratios = [raw.get(name) for name in ("nan_ratio", "non_finite_ratio")]
        if self.check_kind == "lint":
            if any(value is not None for value in ratios):
                raise ValueError("lint cannot supply measured missing-value ratios")
            return None if paper else raw["success"]
        if raw["success"] and any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in ratios):
            raise ValueError("dynamic check response lacks measured missing-value ratios")
        if paper or self.filter_profile == "paper_filter_v1":
            return raw["success"] and max(ratios) <= .01 and elapsed <= 30
        return raw["success"] and ratios[0] <= .01

    def interval(self, phase):
        return {"search": ("2016-01-01", "2021-01-01"),
                "validation": ("2021-01-01", "2022-01-01"),
                "test": ("2022-01-01", "2025-01-01"),
                "check": ("2020-01-01", "2020-01-15")}[phase]


def verify_data_manifest(path: Path, protocol: T3Protocol):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("qualification") != "qualified":
        raise ValueError("data qualification is incomplete; inspect the recorded coverage issues")
    for key in ("source", "archive_sha256", "files_sha256", "calendar_sha256", "universe_sha256",
                "adjustment", "fields", "start", "end", "benchmark", "historical_universe"):
        if not manifest.get(key):
            raise ValueError(f"data manifest lacks {key}")
    if manifest["historical_universe"] is not True:
        raise ValueError("current constituents cannot replace a historical universe")
    if manifest.get("market") != protocol.market or manifest.get("backend") != protocol.backend:
        raise ValueError("data manifest backend/market mismatch")
    if manifest["start"] > "2015-01-01" or manifest["end"] < "2025-01-01":
        raise ValueError("data does not cover lookback, all splits, and forward labels")
    if digest(manifest) != protocol.data_digest:
        raise ValueError("data manifest does not match the frozen protocol")
    return manifest
