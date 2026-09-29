"""Declarative ResearchGym case catalog and template rendering."""

from __future__ import annotations

import importlib.util
import itertools
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

TASK_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_ROOT = TASK_ROOT / "resources"
CATALOG_PATH = RESOURCE_ROOT / "cases" / "catalog.json"
RULES_PATH = RESOURCE_ROOT / "harness" / "tools" / "program_rules.py"
CASE_IDS = (
    "time_series_explanation",
    "continual_learning",
    "cross_modal_retrieval",
    "improving_replay_buffers",
)
_PLACEHOLDER = re.compile(r"\{([a-z_]+)(:int)?\}")


@lru_cache(maxsize=1)
def catalog() -> dict[str, Any]:
    return json.loads(CATALOG_PATH.read_text())


@lru_cache(maxsize=1)
def program_rules():
    """Import the exact rules module that the guest checker runs."""
    spec = importlib.util.spec_from_file_location("researchgym_program_rules", RULES_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class CaseSpec:
    raw: dict[str, Any]

    @property
    def case_id(self) -> str:
        return self.raw["case_id"]

    @property
    def entry(self) -> dict[str, Any]:
        return self.raw["entry"]

    @property
    def candidate_path(self) -> str:
        return self.raw["candidate_path"]

    @property
    def module_name(self) -> str:
        return Path(self.candidate_path).stem

    @property
    def metric(self) -> dict[str, Any]:
        return self.raw["metric"]

    @property
    def protocol(self) -> dict[str, Any]:
        return self.raw["protocol"]

    @property
    def rules(self) -> dict[str, Any]:
        return catalog()["global_rules"]

    def seed_program(self) -> str:
        return (RESOURCE_ROOT / "cases" / self.raw["seed_program"]).read_text()

    def check_program(self, source: Any) -> list[dict[str, str]]:
        return program_rules().check_program(
            source, entry=self.entry, rules=self.rules,
            forbidden_prefixes=tuple(self.raw["forbidden_import_prefixes"]),
        )

    def canonical_key(self, source: str) -> str:
        return program_rules().canonical_key(source)

    def jobs(self, slot: str) -> list[dict[str, Any]]:
        """Expand the job matrix in declaration order with stable indices."""
        spec = self.raw["jobs"]
        names = list(spec["matrix"])
        rows = []
        for index, values in enumerate(itertools.product(*(spec["matrix"][n] for n in names))):
            binding = {"slot": slot, "index": index, **dict(zip(names, values))}
            rows.append({
                "index": index,
                "binding": binding,
                "job_id": "-".join(str(binding[n]) for n in names),
                "expected_outputs": [render(item, binding) for item in spec["expected_outputs"]],
                "generated_files": [
                    {"path": render(item["path"], binding), "json": render(item["json"], binding)}
                    for item in spec.get("generated_files", ())
                ],
                "env": dict(spec.get("env", {})),
            })
        return rows

    def public_contract(self) -> dict[str, Any]:
        """Model-visible case contract; contains no evaluator paths or labels."""
        return {
            "case_id": self.case_id,
            "display_name": self.raw["display_name"],
            "entry": self.entry,
            "candidate_file": self.candidate_path,
            "interface": self.raw["public_context"]["interface"],
            "scoring": self.raw["public_context"]["scoring"],
            "rules": list(self.raw["public_context"]["rules"]),
            "metric": self.metric,
            "protocol": self.protocol,
            "static_rules": {**self.rules, "forbidden_import_prefixes": self.raw["forbidden_import_prefixes"]},
            "job_matrix": self.raw["jobs"]["matrix"],
        }


def render(value: Any, binding: dict[str, Any]) -> Any:
    """Fill {name} placeholders; "{name:int}" alone becomes an integer."""
    if isinstance(value, str):
        whole = _PLACEHOLDER.fullmatch(value)
        if whole and whole.group(2):
            return int(binding[whole.group(1)])

        def replace(match):
            key = match.group(1)
            if key not in binding:
                raise KeyError(f"unbound case template placeholder {{{key}}}")
            return str(binding[key])

        return _PLACEHOLDER.sub(replace, value)
    if isinstance(value, list):
        return [render(item, binding) for item in value]
    if isinstance(value, dict):
        return {key: render(item, binding) for key, item in value.items()}
    return value


def load_case(case_id: str) -> CaseSpec:
    for raw in catalog()["cases"]:
        if raw["case_id"] == case_id:
            return CaseSpec(raw)
    raise ValueError(f"unknown ResearchGym case {case_id!r}; choose from {', '.join(CASE_IDS)}")
