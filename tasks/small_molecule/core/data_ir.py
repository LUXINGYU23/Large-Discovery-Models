"""Small-molecule prompt and trajectory conversion to LDM IR."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from ldm_tts.data.ir import make_complete_design_ir


SMALLMOL_PROMPT_HEADS = [
    "Task",
    "Target context",
    "Background",
    "Molecule context table",
    "How to use the molecule context",
    "Generation principles",
    "SMILES hygiene",
    "Generation focus",
    "History summary",
    "JSON output format",
]

SMALLMOL_ROLE_MAP = {
    "pareto_front": "pareto_front",
    "top_low_vina": "top_objective_0",
    "top_high_activity": "top_objective_1",
    "balanced_elites": "elite",
    "recent_selected": "recent",
}

_SMALLMOL_FORMAT_PROSE = re.compile(
    r"(Use compact minified JSON[^.]*\."
    r"|Return JSON only[^.]*\."
    r"|The top-level JSON value[^.]*\."
    r"|Do not include ids, scores[^.]*\."
    r"|keep each rationale under \d+ words\.?)",
    re.I,
)


def smallmol_ir_from_prompt_response(
    instruction: str,
    output: str,
    *,
    round_idx: int | None = None,
    source_id: str | None = None,
) -> dict[str, Any] | None:
    """Convert one accepted M1 direct-SMILES LLM call into ldm-2.0 IR.

    This adapter is intentionally narrow: it handles the direct SMILES proposal
    prompt used by
    ``tasks.small_molecule.core.ldm_tilted_case2.prompts.build_m1_prompt``. Seed
    planning and non-M1 prompts should use their own adapters rather than being
    forced through this schema.
    """

    sections = _sections(instruction, SMALLMOL_PROMPT_HEADS)
    if "History summary" not in sections or "Task" not in sections:
        return None
    try:
        history = json.loads(sections.get("History summary", "{}"))
    except json.JSONDecodeError:
        return None
    try:
        molecule_context = json.loads(sections.get("Molecule context table", "[]"))
    except json.JSONDecodeError:
        molecule_context = []
    try:
        parsed_output = json.loads(output)
    except json.JSONDecodeError:
        return None
    direct = parsed_output.get("direct_smiles")
    if not isinstance(direct, list):
        return None

    observations_by_design: dict[str, dict[str, Any]] = {}
    for view_name, role in SMALLMOL_ROLE_MAP.items():
        for entry in history.get(view_name, []) or []:
            smiles = entry.get("smiles")
            if not smiles:
                continue
            observation = observations_by_design.setdefault(
                str(smiles),
                {"design": str(smiles), "results": None, "roles": []},
            )
            scores = entry.get("scores")
            if scores and observation["results"] is None:
                observation["results"] = {"vina": scores[0], "activity": scores[1]}
            if role not in observation["roles"]:
                observation["roles"].append(role)

    if isinstance(molecule_context, list):
        for item in molecule_context:
            if not isinstance(item, Mapping):
                continue
            smiles = item.get("smiles")
            if smiles not in observations_by_design:
                continue
            observations_by_design[str(smiles)]["description"] = "; ".join(
                f"{key}={value}"
                for key, value in item.items()
                if key != "smiles" and value is not None
            )

    candidates = [
        {"design": item.get("smiles"), "rationale": item.get("rationale")}
        for item in direct
        if isinstance(item, Mapping) and item.get("smiles")
    ]
    if not candidates:
        return None

    alert = history.get("recent_diversity_alert")
    progress = None
    if isinstance(alert, Mapping):
        progress = {
            "stalled": True,
            "rounds_since_improvement": None,
            "description": alert.get("instruction"),
        }

    request_description = _strip_smallmol_format_prose(
        (sections.get("Task") or "").strip()
        + "\n"
        + (sections.get("Generation focus") or "").strip()
    )
    raw_context = {
        "generation_principles": (sections.get("Generation principles") or "").strip(),
        "how_to_use_context": (sections.get("How to use the molecule context") or "").strip(),
    }
    if source_id:
        raw_context["source_id"] = source_id

    return make_complete_design_ir(
        task_id="smallmol",
        domain="molecule",
        task_description=(
            (sections.get("Target context") or "").strip()
            + "\n\n"
            + (sections.get("Background") or "").strip()
        ).strip(),
        objectives=[
            {
                "name": "vina_docking",
                "direction": "minimize",
                "description": "AutoDock Vina docking score; lower is better.",
            },
            {
                "name": "neural_activity",
                "direction": "maximize",
                "description": "Target-specific activity model prediction; higher is better.",
            },
        ],
        design_space_description=(sections.get("SMILES hygiene") or "").strip(),
        observations=list(observations_by_design.values()),
        candidates=candidates,
        request_description=request_description,
        num_candidates=len(candidates),
        round_idx=round_idx,
        num_evaluated=history.get("n_evaluated"),
        progress=progress,
        do_not_repeat=history.get("avoid_exact_smiles", []) or [],
        allows_new_parameters=True,
        reasoning_available=True,
        raw_context=raw_context,
    )


def smallmol_irs_from_round_record(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract ldm-2.0 rows from a small-molecule round record."""

    rows: list[dict[str, Any]] = []
    round_idx = record.get("round_idx")
    for attempt in record.get("llm_attempts", []) or []:
        if not isinstance(attempt, Mapping):
            continue
        if attempt.get("error"):
            continue
        if attempt.get("stage") and attempt.get("stage") != "m1_direct":
            continue
        instruction = attempt.get("user_prompt")
        output = attempt.get("raw_text") or attempt.get("raw_output")
        if not isinstance(instruction, str) or not isinstance(output, str):
            continue
        ir = smallmol_ir_from_prompt_response(
            instruction,
            output,
            round_idx=int(round_idx) if round_idx is not None else None,
            source_id=str(attempt.get("source_id") or ""),
        )
        if ir is not None:
            rows.append(ir)
    return rows


def _strip_smallmol_format_prose(text: str) -> str:
    return re.sub(r"\s{2,}", " ", _SMALLMOL_FORMAT_PROSE.sub("", text or "")).strip()


def _sections(text: str, headings: Sequence[str]) -> dict[str, str]:
    indexed = []
    for heading in headings:
        idx = text.find(heading + ":")
        if idx >= 0:
            indexed.append((idx, heading))
    indexed.sort()
    sections: dict[str, str] = {}
    for pos, (idx, heading) in enumerate(indexed):
        end = indexed[pos + 1][0] if pos + 1 < len(indexed) else len(text)
        sections[heading] = text[idx + len(heading) + 1 : end].strip()
    return sections
