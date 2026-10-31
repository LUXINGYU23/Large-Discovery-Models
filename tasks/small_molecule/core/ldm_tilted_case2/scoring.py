"""Score molecular candidates and retain per-attempt diagnostics."""

from __future__ import annotations

import json
from typing import Any, Sequence

from ldm_tts.contracts.evaluation import finite_or_none, is_finite_number

MAX_SCORE_ATTEMPTS = 2


def _score_smiles_with_diagnostics(smiles_list, scorers):
    smiles_list = list(smiles_list)
    per_obj = []
    per_obj_diagnostics = []
    for scorer in scorers:
        values, diagnostics = _score_with_retries_with_diagnostics(smiles_list, scorer)
        per_obj.append([finite_or_none(value) for value in values])
        per_obj_diagnostics.append(diagnostics)
    scores = [tuple(values[i] for values in per_obj) for i in range(len(smiles_list))]
    diagnostics = [
        {
            "smiles": smiles,
            "objectives": [
                obj_diagnostics[i]
                for obj_diagnostics in per_obj_diagnostics
                if i < len(obj_diagnostics)
            ],
        }
        for i, smiles in enumerate(smiles_list)
    ]
    return scores, diagnostics


def _score_with_retries_with_diagnostics(smiles_list, scorer):
    smiles_list = list(smiles_list)
    values, call_info = _call_scorer_with_info(smiles_list, scorer)
    diagnostics = [
        _score_diagnostic(value, _attempt_diagnostic(value, call_info, idx))
        for idx, value in enumerate(values)
    ]
    for _attempt in range(1, MAX_SCORE_ATTEMPTS):
        bad_indices = [
            idx for idx, value in enumerate(values) if not is_finite_number(value)
        ]
        if not bad_indices:
            break
        for idx in bad_indices:
            retry_values, retry_info = _call_scorer_with_info([smiles_list[idx]], scorer)
            diagnostics[idx]["attempts"].append(
                _attempt_diagnostic(
                    retry_values[0] if retry_values else float("nan"),
                    retry_info,
                    0,
                )
            )
            if retry_values and is_finite_number(retry_values[0]):
                values[idx] = retry_values[0]
            diagnostics[idx]["final_value"] = finite_or_none(values[idx])
            diagnostics[idx]["final_finite"] = is_finite_number(values[idx])
    return values, diagnostics


def _call_scorer_with_info(smiles_list, scorer):
    smiles_list = list(smiles_list)
    try:
        values = list(scorer(smiles_list))
    except Exception as exc:
        raise RuntimeError(
            f"{_scorer_name(scorer)} failed while scoring {len(smiles_list)} "
            f"SMILES; sample={_sample_smiles_for_error(smiles_list)}: {exc}"
        ) from exc
    details = _scorer_last_results(scorer)
    if len(values) != len(smiles_list):
        raise RuntimeError(
            f"{_scorer_name(scorer)} returned {len(values)} values for "
            f"{len(smiles_list)} SMILES; sample={_sample_smiles_for_error(smiles_list)}"
        )
    return values, {"error": None, "details": details}


def _scorer_name(scorer) -> str:
    if hasattr(scorer, "__class__"):
        class_name = scorer.__class__.__name__
        if class_name and class_name != "function":
            return class_name
    return getattr(scorer, "__name__", type(scorer).__name__)


def _sample_smiles_for_error(smiles_list: Sequence[str], limit: int = 3) -> list[str]:
    return [_short_smiles(smiles) for smiles in smiles_list[:limit]]


def _scorer_last_results(scorer) -> list[Any]:
    details = getattr(scorer, "last_results", None)
    if not isinstance(details, list):
        return []
    return _json_safe(details)


def _score_diagnostic(value, first_attempt: dict[str, Any]) -> dict[str, Any]:
    return {
        "final_value": finite_or_none(value),
        "final_finite": is_finite_number(value),
        "attempts": [first_attempt],
    }


def _attempt_diagnostic(value, call_info: dict[str, Any], idx: int) -> dict[str, Any]:
    attempt = {
        "value": finite_or_none(value),
        "finite": is_finite_number(value),
    }
    if call_info.get("error"):
        attempt["error"] = call_info["error"]
    detail = _detail_at(call_info.get("details"), idx)
    if detail is not None:
        attempt["detail"] = detail
    return attempt


def _detail_at(details, idx: int):
    if not isinstance(details, list) or idx >= len(details):
        return None
    return details[idx]


def _json_safe(value):
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)


def _short_smiles(smiles: str | None, max_len: int = 80) -> str:
    text = str(smiles or "")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
