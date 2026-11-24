"""Case-identity isolation: canonical configuration, opaque per-evaluation IDs and static source checks."""

import ast
import hashlib
import re

CASE_FEEDBACK_MODES = ("none", "opaque", "public")
REQUIRED_CASE_ISOLATION_KEYS = ("anonymize_cases", "canonical_origin", "case_feedback", "refuse_case_ids")
OPTIONAL_CASE_ISOLATION_KEYS = ("randomize_orientation",)
CASE_ISOLATION_KEYS = REQUIRED_CASE_ISOLATION_KEYS + OPTIONAL_CASE_ISOLATION_KEYS

DEFAULT_CASE_ISOLATION = {
    "anonymize_cases": True,
    "canonical_origin": True,
    "case_feedback": "none",
    "refuse_case_ids": True,
    "randomize_orientation": False,
}

LEGACY_CASE_ISOLATION = {
    "anonymize_cases": False,
    "canonical_origin": False,
    "case_feedback": "public",
    "refuse_case_ids": False,
    "randomize_orientation": False,
}

_HEX_ID = re.compile(r"[0-9a-f]{16,}")


def validate_case_isolation(value):
    if not isinstance(value, dict) or not set(REQUIRED_CASE_ISOLATION_KEYS) <= set(value) <= set(CASE_ISOLATION_KEYS):
        raise ValueError("case_isolation requires: " + ", ".join(REQUIRED_CASE_ISOLATION_KEYS)
                         + " (optional: " + ", ".join(OPTIONAL_CASE_ISOLATION_KEYS) + ")")
    normalized = {**value, "randomize_orientation": value.get("randomize_orientation", False)}
    for name in ("anonymize_cases", "canonical_origin", "refuse_case_ids", "randomize_orientation"):
        if type(normalized[name]) is not bool:
            raise ValueError(f"case_isolation.{name} must be a boolean")
    if normalized["case_feedback"] not in CASE_FEEDBACK_MODES:
        raise ValueError(f"case_isolation.case_feedback must be one of {CASE_FEEDBACK_MODES}")
    if normalized["case_feedback"] == "opaque" and not normalized["anonymize_cases"]:
        raise ValueError("case_isolation.case_feedback='opaque' requires anonymize_cases=true")
    return {k: normalized[k] for k in CASE_ISOLATION_KEYS}


def resolve_case_isolation(value):
    if value is None:
        return dict(LEGACY_CASE_ISOLATION)
    return validate_case_isolation(value)


def opaque_case_id(evaluation_key, case_id):
    """Deterministic per-(evaluation, case) opaque identifier; changes for every candidate source."""
    if not isinstance(evaluation_key, str) or not evaluation_key or not isinstance(case_id, str) or not case_id:
        raise ValueError("non-empty evaluation_key and case_id required")
    digest = hashlib.sha256(f"rsi-case-v1:{evaluation_key}:{case_id}".encode("utf-8")).hexdigest()[:16]
    return f"case-{digest}"


def case_orientation(evaluation_key, case_id):
    """Deterministic per-(evaluation, case) horizontal D4 orientation index in 0..7."""
    if not isinstance(evaluation_key, str) or not evaluation_key or not isinstance(case_id, str) or not case_id:
        raise ValueError("non-empty evaluation_key and case_id required")
    return hashlib.sha256(f"rsi-orient-v1:{evaluation_key}:{case_id}".encode("utf-8")).digest()[0] & 7


def export_case_id(index):
    if type(index) is not int or index < 0:
        raise ValueError("non-negative case index required")
    return f"case-{index:04d}"


def forbidden_case_tokens(case_ids, exported_count=0):
    """Tokens a submitted solver must never embed: release IDs, long hex prefixes and workspace IDs."""
    tokens = set()
    for cid in case_ids:
        if not isinstance(cid, str) or not cid:
            continue
        lowered = cid.lower()
        tokens.add(lowered)
        if _HEX_ID.fullmatch(lowered):
            tokens.add(lowered[:8])
    for index in range(max(0, int(exported_count))):
        tokens.add(export_case_id(index))
    return frozenset(tokens)


def check_source_case_ids(source, forbidden):
    """Refuse a solver that embeds any training or workspace case identifier."""
    if not forbidden:
        return
    text = source.decode("utf-8") if isinstance(source, bytes) else source
    tree = ast.parse(text)
    lowered_forbidden = {tok.lower() for tok in forbidden if tok}
    long_tokens = sorted((tok for tok in lowered_forbidden if len(tok) >= 8), key=len, reverse=True)
    hex_tokens = [tok for tok in long_tokens if re.fullmatch(r"[0-9a-f]{8,}", tok)]
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            val = (node.value.decode("utf-8", "ignore") if isinstance(node.value, bytes) else node.value).lower()
            if val in lowered_forbidden or any(tok in val for tok in long_tokens):
                raise ValueError("source references a training case identifier")
    lowered_text = text.lower()
    if any(tok in lowered_text for tok in hex_tokens):
        raise ValueError("source references a training case identifier")
