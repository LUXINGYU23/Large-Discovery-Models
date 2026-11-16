"""The only feedback permitted across the research/evaluator boundary."""

import math
import re


VERSION = "rsi-feedback-v2"
FIELDS = frozenset({"source_sha256", "status", "risk", "risk_by_case", "legal_rate",
                    "failed_solves", "seconds_per_solve", "message"})


def valid_case_id(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value) is not None


def risk_by_case(rows, case_ids):
    grouped = {case_id: [] for case_id in case_ids}
    for row in rows:
        case_id = row["case_id"]
        if row["status"] == "succeeded":
            loss = row["metrics"]["confirmation_loss_total"]
            if type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0:
                raise ValueError("invalid confirmation loss in private receipt")
            value = loss / (1.0 + loss)
        else:
            value = 1.0
        grouped[case_id].append(value)
    if any(not values for values in grouped.values()):
        raise ValueError("private receipt is missing a planned case")
    return {key: sum(values) / len(values) for key, values in sorted(grouped.items())}


def validate_feedback(value):
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise ValueError("feedback contains unexpected or missing fields")
    if (not isinstance(value["source_sha256"], str)
            or re.fullmatch(r"[a-f0-9]{64}", value["source_sha256"]) is None):
        raise ValueError("invalid feedback source identity")
    if value["status"] not in {"valid", "rejected", "failed"}:
        raise ValueError("invalid feedback status")
    if value["message"] not in {"", "interface rejected", "all planned solves failed"}:
        raise ValueError("invalid feedback reason")
    def numeric(number, lo, hi=None):
        return (type(number) in (int, float) and math.isfinite(number)
                and number >= lo and (hi is None or number <= hi))
    if (not numeric(value["risk"], 0, 1) or not numeric(value["legal_rate"], 0, 1)
            or not numeric(value["seconds_per_solve"], 0)
            or type(value["failed_solves"]) is not int or value["failed_solves"] < 0):
        raise ValueError("invalid feedback numbers")
    cases = value["risk_by_case"]
    if (not isinstance(cases, dict) or not cases
            or any(not valid_case_id(k) or not numeric(v, 0, 1) for k, v in cases.items())):
        raise ValueError("invalid public case risks")
    if not math.isclose(sum(cases.values()) / len(cases), value["risk"], abs_tol=1e-10):
        raise ValueError("feedback parent mean differs")
    if value["status"] == "rejected":
        if value["risk"] != 1 or value["legal_rate"] != 0 or value["message"] != "interface rejected":
            raise ValueError("invalid rejected feedback")
    elif value["status"] == "failed":
        if value["risk"] != 1 or value["legal_rate"] != 0 or value["message"] != "all planned solves failed":
            raise ValueError("invalid failed feedback")
    elif value["message"]:
        raise ValueError("valid feedback cannot contain a reason")
    return value
