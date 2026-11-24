"""Reader for the evaluator-side outcome export (contracts §9.1).

Input: one or more tables written by experiments/rsi/export_outcomes.py (T7),
schema "rsi-outcomes-v1": {experiment, protocol_id, sealed, frozen_list_sha256,
unseal_log_sha256, columns, rows}, one row per frozen solver with prefix "full"
or "round_tt". A directory argument means every *.json table in it.

Optional blocks a table may carry (T8 extension; produced after the unseal only):

  body_scores  [{run_id, prefix, split, iou: [float, ...]}]
               sealed IoU per body, in one fixed anonymous body order shared by
               all runs of the protocol; no case or parent ids, no hashes
               (scoped exception to AT-31 for the body-paired bootstrap of spec §7)
  programs     [{run_id, program_id, train_iou}]          hidden training IoU per program
  validations  [{run_id, program_id, risk, iou}]          offline validation of incumbents
  references   {name: {test_id: iou, test_ood: iou}}      domain references (spec §4.3)
  q1           {programs: [{program_id, short_12x1, full_24x1, short_24x2}],
                repeats: [{program_id, risks: [float, ...]}]}

All lookups are keyed by (experiment, run_id): run ids repeat across experiments.
"""

import json
import re
from pathlib import Path

SCHEMA = "rsi-outcomes-v1"
TABLE = {"schema", "experiment", "protocol_id", "sealed", "frozen_list_sha256", "unseal_log_sha256",
         "columns", "rows", "body_scores", "programs", "validations", "diagnostics", "strata_slices",
         "references", "q1"}
FORBIDDEN = re.compile(r"case_id|parent_id|truth|geometry|body_id", re.IGNORECASE)
SPLITS = ("test_id", "test_ood")
LAYOUTS = ("drill", "drill_outcrop")
FULL = "full"


def _scan(value, path="outcomes"):
    if isinstance(value, dict):
        for key, item in value.items():
            if FORBIDDEN.search(str(key)):
                raise ValueError(f"outcome export carries a body-identifying field: {path}.{key}")
            _scan(item, f"{path}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _scan(item, f"{path}[{i}]")


def prefix_round(prefix):
    if prefix == FULL:
        return FULL
    match = re.fullmatch(r"round_(\d+)", str(prefix))
    if match is None:
        raise ValueError("invalid freeze prefix")
    return int(match.group(1))


def _freeze(row):
    """A T7 row as {program_id, validation, sealed}."""
    sealed = {}
    for split in SPLITS:
        if row.get(f"{split}_iou") is None:
            continue
        sealed[split] = {k: row.get(f"{split}_{k}") for k in
                         ("iou", "risk", "legal_rate", "seconds_per_solve", "surface_distance", "volume_error")}
        sealed[split]["by_layout"] = {m: row.get(f"{split}_iou_{m}") for m in LAYOUTS}
    return {"program_id": row["selected_sha256"], "prefix": row["prefix"],
            "validation": {"risk": row.get("validation_risk"), "iou": row.get("validation_iou"),
                           "legal_rate": row.get("validation_legal_rate"),
                           "seconds_per_solve": row.get("validation_seconds_per_solve")},
            "sealed": sealed or None}


class Outcomes:
    def __init__(self, tables):
        self.tables = tables
        self.freezes, self.bodies, self.train_iou, self.validation = {}, {}, {}, {}
        self.diagnostics, self.strata_slices = {}, {}
        self.references, self.q1, self.protocol_ids, self.unseal_logs = {}, None, set(), {}
        for table in tables:
            if table.get("schema") != SCHEMA or set(table) - TABLE:
                raise ValueError("unexpected outcome export schema or fields")
            _scan(table)
            exp = table["experiment"]
            self.protocol_ids.add(table.get("protocol_id"))
            if table.get("unseal_log_sha256"):
                self.unseal_logs[exp] = table["unseal_log_sha256"]
            for row in table.get("rows", []):
                key = (exp, row["run_id"], prefix_round(row["prefix"]))
                if key in self.freezes:
                    raise ValueError("duplicate frozen solver for one run and prefix")
                self.freezes[key] = _freeze(row)
                v = self.freezes[key]["validation"]
                if v["risk"] is not None:
                    self.validation.setdefault((exp, row["run_id"], row["selected_sha256"]), v)
            for row in table.get("body_scores", []):
                self.bodies[(exp, row["run_id"], prefix_round(row["prefix"]), row["split"])] = row["iou"]
            for row in table.get("programs", []):
                self.train_iou[(exp, row["run_id"], row["program_id"])] = row["train_iou"]
            for row in table.get("validations", []):
                self.validation[(exp, row["run_id"], row["program_id"])] = row
            for row in table.get("diagnostics", []):
                self.diagnostics[(exp, row["run_id"])] = row
            for row in table.get("strata_slices", []):
                self.strata_slices[(exp, row["run_id"], prefix_round(row["prefix"]), row["split"])] = {
                    k: row[k] for k in ("family", "modality", "noise") if k in row}
            self.references.update(table.get("references", {}))
            if table.get("q1"):
                self.q1 = table["q1"]

    @classmethod
    def load(cls, path):
        path = Path(path)
        files = sorted(path.glob("*.json")) if path.is_dir() else [path]
        return cls([json.loads(f.read_text()) for f in files])

    @staticmethod
    def files(path):
        path = Path(path)
        return sorted(path.glob("*.json")) if path.is_dir() else [path]

    @property
    def protocol_id(self):
        ids = sorted(p for p in self.protocol_ids if p)
        if len(ids) > 1:
            raise ValueError("outcome tables come from different protocols")
        return ids[0] if ids else None

    # -- lookups; `rounds` is the run's nominal last round, so that "full" answers it
    def prefixes(self, exp, run_id):
        return [p for (e, r, p) in self.freezes if e == exp and r == run_id]

    def freeze(self, exp, run_id, prefix, rounds=None):
        row = self.freezes.get((exp, run_id, prefix))
        if row is None and prefix == rounds:
            row = self.freezes.get((exp, run_id, FULL))
        if row is None and prefix == FULL and rounds is not None:
            row = self.freezes.get((exp, run_id, rounds))
        return row

    def final(self, exp, run_id):
        """The whole-run freeze ("full"), else the latest prefix."""
        prefixes = self.prefixes(exp, run_id)
        if not prefixes:
            return None
        return FULL if FULL in prefixes else max(prefixes)

    def sealed(self, exp, run_id, prefix, split, metric="iou", rounds=None):
        row = self.freeze(exp, run_id, prefix, rounds)
        if not row or not row.get("sealed") or split not in row["sealed"]:
            return None
        return row["sealed"][split].get(metric)

    def body_row(self, exp, run_id, prefix, split, rounds=None):
        row = self.bodies.get((exp, run_id, prefix, split))
        if row is None and prefix == rounds:
            row = self.bodies.get((exp, run_id, FULL, split))
        return row

    def validation_value(self, exp, run_id, prefix, metric="risk", rounds=None):
        row = self.freeze(exp, run_id, prefix, rounds)
        return None if not row else row["validation"].get(metric)
