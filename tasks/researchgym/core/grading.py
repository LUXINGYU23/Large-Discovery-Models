"""Read official grader summaries and enforce the case's complete coverage identity.

Each official grader averages whatever cells it finds. A candidate that crashes
on a hard dataset could otherwise score higher than one that runs everywhere,
so every declared cell, fold, and seed must be present exactly as specified.
"""

from __future__ import annotations

import math
from typing import Any

from .cases import CaseSpec


class CoverageError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("incomplete coverage: " + "; ".join(problems[:6]))


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def score_summary(case: CaseSpec, summary: dict[str, Any]) -> dict[str, float]:
    """Return the objective plus per-cell diagnostics, or raise CoverageError."""
    coverage = case.raw["coverage"]
    parser = {"tse_real": _tse_real, "cl_table": _cl_table, "cmr_row": _cmr_row,
              "rl_environment": _rl_environment}[coverage["kind"]]
    objective, cells, diagnostics, problems = parser(coverage, summary)
    if problems or objective is None:
        raise CoverageError(problems or ["objective is missing"])
    return {case.metric["name"]: objective,
            **{f"cell.{name}": value for name, value in cells.items()}, **diagnostics}


def _tse_real(coverage, summary):
    cells, problems = {}, []
    real = summary.get("real") if isinstance(summary.get("real"), dict) else {}
    for dataset in coverage["datasets"]:
        for baseline in coverage["baselines"]:
            entry = (real.get(dataset) or {}).get(baseline)
            name = f"{dataset}.{baseline}"
            value = _number((entry or {}).get("mean"))
            count = (entry or {}).get("count")
            if value is None:
                problems.append(f"missing {name}")
            elif count != coverage["count"]:
                problems.append(f"{name} has {count} folds, expected {coverage['count']}")
            else:
                cells[name] = value
    extra = sorted(f"{d}.{b}" for d, row in real.items() if isinstance(row, dict) for b in row
                   if f"{d}.{b}" not in {f"{x}.{y}" for x in coverage["datasets"] for y in coverage["baselines"]})
    if extra:
        problems.append(f"unexpected cells {extra}")
    objective = sum(cells.values()) / len(cells) if cells else None
    return objective, cells, {}, problems


def _cl_table(coverage, summary):
    problems, diagnostics = [], {}
    tables = [t for t in summary.get("tables") or [] if t.get("table") == coverage["table"]]
    rows = [row for table in tables for row in table.get("datasets") or []
            if row.get("dataset") == coverage["dataset"] and row.get("num_tasks") == coverage["num_tasks"]]
    if len(rows) != 1:
        return None, {}, {}, [f"expected one {coverage['dataset']} N={coverage['num_tasks']} row, found {len(rows)}"]
    row = rows[0]
    metrics = row.get("metrics") or {}
    acc = _number(metrics.get("acc_mean"))
    if acc is None:
        problems.append("acc_mean is missing")
    if row.get("num_runs") != len(coverage["seeds"]):
        problems.append(f"{row.get('num_runs')} runs, expected {len(coverage['seeds'])}")
    if sorted(row.get("seeds") or []) != sorted(coverage["seeds"]):
        problems.append(f"seeds {row.get('seeds')} != {coverage['seeds']}")
    for name in ("aaa_mean", "acc_se", "aaa_se"):
        value = _number(metrics.get(name))
        if value is not None:
            diagnostics[f"diagnostic.{name}"] = value
    return acc, {}, diagnostics, problems


def _cmr_row(coverage, summary):
    problems, cells = [], {}
    rows = [row for row in summary.get("rows") or [] if row.get("row") == coverage["row"]]
    if len(rows) != 1:
        return None, {}, {}, [f"expected one {coverage['row']} row, found {len(rows)}"]
    columns = {column.get("id"): column for column in rows[0].get("columns") or []}
    for name in coverage["columns"]:
        column = columns.get(name) or {}
        value = _number(column.get("value"))
        sources = column.get("sources") or []
        if value is None:
            problems.append(f"missing {name}")
        elif len(sources) != coverage["sources_per_column"]:
            problems.append(f"{name} has {len(sources)} result sources, expected {coverage['sources_per_column']}")
        else:
            cells[name] = value
    average = _number(rows[0].get("average"))
    if average is None:
        problems.append("row average is missing")
    elif cells and not problems and abs(average - sum(cells.values()) / len(cells)) > 1e-6:
        problems.append("row average does not match its columns")
    return average, cells, {}, problems


def _rl_environment(coverage, summary):
    problems = []
    entry = (summary.get("environments") or {}).get(coverage["environment"]) or {}
    stats = entry.get("average_return") or {}
    mean = _number(stats.get("mean"))
    seeds = sorted(sample.get("seed") for sample in entry.get("samples") or [])
    if mean is None:
        problems.append(f"missing {coverage['environment']} average_return")
    if stats.get("count") != len(coverage["seeds"]) or seeds != sorted(coverage["seeds"]):
        problems.append(f"seeds {seeds} (count {stats.get('count')}) != {coverage['seeds']}")
    extra = sorted(set(summary.get("environments") or {}) - {coverage["environment"]})
    if extra:
        problems.append(f"unexpected environments {extra}")
    diagnostics = {}
    std = _number(stats.get("std"))
    if std is not None:
        diagnostics["diagnostic.return_std"] = std
    cells = {f"{coverage['environment']}.seed{s['seed']}": _number(s.get("average_return"))
             for s in entry.get("samples") or [] if _number(s.get("average_return")) is not None}
    return mean, cells, diagnostics, problems
