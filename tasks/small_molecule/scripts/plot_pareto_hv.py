#!/usr/bin/env python3
"""Visualize small-molecule LDM-TTS Pareto hypervolume curves."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SMALL_MOLECULE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SMALL_MOLECULE_ROOT.parent.parent
for _path in (SMALL_MOLECULE_ROOT, WORKSPACE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

DEFAULT_REF_POINT = (0.0, 5.0)
DEFAULT_MINIMIZE = (True, False)


@dataclass(frozen=True)
class ScoreRow:
    evaluation: int
    smiles: str
    scores: tuple[float, float]


@dataclass(frozen=True)
class RunPlotData:
    label: str
    run_dir: Path
    scores: list[ScoreRow]
    hypervolume: np.ndarray
    summary: dict[str, Any]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ref_point = parse_float_pair(args.ref_point, "--ref-point")
    minimize = parse_bool_pair(args.minimize, "--minimize")
    run_dirs = discover_run_dirs(args)
    if not run_dirs:
        raise SystemExit(
            "No run directories found. Pass a trajectory directory such as "
            "tasks/small_molecule/runs/case2_mock_m1."
        )

    raw_runs = [load_run_scores(run_dir) for run_dir in run_dirs]
    raw_runs = [(run_dir, scores, summary) for run_dir, scores, summary in raw_runs if scores]
    if not raw_runs:
        raise SystemExit("No finite 2-objective scores found in the selected run directories.")

    budget = args.budget if args.budget is not None else max(len(scores) for _run_dir, scores, _summary in raw_runs)
    runs = [
        build_run_plot_data(
            run_dir=run_dir,
            scores=scores,
            summary=summary,
            budget=budget,
            ref_point=ref_point,
            minimize=minimize,
            label=label_for_run(run_dir),
        )
        for run_dir, scores, summary in raw_runs
    ]

    output_dir = resolve_output_dir(args.output_dir, runs, args)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = safe_filename(args.prefix)

    hv_csv = output_dir / f"{prefix}_hypervolume.csv"
    summary_csv = output_dir / f"{prefix}_summary.csv"
    png_path = output_dir / f"{prefix}.png"
    pdf_path = output_dir / f"{prefix}.pdf"

    write_hypervolume_csv(hv_csv, runs)
    write_summary_csv(summary_csv, runs)
    plot_hypervolume(png_path, runs, title=args.title)
    plot_hypervolume(pdf_path, runs, title=args.title)

    print(f"wrote {png_path.resolve()}")
    print(f"wrote {pdf_path.resolve()}")
    print(f"wrote {hv_csv.resolve()}")
    print(f"wrote {summary_csv.resolve()}")
    for run in runs:
        summary_hv = run.summary.get("final_hypervolume")
        suffix = ""
        if isinstance(summary_hv, (int, float)):
            suffix = f" summary_hv={float(summary_hv):.6g}"
        print(
            f"{run.label}: evaluations={len(run.scores)} "
            f"final_hv={float(run.hypervolume[-1]):.6g}{suffix}"
        )
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot cumulative Pareto hypervolume for one or more "
            "small-molecule LDM-TTS trajectory directories."
        )
    )
    parser.add_argument(
        "run_dirs",
        nargs="*",
        help=(
            "Trajectory directories containing history.json or rounds.jsonl. "
            "If omitted, child runs under --runs-root are discovered."
        ),
    )
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Parent directory used when run_dirs are omitted. Relative paths are resolved from tasks/small_molecule/.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Pad/truncate hypervolume curves to this many evaluations. Default: max observed evaluations.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for plots and CSVs. Default: RUN_DIR/plots for one run, otherwise RUNS_ROOT/plots.",
    )
    parser.add_argument("--prefix", default="pareto_hv", help="Output filename prefix.")
    parser.add_argument(
        "--ref-point",
        default="0.0,5.0",
        help="Reference point as 'vina,activity'. Default: 0.0,5.0.",
    )
    parser.add_argument(
        "--minimize",
        default="true,false",
        help="Per-objective minimize flags as 'true,false'. Default: true,false.",
    )
    parser.add_argument(
        "--title",
        default="Small-Molecule LDM-TTS Pareto Hypervolume",
        help="Plot title.",
    )
    return parser.parse_args(argv)


def discover_run_dirs(args: argparse.Namespace) -> list[Path]:
    if args.run_dirs:
        return [resolve_path(raw).resolve() for raw in args.run_dirs]
    runs_root = resolve_path(args.runs_root).resolve()
    if not runs_root.exists():
        return []
    return [
        child.resolve()
        for child in sorted(runs_root.iterdir())
        if child.is_dir() and ((child / "history.json").exists() or (child / "rounds.jsonl").exists())
    ]


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (SMALL_MOLECULE_ROOT / path).resolve()


def resolve_output_dir(raw: str, runs: Sequence[RunPlotData], args: argparse.Namespace) -> Path:
    if raw:
        return resolve_path(raw)
    if len(runs) == 1:
        return runs[0].run_dir / "plots"
    return resolve_path(args.runs_root) / "plots"


def load_run_scores(run_dir: Path) -> tuple[Path, list[ScoreRow], dict[str, Any]]:
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")
    scores = load_history_scores(run_dir / "history.json")
    if not scores:
        scores = load_round_scores(run_dir / "rounds.jsonl")
    return run_dir, scores, load_summary(run_dir / "summary.json")


def load_history_scores(path: Path) -> list[ScoreRow]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    rows: list[ScoreRow] = []
    if not isinstance(payload, list):
        return rows
    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue
        parsed = parse_score_pair(item.get("scores"))
        if parsed is None:
            continue
        rows.append(ScoreRow(evaluation=idx, smiles=str(item.get("smiles", "")), scores=parsed))
    return rows


def load_round_scores(path: Path) -> list[ScoreRow]:
    if not path.exists():
        return []
    rows: list[ScoreRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            selection = record.get("selection_results", {})
            smiles_list = selection.get("selected_smiles") or []
            selected_scores = selection.get("selected_scores") or []
            for smiles, score_pair in zip(smiles_list, selected_scores):
                parsed = parse_score_pair(score_pair)
                if parsed is None:
                    continue
                rows.append(ScoreRow(evaluation=len(rows) + 1, smiles=str(smiles), scores=parsed))
    return rows


def load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def parse_score_pair(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return None
    if value[0] is None or value[1] is None:
        return None
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None


def build_run_plot_data(
    *,
    run_dir: Path,
    scores: list[ScoreRow],
    summary: dict[str, Any],
    budget: int,
    ref_point: tuple[float, float],
    minimize: tuple[bool, bool],
    label: str,
) -> RunPlotData:
    curve = hypervolume_curve([row.scores for row in scores], budget, ref_point, minimize)
    return RunPlotData(
        label=label,
        run_dir=run_dir,
        scores=scores,
        hypervolume=curve,
        summary=summary,
    )


def hypervolume_curve(
    scores: Sequence[tuple[float, float]],
    budget: int,
    ref_point: tuple[float, float],
    minimize: tuple[bool, bool],
) -> np.ndarray:
    values = [0.0]
    finite: list[tuple[float, float]] = []
    for score in scores[:budget]:
        finite.append(score)
        values.append(compute_hypervolume(finite, ref_point, minimize))
    while len(values) < budget + 1:
        values.append(values[-1])
    return np.asarray(values[: budget + 1], dtype=float)


def compute_hypervolume(
    points: Sequence[tuple[float, float]],
    ref_point: tuple[float, float],
    minimize: tuple[bool, bool],
) -> float:
    try:
        from ldm_tts.optimization.acquisition import hypervolume

        return float(hypervolume(points, ref_point, minimize=minimize))
    except Exception:
        return hypervolume_2d_fallback(points, ref_point, minimize)


def pareto_score_indices(
    points: Sequence[tuple[float, float]],
    minimize: tuple[bool, bool],
) -> list[int]:
    indices = []
    for idx, point in enumerate(points):
        if not any(
            other_idx != idx and dominates_2d(other, point, minimize)
            for other_idx, other in enumerate(points)
        ):
            indices.append(idx)
    return indices


def dominates_2d(
    left: tuple[float, float],
    right: tuple[float, float],
    minimize: tuple[bool, bool],
) -> bool:
    better_or_equal = []
    strictly_better = []
    for left_value, right_value, is_minimize in zip(left, right, minimize):
        if is_minimize:
            better_or_equal.append(left_value <= right_value)
            strictly_better.append(left_value < right_value)
        else:
            better_or_equal.append(left_value >= right_value)
            strictly_better.append(left_value > right_value)
    return all(better_or_equal) and any(strictly_better)


def hypervolume_2d_fallback(
    points: Sequence[tuple[float, float]],
    ref_point: tuple[float, float],
    minimize: tuple[bool, bool],
) -> float:
    converted = []
    ref = (
        float(ref_point[0]) if minimize[0] else -float(ref_point[0]),
        float(ref_point[1]) if minimize[1] else -float(ref_point[1]),
    )
    for point in points:
        converted_point = (
            float(point[0]) if minimize[0] else -float(point[0]),
            float(point[1]) if minimize[1] else -float(point[1]),
        )
        if converted_point[0] < ref[0] and converted_point[1] < ref[1]:
            converted.append(converted_point)
    if not converted:
        return 0.0
    front = [converted[idx] for idx in pareto_score_indices(converted, (True, True))]
    front.sort(key=lambda item: item[0])
    volume = 0.0
    for idx, point in enumerate(front):
        next_x = front[idx + 1][0] if idx + 1 < len(front) else ref[0]
        volume += max(0.0, next_x - point[0]) * max(0.0, ref[1] - point[1])
    return float(volume)


def write_hypervolume_csv(path: Path, runs: Sequence[RunPlotData]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["evaluation", "run", "hypervolume"])
        for run in runs:
            for evaluation, value in enumerate(run.hypervolume):
                writer.writerow([evaluation, run.label, float(value)])
        if len(runs) > 1:
            arr = np.stack([run.hypervolume for run in runs], axis=0)
            for evaluation in range(arr.shape[1]):
                writer.writerow([evaluation, "mean", float(arr[:, evaluation].mean())])
                writer.writerow([evaluation, "std", float(arr[:, evaluation].std())])


def write_summary_csv(path: Path, runs: Iterable[RunPlotData]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "run",
            "run_dir",
            "evaluations",
            "final_hypervolume",
            "summary_final_hypervolume",
            "method",
            "llm_call_count",
            "round_count",
        ])
        for run in runs:
            writer.writerow([
                run.label,
                str(run.run_dir),
                len(run.scores),
                float(run.hypervolume[-1]),
                run.summary.get("final_hypervolume", ""),
                run.summary.get("method", ""),
                run.summary.get("llm_call_count", ""),
                run.summary.get("round_count", ""),
            ])


def plot_hypervolume(
    path: Path,
    runs: Sequence[RunPlotData],
    *,
    title: str,
) -> None:
    configure_matplotlib_cache(path.parent)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    colors = plt.get_cmap("tab10")

    for run_idx, run in enumerate(runs):
        color = colors(run_idx % 10)
        ax.plot(
            np.arange(len(run.hypervolume)),
            run.hypervolume,
            linewidth=2.4 if len(runs) == 1 else 1.45,
            alpha=0.98 if len(runs) == 1 else 0.55,
            color=color,
            label=run.label,
        )

    if len(runs) > 1:
        arr = np.stack([run.hypervolume for run in runs], axis=0)
        x = np.arange(arr.shape[1])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ax.plot(x, mean, color="black", linewidth=2.8, label="mean")
        ax.fill_between(x, mean - std, mean + std, color="black", alpha=0.12, linewidth=0)

    ax.set_xlabel("Expensive evaluations")
    ax.set_ylabel("Pareto hypervolume")
    ax.set_title("Cumulative Pareto Hypervolume")
    ax.grid(alpha=0.28)
    ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def configure_matplotlib_cache(output_dir: Path) -> None:
    cache_dir = output_dir / ".plot_cache"
    mpl_cache = cache_dir / "matplotlib"
    xdg_cache = cache_dir / "xdg"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))


def parse_float_pair(raw: str, name: str) -> tuple[float, float]:
    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    if len(parts) != 2:
        raise SystemExit(f"{name} must contain exactly two comma-separated values.")
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError as exc:
        raise SystemExit(f"{name} contains a non-float value: {raw!r}") from exc


def parse_bool_pair(raw: str, name: str) -> tuple[bool, bool]:
    parts = [part.strip().lower() for part in str(raw).split(",") if part.strip()]
    if len(parts) != 2:
        raise SystemExit(f"{name} must contain exactly two comma-separated booleans.")
    return (parse_bool(parts[0], name), parse_bool(parts[1], name))


def parse_bool(raw: str, name: str) -> bool:
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    raise SystemExit(f"{name} contains a non-boolean value: {raw!r}")


def label_for_run(run_dir: Path) -> str:
    return run_dir.name or str(run_dir)


def safe_filename(raw: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in raw.strip())
    return cleaned.strip("._-") or "pareto_hv"


if __name__ == "__main__":
    raise SystemExit(main())
