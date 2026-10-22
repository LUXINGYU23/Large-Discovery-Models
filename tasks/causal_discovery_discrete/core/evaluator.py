"""Mock and pinned MLS-Bench evaluators for discrete causal discovery."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from ldm_tts.contracts import Candidate, EvaluationResult
from ldm_tts.engine.run_store import atomic_json_write


OFFICIAL_COMMIT = "cfd57a7e0139c72753e32e31bca593719b098717"
TASK_PATH = "tasks/causal-discovery-discrete"
HARBOR_TASK_PATH = "harbor/tasks/mls-bench__causal-discovery-discrete"
OFFICIAL_CASES = {
    "Cancer": ("cancer", 500),
    "Child": ("child", 2000),
    "Alarm": ("alarm", 5000),
    "Hailfinder": ("hailfinder", 10000),
    "Win95pts": ("win95pts", 10000),
}
UPSTREAM_SHA256 = {
    "tasks/causal-discovery-discrete/config.json": "d4b5a9027c0170e965f5ff06525335802459e2675e93103fb7d8e193babf1657",
    "tasks/causal-discovery-discrete/score_spec.py": "c9bb34fdea2284bb6031abdb45efd70f96525a923cfc67ccac204900e74a5ea2",
    "tasks/causal-discovery-discrete/leaderboard.csv": "37cfeb699013ab1baaac42e0b1b6338b1fca1847768c9cbcb3d586e44f65e30c",
    "harbor/tasks/mls-bench__causal-discovery-discrete/tests/meta/dgp.py": "3eaccae6f9316af2bbdf7b28534c21142b902e4336e3c8035937b4dd1cd45608",
    "harbor/tasks/mls-bench__causal-discovery-discrete/tests/meta/score_spec.py": "c9bb34fdea2284bb6031abdb45efd70f96525a923cfc67ccac204900e74a5ea2",
}


class MockCausalEvaluator:
    def evaluate(self, candidate: Candidate) -> EvaluationResult:
        spec = candidate.payload["spec"]
        threshold = float(spec["min_association"])
        degree = int(spec["max_degree"])
        score = max(0.0, 0.72 - abs(threshold - 0.07) * 2.0 - abs(degree - 6) * 0.01)
        return EvaluationResult(
            candidate.candidate_id,
            "succeeded",
            metrics={"selection_score": score, "edge_budget": float(degree)},
            resource_usage={"benchmark_jobs": 1},
            metadata={"evaluator": "deterministic_mock", "official": False},
        )


class MLSBenchCausalEvaluator:
    def __init__(
        self,
        *,
        upstream_root: Path,
        run_dir: Path,
        timeout_seconds: float = 3540.0,
        evaluator_python: str | Path = sys.executable,
    ) -> None:
        self.upstream_root = Path(upstream_root).resolve()
        self.run_dir = Path(run_dir)
        self.timeout_seconds = min(float(timeout_seconds), 3540.0)
        if not math.isfinite(float(timeout_seconds)) or self.timeout_seconds <= 0:
            raise ValueError("evaluation timeout must be finite and positive")
        self.evaluator_python = str(evaluator_python)
        validate_upstream_contract(self.upstream_root)
        self._prepared = self._prepare_cases()

    def evaluate(self, candidate: Candidate) -> EvaluationResult:
        evaluation_dir = self.run_dir / "evaluations" / candidate.candidate_id
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = evaluation_dir / "evaluation_manifest.json"
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "candidate_id": candidate.candidate_id,
            "source_commit": OFFICIAL_COMMIT,
            "cases": list(OFFICIAL_CASES),
            "status": "running",
        }
        atomic_json_write(manifest_path, manifest)
        started = time.monotonic()
        case_records = []
        try:
            metrics: dict[str, float] = {}
            for label, prepared in self._prepared.items():
                record = {"label": label, "status": "running"}
                case_records.append(record)
                values = _evaluate_case(prepared, candidate.payload["spec"], self.timeout_seconds)
                for name, value in values.items():
                    metrics[f"{name}_{label}"] = float(value)
                record.update(status="succeeded", metrics=values)
            scoring = score_with_pinned_mlsbench(
                self.upstream_root,
                metrics,
                evaluator_python=self.evaluator_python,
                timeout_seconds=min(30.0, self.timeout_seconds),
            )
            metrics["official_score"] = scoring["official_score"]
            metrics["selection_score"] = scoring["official_score"]
            metrics["runtime_seconds"] = time.monotonic() - started
            manifest.update(
                status="succeeded",
                candidate_spec=candidate.payload["spec"],
                metrics=metrics,
                cases=case_records,
                scoring=scoring,
            )
            atomic_json_write(manifest_path, manifest)
            return EvaluationResult(
                candidate.candidate_id,
                "succeeded",
                metrics=metrics,
                artifacts={"evaluation_manifest": self._artifact_path(manifest_path)},
                resource_usage={
                    "wall_seconds": time.monotonic() - started,
                    "benchmark_jobs": len(case_records),
                },
                metadata={
                    "evaluator": "pinned_mlsbench_causal_discovery_discrete",
                    "official": True,
                    "source_commit": OFFICIAL_COMMIT,
                },
            )
        except Exception as exc:
            status = "timed_out" if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)) else "failed"
            if case_records and case_records[-1]["status"] == "running":
                case_records[-1]["status"] = status
            manifest.update(
                status=status,
                cases=case_records,
                error=str(exc) or repr(exc),
                error_type=type(exc).__name__,
            )
            atomic_json_write(manifest_path, manifest)
            return EvaluationResult(
                candidate.candidate_id,
                status,
                error=str(exc),
                artifacts={"evaluation_manifest": self._artifact_path(manifest_path)},
                resource_usage={"benchmark_jobs": len(case_records), "wall_seconds": time.monotonic() - started},
                metadata={"evaluator": "pinned_mlsbench_causal_discovery_discrete"},
            )

    def _artifact_path(self, path: Path) -> str:
        return path.resolve().relative_to(self.run_dir.resolve()).as_posix()

    def _prepare_cases(self) -> dict[str, dict[str, Any]]:
        from causallearn.graph.Dag import Dag
        from causallearn.graph.GraphNode import GraphNode
        from causallearn.utils.DAG2CPDAG import dag2cpdag
        from pgmpy.sampling import BayesianModelSampling
        from pgmpy.utils import get_example_model

        prepared = {}
        for label, (network_name, n_samples) in OFFICIAL_CASES.items():
            model = get_example_model(network_name)
            frame = BayesianModelSampling(model).forward_sample(size=n_samples, seed=42)
            node_names = sorted(model.nodes())
            data = np.zeros((n_samples, len(node_names)), dtype=np.int64)
            for index, name in enumerate(node_names):
                values = frame[name].to_numpy()
                categories = {value: i for i, value in enumerate(sorted(set(values)))}
                data[:, index] = [categories[value] for value in values]
            nodes = [GraphNode(f"X{i + 1}") for i in range(len(node_names))]
            dag = Dag(nodes)
            lookup = {name: index for index, name in enumerate(node_names)}
            for parent, child in model.edges():
                dag.add_directed_edge(nodes[lookup[parent]], nodes[lookup[child]])
            prepared[label] = {
                "nodes": nodes,
                "association": _association_matrix(data),
                "true_cpdag": dag2cpdag(dag),
            }
        return prepared


def _case_worker(connection, prepared, spec):
    try:
        estimate = _build_estimated_graph(prepared, spec)
        connection.send({"metrics": _score_graph(prepared["true_cpdag"], estimate)})
    except Exception as exc:
        connection.send({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def _evaluate_case(prepared, spec, timeout_seconds):
    # A process deadline also interrupts graph code blocked inside native libraries.
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_case_worker, args=(sender, prepared, spec))
    deadline = time.monotonic() + timeout_seconds
    try:
        process.start()
        sender.close()
        if not receiver.poll(max(0, deadline - time.monotonic())):
            raise TimeoutError(f"causal network evaluation exceeded {timeout_seconds} seconds")
        try:
            result = receiver.recv()
        except EOFError as exc:
            raise RuntimeError("causal evaluation worker exited without a result") from exc
        if "error" in result:
            raise RuntimeError(result["error"])
        return result["metrics"]
    finally:
        receiver.close()
        sender.close()
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join()
        process.close()


def _association_matrix(data: np.ndarray) -> np.ndarray:
    count = data.shape[1]
    result = np.eye(count, dtype=float)
    entropies = [_entropy(data[:, index]) for index in range(count)]
    for left in range(count):
        for right in range(left + 1, count):
            x = data[:, left]
            y = data[:, right]
            nx = int(x.max()) + 1
            ny = int(y.max()) + 1
            joint = np.bincount(x * ny + y, minlength=nx * ny).reshape(nx, ny).astype(float)
            joint /= float(len(x))
            px = joint.sum(axis=1, keepdims=True)
            py = joint.sum(axis=0, keepdims=True)
            mask = joint > 0
            denom = px @ py
            mi = float(np.sum(joint[mask] * np.log(joint[mask] / denom[mask])))
            scale = math.sqrt(entropies[left] * entropies[right])
            result[left, right] = result[right, left] = 0.0 if scale == 0 else mi / scale
    return result


def _entropy(values: np.ndarray) -> float:
    counts = np.bincount(values).astype(float)
    probabilities = counts[counts > 0] / float(len(values))
    return float(-np.sum(probabilities * np.log(probabilities)))


def _build_estimated_graph(prepared: dict[str, Any], spec: dict[str, Any]):
    from causallearn.graph.Edge import Edge
    from causallearn.graph.Endpoint import Endpoint
    from causallearn.graph.GeneralGraph import GeneralGraph

    nodes = prepared["nodes"]
    association = prepared["association"]
    graph = GeneralGraph(nodes)
    degrees = np.zeros(len(nodes), dtype=int)
    edges = sorted(
        (
            (float(association[left, right]), left, right)
            for left in range(len(nodes))
            for right in range(left + 1, len(nodes))
            if association[left, right] >= float(spec["min_association"])
        ),
        reverse=True,
    )
    for _score, left, right in edges:
        if degrees[left] >= int(spec["max_degree"]) or degrees[right] >= int(spec["max_degree"]):
            continue
        added = graph.add_edge(
            Edge(nodes[left], nodes[right], Endpoint.TAIL, Endpoint.TAIL)
        )
        if added is False:
            raise ValueError(f"could not add undirected edge {left}-{right}")
        degrees[left] += 1
        degrees[right] += 1
    return graph


def _score_graph(true_cpdag, estimate) -> dict[str, float]:
    from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
    from causallearn.graph.ArrowConfusion import ArrowConfusion
    from causallearn.graph.SHD import SHD

    adjacency = AdjacencyConfusion(true_cpdag, estimate)
    arrows = ArrowConfusion(true_cpdag, estimate)
    return {
        "shd": float(SHD(true_cpdag, estimate).get_shd()),
        "adj_precision": _safe_div(adjacency.get_adj_tp(), adjacency.get_adj_tp() + adjacency.get_adj_fp()),
        "adj_recall": _safe_div(adjacency.get_adj_tp(), adjacency.get_adj_tp() + adjacency.get_adj_fn()),
        "arrow_precision": _safe_div(arrows.get_arrows_tp(), arrows.get_arrows_tp() + arrows.get_arrows_fp()),
        "arrow_recall": _safe_div(arrows.get_arrows_tp(), arrows.get_arrows_tp() + arrows.get_arrows_fn()),
    }


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def validate_upstream_contract(upstream_root: Path) -> None:
    upstream_root = Path(upstream_root).resolve()
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=upstream_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15,
        check=False,
    )
    if completed.returncode or completed.stdout.strip() != OFFICIAL_COMMIT:
        raise ValueError(f"MLS-Bench checkout must be pinned at {OFFICIAL_COMMIT}")
    for relative, expected in UPSTREAM_SHA256.items():
        path = upstream_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"pinned upstream file is missing: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"pinned upstream file digest mismatch: {relative}")


def score_with_pinned_mlsbench(
    upstream_root: Path,
    metrics: dict[str, float],
    *,
    evaluator_python: str | Path = sys.executable,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    meta = Path(upstream_root) / HARBOR_TASK_PATH / "tests" / "meta"
    source = Path(upstream_root) / HARBOR_TASK_PATH / "tests" / "mlsbench_src"
    scorer = """
import json
from pathlib import Path
import sys
from mlsbench.scoring.anchors import BaselineAnchors
from mlsbench.scoring.evaluate import load_expanded_spec, score_record_details
task_dir = Path(sys.argv[1])
record = json.load(sys.stdin)
anchors = BaselineAnchors(task_dir)
spec = load_expanded_spec(task_dir, anchors)
score, settings, valid = score_record_details(spec, record, anchors)
print(json.dumps({"official_score": score, "record_valid": valid,
                  "setting_scores": {item.name: item.score for item in settings}}, sort_keys=True))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    completed = subprocess.run(
        [str(evaluator_python), "-c", scorer, str(meta)],
        cwd=upstream_root,
        env=env,
        input=json.dumps(metrics),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError("pinned MLS-Bench scorer failed: " + completed.stderr[-2000:])
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    score = float(payload.get("official_score", float("nan")))
    if not payload.get("record_valid") or not math.isfinite(score):
        raise ValueError("pinned MLS-Bench scorer rejected the evaluation record")
    return {
        "engine": "pinned_mlsbench.scoring.evaluate.score_record_details",
        "official_score": score,
        "record_valid": True,
        "setting_scores": payload["setting_scores"],
    }
