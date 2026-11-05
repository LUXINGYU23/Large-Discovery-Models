"""Trusted whole-suite adapter for the existing LDM evaluator protocol.

The manifest is controller-owned. Research agents receive only public problems
and aggregate training feedback, never this manifest or its truth directories.
"""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from ldm_tts.contracts import EvaluationResult
from ldm_tts.contracts.evaluation import EVALUATION_ATTEMPT_RECEIPT_KEY, EvaluationPaused
from ldm_tts.engine.run_store import atomic_json_write
from ..core.execution import digest, source_identity, freeze_model
from ..core.geometry import read_model
from .data import read_problem, save_problem


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate(rows):
    """One vote per geology parent, including every planned failed solve."""
    if not rows:
        raise ValueError("empty evaluation suite")
    grouped={}
    for row in rows:
        grouped.setdefault(row["parent_id"],[]).append(row)
    parent_ious=[float(np.mean([r["metrics"]["iou"] if r["status"]=="succeeded" else 0.
                              for r in members])) for members in grouped.values()]
    mean=float(np.mean(parent_ious))
    def utility(row):
        if row['status']!='succeeded':return 0.
        loss=row['metrics'].get('confirmation_loss_total')
        if not isinstance(loss,(int,float)) or not np.isfinite(loss) or loss<0:
            raise ValueError('independent confirmation objective is missing or invalid')
        return 1./(1.+loss)
    # Bound the loss without dropping failed solves or choosing a numerical
    # penalty scale. Each parent has equal weight. IoU is diagnostic only.
    measurement_utility=float(np.mean([np.mean([utility(r) for r in members]) for members in grouped.values()]))
    return dict(mean_iou=mean,loss_total=1-measurement_utility,research_utility=measurement_utility,
        legal_rate=float(np.mean([r["status"]=="succeeded" for r in rows])),
        parent_count=len(grouped),solve_count=len(rows),
        failed_solves=sum(r["status"]!="succeeded" for r in rows),
        mean_solver_wall_seconds=float(np.mean([r["solver_wall_seconds"] for r in rows])))


def load_manifest(path):
    path=Path(path).resolve()
    manifest=json.loads(path.read_text())
    if set(manifest)!={"schema","stage","split","seeds","cases"} or manifest["schema"]!="copper-suite-v2":
        raise ValueError("invalid suite manifest schema")
    if manifest["stage"] not in {"development","formal"} or manifest["split"] not in {"train","validation","test_id","test_ood"}:
        raise ValueError("invalid suite scope")
    seeds=manifest["seeds"]
    if not seeds or len(set(seeds))!=len(seeds) or any(type(s) is not int or s<0 for s in seeds):
        raise ValueError("invalid execution seeds")
    if not manifest["cases"]:raise ValueError("empty suite")
    seen=set();parents={}; parent_geometry={}
    for row in manifest["cases"]:
        if set(row)!={"case_id","parent_id","modality","public","truth","sha256"}:
            raise ValueError("invalid suite member")
        if row["case_id"] in seen:raise ValueError("duplicate case")
        seen.add(row["case_id"])
        parents.setdefault(row["parent_id"],[]).append(row["modality"])
        for name in ("public","truth"):
            location=(path.parent/row[name]).resolve()
            if not location.is_relative_to(path.parent):raise ValueError("suite path escapes dataset root")
        expected={"problem.json","problem.npz","truth.npy"}
        if manifest["stage"]=="formal" or any(k.startswith('confirmation/') for k in row['sha256']):
            expected|={"confirmation/problem.json","confirmation/problem.npz"}
        if set(row["sha256"])!=expected:raise ValueError("missing dataset digests")
        for filename in expected:
            folder=row["truth"] if filename=="truth.npy" or filename.startswith("confirmation/") else row["public"]
            if file_hash(path.parent/folder/filename)!=row["sha256"][filename]:
                raise ValueError("dataset digest mismatch")
        problem=read_problem(path.parent/row["public"])
        if problem["case_id"]!=row["case_id"]:raise ValueError("public request id differs")
        geometry=digest(dict(truth=row["sha256"]["truth.npy"],shape=problem["shape"],
            origin=problem["origin"],spacing=problem["spacing"],
            domain=hashlib.sha256(problem["domain_mask"].tobytes()).hexdigest(),
            forbidden=hashlib.sha256(problem["forbidden_mask"].tobytes()).hexdigest()))
        previous=parent_geometry.setdefault(row["parent_id"],geometry)
        if previous!=geometry:raise ValueError("one parent has inconsistent geometry across observation packs")

    if any(sorted(values)!=["drill","drill_outcrop"] for values in parents.values()):
        raise ValueError("each parent requires exactly two observation packs")
    return manifest


class SuiteEvaluator:
    def __init__(self,root,execution,manifest_path,seconds=120,slots=None):
        self.root=Path(root);self.execution=execution
        self.slots=slots
        self.manifest_path=Path(manifest_path).resolve()
        self.manifest=load_manifest(self.manifest_path)
        if any('confirmation/problem.npz' not in case['sha256'] for case in self.manifest['cases']):
            raise ValueError('independent confirmation measurements required for evaluation')
        self.seconds=seconds
        if self.manifest["stage"]=="formal" and seconds!=120:
            raise ValueError("formal solver budget is fixed at 120 seconds")
        if seconds<=0:raise ValueError("invalid solver budget")
        self.context=dict(manifest=digest(self.manifest),implementation=source_identity(),case_workers=4,
            image=execution.image,seconds=seconds,cpus=2,memory_mb=4096,aggregation="parent-macro-confirmation-utility-v1")

    def evaluation_attempt_usage_key(self,candidate):
        return digest(dict(context=self.context,source=candidate.canonical_key))

    def evaluate(self,candidate):
        # Verify even on resume; a path being unchanged does not imply data are.
        if load_manifest(self.manifest_path)!=self.manifest or source_identity()!=self.context["implementation"]:
            raise EvaluationPaused("suite scientific identity changed")
        source=self.root/candidate.payload["source_path"]
        if file_hash(source)!=candidate.canonical_key:raise EvaluationPaused("source changed")
        if self.slots is not None:self.slots.reserve(candidate.canonical_key)
        key=self.evaluation_attempt_usage_key(candidate)
        root=self.root/"evaluations"/key;root.mkdir(parents=True,exist_ok=True)
        result_path=root/"result.json"
        if result_path.exists():return EvaluationResult(**json.loads(result_path.read_text()))
        rows=[];jobs=0;wall=0.
        def solve_one(item):
            case,seed=item
            identity=dict(evaluation=key,case=case["case_id"],seed=seed)
            case_root=root/case["case_id"]/str(seed);case_root.mkdir(parents=True,exist_ok=True)
            row_path=case_root/"receipt.json"
            if row_path.exists():
                row=json.loads(row_path.read_text())
            else:
                inputs=case_root/"input"
                problem=read_problem(self.manifest_path.parent/case["public"])
                save_problem(problem,inputs)
                atomic_json_write(inputs/"budget.json",dict(seed=seed,wall_time_seconds=self.seconds,cpu_count=2,memory_mb=4096))
                job,r=self.execution.run(identity,phase="solver",inputs=inputs,source=source.parent,
                    timeout=self.seconds,cpus=2,memory_mb=4096,science_module="v2")
                row=dict(parent_id=case["parent_id"],case_id=case["case_id"],seed=seed,
                    status="failed",metrics={"iou":0.},solver_wall_seconds=r["wall_seconds"],jobs=1,
                    solver_resources=r.get("workload_resources"),error="solver failed")
                if r["reason"]=="timeout":row["status"]="timed_out"
                if not (r["exit_code"] or r["reason"] or r["oom"]):
                    try:
                        try:
                            read_model(job/"output/model.npy",problem["shape"])
                        except (FileNotFoundError,PermissionError) as exc:
                            raise ValueError("candidate did not produce a readable model") from exc
                        sha=freeze_model(job/"output/model.npy",case_root/"model",problem["shape"])
                    except OSError as exc:
                        raise EvaluationPaused("model freeze infrastructure I/O failed") from exc
                    except ValueError as exc:
                        row.update(status="invalid",error=str(exc))
                    else:
                        score_root,s=self.execution.run(dict(identity=identity,model=sha),phase="scorer",inputs=inputs,
                            model=case_root/"model",truth=self.manifest_path.parent/case["truth"],timeout=30,science_module="v2")
                        if s["exit_code"] or s["reason"] or s["oom"]:
                            raise EvaluationPaused("independent scorer infrastructure failure")
                        outcome=json.loads((score_root/"output/result.json").read_text())
                        if outcome.get("status") not in {"invalid","succeeded"}:
                            raise EvaluationPaused("invalid independent scorer receipt")
                        value=outcome.get("metrics",{}).get("iou")
                        if not isinstance(value,(int,float)) or not np.isfinite(value) or not 0<=value<=1:
                            raise EvaluationPaused("invalid hidden geometry metric")
                        row.update(outcome);row["jobs"]=2
                        if row["status"]=="succeeded":row["error"]=""
                atomic_json_write(row_path,row)
            return row
        plan=[(case,seed) for case in self.manifest["cases"] for seed in self.manifest["seeds"]]
        with ThreadPoolExecutor(max_workers=4) as pool:
            rows=list(pool.map(solve_one,plan))
        jobs=sum(row["jobs"] for row in rows)
        wall=sum(row["solver_wall_seconds"] for row in rows)
        metrics=aggregate(rows)
        # Individual solver failures are scored zero within a successfully
        # completed suite. They must not disappear from BO's observed dataset.
        result=EvaluationResult(candidate.candidate_id,"succeeded",metrics=metrics,
            resource_usage=dict(benchmark_jobs=jobs,solver_wall_seconds=wall,suite_evaluations=1),
            artifacts={},metadata={EVALUATION_ATTEMPT_RECEIPT_KEY:key,"suite_digest":self.context["manifest"]})
        atomic_json_write(root/"private-case-results.json",rows)
        atomic_json_write(result_path,result.to_dict())
        return result
