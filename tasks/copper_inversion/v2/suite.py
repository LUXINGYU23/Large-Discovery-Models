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


OBJECTIVES=("confirmation","oracle_iou")
AGGREGATIONS={"confirmation":"parent-macro-confirmation-utility-v1","oracle_iou":"parent-macro-oracle-iou-v1"}
TIER_FIELDS={"solver_seconds","train_manifest","train_bodies","validation_manifest","sealed_manifests","slots"}


def aggregate(rows):
    """One vote per geology parent, including every planned failed solve."""
    if not rows:
        raise ValueError("empty evaluation suite")
    # Oracle receipts name their objective; legacy confirmation rows do not.
    objectives={row.get("objective","confirmation") for row in rows}
    if len(objectives)!=1 or not objectives<=set(OBJECTIVES):
        raise ValueError("mixed or unknown evaluation objective")
    objective=objectives.pop()
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
    # penalty scale. Each parent has equal weight. IoU is diagnostic only,
    # except for the deliberately unblinded oracle judge (risk = 1 - IoU).
    if objective=="oracle_iou":measurement_utility=mean
    else:measurement_utility=float(np.mean([np.mean([utility(r) for r in members]) for members in grouped.values()]))
    return dict(mean_iou=mean,loss_total=1-measurement_utility,research_utility=measurement_utility,
        legal_rate=float(np.mean([r["status"]=="succeeded" for r in rows])),
        parent_count=len(grouped),solve_count=len(rows),
        failed_solves=sum(r["status"]!="succeeded" for r in rows),
        mean_solver_wall_seconds=float(np.mean([r["solver_wall_seconds"] for r in rows])))


def release_record(path,stage):
    """Receipt of the release holding <release>/<suite>/manifest.json, if any."""
    release=Path(path).resolve().parent.parent
    names=("qualification.json",) if stage=="formal" else ("development-receipt.json","qualification.json")
    for name in names:
        if (release/name).is_file():return release,json.loads((release/name).read_text())
    return release,None


def release_tiers(path,stage):
    """The sealed tier table (protocol.json tiers shape), or None for old releases."""
    release,record=release_record(path,stage)
    tiers=None if record is None else record.get("tiers")
    if tiers is None:return release,None
    if not isinstance(tiers,dict) or not tiers:raise ValueError("invalid release tier table")
    for row in tiers.values():
        if not isinstance(row,dict) or set(row)!=TIER_FIELDS or not isinstance(row["sealed_manifests"],list):
            raise ValueError("invalid release tier table")
        if any(type(row[k]) is not int or row[k]<1 for k in ("solver_seconds","train_bodies","slots")):
            raise ValueError("invalid release tier budget")
        for name in [row["train_manifest"],row["validation_manifest"],*row["sealed_manifests"]]:
            if not isinstance(name,str) or not (release/name).resolve().is_relative_to(release):
                raise ValueError("tier manifest escapes release")
    return release,tiers


def release_packs(path,stage):
    """Observation packs per body fixed by the release layout, or None."""
    _,record=release_record(path,stage)
    layout=None if record is None else record.get("layout")
    if layout is None:return None
    value=layout.get("packs_per_body") if isinstance(layout,dict) else None
    if type(value) is not int or value not in (1,2):raise ValueError("invalid release observation layout")
    return value


def public_projection(manifest):
    """Bodies, public data and truth; judge variants differ only in confirmation."""
    return dict(stage=manifest["stage"],split=manifest["split"],seeds=manifest["seeds"],
        cases=sorted((c["case_id"],c["parent_id"],c["modality"],c["sha256"]["problem.json"],
                      c["sha256"]["problem.npz"],c["sha256"]["truth.npy"]) for c in manifest["cases"]))


def manifest_tiers(path,manifest,release,tiers,roles=("train","validation","sealed")):
    """Tiers listing this manifest, or a judge variant of a listed one, in a role."""
    path=Path(path).resolve();own=public_projection(manifest);names=[]
    for name,row in sorted(tiers.items()):
        listed=([row["train_manifest"]] if "train" in roles else [])+(
            [row["validation_manifest"]] if "validation" in roles else [])+(
            list(row["sealed_manifests"]) if "sealed" in roles else [])
        for relative in listed:
            other=(release/relative).resolve()
            if other==path or (other.is_file() and public_projection(json.loads(other.read_text()))==own):
                names.append(name);break
    return names


def load_manifest(path,packs_per_body=None):
    """packs_per_body overrides the release layout, for release tools writing it."""
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

    packs=release_packs(path,manifest["stage"]) if packs_per_body is None else packs_per_body
    if packs==1:
        if any(values not in (["drill"],["drill_outcrop"]) for values in parents.values()):
            raise ValueError("each parent requires exactly one observation pack")
    elif packs==2 or manifest["stage"]=="formal":
        if any(sorted(values)!=["drill","drill_outcrop"] for values in parents.values()):
            raise ValueError("each parent requires exactly two observation packs")
    elif packs is not None:
        raise ValueError("invalid observation layout")
    # Development data without a recorded layout keep the rsi-dev tolerance.
    elif any(sorted(values) not in (["drill"], ["drill_outcrop"], ["drill","drill_outcrop"])
             for values in parents.values()):
        raise ValueError("each parent requires one or both distinct observation packs")
    return manifest


class SuiteEvaluator:
    def __init__(self,root,execution,manifest_path,seconds=120,slots=None,objective="confirmation"):
        self.root=Path(root);self.execution=execution
        self.slots=slots
        self.manifest_path=Path(manifest_path).resolve()
        self.manifest=load_manifest(self.manifest_path)
        if any('confirmation/problem.npz' not in case['sha256'] for case in self.manifest['cases']):
            raise ValueError('independent confirmation measurements required for evaluation')
        if objective not in OBJECTIVES:raise ValueError("unknown evaluation objective")
        self.seconds=seconds;self.objective=objective
        release,tiers=release_tiers(self.manifest_path,self.manifest["stage"])
        if tiers is not None:
            allowed={tiers[name]["solver_seconds"] for name in manifest_tiers(self.manifest_path,self.manifest,release,tiers)}
            if seconds not in allowed:raise ValueError("solver budget differs from the release tier table")
        elif self.manifest["stage"]=="formal" and seconds!=120:
            raise ValueError("formal solver budget is fixed at 120 seconds")
        if seconds<=0:raise ValueError("invalid solver budget")
        self.context=dict(manifest=digest(self.manifest),implementation=source_identity(),case_workers=4,
            image=execution.image,seconds=seconds,cpus=2,memory_mb=4096,aggregation=AGGREGATIONS[objective])

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
                if self.objective!="confirmation":row["objective"]=self.objective
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
