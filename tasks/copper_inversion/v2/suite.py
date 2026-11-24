"""Trusted whole-suite adapter for the existing LDM evaluator protocol.

The manifest is controller-owned. Research agents receive only public problems
and aggregate training feedback, never this manifest or its truth directories.
"""
import hashlib
import json
import math
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from ldm_tts.contracts import EvaluationResult
from ldm_tts.contracts.evaluation import EVALUATION_ATTEMPT_RECEIPT_KEY, EvaluationPaused
from ldm_tts.engine.run_store import atomic_json_write
from ..core.execution import digest, source_identity, freeze_model
from ..core.geometry import read_model
from ..rsi.case_isolation import case_orientation, opaque_case_id
from .contract import LEGACY_OBJECTIVE, OBJECTIVE, OBJECTIVE_VERSIONS
from .contract import validate as validate_problem
from .data import canonical_problem, invert_orientation, read_problem, save_problem


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


OBJECTIVES=("confirmation","oracle_iou")
AGGREGATIONS={"confirmation":"parent-macro-line-jaccard-v1","oracle_iou":"parent-macro-oracle-iou-v1"}
LEGACY_CONFIRMATION_AGGREGATION="parent-macro-confirmation-utility-v1"
TIER_FIELDS={"solver_seconds","train_manifest","train_bodies","validation_manifest","sealed_manifests","slots"}
_PACK=re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,31}")


def confirmation_resample_seed(evaluation_key,case_id):
    """Deterministic 32-bit seed per (evaluation, case) for resampled confirmation thinning."""
    if not isinstance(evaluation_key,str) or not evaluation_key or not isinstance(case_id,str) or not case_id:
        raise ValueError("non-empty evaluation_key and case_id required")
    return int.from_bytes(hashlib.sha256(f"rsi-resample-v1:{evaluation_key}:{case_id}".encode("utf-8")).digest()[:4],"big")


def validate_resample(spec):
    if spec is None:return None
    if not isinstance(spec,dict) or not {"fraction"}<=set(spec)<={"fraction","outcrop_grid"}:
        raise ValueError("invalid confirmation resample specification")
    fraction=spec["fraction"]
    if type(fraction) not in (int,float) or not math.isfinite(fraction) or not 0<fraction<1:
        raise ValueError("confirmation resample fraction must lie in (0, 1)")
    out={"fraction":float(fraction)}
    if "outcrop_grid" in spec:
        grid=spec["outcrop_grid"]
        if type(grid) is not int or grid<1:
            raise ValueError("confirmation resample outcrop_grid must be a positive integer")
        out["outcrop_grid"]=grid
    return out


def validate_confirmation_packs(packs):
    if packs is None:return None
    if packs=="all":return "all"
    if (not isinstance(packs,(list,tuple)) or not packs or len(set(packs))!=len(packs)
            or any(not isinstance(p,str) or _PACK.fullmatch(p) is None for p in packs)):
        raise ValueError("confirmation_packs must be 'all' or a list of distinct pack names")
    return list(packs)


def merge_confirmations(problems):
    """Union of multiple confirmation packs for the same parent, with non-colliding group ids."""
    if not problems:raise ValueError("at least one confirmation pack is required")
    if len(problems)==1:return problems[0]
    first=problems[0]
    for other in problems[1:]:
        for key in ("schema_version","geometry_version","objective_version","shape","origin","spacing",
                    "open_boundary_faces","priors"):
            if other[key]!=first[key]:raise ValueError(f"confirmation packs differ on {key}")
        for key in ("domain_mask","forbidden_mask"):
            if not np.array_equal(other[key],first[key]):raise ValueError(f"confirmation packs differ on {key}")
        if [o["name"] for o in other["observations"]]!=[o["name"] for o in first["observations"]]:
            raise ValueError("confirmation packs have different observation modalities")
    observations=[]
    for index,base_obs in enumerate(first["observations"]):
        pts,lbls,grps,offset=[],[],[],0
        for pack in problems:
            obs=pack["observations"][index]
            for key in ("weight","transition_scale","contact_sigma","label_error"):
                if obs[key]!=base_obs[key]:raise ValueError(f"confirmation packs differ on observation {key}")
            pts.append(np.asarray(obs["points"],dtype=float))
            lbls.append(np.asarray(obs["labels"],dtype=float))
            g=np.asarray(obs["groups"],dtype=int)
            grps.append(g+offset)
            if len(g):offset+=int(g.max())+1
        observations.append({**base_obs,"points":np.concatenate(pts,axis=0),
                             "labels":np.concatenate(lbls,axis=0),"groups":np.concatenate(grps,axis=0)})
    merged={**first,"observations":observations}
    validate_problem(merged)
    return merged


def _load_confirmation(truth_root,confirmation_packs=None):
    base=Path(truth_root)/"confirmation"
    if confirmation_packs is None:
        if (base/"problem.json").is_file():return read_problem(base)
        if (base/"c0"/"problem.json").is_file():return read_problem(base/"c0")
        raise ValueError("confirmation measurements missing")
    if confirmation_packs=="all":
        dirs=sorted(p for p in base.iterdir() if p.is_dir() and (p/"problem.json").is_file()) if base.is_dir() else []
        if not dirs and (base/"problem.json").is_file():dirs=[base]
        if not dirs:raise ValueError("no confirmation packs found")
        return merge_confirmations([read_problem(p) for p in dirs])
    loaded=[]
    for name in confirmation_packs:
        folder=base/name
        if (folder/"problem.json").is_file():loaded.append(read_problem(folder))
        elif name=="c0" and (base/"problem.json").is_file():loaded.append(read_problem(base))
        else:raise ValueError(f"confirmation pack {name!r} is missing")
    return merge_confirmations(loaded)


def _chmod_readable(root):
    root=Path(root)
    for path in [root,*root.rglob("*")]:
        if not path.is_symlink():
            path.chmod(0o755 if path.is_dir() else 0o644)


def row_objective_version(row,default=LEGACY_OBJECTIVE):
    metrics=row.get("metrics") if isinstance(row.get("metrics"),dict) else {}
    return row.get("objective_version") or metrics.get("confirmation_objective_version") or metrics.get("objective_version") or default


def confirmation_utility(loss,objective_version=OBJECTIVE):
    if objective_version==OBJECTIVE:return float(np.clip(1.-loss,0.,1.))
    if objective_version==LEGACY_OBJECTIVE:return 1./(1.+loss)
    raise ValueError("unknown evaluation objective version")


def aggregate(rows,*,objective_version=None):
    """One vote per geology parent, including every planned failed solve."""
    if not rows:
        raise ValueError("empty evaluation suite")
    # Oracle receipts name their objective; legacy confirmation rows do not.
    objectives={row.get("objective","confirmation") for row in rows}
    if len(objectives)!=1 or not objectives<=set(OBJECTIVES):
        raise ValueError("mixed or unknown evaluation objective")
    objective=objectives.pop()
    if objective=="confirmation":
        if objective_version is None:
            versions={row_objective_version(row) for row in rows}
            if len(versions)!=1 or not versions<=set(OBJECTIVE_VERSIONS):
                raise ValueError("mixed or unknown evaluation objective version")
            objective_version=versions.pop()
        elif objective_version not in OBJECTIVE_VERSIONS:
            raise ValueError("unknown evaluation objective version")
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
        return confirmation_utility(loss,objective_version)
    # Bound the loss without dropping failed solves or choosing a numerical
    # penalty scale. Each parent has equal weight. IoU is diagnostic only,
    # except for the deliberately unblinded oracle judge (risk = 1 - IoU).
    if objective=="oracle_iou":measurement_utility=mean
    else:measurement_utility=float(np.mean([np.mean([utility(r) for r in members]) for members in grouped.values()]))
    summary=dict(mean_iou=mean,loss_total=1-measurement_utility,research_utility=measurement_utility,
        legal_rate=float(np.mean([r["status"]=="succeeded" for r in rows])),
        parent_count=len(grouped),solve_count=len(rows),
        failed_solves=sum(r["status"]!="succeeded" for r in rows),
        mean_solver_wall_seconds=float(np.mean([r["solver_wall_seconds"] for r in rows])))
    if objective!="oracle_iou" and any(
            r["status"]=="succeeded" and "full_confirmation_loss_total" in r["metrics"] for r in rows):
        def full_utility(row):
            if row["status"]!="succeeded":return 0.
            loss=row["metrics"].get("full_confirmation_loss_total")
            if not isinstance(loss,(int,float)) or not np.isfinite(loss) or loss<0:
                raise ValueError("full confirmation objective is missing or invalid")
            return confirmation_utility(loss,objective_version)
        summary["full_loss_total"]=1.-float(np.mean([np.mean([full_utility(r) for r in members]) for members in grouped.values()]))
    return summary


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
    release,record=release_record(path,stage)
    layout=None if record is None else record.get("layout")
    if layout is None:return None
    if not isinstance(layout,dict):raise ValueError("invalid release observation layout")
    special=layout.get("packs_per_body_by_manifest",{})
    if not isinstance(special,dict):raise ValueError("invalid release observation layout")
    relative=str(Path(path).resolve().relative_to(release)) if Path(path).resolve().is_relative_to(release) else None
    value=special.get(relative,layout.get("packs_per_body"))
    if type(value) is not int or value not in (1,2):raise ValueError("invalid release observation layout")
    return value


Q1_FIELDS={"execution_seeds","execution_seeds_manifest","both_packs_manifest","solver_seconds"}


def q1_seconds(path,manifest):
    """Solver budget of a Q1 measurement suite; Q1 is never a tier or a discovery loop."""
    release,record=release_record(path,manifest["stage"])
    q1=None if record is None else record.get("q1")
    if not isinstance(q1,dict) or not Q1_FIELDS<=set(q1):raise ValueError("release has no Q1 measurement record")
    if type(q1["solver_seconds"]) is not int or q1["solver_seconds"]<1:raise ValueError("invalid Q1 solver budget")
    path=Path(path).resolve()
    if path==(release/q1["execution_seeds_manifest"]).resolve():
        if manifest["seeds"]!=q1["execution_seeds"]:raise ValueError("Q1 execution seeds differ from the release record")
    elif path!=(release/q1["both_packs_manifest"]).resolve():
        raise ValueError("manifest is not a Q1 measurement suite")
    return q1["solver_seconds"]


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
    if manifest["stage"] not in {"development","formal"} or manifest["split"] not in {
            "train","calibration","calibration_ood","validation","test_id","test_ood"}:
        raise ValueError("invalid suite scope")
    seeds=manifest["seeds"]
    if not seeds or len(set(seeds))!=len(seeds) or any(type(s) is not int or s<0 for s in seeds):
        raise ValueError("invalid execution seeds")
    if not manifest["cases"]:raise ValueError("empty suite")
    seen=set();parents={}; parent_geometry={}; objective_versions=set()
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
        conf_keys={k for k in row["sha256"] if k.startswith("confirmation/")}
        if manifest["stage"]=="formal" and not conf_keys:raise ValueError("missing dataset digests")
        if conf_keys:
            if {"confirmation/problem.json","confirmation/problem.npz"} & conf_keys:
                expected|={"confirmation/problem.json","confirmation/problem.npz"}
            packs=sorted({k.split("/")[1] for k in conf_keys if len(k.split("/"))==3})
            for pack in packs:
                if _PACK.fullmatch(pack) is None:raise ValueError("invalid confirmation pack name")
                expected|={f"confirmation/{pack}/problem.json",f"confirmation/{pack}/problem.npz"}
        if set(row["sha256"])!=expected:raise ValueError("missing dataset digests")
        for filename in expected:
            folder=row["truth"] if filename=="truth.npy" or filename.startswith("confirmation/") else row["public"]
            if file_hash(path.parent/folder/filename)!=row["sha256"][filename]:
                raise ValueError("dataset digest mismatch")
        problem=read_problem(path.parent/row["public"])
        if problem["case_id"]!=row["case_id"]:raise ValueError("public request id differs")
        objective_versions.add(problem["objective_version"])
        geometry=digest(dict(truth=row["sha256"]["truth.npy"],shape=problem["shape"],
            origin=problem["origin"],spacing=problem["spacing"],
            domain=hashlib.sha256(problem["domain_mask"].tobytes()).hexdigest(),
            forbidden=hashlib.sha256(problem["forbidden_mask"].tobytes()).hexdigest()))
        previous=parent_geometry.setdefault(row["parent_id"],geometry)
        if previous!=geometry:raise ValueError("one parent has inconsistent geometry across observation packs")
    if len(objective_versions)!=1:
        raise ValueError("mixed objective versions across suite cases")

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
    # Scheduling only: cases solved at once (scheduling policy case_workers), never in the key.
    case_workers=4

    def __init__(self,root,execution,manifest_path,seconds=120,slots=None,objective="confirmation",purpose=None,
                 *,anonymize_cases=False,canonical_origin=False,randomize_orientation=False,
                 resample=None,confirmation_packs=None,objective_version=None):
        if any(type(v) is not bool for v in (anonymize_cases,canonical_origin,randomize_orientation)):
            raise ValueError("invalid case isolation flags")
        self.root=Path(root);self.execution=execution
        self.slots=slots
        self.manifest_path=Path(manifest_path).resolve()
        self.manifest=load_manifest(self.manifest_path)
        if any('confirmation/problem.npz' not in case['sha256'] and 'confirmation/c0/problem.npz' not in case['sha256']
               for case in self.manifest['cases']):
            raise ValueError('independent confirmation measurements required for evaluation')
        if objective not in OBJECTIVES:raise ValueError("unknown evaluation objective")
        self.resample=validate_resample(resample)
        self.confirmation_packs=validate_confirmation_packs(confirmation_packs)
        if (self.resample is not None or self.confirmation_packs is not None) and objective!="confirmation":
            raise ValueError("confirmation resample and packs require confirmation objective")
        manifest_objective=read_problem(self.manifest_path.parent/self.manifest["cases"][0]["public"])["objective_version"]
        if objective_version is None:
            self.objective_version=manifest_objective
        elif objective_version not in OBJECTIVE_VERSIONS:
            raise ValueError("unknown evaluation objective version")
        else:
            self.objective_version=objective_version
        self._explicit_objective_version=objective_version is not None
        self.seconds=seconds;self.objective=objective
        self.anonymize_cases=anonymize_cases;self.canonical_origin=canonical_origin
        self.randomize_orientation=randomize_orientation
        release,tiers=release_tiers(self.manifest_path,self.manifest["stage"])
        if purpose not in (None,"q1"):raise ValueError("unknown evaluation purpose")
        if purpose=="q1":
            # Explicit opt-in: Q1 suites are measured outside any discovery loop.
            if seconds!=q1_seconds(self.manifest_path,self.manifest):
                raise ValueError("solver budget differs from the release Q1 record")
        elif tiers is not None:
            allowed={tiers[name]["solver_seconds"] for name in manifest_tiers(self.manifest_path,self.manifest,release,tiers)}
            if seconds not in allowed:raise ValueError("solver budget differs from the release tier table")
        elif self.manifest["stage"]=="formal" and seconds!=120:
            raise ValueError("formal solver budget is fixed at 120 seconds")
        if seconds<=0:raise ValueError("invalid solver budget")
        aggregation=(LEGACY_CONFIRMATION_AGGREGATION
                     if objective=="confirmation" and self.objective_version==LEGACY_OBJECTIVE
                     else AGGREGATIONS[objective])
        self.context=dict(manifest=digest(self.manifest),implementation=source_identity(),
            image=execution.image,seconds=seconds,cpus=2,memory_mb=4096,aggregation=aggregation)
        if self._explicit_objective_version or self.objective_version!=LEGACY_OBJECTIVE:
            self.context["objective_version"]=self.objective_version
        if purpose is not None:self.context["purpose"]=purpose
        if anonymize_cases:self.context["anonymize_cases"]=True
        if canonical_origin:self.context["canonical_origin"]=True
        if randomize_orientation:self.context["randomize_orientation"]=True
        if self.resample is not None:self.context["resample"]=dict(self.resample)
        if self.confirmation_packs is not None:
            self.context["confirmation_packs"]=list(self.confirmation_packs) if isinstance(self.confirmation_packs,list) else self.confirmation_packs

    def evaluation_attempt_usage_key(self,candidate):
        return digest(dict(context=self.context,source=candidate.canonical_key))

    def _prepare_score_truth(self,key,case,case_root):
        raw_truth=self.manifest_path.parent/case["truth"]
        root_conf=raw_truth/"confirmation/problem.json"
        restamp=root_conf.is_file() and json.loads(root_conf.read_text())["objective_version"]!=self.objective_version
        if (self.resample is not None or self.confirmation_packs is not None or restamp
                or not root_conf.is_file()):
            score_truth=case_root/"score-truth"
            score_truth.mkdir(parents=True,exist_ok=True)
            truth_dst=score_truth/"truth.npy"
            if not truth_dst.exists() and (raw_truth/"truth.npy").exists():
                try:os.link(raw_truth/"truth.npy",truth_dst)
                except OSError:shutil.copy2(raw_truth/"truth.npy",truth_dst)
            base_conf=_load_confirmation(raw_truth,self.confirmation_packs)
            if base_conf["objective_version"]!=self.objective_version:
                base_conf={**base_conf,"objective_version":self.objective_version}
            if self.resample is not None:
                from ..experiments.rsi.measurements import thin_confirmation
                rseed=confirmation_resample_seed(key,case["case_id"])
                thinned,_=thin_confirmation(base_conf,rseed,self.resample["fraction"],
                                            self.resample.get("outcrop_grid",4))
                save_problem(thinned,score_truth/"confirmation")
                save_problem(base_conf,score_truth/"full_confirmation")
            else:
                save_problem(base_conf,score_truth/"confirmation")
            _chmod_readable(score_truth)
            return score_truth
        return raw_truth

    def evaluate_precomputed(self,candidate_id,models_by_case_seed):
        """Score frozen precomputed per-(case, seed) models with the independent scorer."""
        if load_manifest(self.manifest_path)!=self.manifest or source_identity()!=self.context["implementation"]:
            raise EvaluationPaused("suite scientific identity changed")
        if self.anonymize_cases or self.canonical_origin or self.randomize_orientation:
            raise ValueError("precomputed field evaluation uses raw public case coordinates")
        key=digest(dict(context=self.context,precomputed=candidate_id))
        root=self.root/"evaluations"/key;root.mkdir(parents=True,exist_ok=True)
        result_path=root/"result.json"
        if result_path.exists():return EvaluationResult(**json.loads(result_path.read_text()))
        merged=getattr(self.execution,"merged",False) is True
        def prepare_one(item):
            case,seed=item
            identity=dict(evaluation=key,case=case["case_id"],seed=seed)
            case_root=root/case["case_id"]/str(seed);case_root.mkdir(parents=True,exist_ok=True)
            row_path=case_root/"receipt.json"
            if row_path.exists():return json.loads(row_path.read_text())
            entry=models_by_case_seed[case["case_id"]][str(seed)]
            model_src=Path(entry["path"])
            if model_src.is_symlink() or file_hash(model_src)!=entry["sha256"]:
                raise EvaluationPaused("frozen model changed")
            raw_problem=read_problem(self.manifest_path.parent/case["public"])
            score_inputs=case_root/"input"
            save_problem(raw_problem,score_inputs)
            wall_s=float(entry.get("wall_seconds",0.))
            row=dict(parent_id=case["parent_id"],case_id=case["case_id"],seed=seed,
                status="failed",metrics={"iou":0.},solver_wall_seconds=wall_s,jobs=1,
                solver_resources=None,error="invalid precomputed model")
            if self.objective!="confirmation":row["objective"]=self.objective
            try:
                sha=freeze_model(model_src,case_root/"model",raw_problem["shape"])
            except OSError as exc:
                raise EvaluationPaused("model freeze infrastructure I/O failed") from exc
            except ValueError as exc:
                row.update(status="invalid",error=str(exc))
                atomic_json_write(row_path,row)
                return row
            score_truth=self._prepare_score_truth(key,case,case_root)
            scoring=dict(identity=identity,model=sha,inputs=score_inputs,model_dir=case_root/"model",
                truth=score_truth,row_path=row_path)
            if merged:return row,scoring
            score_root,s=self.execution.run(dict(identity=identity,model=sha),phase="scorer",inputs=score_inputs,
                model=case_root/"model",truth=score_truth,timeout=30,science_module="v2")
            if s["exit_code"] or s["reason"] or s["oom"]:
                raise EvaluationPaused("independent scorer infrastructure failure")
            scored(row,json.loads((score_root/"output/result.json").read_text()))
            atomic_json_write(row_path,row)
            return row
        def scored(row,outcome):
            if outcome.get("status") not in {"invalid","succeeded"}:
                raise EvaluationPaused("invalid independent scorer receipt")
            value=outcome.get("metrics",{}).get("iou")
            if not isinstance(value,(int,float)) or not np.isfinite(value) or not 0<=value<=1:
                raise EvaluationPaused("invalid hidden geometry metric")
            row.update(outcome);row["jobs"]=1
            if row["status"]=="succeeded":row["error"]=""
        plan=[(case,seed) for case in self.manifest["cases"] for seed in self.manifest["seeds"]]
        with ThreadPoolExecutor(max_workers=self.case_workers) as pool:
            rows=list(pool.map(prepare_one,plan))
        pending=[(i,item) for i,item in enumerate(rows) if isinstance(item,tuple)]
        if pending:
            jobs=[job for _,(_,job) in pending]
            score_root,s=self.execution.run(dict(evaluation=key,scoring=[dict(identity=j["identity"],model=j["model"]) for j in jobs]),
                phase="scorer",cases=[dict(inputs=j["inputs"],model=j["model_dir"],truth=j["truth"]) for j in jobs],
                timeout=30+5*len(jobs),science_module="v2")
            if s["exit_code"] or s["reason"] or s["oom"]:
                raise EvaluationPaused("independent scorer infrastructure failure")
            outcomes=json.loads((score_root/"output/results.json").read_text())
            if not isinstance(outcomes,list) or len(outcomes)!=len(jobs):
                raise EvaluationPaused("invalid independent scorer receipt")
            for (index,(row,job)),outcome in zip(pending,outcomes):
                if not isinstance(outcome,dict):raise EvaluationPaused("invalid independent scorer receipt")
                scored(row,outcome)
                atomic_json_write(job["row_path"],row)
                rows[index]=row
        jobs=sum(row["jobs"] for row in rows)
        wall=sum(row["solver_wall_seconds"] for row in rows)
        metrics=aggregate(rows)
        result=EvaluationResult(candidate_id,"succeeded",metrics=metrics,
            resource_usage=dict(benchmark_jobs=jobs,solver_wall_seconds=wall,suite_evaluations=1),
            artifacts={},metadata={EVALUATION_ATTEMPT_RECEIPT_KEY:key,"suite_digest":self.context["manifest"]})
        atomic_json_write(root/"private-case-results.json",rows)
        atomic_json_write(result_path,result.to_dict())
        return result

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
        merged=getattr(self.execution,"merged",False) is True
        def solve_one(item):
            # Returns a finished row, or (row, scoring job) when the model needs scoring.
            case,seed=item
            identity=dict(evaluation=key,case=case["case_id"],seed=seed)
            case_root=root/case["case_id"]/str(seed);case_root.mkdir(parents=True,exist_ok=True)
            row_path=case_root/"receipt.json"
            if row_path.exists():return json.loads(row_path.read_text())
            inputs=case_root/"input"
            raw_problem=read_problem(self.manifest_path.parent/case["public"])
            scored_problem=(raw_problem if raw_problem["objective_version"]==self.objective_version
                            else {**raw_problem,"objective_version":self.objective_version})
            orientation=case_orientation(key,case["case_id"]) if self.randomize_orientation else 0
            if self.anonymize_cases or self.canonical_origin or orientation:
                cid=opaque_case_id(key,case["case_id"]) if self.anonymize_cases else None
                problem=canonical_problem(scored_problem,case_id=cid,canonical_origin=self.canonical_origin,
                                          orientation=orientation)
                save_problem(problem,inputs)
                score_inputs=case_root/"score-input"
                save_problem(scored_problem,score_inputs)
            elif scored_problem is not raw_problem:
                problem=scored_problem
                save_problem(problem,inputs)
                score_inputs=inputs
            else:
                problem=raw_problem
                save_problem(problem,inputs)
                score_inputs=inputs
            atomic_json_write(inputs/"budget.json",dict(seed=seed,wall_time_seconds=self.seconds,cpu_count=2,memory_mb=4096))
            job,r=self.execution.run(identity,phase="solver",inputs=inputs,source=source.parent,
                timeout=self.seconds,cpus=2,memory_mb=4096,science_module="v2")
            row=dict(parent_id=case["parent_id"],case_id=case["case_id"],seed=seed,
                status="failed",metrics={"iou":0.},solver_wall_seconds=r["wall_seconds"],jobs=1,
                solver_resources=r.get("workload_resources"),error="solver failed")
            if self.objective!="confirmation":row["objective"]=self.objective
            if self._explicit_objective_version or self.objective_version!=LEGACY_OBJECTIVE:
                row["objective_version"]=self.objective_version
            if r["reason"]=="timeout":row["status"]="timed_out"
            if not (r["exit_code"] or r["reason"] or r["oom"]):
                try:
                    try:
                        field=read_model(job/"output/model.npy",problem["shape"])
                    except (FileNotFoundError,PermissionError) as exc:
                        raise ValueError("candidate did not produce a readable model") from exc
                    if orientation:
                        restored_path=case_root/"restored-model.npy"
                        np.save(restored_path,invert_orientation(field,orientation))
                        try:
                            sha=freeze_model(restored_path,case_root/"model",raw_problem["shape"])
                        finally:
                            restored_path.unlink(missing_ok=True)
                    else:
                        sha=freeze_model(job/"output/model.npy",case_root/"model",problem["shape"])
                except OSError as exc:
                    raise EvaluationPaused("model freeze infrastructure I/O failed") from exc
                except ValueError as exc:
                    row.update(status="invalid",error=str(exc))
                else:
                    score_truth=self._prepare_score_truth(key,case,case_root)
                    scoring=dict(identity=identity,model=sha,inputs=score_inputs,model_dir=case_root/"model",
                        truth=score_truth,row_path=row_path)
                    if merged:return row,scoring
                    score_root,s=self.execution.run(dict(identity=identity,model=sha),phase="scorer",inputs=score_inputs,
                        model=case_root/"model",truth=scoring["truth"],timeout=30,science_module="v2")
                    if s["exit_code"] or s["reason"] or s["oom"]:
                        raise EvaluationPaused("independent scorer infrastructure failure")
                    scored(row,json.loads((score_root/"output/result.json").read_text()))
            atomic_json_write(row_path,row)
            return row
        def scored(row,outcome):
            if outcome.get("status") not in {"invalid","succeeded"}:
                raise EvaluationPaused("invalid independent scorer receipt")
            value=outcome.get("metrics",{}).get("iou")
            if not isinstance(value,(int,float)) or not np.isfinite(value) or not 0<=value<=1:
                raise EvaluationPaused("invalid hidden geometry metric")
            # jobs counts solve + score per case, also when scoring is merged.
            row.update(outcome);row["jobs"]=2
            if row["status"]=="succeeded":row["error"]=""
        plan=[(case,seed) for case in self.manifest["cases"] for seed in self.manifest["seeds"]]
        # Case concurrency is a scheduling parameter (case_workers), not part of the key.
        with ThreadPoolExecutor(max_workers=self.case_workers) as pool:
            rows=list(pool.map(solve_one,plan))
        pending=[(i,item) for i,item in enumerate(rows) if isinstance(item,tuple)]
        if pending:
            # Every model of this evaluation in one scorer container (parallelism.md §4.1).
            jobs=[job for _,(_,job) in pending]
            score_root,s=self.execution.run(dict(evaluation=key,scoring=[dict(identity=j["identity"],model=j["model"]) for j in jobs]),
                phase="scorer",cases=[dict(inputs=j["inputs"],model=j["model_dir"],truth=j["truth"]) for j in jobs],
                timeout=30+5*len(jobs),science_module="v2")
            if s["exit_code"] or s["reason"] or s["oom"]:
                raise EvaluationPaused("independent scorer infrastructure failure")
            outcomes=json.loads((score_root/"output/results.json").read_text())
            if not isinstance(outcomes,list) or len(outcomes)!=len(jobs):
                raise EvaluationPaused("invalid independent scorer receipt")
            for (index,(row,job)),outcome in zip(pending,outcomes):
                if not isinstance(outcome,dict):raise EvaluationPaused("invalid independent scorer receipt")
                scored(row,outcome)
                atomic_json_write(job["row_path"],row)
                rows[index]=row
        jobs=sum(row["jobs"] for row in rows)
        wall=sum(row["solver_wall_seconds"] for row in rows)
        metrics=aggregate(rows,objective_version=self.objective_version if self.objective=="confirmation" else None)
        # Individual solver failures are scored zero within a successfully
        # completed suite. They must not disappear from BO's observed dataset.
        meta={EVALUATION_ATTEMPT_RECEIPT_KEY:key,"suite_digest":self.context["manifest"]}
        if self._explicit_objective_version or self.objective_version!=LEGACY_OBJECTIVE:
            meta["objective_version"]=self.objective_version
        result=EvaluationResult(candidate.candidate_id,"succeeded",metrics=metrics,
            resource_usage=dict(benchmark_jobs=jobs,solver_wall_seconds=wall,suite_evaluations=1),
            artifacts={},metadata=meta)
        atomic_json_write(root/"private-case-results.json",rows)
        atomic_json_write(result_path,result.to_dict())
        return result
