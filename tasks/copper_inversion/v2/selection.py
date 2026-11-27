"""Trusted frozen validation/test protocol, reusing suite execution and receipts.

Development qualification only. This is not an access-control service: formal
publication still requires the independently owned evaluator and dataset gate.
"""
import fcntl
import json
import math
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
from ldm_tts.contracts import RawProposal,CandidateRejection
from ldm_tts.engine.run_store import atomic_json_write
from ..core.candidate import seal,validate_source
from ..core.execution import Execution,digest,source_identity,freeze_model
from ..core.geometry import read_model,MAX_MODEL_BYTES
from .contract import OBJECTIVE_VERSIONS
from .data import read_problem
from .suite import SuiteEvaluator,load_manifest,file_hash,release_tiers,manifest_tiers,OBJECTIVES
from .domain import CheckedDomain
from .interface_checks import InterfaceChecks
from .training_receipts import eligible


SELECTION_RULES={"confirmation":"minimum (confirmation_loss, -legal_rate, mean_solver_wall_seconds, source_sha256)",
    "oracle_iou":"minimum (1 - validation IoU, -legal_rate, mean_solver_wall_seconds, source_sha256)"}


@contextmanager
def locked(root):
    with (Path(root)/"protocol.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        yield


def selection_seconds(training_manifest,training,manifests,loaded):
    """Solver budgets of tiers listing all four suites; old formal releases fix 120 s."""
    release,tiers=release_tiers(training_manifest,training["stage"])
    if tiers is None:return {120} if training["stage"]=="formal" else None
    names=set(manifest_tiers(training_manifest,training,release,tiers,roles=("train",)))
    names&=set(manifest_tiers(manifests[0],loaded[0],release,tiers,roles=("validation",)))
    for path,manifest in zip(manifests[1:],loaded[1:]):
        names&=set(manifest_tiers(path,manifest,release,tiers,roles=("sealed",)))
    if not names:raise ValueError("suites do not belong to one release tier")
    return {tiers[name]["solver_seconds"] for name in names}


def freeze(root,sources,validation,tests,*,seconds,image,training_run,training_manifest,objective="confirmation",
           max_candidates=3,anonymize_cases=False,canonical_origin=False,randomize_orientation=False,
           objective_version=None,seed_override=None):
    """objective is the run judge's validation scoring; sealed outcomes stay IoU."""
    root=Path(root)
    if any(type(v) is not bool for v in (anonymize_cases,canonical_origin,randomize_orientation)):
        raise ValueError("invalid case isolation flags")
    if seed_override is not None and (type(seed_override) is not int or seed_override<0):
        raise ValueError("invalid seed_override")
    if objective not in OBJECTIVES:raise ValueError("unknown selection objective")
    if objective_version is not None and objective_version not in OBJECTIVE_VERSIONS:
        raise ValueError("unknown selection objective version")
    if not sources or (max_candidates is not None and len(sources)>max_candidates):
        raise ValueError("freeze one to three candidates before validation")
    if len(tests)!=2:raise ValueError("both ID and OOD test suites required")
    manifests=[Path(validation).resolve(),*[Path(p).resolve() for p in tests]]
    loaded=[load_manifest(p) for p in manifests]
    training=load_manifest(training_manifest)
    admissible=eligible(training_run,training_manifest)
    if [m["split"] for m in loaded]!=["validation","test_id","test_ood"]:
        raise ValueError("ordered validation, test-ID and test-OOD manifests required")
    if len({m["stage"] for m in [training,*loaded]})!=1:raise ValueError("mixed evaluation stages")
    if training["stage"]=="formal":
        from .qualification import verify_release
        for path in [training_manifest,*manifests]:verify_release(path)
    allowed=selection_seconds(training_manifest,training,manifests,loaded)
    if allowed is not None and seconds not in allowed:
        raise ValueError("selection solver budget differs from the release tier table")
    parents=[{c["parent_id"] for c in m["cases"]} for m in [training,*loaded]]
    if any(parents[i]&parents[j] for i in range(4) for j in range(i)):
        raise ValueError("held-out suites share geology parents")
    truth_hashes=[{c["sha256"]["truth.npy"] for c in m["cases"]} for m in [training,*loaded]]
    if any(truth_hashes[i]&truth_hashes[j] for i in range(4) for j in range(i)):
        raise ValueError("held-out suites share truth geometry")
    if not math.isfinite(seconds) or seconds<=0:raise ValueError("positive solver budget required")
    if not image.startswith("sha256:") or len(image)!=71:raise ValueError("resolved Docker image digest required")
    bodies=[Path(p).read_bytes() for p in sources]
    source_ids=[validate_source(b) for b in bodies]
    if any(s not in admissible for s in source_ids):raise ValueError("candidate lacks complete training evidence")
    # Exclusive root means there is no append-candidate or reset-test operation.
    root.mkdir(parents=True,exist_ok=False)
    candidates=[seal(root,b) for b in bodies]
    if len({c["sha256"] for c in candidates})!=len(candidates):raise ValueError("duplicate frozen candidates")
    frozen=dict(schema="copper-selection-v2",stage=training["stage"],candidates=candidates,
        training=dict(manifest_path=str(Path(training_manifest).resolve()),manifest_sha256=file_hash(training_manifest),
            run=str(Path(training_run).resolve()),selected_receipts={s:admissible[s] for s in source_ids}),
        manifests=[dict(path=str(p),sha256=file_hash(p),contents_sha256=digest(m)) for p,m in zip(manifests,loaded)],
        implementation_sha256=source_identity(),image=image,seconds=seconds,selection_objective=objective,
        selection_rule=SELECTION_RULES[objective])
    if anonymize_cases:frozen["anonymize_cases"]=True
    if canonical_origin:frozen["canonical_origin"]=True
    if randomize_orientation:frozen["randomize_orientation"]=True
    if objective_version is not None:frozen["objective_version"]=objective_version
    if seed_override is not None:frozen["seed_override"]=seed_override
    atomic_json_write(root/"frozen.json",frozen)
    atomic_json_write(root/"state.json",dict(phase="frozen",frozen_sha256=digest(frozen)))
    return frozen


def precomputed_manifest(fields_dir):
    """Canonical digest of a precomputed offline-reference directory (model.npy + optional meta.json)."""
    base=Path(fields_dir)
    if base.is_symlink() or not base.is_dir():
        raise ValueError("precomputed fields directory missing or symlinked")
    base=base.resolve()
    files={};model_count=0
    for item in sorted(base.rglob("*")):
        if item.is_symlink():
            raise ValueError("precomputed fields directory contains a symlink")
        if item.is_dir():
            continue
        if item.name not in {"model.npy","meta.json"}:
            raise ValueError(f"unexpected file in precomputed fields directory: {item.name!r}")
        if item.name=="model.npy":
            model_count+=1
        files[item.relative_to(base).as_posix()]=file_hash(item)
    if not model_count:
        raise ValueError("precomputed fields directory contains no model.npy files")
    manifest=dict(schema="copper-precomputed-fields-v1",files=files)
    return manifest,digest(manifest)


def _evaluate_precomputed_with_evaluator(evaluator,sha,split_models):
    if hasattr(evaluator,"evaluate_precomputed"):
        return evaluator.evaluate_precomputed(sha,split_models)
    return evaluator.evaluate(SimpleNamespace(canonical_key=sha,candidate_id=sha,
        payload=dict(sha256=sha,kind="precomputed")))


def freeze_precomputed(root,fields_dir,validation,tests,*,seconds,image,training_manifest,objective="confirmation",
                       objective_version=None):
    """Freeze precomputed offline fields (M4/M7/RF) across train, validation and sealed splits for REF."""
    root=Path(root)
    if objective not in OBJECTIVES:raise ValueError("unknown selection objective")
    if objective_version is not None and objective_version not in OBJECTIVE_VERSIONS:
        raise ValueError("unknown selection objective version")
    if len(tests)!=2:raise ValueError("both ID and OOD test suites required")
    training_path=Path(training_manifest).resolve()
    manifests=[Path(validation).resolve(),*[Path(p).resolve() for p in tests]]
    loaded=[load_manifest(p) for p in manifests]
    training=load_manifest(training_path)
    if [m["split"] for m in loaded]!=["validation","test_id","test_ood"]:
        raise ValueError("ordered validation, test-ID and test-OOD manifests required")
    if len({m["stage"] for m in [training,*loaded]})!=1:raise ValueError("mixed evaluation stages")
    if training["stage"]=="formal":
        from .qualification import verify_release
        for path in [training_path,*manifests]:verify_release(path)
    allowed=selection_seconds(training_path,training,manifests,loaded)
    if allowed is not None and seconds not in allowed:
        raise ValueError("selection solver budget differs from the release tier table")
    parents=[{c["parent_id"] for c in m["cases"]} for m in [training,*loaded]]
    if any(parents[i]&parents[j] for i in range(4) for j in range(i)):
        raise ValueError("held-out suites share geology parents")
    truth_hashes=[{c["sha256"]["truth.npy"] for c in m["cases"]} for m in [training,*loaded]]
    if any(truth_hashes[i]&truth_hashes[j] for i in range(4) for j in range(i)):
        raise ValueError("held-out suites share truth geometry")
    if not math.isfinite(seconds) or seconds<=0:raise ValueError("positive solver budget required")
    if not image.startswith("sha256:") or len(image)!=71:raise ValueError("resolved Docker image digest required")

    f_manifest,f_sha=precomputed_manifest(fields_dir)
    base=Path(fields_dir).resolve()
    resolved_sources={}
    for m_path,m in [(training_path,training),*zip(manifests,loaded)]:
        split=m["split"]
        resolved_sources[split]={}
        for case in m["cases"]:
            pub=read_problem(m_path.parent/case["public"])
            resolved_sources[split][case["case_id"]]={}
            for seed in m["seeds"]:
                per_seed=base/split/case["case_id"]/str(seed)/"model.npy"
                shared=base/split/case["case_id"]/"model.npy"
                src=per_seed if per_seed.is_file() else (shared if shared.is_file() else None)
                if src is None:
                    raise ValueError(f"missing precomputed model for {split}/{case['case_id']}/{seed}")
                if src.stat().st_size>MAX_MODEL_BYTES:
                    raise ValueError("oversized model")
                read_model(src,pub["shape"])
                meta_path=src.parent/"meta.json"
                wall_s=0.
                if meta_path.is_file():
                    meta=json.loads(meta_path.read_text())
                    wall_s=float(meta.get("wall_seconds",0.))
                    if not math.isfinite(wall_s) or wall_s<0:
                        raise ValueError("invalid precomputed wall_seconds")
                resolved_sources[split][case["case_id"]][str(seed)]=(src,pub["shape"],wall_s)

    root.mkdir(parents=True,exist_ok=False)
    fields_rel=f"candidates/{f_sha}/fields.json"
    fields_dst=root/fields_rel
    fields_dst.parent.mkdir(parents=True,exist_ok=True)
    atomic_json_write(fields_dst,f_manifest)
    fields_dst.chmod(0o444)

    models={}
    for split,by_case in sorted(resolved_sources.items()):
        models[split]={}
        for cid,by_seed in sorted(by_case.items()):
            models[split][cid]={}
            for seed_s,(src,shape,wall_s) in sorted(by_seed.items()):
                rel_dir=f"models/{split}/{cid}/{seed_s}"
                m_sha=freeze_model(src,root/rel_dir,shape)
                models[split][cid][seed_s]=dict(path=f"{rel_dir}/model.npy",sha256=m_sha,wall_seconds=wall_s)

    train_work=root/"training"
    train_work.mkdir(parents=True,exist_ok=True)
    execution=Execution(train_work/"jobs",image)
    if execution.image!=image:raise ValueError("freeze a resolved image digest")
    opts={} if objective=="confirmation" else {"objective":objective}
    if objective_version is not None:opts["objective_version"]=objective_version
    evaluator=SuiteEvaluator(train_work,execution,training_path,seconds,**opts)
    train_models={cid:{s:{**e,"path":str(root/e["path"])} for s,e in seeds.items()}
                  for cid,seeds in models["train"].items()}
    train_res=_evaluate_precomputed_with_evaluator(evaluator,f_sha,train_models)
    eval_results=sorted((train_work/"evaluations").glob("*/result.json"))
    if len(eval_results)!=1:
        raise ValueError("expected exactly one precomputed training evaluation receipt")
    res_path=eval_results[0]
    cases_path=res_path.parent/"private-case-results.json"
    train_record=dict(fields_path=str(fields_dst.resolve()),metrics=dict(train_res.metrics),
        result_path=str(res_path.resolve()),result_sha256=file_hash(res_path),
        cases_path=str(cases_path.resolve()),cases_sha256=file_hash(cases_path))

    candidates=[dict(sha256=f_sha,kind="precomputed",fields_path=fields_rel,models=models)]
    frozen=dict(schema="copper-selection-v2",stage=training["stage"],candidates=candidates,
        training=dict(manifest_path=str(training_path),manifest_sha256=file_hash(training_path),
            run=str(train_work.resolve()),selected_receipts={f_sha:train_record}),
        manifests=[dict(path=str(p),sha256=file_hash(p),contents_sha256=digest(m)) for p,m in zip(manifests,loaded)],
        implementation_sha256=source_identity(),image=image,seconds=seconds,selection_objective=objective,
        selection_rule=SELECTION_RULES[objective])
    if objective_version is not None:frozen["objective_version"]=objective_version
    atomic_json_write(root/"frozen.json",frozen)
    atomic_json_write(root/"state.json",dict(phase="frozen",frozen_sha256=digest(frozen)))
    return frozen


def verify(root):
    root=Path(root);frozen=json.loads((root/"frozen.json").read_text())
    state=json.loads((root/"state.json").read_text())
    # The frozen record is tamper-evident through its digest; the code commit it was frozen under is
    # recorded in it (implementation_sha256) but need not equal the current checkout (authors 2026-11-22).
    if digest(frozen)!=state["frozen_sha256"]:
        raise ValueError("frozen scientific identity changed")
    for c in frozen["candidates"]:
        if c.get("kind")=="precomputed":
            fields_file=root/c["fields_path"]
            if fields_file.is_symlink() or not fields_file.is_file() or digest(json.loads(fields_file.read_text()))!=c["sha256"]:
                raise ValueError("frozen source changed")
            for by_case in c["models"].values():
                for by_seed in by_case.values():
                    for entry in by_seed.values():
                        mfile=root/entry["path"]
                        if mfile.is_symlink() or not mfile.is_file() or file_hash(mfile)!=entry["sha256"]:
                            raise ValueError("frozen source changed")
        elif file_hash(root/c["source_path"])!=c["sha256"]:
            raise ValueError("frozen source changed")
    training=frozen["training"]
    if file_hash(training["manifest_path"])!=training["manifest_sha256"]:raise ValueError("training manifest changed")
    for record in training["selected_receipts"].values():
        if file_hash(record["result_path"])!=record["result_sha256"] or file_hash(record["cases_path"])!=record["cases_sha256"]:
            raise ValueError("training evidence changed after freeze")
    for item in frozen["manifests"]:
        if file_hash(item["path"])!=item["sha256"] or digest(load_manifest(item["path"]))!=item["contents_sha256"]:
            raise ValueError("frozen dataset changed")
    if state["phase"] in {"selected","tested"}:
        report=json.loads((root/"selection.json").read_text())
        if digest(report)!=state["selection_sha256"]:raise ValueError("frozen selection changed")
        ids=[c["sha256"] for c in frozen["candidates"]]
        rows=report["candidates"]
        if (len(rows)!=len(ids) or {r["source_sha256"] for r in rows}!=set(ids)
                or report["selected_sha256"]!=min(rows,key=ranking)["source_sha256"]):
            raise ValueError("validation does not cover the frozen candidates")
    return frozen,state


def ranking(row):
    m=row["metrics"]
    return (m["loss_total"],-m["legal_rate"],m["mean_solver_wall_seconds"],row["source_sha256"])


def run(root,phase):
    if phase not in {"validation","test"}:raise ValueError("unknown sealed evaluation phase")
    root=Path(root)
    with locked(root):
        frozen,state=verify(root)
        if phase=="test" and state["phase"] not in {"selected","tested"}:
            raise ValueError("freeze validation selection before test")
        if phase=="validation" and state["phase"] in {"selected","tested"}:
            report=json.loads((root/"selection.json").read_text())
            if digest(report)!=state["selection_sha256"]:raise ValueError("frozen selection changed")
            return report
        if phase=="test" and state["phase"]=="tested":
            report=json.loads((root/"test-result.json").read_text())
            if digest(report)!=state["test_result_sha256"]:raise ValueError("frozen test result changed")
            return report
        selected=None
        if phase=="test":
            selection=json.loads((root/"selection.json").read_text())
            if digest(selection)!=state["selection_sha256"]:raise ValueError("frozen selection changed")
            selected=selection["selected_sha256"]
        rows=[]
        for payload in frozen["candidates"]:
            if selected is not None and payload["sha256"]!=selected:continue
            for index in ([0] if phase=="validation" else [1,2]):
                split=("validation","test_id","test_ood")[index]
                work=root/phase/payload["sha256"]/split
                if payload.get("kind")=="precomputed":
                    work.mkdir(parents=True,exist_ok=True)
                    execution=Execution(work/"jobs",frozen["image"])
                    if execution.image!=frozen["image"]:raise ValueError("freeze a resolved image digest")
                    objective=frozen.get("selection_objective","confirmation") if phase=="validation" else "confirmation"
                    options={} if objective=="confirmation" else {"objective":objective}
                    if frozen.get("objective_version"):options["objective_version"]=frozen["objective_version"]
                    evaluator=SuiteEvaluator(work,execution,frozen["manifests"][index]["path"],frozen["seconds"],**options)
                    split_models={cid:{s:{**e,"path":str(root/e["path"])} for s,e in seeds.items()}
                                  for cid,seeds in payload["models"][split].items()}
                    result=_evaluate_precomputed_with_evaluator(evaluator,payload["sha256"],split_models)
                    rows.append(dict(source_sha256=payload["sha256"],split=split,metrics=dict(result.metrics)))
                    continue
                copied=seal(work,(root/payload["source_path"]).read_bytes())
                execution=Execution(work/"jobs",frozen["image"])
                # The resolved Docker image digest must match every phase.
                if execution.image!=frozen["image"]:raise ValueError("freeze a resolved image digest")
                candidate=CheckedDomain(work,InterfaceChecks(work,execution)).admit(RawProposal(copied,"frozen_selection"))
                if isinstance(candidate,CandidateRejection):
                    rows.append(dict(source_sha256=payload["sha256"],split=split,interface_rejected=True,
                        metrics=dict(loss_total=1.,mean_iou=0.,legal_rate=0.,mean_solver_wall_seconds=0.)))
                    continue
                # Validation uses the judge's objective; sealed tests always use the
                # standard scoring, whose receipts carry the IoU outcome for every judge.
                objective=frozen.get("selection_objective","confirmation") if phase=="validation" else "confirmation"
                options={} if objective=="confirmation" else {"objective":objective}
                if frozen.get("anonymize_cases"):options["anonymize_cases"]=True
                if frozen.get("canonical_origin"):options["canonical_origin"]=True
                if frozen.get("randomize_orientation"):options["randomize_orientation"]=True
                if frozen.get("objective_version"):options["objective_version"]=frozen["objective_version"]
                if frozen.get("seed_override") is not None:options["seed_override"]=frozen["seed_override"]
                evaluator=SuiteEvaluator(work,execution,frozen["manifests"][index]["path"],frozen["seconds"],**options)
                result=evaluator.evaluate(candidate)
                rows.append(dict(source_sha256=payload["sha256"],split=split,metrics=dict(result.metrics)))
        if phase=="validation":
            report=dict(selected_sha256=min(rows,key=ranking)["source_sha256"],candidates=rows,
                rule=frozen["selection_rule"],objective=frozen.get("selection_objective","confirmation"))
            atomic_json_write(root/"selection.json",report)
            atomic_json_write(root/"state.json",dict(phase="selected",frozen_sha256=digest(frozen),selection_sha256=digest(report)))
        else:
            report=dict(selected_sha256=selected,development_only=frozen["stage"]=="development",results=rows,
                disclosure="test results now seen; these datasets cannot be reused as unseen evaluation")
            atomic_json_write(root/"test-result.json",report)
            atomic_json_write(root/"state.json",{**state,"phase":"tested","test_result_sha256":digest(report)})
        return report
