"""Trusted frozen validation/test protocol, reusing suite execution and receipts.

Development qualification only. This is not an access-control service: formal
publication still requires the independently owned evaluator and dataset gate.
"""
import fcntl
import json
import math
from pathlib import Path
from contextlib import contextmanager
from ldm_tts.contracts import RawProposal,CandidateRejection
from ldm_tts.engine.run_store import atomic_json_write
from ..core.candidate import seal,validate_source
from ..core.execution import Execution,digest,source_identity
from .contract import OBJECTIVE_VERSIONS
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
           objective_version=None):
    """objective is the run judge's validation scoring; sealed outcomes stay IoU."""
    root=Path(root)
    if any(type(v) is not bool for v in (anonymize_cases,canonical_origin,randomize_orientation)):
        raise ValueError("invalid case isolation flags")
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
        if file_hash(root/c["source_path"])!=c["sha256"]:raise ValueError("frozen source changed")
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
