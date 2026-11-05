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
from .suite import SuiteEvaluator,load_manifest,file_hash
from .domain import CheckedDomain
from .interface_checks import InterfaceChecks
from .training_receipts import eligible


@contextmanager
def locked(root):
    with (Path(root)/"protocol.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        yield


def freeze(root,sources,validation,tests,*,seconds,image,training_run,training_manifest):
    root=Path(root)
    if not 1<=len(sources)<=3:raise ValueError("freeze one to three candidates before validation")
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
        if seconds!=120:raise ValueError("formal selection requires 120 seconds")
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
        implementation_sha256=source_identity(),image=image,seconds=seconds,
        selection_rule="minimum (confirmation_loss, -legal_rate, mean_solver_wall_seconds, source_sha256)")
    atomic_json_write(root/"frozen.json",frozen)
    atomic_json_write(root/"state.json",dict(phase="frozen",frozen_sha256=digest(frozen)))
    return frozen


def verify(root):
    root=Path(root);frozen=json.loads((root/"frozen.json").read_text())
    state=json.loads((root/"state.json").read_text())
    if digest(frozen)!=state["frozen_sha256"] or source_identity()!=frozen["implementation_sha256"]:
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
                evaluator=SuiteEvaluator(work,execution,frozen["manifests"][index]["path"],frozen["seconds"])
                result=evaluator.evaluate(candidate)
                rows.append(dict(source_sha256=payload["sha256"],split=split,metrics=dict(result.metrics)))
        if phase=="validation":
            report=dict(selected_sha256=min(rows,key=ranking)["source_sha256"],candidates=rows,
                rule=frozen["selection_rule"])
            atomic_json_write(root/"selection.json",report)
            atomic_json_write(root/"state.json",dict(phase="selected",frozen_sha256=digest(frozen),selection_sha256=digest(report)))
        else:
            report=dict(selected_sha256=selected,development_only=frozen["stage"]=="development",results=rows,
                disclosure="test results now seen; these datasets cannot be reused as unseen evaluation")
            atomic_json_write(root/"test-result.json",report)
            atomic_json_write(root/"state.json",{**state,"phase":"tested","test_result_sha256":digest(report)})
        return report
