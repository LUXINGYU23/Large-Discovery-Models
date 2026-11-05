"""State-machine unit coverage; real container recovery lives in test_v2_suite."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import numpy as np
from tasks.copper_inversion.tests.test_v2_science import fixture,model
from tasks.copper_inversion.v2.data import save_problem
from tasks.copper_inversion.v2.suite import file_hash,aggregate
from tasks.copper_inversion.core.execution import digest
from tasks.copper_inversion.core.candidate import seal
from tasks.copper_inversion.v2.selection import freeze,run,verify


def inputs(root):
    manifests=[]
    for index,split in enumerate(("train","validation","test_id","test_ood")):
        base=root/split;base.mkdir();truth=base/"truth";truth.mkdir()
        mask=model(fixture())>0;mask[:index+1,:,:]=False
        np.save(truth/"truth.npy",mask)
        rows=[]
        for modality in ("drill","drill_outcrop"):
            problem=fixture();problem["case_id"]=split+modality
            public=base/modality;save_problem(problem,public)
            rows.append(dict(case_id=problem["case_id"],parent_id=split,modality=modality,public=modality,truth="truth",
                sha256={name:file_hash((truth if name=="truth.npy" else public)/name) for name in ("problem.json","problem.npz","truth.npy")}))
        path=base/"manifest.json"
        path.write_text(json.dumps(dict(schema="copper-suite-v2",stage="development",split=split,seeds=[47],cases=rows)))
        manifests.append(path)
    sources=[]
    for i in range(3):
        path=root/f"solver-{i}.py";path.write_text(f"def solve(problem,budget,seed):\n    return {i}\n")
        sources.append(path)
        payload=seal(root/"training",path.read_bytes())
        evaluation=root/"training/evaluations"/payload["sha256"];evaluation.mkdir(parents=True)
        train=json.loads(manifests[0].read_text())
        records=[]
        for case in train["cases"]:
            row=dict(case_id=case["case_id"],seed=47,parent_id=case["parent_id"],status="succeeded",metrics={"iou":.2,"confirmation_loss_total":.4},solver_wall_seconds=1.)
            receipt=evaluation/case["case_id"]/"47/receipt.json";receipt.parent.mkdir(parents=True)
            receipt.write_text(json.dumps(row));records.append(row)
        (evaluation/"private-case-results.json").write_text(json.dumps(records))
        (evaluation/"result.json").write_text(json.dumps(dict(candidate_id=payload["sha256"],status="succeeded",metrics=aggregate(records),metadata={"suite_digest":digest(train)})))
    return sources,manifests[1:]


def test_freeze_select_one_and_replay_without_new_evaluations(tmp_path):
    sources,manifests=inputs(tmp_path);root=tmp_path/"protocol";image="sha256:"+"a"*64
    provenance=dict(training_run=tmp_path/"training",training_manifest=tmp_path/"train/manifest.json")
    frozen=freeze(root,sources,manifests[0],manifests[1:],seconds=10,image=image,**provenance)
    expected=frozen["candidates"][1]["sha256"]
    with pytest.raises(ValueError,match="before test"):run(root,"test")
    calls=[]
    class Evaluator:
        def __init__(self,root,execution,manifest,seconds):self.manifest=manifest
        def evaluate(self,candidate):
            calls.append((candidate.canonical_key,self.manifest))
            # Better independent data fit must win even with worse IoU.
            return SimpleNamespace(metrics=dict(loss_total=.1 if candidate.canonical_key==expected else .9,
                mean_iou=.1 if candidate.canonical_key==expected else .9,legal_rate=1.,mean_solver_wall_seconds=1.))
    with patch("tasks.copper_inversion.v2.selection.Execution",lambda root,img:SimpleNamespace(image=img)),patch("tasks.copper_inversion.v2.selection.SuiteEvaluator",Evaluator),patch("tasks.copper_inversion.v2.selection.InterfaceChecks",lambda root,execution:lambda payload:True):
        selection=run(root,"validation")
        assert selection["selected_sha256"]==expected and len(calls)==3
        assert run(root,"validation")==selection and len(calls)==3
        result=run(root,"test")
        assert len(calls)==5 and {r["source_sha256"] for r in result["results"]}=={expected}
        assert run(root,"test")==result and len(calls)==5
    with pytest.raises(FileExistsError):freeze(root,sources,manifests[0],manifests[1:],seconds=10,image=image,**provenance)
    source=root/frozen["candidates"][0]["source_path"];source.chmod(0o644);source.write_text("changed")
    with pytest.raises(ValueError,match="source changed"):verify(root)


def test_reject_unbounded_selection_and_unresolved_image(tmp_path):
    sources,manifests=inputs(tmp_path)
    provenance=dict(training_run=tmp_path/"training",training_manifest=tmp_path/"train/manifest.json")
    with pytest.raises(ValueError,match="one to three"):
        freeze(tmp_path/"p",sources*2,manifests[0],manifests[1:],seconds=10,image="sha256:"+"a"*64,**provenance)
    with pytest.raises(ValueError,match="image"):
        freeze(tmp_path/"p",sources,manifests[0],manifests[1:],seconds=10,image="mutable-tag",**provenance)


def test_untrained_source_and_incomplete_receipts_rejected(tmp_path):
    from tasks.copper_inversion.v2.training_receipts import eligible
    sources,manifests=inputs(tmp_path)
    training=tmp_path/"training";manifest=tmp_path/"train/manifest.json"
    assert len(eligible(training,manifest))==3
    sources[0].write_text("def solve(problem,budget,seed):\n    return 42\n")
    with pytest.raises(ValueError,match="training evidence"):
        freeze(tmp_path/"p",sources,manifests[0],manifests[1:],seconds=10,image="sha256:"+"a"*64,
            training_run=training,training_manifest=manifest)
    rows_path=next(training.glob("evaluations/*/private-case-results.json"))
    rows=json.loads(rows_path.read_text());rows_path.write_text(json.dumps(rows[:1]))
    with pytest.raises(ValueError,match="missing or duplicating"):eligible(training,manifest)
