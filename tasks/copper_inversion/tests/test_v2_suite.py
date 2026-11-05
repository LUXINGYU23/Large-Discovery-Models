"""Remote suite aggregation and real-container receipt replay checks."""
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from ldm_tts.contracts.evaluation import EvaluationPaused
from ldm_tts.engine.run_store import BudgetLedger
from tasks.copper_inversion.v2.budget import ScientificSlots
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from tasks.copper_inversion.tests.test_v2_science import fixture,model
from tasks.copper_inversion.v2.data import save_problem
from tasks.copper_inversion.v2.suite import aggregate,load_manifest,SuiteEvaluator,file_hash
from tasks.copper_inversion.core.execution import Execution


class SuiteV2(unittest.TestCase):
    def test_fusion_task_contract(self):
        from tasks.copper_inversion.v2.task_spec import describe_task
        spec=describe_task("a"*64)
        self.assertEqual(spec.surrogate.dimension,1280)
        self.assertEqual(spec.metadata["encoding"],"fusion")
        self.assertEqual(spec.metadata["block_weights"],[.5,.5])
        self.assertIn("independent measurement",spec.objectives[0].description)

    def test_parent_macro_includes_failures(self):
        rows=[dict(parent_id="a",status="succeeded",metrics={"iou":1.,"confirmation_loss_total":0.},solver_wall_seconds=1.)]*4
        rows += [dict(parent_id="b",status="failed",metrics={},solver_wall_seconds=1.)]
        result=aggregate(rows)
        self.assertEqual(result["mean_iou"],.5)
        self.assertEqual(result["loss_total"],.5)
        self.assertEqual(result["legal_rate"],.8)
        self.assertEqual(result["failed_solves"],1)

    def test_measurement_objective_is_independent_of_iou(self):
        def row(iou,loss):
            return dict(parent_id='a',status='succeeded',metrics={'iou':iou,'confirmation_loss_total':loss},solver_wall_seconds=1.)
        good=aggregate([row(.1,.25)]);bad=aggregate([row(.9,4.)])
        self.assertAlmostEqual(good['loss_total'],.2)
        self.assertAlmostEqual(bad['loss_total'],.8)
        self.assertLess(good['loss_total'],bad['loss_total'])
        with self.assertRaisesRegex(ValueError,'confirmation'):
            aggregate([dict(parent_id='a',status='succeeded',metrics={'iou':1.},solver_wall_seconds=1.)])

    def test_real_suite_resume_and_data_identity(self):
        with tempfile.TemporaryDirectory(prefix="copper-suite-") as tmp:
            root=Path(tmp);root.chmod(0o755)
            dataset=root/"data";dataset.mkdir()
            truth=dataset/"truth";truth.mkdir();np.save(truth/"truth.npy",model(fixture())>0)
            confirmation=fixture();confirmation['observations'][0]['points'][:,0]+=.01
            save_problem(confirmation,truth/'confirmation')
            rows=[]
            for modality in ("drill","drill_outcrop"):
                problem=fixture();problem["case_id"]=modality
                public=dataset/modality;save_problem(problem,public)
                rows.append(dict(case_id=modality,parent_id="parent",modality=modality,public=modality,truth="truth",
                    sha256={name:file_hash((truth if name=="truth.npy" or name.startswith('confirmation/') else public)/name)
                            for name in ("problem.json","problem.npz","truth.npy","confirmation/problem.json","confirmation/problem.npz")}))
            manifest=dataset/"manifest.json"
            manifest.write_text(json.dumps(dict(schema="copper-suite-v2",stage="development",split="train",seeds=[47],cases=rows)))
            source=root/"candidates"/"solver.py";source.parent.mkdir()
            # Illegal outputs are still measured outer-loop observations, zero
            # on both cases, not omitted as failed BO observations.
            source.write_text("import numpy as np\ndef solve(problem,budget,seed):\n    return np.ones(problem['shape'])\n")
            candidate=SimpleNamespace(candidate_id="c",canonical_key=file_hash(source),payload={"source_path":"candidates/solver.py"})
            execution=Execution(root/"jobs","ldm-copper-baselines:20261024")
            ledger=BudgetLedger(limits={"algorithm_feedback_slots":1},path=root/"budget.json")
            evaluator=SuiteEvaluator(root,execution,manifest,seconds=10,slots=ScientificSlots(ledger.consume_many))
            with patch("tasks.copper_inversion.v2.suite.freeze_model",side_effect=OSError("disk full")):
                with self.assertRaisesRegex(EvaluationPaused,"infrastructure"):
                    evaluator.evaluate(candidate)
            self.assertFalse(list((root/"evaluations").glob("*/*/*/receipt.json")))
            result=evaluator.evaluate(candidate)
            self.assertEqual(result.metrics["mean_iou"],0.)
            self.assertEqual(result.metrics["failed_solves"],2)
            self.assertEqual(result.metrics["loss_total"],1.)
            self.assertEqual(result.resource_usage["benchmark_jobs"],4)
            jobs=list((root/"jobs").glob("*/job.json"))
            before={str(p):p.read_bytes() for p in jobs}
            # Simulate interruption after child receipts but before suite commit.
            for result_path in (root/"evaluations").glob("*/result.json"):result_path.unlink()
            replay=evaluator.evaluate(candidate)
            self.assertEqual(replay.to_dict(),result.to_dict())
            self.assertEqual(before,{str(p):p.read_bytes() for p in jobs})
            self.assertEqual(ledger.counters["algorithm_feedback_slots"],1)
            (dataset/"drill"/"problem.json").write_text("{}")
            with self.assertRaisesRegex(ValueError,"digest"):
                load_manifest(manifest)


if __name__=="__main__":unittest.main()
