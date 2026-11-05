"""Remote Docker integration checks; all data here are public test fixtures."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from tasks.copper_inversion.tests.test_v2_science import fixture, model
from tasks.copper_inversion.v2.data import save_problem
from tasks.copper_inversion.core.execution import Execution, freeze_model


class ExecutionV2(unittest.TestCase):
    def test_solver_scorer_boundary(self):
        with tempfile.TemporaryDirectory(prefix="copper-v2-") as tmp:
            root = Path(tmp); root.chmod(0o755)
            inputs = root / "input"; save_problem(fixture(), inputs)
            (inputs / "budget.json").write_text(json.dumps(dict(seed=47, wall_time_seconds=10, cpu_count=2, memory_mb=4096)))
            source = root / "source"; source.mkdir()
            (source / "solver.py").write_text('''import numpy as np
from pathlib import Path
from tasks.copper_inversion.v2.baselines import initialize

def solve(problem, budget, seed):
    assert not Path('/truth').exists(), 'hidden truth exposed'
    try:
        Path('/input/problem.json').write_text('corrupt')
    except OSError:
        pass
    else:
        raise AssertionError('input writable')
    return initialize(problem)
''')
            execution = Execution(root / "jobs", "ldm-copper-baselines:20261024")
            job, receipt = execution.run("valid", phase="solver", inputs=inputs, timeout=10,
                cpus=2, memory_mb=4096, source=source, science_module="v2")
            self.assertEqual(receipt["exit_code"], 0, (job / "log.txt").read_text())
            freeze_model(job / "output/model.npy", root / "model", fixture()["shape"])
            truth = root / "truth"; truth.mkdir(); np.save(truth / "truth.npy", model(fixture()) > 0)
            save_problem(fixture(), truth / "confirmation")
            scored, receipt = execution.run("valid-score", phase="scorer", inputs=inputs,
                timeout=10, model=root / "model", truth=truth, science_module="v2")
            self.assertEqual(receipt["exit_code"], 0, (scored / "log.txt").read_text())
            outcome = json.loads((scored / "output/result.json").read_text())
            self.assertEqual(outcome["status"], "succeeded")
            self.assertTrue(0 <= outcome["metrics"]["iou"] <= 1)
            self.assertIn("confirmation_loss_binary", outcome["metrics"])
            self.assertIn("confirmation_loss_contact", outcome["metrics"])
            bad = root / "bad"; bad.mkdir(); np.save(bad / "model.npy", np.ones(fixture()["shape"]))
            scored, receipt = execution.run("invalid-score", phase="scorer", inputs=inputs,
                timeout=10, model=bad, truth=truth, science_module="v2")
            self.assertEqual(receipt["exit_code"], 0)
            outcome = json.loads((scored / "output/result.json").read_text())
            self.assertEqual(outcome["status"], "invalid")
            self.assertEqual(outcome["metrics"]["iou"], 0.)
            with self.assertRaisesRegex(ValueError, "restricted"):
                execution.run("leak", phase="solver", inputs=inputs, timeout=10,
                    source=source, truth=truth, science_module="v2")


if __name__ == "__main__":
    unittest.main()
