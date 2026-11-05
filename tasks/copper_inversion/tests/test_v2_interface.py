"""Run public candidate admission checks on the remote Docker host."""
import tempfile
import unittest
from unittest.mock import Mock
from ldm_tts.contracts.evaluation import EvaluationPaused
from pathlib import Path
import numpy as np
from ldm_tts.contracts import RawProposal, CandidateRejection
from tasks.copper_inversion.core.candidate import seal
from tasks.copper_inversion.core.execution import Execution
from tasks.copper_inversion.v2.fixtures import fixtures
from tasks.copper_inversion.v2.interface_checks import InterfaceChecks
from tasks.copper_inversion.v2.domain import CheckedDomain


class InterfaceV2(unittest.TestCase):
    def test_infrastructure_failure_does_not_charge_rejection(self):
        slots=Mock()
        checks=InterfaceChecks("unused",Mock(),slots)
        checks._check=Mock(side_effect=OSError("disk full"))
        with self.assertRaises(EvaluationPaused):checks({"sha256":"candidate"})
        slots.reserve.assert_not_called()
        checks._check=Mock(side_effect=ValueError("invalid candidate"))
        with self.assertRaises(ValueError):checks({"sha256":"candidate"})
        slots.reserve.assert_called_once_with("candidate",interface_rejected=True)

    def test_fixture_coverage(self):
        problems = list(fixtures())
        self.assertEqual(len(problems), 6)
        self.assertEqual([int(np.count_nonzero(p["observations"][0]["labels"] == .5)) for p in problems], [0,1,3,4,7,2])
        self.assertEqual(problems[-1]["open_boundary_faces"], [])
        self.assertNotEqual(problems[0]["spacing"], problems[4]["spacing"])
        self.assertIsNotNone(problems[0]['priors']['hard_volume_bounds'])
        self.assertEqual(problems[5]['priors']['max_components'],4)

    def test_admission_gate_and_receipt_reuse(self):
        with tempfile.TemporaryDirectory(prefix="copper-interface-") as tmp:
            root = Path(tmp); root.chmod(0o755)
            execution = Execution(root / "jobs", "ldm-copper-baselines:20261024")
            domain = CheckedDomain(root, InterfaceChecks(root, execution))
            good = seal(root, b"from tasks.copper_inversion.v2.baselines import initialize\ndef solve(problem,budget,seed):\n    return initialize(problem)\n")
            result = domain.admit(RawProposal(good, "test"))
            self.assertNotIsInstance(result, CandidateRejection)
            jobs = list((root / "jobs").glob("*/job.json"))
            self.assertEqual(len(jobs), 6)
            before = {str(p): p.read_bytes() for p in jobs}
            self.assertNotIsInstance(domain.admit(RawProposal(good, "test")), CandidateRejection)
            self.assertEqual(before, {str(p): p.read_bytes() for p in jobs})
            bad = seal(root, b"import numpy as np\ndef solve(problem,budget,seed):\n    return np.ones(problem['shape'])\n")
            rejected = domain.admit(RawProposal(bad, "test"))
            self.assertIsInstance(rejected, CandidateRejection)


if __name__ == "__main__":
    unittest.main()
