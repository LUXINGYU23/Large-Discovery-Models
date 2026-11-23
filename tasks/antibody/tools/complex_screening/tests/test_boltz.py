import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tasks.antibody.tools.complex_screening.boltz import predict


class BoltzArtifactsTests(unittest.TestCase):
    def test_successful_exit_without_prediction_is_failed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'case.yaml'
            source.write_text('sequences: []\n')
            with patch('subprocess.run'), self.assertRaisesRegex(RuntimeError, 'received 0'):
                predict(source, root / 'out', '/isolated/boltz', root / 'cache', 0)
            record = json.loads((root / 'out/run.json').read_text())
            self.assertEqual(record['status'], 'failed')
            self.assertFalse(record['hard_filter_eligible'])

    def test_all_samples_need_native_confidence_and_pae(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'case.yaml'
            source.write_text('sequences: []\n')

            def partial_success(*args, **kwargs):
                folder = root / 'out/prediction/boltz_results_case/predictions/case'
                folder.mkdir(parents=True)
                (folder / 'case_model_0.cif').write_text('fixture')
                (folder / 'confidence_case_model_0.json').write_text('{"iptm": 0.5}')

            with patch('subprocess.run', side_effect=partial_success):
                with self.assertRaisesRegex(RuntimeError, 'Missing native PAE'):
                    predict(source, root / 'out', '/isolated/boltz', root / 'cache', 0)


if __name__ == '__main__':
    unittest.main()
