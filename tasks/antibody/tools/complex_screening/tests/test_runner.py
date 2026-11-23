import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from tasks.antibody.tools.complex_screening.__main__ import run

class RunnerTests(unittest.TestCase):
    def test_boltz_native_confidence_companions_are_kept(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); script=root/'script.py'; pae=root/'pae_case.npz'; structure=root/'case.cif'
            script.write_text('import pathlib,sys\np=pathlib.Path(sys.argv[1]);assert pathlib.Path(str(p).replace("pae", "plddt")).is_file()\npathlib.Path(sys.argv[2]).with_suffix(".score.txt").write_text("Chn1 Chn2 Type ipSAE\\nA B asym 0.5\\n")\n')
            pae.write_bytes(b'fixture'); structure.write_text('fixture')
            a=argparse.Namespace(tool='ipsae',out=root/'pae-score',timeout=10,script=script,pae=pae,structure=structure,python=sys.executable,pae_cutoff=10,distance_cutoff=10)
            with self.assertRaisesRegex(ValueError, 'pLDDT'): run(a)
            (root/'plddt_case.npz').write_bytes(b'fixture')
            record=run(a)
            self.assertEqual(record['status'],'succeeded')
            self.assertIn('plddt',record['inputs'])
    def test_protenix_partial_multiseed_is_failed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'input.json'
            source.write_text(json.dumps([{'name': 'case', 'sequences': [
                {'proteinChain': {'sequence': 'ACD'}}, {'proteinChain': {'sequence': 'EFG'}}]}]))
            a = argparse.Namespace(tool='protenix', out=root/'out', timeout=10,
                input=source, model='protenix_base_default_v1.0.0', seeds=[19, 43], seed=19,
                samples=1, gpu=0, use_msa=False, executable='/isolated/protenix')

            def partial_success(*args, **kwargs):
                p = root / 'out/prediction/case/seed_19/predictions'
                p.mkdir(parents=True)
                (p/'case_sample_0.cif').write_text('fixture')
                (p/'case_full_data_sample_0.json').write_text('{}')
                (p/'case_summary_confidence_sample_0.json').write_text('{}')

            with patch('subprocess.run', side_effect=partial_success):
                with self.assertRaisesRegex(RuntimeError, 'seed 43'):
                    run(a)
            record = json.loads((root/'out/run.json').read_text())
            self.assertEqual(record['status'], 'failed')
            self.assertFalse(record['hard_filter_eligible'])

    def test_ipsae_outputs_are_isolated(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); script=root/'script.py'; pae=root/'pae.json'; structure=root/'model.cif'
            script.write_text('import pathlib,sys\np=pathlib.Path(sys.argv[2]);p.with_suffix(".score.txt").write_text("Chn1 Chn2 Type ipSAE\\nA B asym 0.5\\n")\n')
            pae.write_text('{}'); structure.write_text('fixture')
            a=argparse.Namespace(tool='ipsae',out=root/'out',timeout=10,script=script,pae=pae,structure=structure,python=sys.executable,pae_cutoff=10,distance_cutoff=10)
            record=run(a)
            self.assertEqual(record['status'],'succeeded')
            self.assertFalse((root/'model.score.txt').exists())
            self.assertTrue((root/'out/model.score.txt').exists())
            with self.assertRaises(ValueError): run(a)
    def test_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); script=root/'bad.py'; script.write_text('raise SystemExit(3)')
            pae=root/'pae.json'; pae.write_text('{}'); structure=root/'model.cif'; structure.write_text('fixture')
            a=argparse.Namespace(tool='ipsae',out=root/'out',timeout=10,script=script,pae=pae,structure=structure,python=sys.executable,pae_cutoff=10,distance_cutoff=10)
            import subprocess
            with self.assertRaises(subprocess.CalledProcessError): run(a)
            self.assertEqual(json.loads((root/'out/run.json').read_text())['status'],'failed')
if __name__=='__main__': unittest.main()
