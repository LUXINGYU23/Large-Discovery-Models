import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from tasks.antibody.tools.complex_screening.__main__ import run

class RunnerTests(unittest.TestCase):
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
