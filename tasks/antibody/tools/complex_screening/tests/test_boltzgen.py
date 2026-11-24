import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from tasks.antibody.tools.complex_screening.boltzgen_design import generate


class InterpreterTests(unittest.TestCase):
    def test_venv_interpreter_symlink_must_not_be_resolved(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            binary = root/'venv/bin/python'
            binary.parent.mkdir(parents=True)
            binary.symlink_to(sys.executable)
            source = root/'source'
            source.write_text('fixture')
            a = argparse.Namespace(out=root/'out', python=binary, gpu=0, seed=19, samples=1,
                timeout=10, memory_limit_gib=24, spec=source, scaffold=source, checkpoint=source, inverse_fold_checkpoint=source, moldir=source,
                heavy='B', light='A', heavy_stop=120, light_stop=107, h3_start=99, h3_stop=109)

            def fake_generation(command, **kwargs):
                self.assertEqual(command[0], str(binary.absolute()))
                self.assertEqual(json.loads((a.out/"job.json").read_text())["memory_limit_gib"], 24)
                folder = a.out/'intermediate_designs'
                folder.mkdir()
                (folder/'backbone.cif').write_text('fixture')
                folded = a.out/'intermediate_designs_inverse_folded'
                folded.mkdir()
                (folded/'sample.cif').write_text('fixture')

            def fake_projection(structure, scaffold, output, *args):
                output.parent.mkdir(parents=True)
                output.write_text('fixture')
                return {'restored_atom_count': 1}

            with patch('subprocess.run', side_effect=fake_generation), patch(
                    'tasks.antibody.tools.complex_screening.h3_from_complex.extract',
                    return_value={'h3': 'WGGDGFYAMDY'}), patch(
                    'tasks.antibody.tools.complex_screening.h3_from_complex.framework_rmsd', side_effect=[0.1, 0]), patch(
                    'tasks.antibody.tools.complex_screening.h3_from_complex.project_fixed_framework', side_effect=fake_projection):
                self.assertEqual(generate(a)['status'], 'succeeded')
            a.out = root/'drift'
            with patch('subprocess.run', side_effect=fake_generation), patch(
                    'tasks.antibody.tools.complex_screening.h3_from_complex.extract',
                    return_value={'h3': 'WGGDGFYAMDY'}), patch(
                    'tasks.antibody.tools.complex_screening.h3_from_complex.framework_rmsd', return_value=2):
                with self.assertRaisesRegex(RuntimeError, 'RMSD limit'):
                    generate(a)


if __name__ == '__main__':
    unittest.main()
