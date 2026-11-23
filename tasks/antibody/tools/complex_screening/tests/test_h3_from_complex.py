import importlib.util
from pathlib import Path
import tempfile
import unittest
from tasks.antibody.tools.complex_screening.h3_from_complex import extract


@unittest.skipUnless(importlib.util.find_spec('Bio'), 'Biopython required')
class FixedFrameworkTests(unittest.TestCase):
    def test_only_h3_can_change_and_renamed_chains_are_mapped(self):
        def cif(path, chains):
            lines = ['data_model', 'loop_', '_atom_site.label_asym_id',
                     '_atom_site.label_seq_id', '_atom_site.label_comp_id', '_atom_site.label_atom_id']
            for chain, sequence in chains.items():
                lines.extend(f'{chain} {i} {aa} CA' for i, aa in enumerate(sequence, 1))
            path.write_text('\n'.join(lines) + '\n')

        with tempfile.TemporaryDirectory() as d:
            native, predicted = Path(d)/'native.cif', Path(d)/'predicted.cif'
            cif(native, {'B': ['ALA', 'CYS', 'GLU', 'ASP'], 'A': ['LEU', 'GLY']})
            cif(predicted, {'X': ['ALA', 'CYS', 'GLY', 'ASP'], 'Z': ['LEU', 'GLY']})
            result = extract(predicted, native, heavy_stop=4, light_stop=2, h3_start=3, h3_stop=3)
            self.assertEqual(result['h3'], 'G')
            self.assertEqual(result['heavy_chain'], 'X')
            predicted.write_text(predicted.read_text().replace('X 3 GLY', 'X 7 GLY').replace('X 4 ASP', 'X 8 ASP'))
            result = extract(predicted, native, heavy_stop=4, light_stop=2, h3_start=3, h3_stop=3)
            self.assertEqual(result['h3_label_seq_ids'], [7])
            cif(predicted, {'X': ['MET', 'CYS', 'GLY', 'ASP'], 'Z': ['LEU', 'GLY']})
            with self.assertRaisesRegex(ValueError, 'unchanged'):
                extract(predicted, native, heavy_stop=4, light_stop=2, h3_start=3, h3_stop=3)


if __name__ == '__main__':
    unittest.main()
