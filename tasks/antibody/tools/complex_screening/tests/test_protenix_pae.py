import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from tasks.antibody.tools.complex_screening.protenix_pae import convert


@unittest.skipUnless(importlib.util.find_spec('Bio') and importlib.util.find_spec('numpy'), 'Predictor dependencies required')
class MappingTests(unittest.TestCase):
    def test_confidence_must_match_cif_and_chain_partition(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            structure = root / 'model.cif'
            structure.write_text('data_model\nloop_\n_atom_site.group_PDB\n_atom_site.label_atom_id\n_atom_site.label_asym_id\n_atom_site.label_seq_id\nATOM CA A 1\nATOM CA B 1\n')
            confidence = root / 'confidence.json'
            data = dict(token_pair_pae=[[0, 5], [6, 0]], atom_to_token_idx=[0, 1],
                        atom_plddt=[0.9, 0.8], token_asym_id=[1, 2])
            confidence.write_text(json.dumps(data))
            output = root / 'converted.json'
            convert(confidence, structure, output)
            self.assertEqual(json.loads(output.read_text())['atom_plddts'], [90, 80])
            data['token_asym_id'] = [1, 1]
            confidence.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'Chain mapping'):
                convert(confidence, structure, root / 'bad-chain.json')
            data['token_asym_id'] = [1, 2]
            data['atom_to_token_idx'] = [0, 0]
            confidence.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'ordering'):
                convert(confidence, structure, root / 'bad-order.json')


if __name__ == '__main__':
    unittest.main()
