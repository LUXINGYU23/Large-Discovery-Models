import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from tasks.antibody.tools.complex_screening.esmfold2_pae import convert


@unittest.skipUnless(importlib.util.find_spec('Bio') and importlib.util.find_spec('numpy'),'Predictor dependencies required')
class NativeMappingTests(unittest.TestCase):
    def test_native_atom_ranges_chain_partition_and_scale(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);structure=root/'model.cif';native=root/'native.npz'
            structure.write_text('data_model\nloop_\n_atom_site.group_PDB\n_atom_site.label_atom_id\n_atom_site.label_asym_id\n_atom_site.label_seq_id\nATOM CA A 1\nATOM CB A 1\nATOM CA B 1\n')
            data=dict(pae=[[0,5],[6,0]],plddt=[.9,.8],token_to_atoms=[[0,2],[2,3]],token_chain_ids=[0,1])
            np.savez(native,**data);convert(native,structure,root/'good.json')
            self.assertEqual(json.loads((root/'good.json').read_text())['atom_plddts'],[90,90,80])
            for key,bad in [('token_to_atoms',[[0,1],[1,3]]),('token_to_atoms',[[0,2.1],[2.1,3]]),('token_chain_ids',[0,0]),('plddt',[90,80]),('pae',[[0,float('nan')],[6,0]])]:
                altered=dict(data);altered[key]=bad;np.savez(native,**altered)
                with self.assertRaises(ValueError):convert(native,structure,root/'bad.json')
            self.assertFalse((root/'bad.json').exists())


if __name__=='__main__':unittest.main()
