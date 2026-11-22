"""Convert native Protenix confidence to ipSAE-readable JSON for protein-only CIFs.

Requires Biopython and NumPy in the predictor environment. Does not infer PAE.
"""
import argparse
import hashlib
import json
from pathlib import Path


def convert(confidence, structure, output):
    import numpy as np
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    confidence, structure, output = map(Path, (confidence, structure, output))
    if output.exists():
        raise FileExistsError(output)
    data = json.loads(confidence.read_text())
    cif = MMCIF2Dict(str(structure))
    atom_names = cif['_atom_site.label_atom_id']
    n_atoms = len(atom_names)
    if set(cif['_atom_site.group_PDB']) != {'ATOM'}:
        raise ValueError('Only standard protein ATOM records supported')
    residues = list(zip(cif['_atom_site.label_asym_id'], cif['_atom_site.label_seq_id']))
    ordered = list(dict.fromkeys(residues))
    if len(set(cif.get('_atom_site.pdbx_PDB_model_num', ['1']))) != 1:
        raise ValueError('Multiple models unsupported')
    ca = [r for r,a in zip(residues, atom_names) if a == 'CA']
    if ca != ordered:
        raise ValueError('Expected exactly one ordered CA per protein token')
    pae = np.asarray(data['token_pair_pae'], dtype=float)
    mapping = np.asarray(data['atom_to_token_idx'], dtype=int)
    if not np.array_equal(mapping, np.asarray(data['atom_to_token_idx'])):
        raise ValueError('Non-integral atom/token mapping')
    plddt = np.asarray(data['atom_plddt'], dtype=float)
    token_ids = np.asarray(data['token_asym_id'])
    if pae.shape != (len(ordered), len(ordered)) or mapping.shape != (n_atoms,) or plddt.shape != (n_atoms,):
        raise ValueError('Confidence/CIF dimensions mismatch')
    expected = {r:i for i,r in enumerate(ordered)}
    if not np.array_equal(mapping, [expected[r] for r in residues]):
        raise ValueError('Confidence/CIF token ordering mismatch')
    chains = list(dict.fromkeys(r[0] for r in ordered))
    if len(token_ids) != len(ordered) or len(set(token_ids)) != len(chains) or any(len(set(token_ids[[i for i,r in enumerate(ordered) if r[0]==chain]])) != 1 for chain in chains):
        raise ValueError('Chain mapping mismatch')
    if not np.isfinite(pae).all() or not np.isfinite(plddt).all() or (pae < 0).any() or (plddt < 0).any() or (plddt > 1).any():
        raise ValueError('Unexpected confidence values or pLDDT scale')
    result = {'pae': pae.tolist(), 'atom_plddts': (plddt * 100).tolist(),
              'token_chain_ids': [r[0] for r in ordered],
              'provenance': {'source': 'Protenix native confidence; format conversion only',
                             'confidence_sha256': hashlib.sha256(confidence.read_bytes()).hexdigest(),
                             'structure_sha256': hashlib.sha256(structure.read_bytes()).hexdigest()}}
    output.write_text(json.dumps(result)+'\n')
    return output

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('confidence');p.add_argument('structure');p.add_argument('output')
    a=p.parse_args();print(convert(a.confidence,a.structure,a.output))
