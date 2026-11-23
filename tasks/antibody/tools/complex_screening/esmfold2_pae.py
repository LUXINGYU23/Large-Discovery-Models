"""Convert native ESMFold2 protein confidence to ipSAE JSON without estimating PAE."""
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
    with np.load(confidence, allow_pickle=False) as native:
        pae = np.asarray(native['pae'], dtype=float)
        plddt = np.asarray(native['plddt'], dtype=float)
        ranges = native['token_to_atoms'].copy()
        chain_ids = native['token_chain_ids'].copy()
    cif = MMCIF2Dict(str(structure))
    names = cif['_atom_site.label_atom_id']
    residues = list(zip(cif['_atom_site.label_asym_id'], cif['_atom_site.label_seq_id']))
    tokens = list(dict.fromkeys(residues))
    n = len(tokens)
    if set(cif['_atom_site.group_PDB']) != {'ATOM'} or len(set(cif.get('_atom_site.pdbx_PDB_model_num', ['1']))) != 1:
        raise ValueError('Expected a single protein-only model')
    if [r for r,a in zip(residues,names) if a == 'CA'] != tokens:
        raise ValueError('Expected exactly one ordered CA per token')
    if pae.shape != (n,n) or plddt.shape != (n,) or ranges.shape != (n,2) or chain_ids.shape != (n,):
        raise ValueError('Confidence/CIF dimensions mismatch')
    if not np.isfinite(ranges).all() or not np.array_equal(ranges, ranges.astype(int)):
        raise ValueError('Non-integral token atom ranges')
    expected = np.zeros((n,2), dtype=int)
    offset = 0
    for i,token in enumerate(tokens):
        size = residues.count(token)
        expected[i] = [offset, offset + size]
        if residues[offset:offset+size] != [token]*size:
            raise ValueError('Non-contiguous residue atoms')
        offset += size
    if not np.array_equal(ranges, expected):
        raise ValueError('Native atom ranges do not match CIF ordering')
    chains = list(dict.fromkeys(t[0] for t in tokens))
    if len(set(chain_ids)) != len(chains) or any(len(set(chain_ids[[i for i,t in enumerate(tokens) if t[0]==c]])) != 1 for c in chains):
        raise ValueError('Native chain partition does not match CIF')
    if not np.isfinite(pae).all() or not np.isfinite(plddt).all() or (pae < 0).any() or (plddt < 0).any() or (plddt > 1).any():
        raise ValueError('Invalid PAE or normalized pLDDT')
    atom_plddt = np.repeat(plddt, expected[:,1]-expected[:,0]) * 100
    output.write_text(json.dumps(dict(pae=pae.tolist(), atom_plddts=atom_plddt.tolist(),
        token_chain_ids=[t[0] for t in tokens], provenance=dict(
            source='ESMFold2 native confidence; token pLDDT broadcast to its native atoms',
            confidence_sha256=hashlib.sha256(confidence.read_bytes()).hexdigest(),
            structure_sha256=hashlib.sha256(structure.read_bytes()).hexdigest())))+'\n')
    return output


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('confidence');p.add_argument('structure');p.add_argument('output')
    a=p.parse_args(); print(convert(a.confidence,a.structure,a.output))
