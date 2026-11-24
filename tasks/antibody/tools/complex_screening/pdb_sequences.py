"""Build native Protenix input from complete PDB SEQRES, not observed ATOMs."""
import argparse
import json
from pathlib import Path


def prepare(pdb, chains, name):
    from Bio import SeqIO
    records = {record.annotations['chain']: str(record.seq) for record in SeqIO.parse(str(pdb), 'pdb-seqres')}
    if len(chains) < 2 or len(set(chains)) != len(chains):
        raise ValueError('Provide at least two unique chains')
    sequences = []
    for chain in chains:
        seq = records.get(chain)
        if not seq or set(seq) - set('ACDEFGHIKLMNPQRSTVWY'):
            raise ValueError('Missing or noncanonical SEQRES for chain ' + chain)
        sequences.append({'proteinChain': {'sequence': seq, 'count': 1, 'id': [chain]}})
    return [{'name': name, 'sequences': sequences}]

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('pdb',type=Path);p.add_argument('--chains',nargs='+',required=True)
    p.add_argument('--name',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    with a.output.open('x') as handle: json.dump(prepare(a.pdb,a.chains,a.name),handle,indent=2)
