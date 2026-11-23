"""Prepare an explicit Fab/Fv x provided-MSA/no-MSA diagnostic panel.

Fv boundaries use the supplied human Fab constant-region junction motifs; this
is a declared construct change, never an unresolved-ATOM-based sequence trim.
"""
import argparse
import hashlib
import json
from pathlib import Path


def crop_a3m(source, sequence, stop, destination):
    records=[]
    for line in Path(source).read_text().splitlines():
        if line.startswith('>'): records.append([line,''])
        elif line.strip():
            if not records: raise ValueError('A3M sequence without header')
            records[-1][1]+=line.strip()
    if not records or ''.join(c for c in records[0][1] if c.isupper() or c=='-')!=sequence:
        raise ValueError('MSA query does not match complete declared sequence')
    selected=[]
    for header,row in records:
        col=0; cropped=[]
        for char in row:
            if char.islower() or char=='.':
                if 0<col<=stop:cropped.append(char)
            elif char.isupper() or char=='-':
                if col<stop:cropped.append(char)
                col+=1
            else:raise ValueError('Unsupported A3M character')
        if col!=len(sequence):raise ValueError('A3M alignment columns mismatch')
        if any(c.isupper() for c in cropped):selected.append((header,''.join(cropped)))
    if ''.join(c for c in selected[0][1] if c.isupper() or c=='-')!=sequence[:stop]:
        raise ValueError('Cropped query mismatch')
    Path(destination).write_text(''.join(h+'\n'+s+'\n' for h,s in selected))
    return dict(source=str(source),sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        rows_before=len(records),rows_after=len(selected),query_columns_before=len(sequence),query_columns_after=stop)


def prepare(source,output):
    import yaml
    source,output=Path(source).resolve(),Path(output).resolve()
    cases=json.loads(source.read_text())
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest=[]
    for case in cases:
        proteins=[e['proteinChain'] for e in case['sequences']]
        if [p['id'] for p in proteins]!=[['A'],['B'],['Y']]:
            raise ValueError('Explicit heavy A, light B, antigen Y mapping required')
        boundaries=[]
        for p,motif in zip(proteins[:2],['ASTKGPS','RTVAAPSV']):
            sequence=p['sequence']
            if sequence.count(motif)!=1:raise ValueError('Nonunique or absent declared constant junction')
            boundaries.append(sequence.index(motif))
        for construct in ['fab','fv']:
            seqs=[];msa_manifest=[]
            for i,p in enumerate(proteins):
                sequence=p['sequence'];stop=boundaries[i] if construct=='fv' and i<2 else len(sequence)
                msa=Path(p['unpairedMsaPath'])
                if not msa.is_absolute():raise ValueError('Precomputed MSA must be absolute')
                dest=output/(case['name']+'-'+construct+'-'+p['id'][0]+'.a3m')
                msa_manifest.append(crop_a3m(msa,sequence,stop,dest))
                seqs.append(dict(id=p['id'][0],sequence=sequence[:stop],msa=str(dest)))
            name=case['name']+'-'+construct
            native=dict(name=name,sequences=[{'proteinChain':dict(id=[s['id']],sequence=s['sequence'])} for s in seqs])
            (output/(name+'.json')).write_text(json.dumps([native],indent=2)+'\n')
            for use_msa in (True,False):
                yaml_name=name+('-msa' if use_msa else '-single')+'.yaml'
                data=dict(version=1,sequences=[{'protein':dict(s,msa=s['msa'] if use_msa else 'empty')} for s in seqs])
                (output/yaml_name).write_text(yaml.safe_dump(data,sort_keys=False))
            manifest.append(dict(name=name,construct=construct,chain_lengths={s['id']:len(s['sequence']) for s in seqs},msa=msa_manifest))
    (output/'manifest.json').write_text(json.dumps(dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),cases=manifest),indent=2)+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('output')
    a=p.parse_args();print(json.dumps(prepare(a.source,a.output),indent=2))
