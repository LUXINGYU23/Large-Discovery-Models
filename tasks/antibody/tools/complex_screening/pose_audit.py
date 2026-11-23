"""Audit positive-control poses against explicit native chains, without confidence proxies.

Reports CA placement after antigen-only alignment and recovery of native
heavy-atom residue contacts at 5 A. These metrics are not DockQ or affinity.
"""
import argparse
import hashlib
import json
from pathlib import Path


def residues(path,chain):
    from Bio.PDB import MMCIFParser
    from Bio.SeqUtils import seq1
    c=MMCIFParser(QUIET=True).get_structure('m',str(path))[0][chain]
    result=[]
    for r in c:
        if r.id[0]!=' ' or 'CA' not in r:continue
        letter=seq1(r.resname)
        if letter not in 'ACDEFGHIKLMNPQRSTVWY':raise ValueError('Noncanonical reference/prediction residue')
        result.append((letter,r))
    return result


def matched(ref,pred):
    from Bio.Align import PairwiseAligner
    a,b=''.join(x[0] for x in ref),''.join(x[0] for x in pred)
    aligner=PairwiseAligner(mode='global',match_score=2,mismatch_score=-3,open_gap_score=-8,extend_gap_score=-.1)
    alignment=aligner.align(a,b)[0];pairs=[];mismatches=0
    for (x0,x1),(y0,y1) in zip(*alignment.aligned):
        for x,y in zip(range(x0,x1),range(y0,y1)):
            if a[x]==b[y]:pairs.append((x,y))
            else:mismatches+=1
    if len(pairs)<.9*min(len(ref),len(pred)) or mismatches>max(1,.02*len(pairs)):
        raise ValueError('Sequence mapping is not a matched positive-control construct')
    return pairs


def contact(a,b,cutoff=5):
    import numpy as np
    x=np.asarray([i.coord for i in a.get_atoms() if i.element not in ('H','D')]);y=np.asarray([i.coord for i in b.get_atoms() if i.element not in ('H','D')])
    return bool((((x[:,None]-y)**2).sum(-1)<cutoff**2).any())


def audit(reference,prediction,reference_chains=('A','B','Y'),prediction_chains=('A','B','Y')):
    import numpy as np
    ref=[residues(reference,c) for c in reference_chains];pred=[residues(prediction,c) for c in prediction_chains]
    mappings=[matched(a,b) for a,b in zip(ref,pred)]
    target=mappings[2];x=np.asarray([pred[2][j][1]['CA'].coord for i,j in target]);y=np.asarray([ref[2][i][1]['CA'].coord for i,j in target])
    xm,ym=x.mean(0),y.mean(0);u,_,v=np.linalg.svd((x-xm).T@(y-ym));u[:,-1]*=np.linalg.det(u@v);rot=u@v;trans=ym-xm@rot
    rmsd=lambda xx,yy:float(np.sqrt(np.mean(np.sum((np.asarray(xx)@rot+trans-np.asarray(yy))**2,axis=1))))
    contacts=[];recovered=0;pred_contacts=0;fv_x=[];fv_y=[];interface_x=[];interface_y=[]
    for k,motif in enumerate(['ASTKGPS','RTVAAPSV']):
        seq=''.join(p[0] for p in pred[k]);stop=seq.index(motif) if motif in seq else len(seq)
        for i,j in mappings[k]:
            if j>=stop:continue
            rr,pp=ref[k][i][1],pred[k][j][1];fv_x.append(pp['CA'].coord);fv_y.append(rr['CA'].coord);is_interface=False
            for ti,tj in target:
                native_contact=contact(rr,ref[2][ti][1]);predicted_contact=contact(pp,pred[2][tj][1]);pred_contacts+=predicted_contact
                if native_contact:
                    contacts.append([k,i,ti]);recovered+=predicted_contact;is_interface=True
            if is_interface:interface_x.append(pp['CA'].coord);interface_y.append(rr['CA'].coord)
    if not contacts:raise ValueError('Explicit reference chains have no native variable-domain interface; check biological copy mapping')
    return dict(reference=str(reference),prediction=str(prediction),reference_sha256=hashlib.sha256(Path(reference).read_bytes()).hexdigest(),prediction_sha256=hashlib.sha256(Path(prediction).read_bytes()).hexdigest(),reference_auth_chains=list(reference_chains),prediction_auth_chains=list(prediction_chains),antigen_ca_rmsd=rmsd(x,y),antigen_aligned_fv_ca_rmsd=rmsd(fv_x,fv_y),antigen_aligned_interface_antibody_ca_rmsd=rmsd(interface_x,interface_y),fv_ca_count=len(fv_x),native_interface_antibody_ca_count=len(interface_x),native_contacts=len(contacts),recovered_native_contacts=int(recovered),fraction_native_contacts=recovered/len(contacts),predicted_contacts=int(pred_contacts),contact_precision=recovered/pred_contacts if pred_contacts else 0,contact_definition='Residue pairs with any heavy-atom distance <5 A; mapped Fv/antigen observed residues only',is_dockq=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('reference');p.add_argument('prediction');p.add_argument('--reference-chains',nargs=3,default=['A','B','Y']);p.add_argument('--prediction-chains',nargs=3,default=['A','B','Y']);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=audit(a.reference,a.prediction,a.reference_chains,a.prediction_chains);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
