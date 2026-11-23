import json,pathlib,sys,numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
r=pathlib.Path('/Aidata/shared/ldm/antibody-poc');sys.path.insert(0,str(r/'tmp/expanded-protein-tools'))
from tasks.antibody.tools.complex_screening.pose_audit import residues,matched,contact
jobs=[(r/'runs/her2-complex-calibration-001/boltz2-HER2_trastuzumab_positive-seed19',r/'runs/boltzgen-validation-001/1N8Z.cif',('B','A','C')),(r/'runs/complex-construct-calibration-001/boltz2-PD1_nivolumab_positive-fv-single-seed19',r/'runs/boltzgen-validation-001/5GGR.cif',('A','B','Y'))]
results=[]
for run,reference,refchains in jobs:
 record=json.loads((run/'run.json').read_text());pair=record['predictions'][0];structure=run/pair['structure'];pae=np.load(run/pair['pae'])['pae'];plddt=np.load(run/pair['plddt'])['plddt'];cif=MMCIF2Dict(str(structure));ca=[i for i,a in enumerate(cif['_atom_site.label_atom_id']) if a=='CA'];ids=[cif['_atom_site.auth_asym_id'][i] for i in ca];bf=np.array([float(cif['_atom_site.B_iso_or_equiv'][i]) for i in ca]);bf=bf/100 if bf.max()>1 else bf
 assert pae.shape==(len(ca),len(ca)) and plddt.shape==(len(ca),)
 native=[residues(reference,c) for c in refchains];pred=[residues(structure,c) for c in ('A','B','Y')];maps=[matched(a,b) for a,b in zip(native,pred)];indices={c:[i for i,chain in enumerate(ids) if chain==c] for c in ('A','B','Y')};values=[[],[]];geometric=0
 for k in (0,1):
  for i,j in maps[k]:
   for ti,tj in maps[2]:
    if contact(native[k][i][1],native[2][ti][1]):
     a=indices[('A','B')[k]][j];b=indices['Y'][tj];values[0].append(float(pae[a,b]));values[1].append(float(pae[b,a]));geometric+=contact(pred[k][j][1],pred[2][tj][1])
 results.append(dict(run_dir=str(run),pae_shape=list(pae.shape),ca_count=len(ca),plddt_range=[float(plddt.min()),float(plddt.max())],plddt_vs_cif_max_normalized_difference=float(abs(plddt-bf).max()),native_contact_pairs=len(values[0]),native_contacts_geometrically_recovered=geometric,directional_native_contact_pae=[dict(median=float(np.median(v)),min=float(min(v)),max=float(max(v)),fraction_under_10=float(np.mean(np.array(v)<10)),fraction_under_15=float(np.mean(np.array(v)<15)),fraction_under_20=float(np.mean(np.array(v)<20))) for v in values]))
(r/'runs/complex-construct-calibration-001/boltz-confidence-audit.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
