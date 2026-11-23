import pathlib,json,hashlib,sys
r=pathlib.Path('/Aidata/shared/ldm/antibody-poc');out=r/'runs/her2-complex-calibration-001';(out/'inputs').mkdir(parents=True,exist_ok=False)
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import yaml
source=r/'runs/boltzgen-validation-001/1N8Z.cif';d=MMCIF2Dict(str(source));seq={}
for chains,s in zip(d['_entity_poly.pdbx_strand_id'],d['_entity_poly.pdbx_seq_one_letter_code_can']):
 s=''.join(s.split());assert not set(s)-set('ACDEFGHIKLMNPQRSTVWY')
 for chain in chains.split(','):seq[chain.strip()]=s
nivo=json.loads((r/'runs/protenix-validation/calibration-001/prepared-controls.json').read_text())[0]
pairs=[('HER2_trastuzumab_positive',[seq['B'],seq['A']]),('HER2_nivolumab_mismatch',[e['proteinChain']['sequence'] for e in nivo['sequences'][:2]])]
manifest=[]
for name,antibody in pairs:
 stops=[];fv=[]
 for s,motif in zip(antibody,['ASTKGPS','RTVAAPSV']):
  assert s.count(motif)==1;stop=s.index(motif);stops.append(stop);fv.append(s[:stop])
 sequences=[dict(id=i,sequence=s) for i,s in zip(['A','B','Y'],fv+[seq['C']])]
 case=[dict(name=name,sequences=[{'proteinChain':dict(id=[s['id']],sequence=s['sequence'])} for s in sequences])]
 (out/'inputs'/(name+'.json')).write_text(json.dumps(case,indent=2)+'\n')
 (out/'inputs'/(name+'.yaml')).write_text(yaml.safe_dump(dict(version=1,sequences=[{'protein':dict(s,msa='empty')} for s in sequences]),sort_keys=False))
 manifest.append(dict(name=name,lengths={s['id']:len(s['sequence']) for s in sequences},antibody_prefix_stops=stops))
plan=dict(reference=str(source),reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),reference_auth_chains=['B','A','C'],prediction_auth_chains=['A','B','Y'],seeds=[19,43,101],models=['boltz2','esmfold2'],samples=1,msa=False,construct='Fv with complete HER2 extracellular entity sequence',esm_loops=20,esm_steps=200,cases=manifest,mismatch_is_proven_nonbinder=False,training_overlap_not_excluded=True,hard_filter_eligible=False)
(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(plan,indent=2))
