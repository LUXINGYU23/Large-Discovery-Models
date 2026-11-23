import pathlib,json,sys
r=pathlib.Path('/Aidata/shared/ldm/antibody-poc');sys.path.insert(0,str(r/'tmp/expanded-protein-tools'))
from tasks.antibody.tools.complex_screening.pose_audit import audit
results=[]
for d in sorted((r/'runs/boltz2-validation-001/calibration-001').glob('*positive*')):
 n=json.loads((d/'run.json').read_text());structure=d/n['predictions'][0]['structure'];a=audit(r/'runs/boltzgen-validation-001/5GGR.cif',structure);(d/'native-pose-audit.json').write_text(json.dumps(a,indent=2)+'\n');results.append(dict(model='boltz2',seed=n['seed'],run_dir=str(d),audit=a))
for d in sorted((r/'runs/esmfold2-validation-001').glob('wrapper-positive-*')):
 n=json.loads((d/'run.json').read_text());a=audit(r/'runs/boltzgen-validation-001/5GGR.cif',d/'model.cif');(d/'native-pose-audit.json').write_text(json.dumps(a,indent=2)+'\n');results.append(dict(model='esmfold2',seed=n['seed'],run_dir=str(d),audit=a))
(r/'runs/complex-construct-calibration-001/baseline-positive-poses.json').write_text(json.dumps(results,indent=2)+'\n')
print([(x['model'],x['seed'],round(x['audit']['antigen_aligned_fv_ca_rmsd'],2),round(x['audit']['fraction_native_contacts'],3)) for x in results])
