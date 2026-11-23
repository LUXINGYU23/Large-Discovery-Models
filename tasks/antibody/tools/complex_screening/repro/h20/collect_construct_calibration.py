import pathlib,json
r=pathlib.Path('/Aidata/shared/ldm/antibody-poc');root=r/'runs/complex-construct-calibration-001'
def read(p):return json.loads(p.read_text())
def summarise(d,model,condition,case):
 n=read(d/'run.json');s=read(d/'ipsae/run.json');scores=[x for x in s['chain_pair_scores'] if x['Type']=='asym'];pair={(x['Chn1'],x['Chn2']):x['ipSAE'] for x in scores}
 result=dict(model=model,condition=condition,case=case,seed=n['seed'],run_dir=str(d),prediction_status=n['status'],ipsae_status=s['status'],score_inputs=s['inputs'],input_sha256=n.get('input_sha256'),directional_scores=scores,heavy_antigen_ipsae_min=min(pair['A','Y'],pair['Y','A']),heavy_antigen_ipsae_max=max(pair['A','Y'],pair['Y','A']),light_antigen_ipsae_min=min(pair['B','Y'],pair['Y','B']),seconds=n['seconds'])
 if (d/'native-pose-audit.json').is_file():result['native_pose_audit']=read(d/'native-pose-audit.json')
 return result
results=[]
for d in sorted((r/'runs/boltz2-validation-001/calibration-001').iterdir()):
 results.append(summarise(d,'boltz2','fab-msa',d.name.rsplit('-seed',1)[0]))
for d in sorted((r/'runs/esmfold2-validation-001').glob('wrapper-*')):
 if not (d/'run.json').is_file():continue
 score=d.parent/('ipsae-'+d.name)
 # Reuse the same summary logic without modifying completed original layouts.
 n=read(d/'run.json');s=read(score/'run.json');scores=[x for x in s['chain_pair_scores'] if x['Type']=='asym'];pair={(x['Chn1'],x['Chn2']):x['ipSAE'] for x in scores}
 result=dict(model='esmfold2',condition='fab-single',case='PD1_nivolumab_positive' if 'positive' in d.name else 'PD1_trastuzumab_mismatch',seed=n['seed'],run_dir=str(d),prediction_status=n['status'],ipsae_status=s['status'],score_inputs=s['inputs'],input_sha256=n.get('input_sha256'),directional_scores=scores,heavy_antigen_ipsae_min=min(pair['A','Y'],pair['Y','A']),heavy_antigen_ipsae_max=max(pair['A','Y'],pair['Y','A']),light_antigen_ipsae_min=min(pair['B','Y'],pair['Y','B']),seconds=n['seconds'])
 if (d/'native-pose-audit.json').is_file():result['native_pose_audit']=read(d/'native-pose-audit.json')
 results.append(result)
plan=read(root/'plan.json');missing=[]
for j in plan['jobs']:
 name=f"{j['model']}-{j['case']}-{j['condition']}-seed{j['seed']}";d=root/name
 if not (d/'ipsae/run.json').is_file():missing.append(name);continue
 if j['case'].endswith('positive') and not (d/'native-pose-audit.json').is_file():missing.append(name+'-pose');continue
 results.append(summarise(d,j['model'],j['condition'],j['case']))
report=dict(scope='Declared Fab/Fv and supplied-MSA/no-MSA PD1 diagnostics; native-pose audit only for genuine positives',plan=plan,input_manifest=read(root/'inputs/manifest.json'),native_self_check=read(root/'native-self-check.json'),hard_filter_eligible=False,training_overlap_not_excluded=True,missing=missing,runs=results)
(root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print('Completed',len(results),'of36; missing',len(missing))
for model in ['boltz2','esmfold2']:
 for condition in ['fab-msa','fab-single','fv-msa','fv-single']:
  group=sorted([x for x in results if x['model']==model and x['condition']==condition],key=lambda x:(x['case'],x['seed']))
  if group:print(model,condition,[(x['case'].split('_')[1],x['seed'],round(x['heavy_antigen_ipsae_min'],3),round(x.get('native_pose_audit',{}).get('fraction_native_contacts',-1),3)) for x in group])
