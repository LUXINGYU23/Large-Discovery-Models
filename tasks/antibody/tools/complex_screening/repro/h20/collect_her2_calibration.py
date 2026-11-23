import json,pathlib
r=pathlib.Path('/Aidata/shared/ldm/antibody-poc');root=r/'runs/her2-complex-calibration-001';plan=json.loads((root/'plan.json').read_text());runs=[];missing=[]
for model in plan['models']:
 for case in plan['cases']:
  for seed in plan['seeds']:
   name=f"{model}-{case['name']}-seed{seed}";d=root/name
   if not (d/'ipsae/run.json').is_file():missing.append(name);continue
   n=json.loads((d/'run.json').read_text());s=json.loads((d/'ipsae/run.json').read_text())
   if n['status']!='succeeded' or s['status']!='succeeded':missing.append(name);continue
   scores=[x for x in s['chain_pair_scores'] if x['Type']=='asym'];pairs={(x['Chn1'],x['Chn2']):x['ipSAE'] for x in scores}
   result=dict(model=model,case=case['name'],seed=seed,run_dir=str(d),prediction_status=n['status'],ipsae_status=s['status'],score_inputs=s['inputs'],input_sha256=n.get('input_sha256'),seconds=n['seconds'],directional_scores=scores,heavy_antigen_ipsae_min=min(pairs['A','Y'],pairs['Y','A']),heavy_antigen_ipsae_max=max(pairs['A','Y'],pairs['Y','A']),light_antigen_ipsae_min=min(pairs['B','Y'],pairs['Y','B']))
   if (d/'native-pose-audit.json').is_file():result['native_pose_audit']=json.loads((d/'native-pose-audit.json').read_text())
   elif case['name'].endswith('positive'):missing.append(name+'-pose');continue
   runs.append(result)
report=dict(scope='Second antigen HER2 complete extracellular sequence plus Fv, no MSA, both models, fixed 3 seeds',plan=plan,native_self_check=json.loads((root/'native-self-check.json').read_text()),hard_filter_eligible=False,missing=missing,runs=runs)
(root/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print('HER2 completed',len(runs),'of12; missing',len(missing))
for x in runs:print(x['model'],x['case'],x['seed'],round(x['heavy_antigen_ipsae_min'],3),round(x.get('native_pose_audit',{}).get('fraction_native_contacts',-1),3),round(x['seconds'],1))
