import concurrent.futures,pathlib,json,subprocess,os,time
r=pathlib.Path('/Aidata/shared/ldm/antibody-poc');code=r/'tmp/expanded-protein-tools';py=r/'env/boltzgen/bin/python';root=r/'runs/complex-pae-sensitivity-001';root.mkdir(exist_ok=False);jobs=[]
for antigen,models in [('PD1',['boltz2','esmfold2']),('HER2',['boltz2','esmfold2'])]:
 for model in models:
  cases=['PD1_nivolumab_positive','PD1_trastuzumab_mismatch'] if antigen=='PD1' else ['HER2_trastuzumab_positive','HER2_nivolumab_mismatch']
  for case in cases:
   for seed in [19,43,101]:
    name=f'{model}-{case}-'+('fv-single-' if antigen=='PD1' else '')+f'seed{seed}'
    d=r/'runs'/('complex-construct-calibration-001' if antigen=='PD1' else 'her2-complex-calibration-001')/name
    for cutoff in [15,20]:jobs.append(dict(model=model,case=case,seed=seed,pae_cutoff=cutoff,run_dir=str(d)))
(root/'plan.json').write_text(json.dumps(dict(exploratory=True,trigger='HER2 Boltz2 seed19 native-like pose with native-contact PAE median near 11.5 A; post-hoc sensitivity analysis, not threshold validation',distance_cutoff=10,cutoffs=[15,20],baseline_cutoff=10,hard_filter_eligible=False,jobs=jobs),indent=2)+'\n')
# Wait for GPU panel completion; do not race generation of genuine confidence artifacts.
start=time.monotonic()
while not (r/'runs/her2-complex-calibration-001/completed.json').is_file():
 if time.monotonic()-start>3600:raise TimeoutError('HER2 prerequisite did not finish')
 time.sleep(10)
def score(j):
 d=pathlib.Path(j['run_dir']);n=json.loads((d/'run.json').read_text())
 if j['model']=='boltz2':
  pair=n['predictions'][0];pae=d/pair['pae'];structure=d/pair['structure']
 else:pae=d/'ipsae-confidence.json';structure=d/'model.cif'
 out=root/(d.name+'-pae'+str(j['pae_cutoff']))
 with (root/(out.name+'.driver.log')).open('w') as log:
  subprocess.run([str(py),'-m','tasks.antibody.tools.complex_screening','--out',str(out),'ipsae','--python',str(py),'--script',str(r/'env/sources/IPSAE/ipsae.py'),'--pae',str(pae),'--structure',str(structure),'--pae-cutoff',str(j['pae_cutoff']),'--distance-cutoff','10'],cwd=code,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
 s=json.loads((out/'run.json').read_text());scores=[v for v in s['chain_pair_scores'] if v['Type']=='asym'];p={(v['Chn1'],v['Chn2']):v['ipSAE'] for v in scores}
 return dict(j,status=s['status'],score_dir=str(out),directional_scores=scores,heavy_antigen_ipsae_min=min(p['A','Y'],p['Y','A']),heavy_antigen_ipsae_max=max(p['A','Y'],p['Y','A']))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(score,jobs))
(root/'summary.json').write_text(json.dumps(dict(exploratory=True,hard_filter_eligible=False,plan=json.loads((root/'plan.json').read_text()),runs=results),indent=2)+'\n');print('ALL 48 SENSITIVITY SCORES COMPLETED',flush=True)
for model in ['boltz2','esmfold2']:
 for case in ['PD1_nivolumab_positive','PD1_trastuzumab_mismatch','HER2_trastuzumab_positive','HER2_nivolumab_mismatch']:
  for cutoff in [15,20]:print(model,case,cutoff,[(v['seed'],round(v['heavy_antigen_ipsae_min'],3)) for v in results if v['model']==model and v['case']==case and v['pae_cutoff']==cutoff],flush=True)
