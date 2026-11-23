import concurrent.futures,json,os,pathlib,subprocess,time
r=pathlib.Path('/Aidata/shared/ldm/antibody-poc');code=r/'tmp/expanded-protein-tools';root=r/'runs/complex-construct-calibration-001';inputs=root/'inputs';py=r/'env/boltzgen/bin/python';reference=r/'runs/boltzgen-validation-001/5GGR.cif'
seeds=[19,43,101];cases=['PD1_nivolumab_positive','PD1_trastuzumab_mismatch'];jobs=[]
for condition in ['fab-single','fv-msa','fv-single']:
 for case in cases:
  for seed in seeds:jobs.append(dict(model='boltz2',condition=condition,case=case,seed=seed))
for case in cases:
 for seed in seeds:jobs.append(dict(model='esmfold2',condition='fv-single',case=case,seed=seed))
plan=dict(seeds=seeds,samples_per_case_seed=1,boltz_model='boltz2',esm_loops=20,esm_sampling_steps=200,antigen_unchanged=True,reference_auth_chains=['A','B','Y'],mismatch_is_proven_nonbinder=False,hard_filter_eligible=False,jobs=jobs)
(root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
def run(command,log):
 with log.open('w') as f:subprocess.run([str(x) for x in command],cwd=code,env=dict(os.environ,OMP_NUM_THREADS='4'),stdout=f,stderr=subprocess.STDOUT,check=True)
def execute(job,gpu):
 name=f"{job['model']}-{job['case']}-{job['condition']}-seed{job['seed']}";out=root/name
 start=time.monotonic();print('START',name,'GPU',gpu,flush=True)
 if job['model']=='boltz2':
  run([py,'-m','tasks.antibody.tools.complex_screening.boltz','--input',inputs/f"{job['case']}-{job['condition']}.yaml",'--out',out,'--executable',r/'env/boltz2/bin/boltz','--cache',r/'cache/boltz','--gpu',gpu,'--seed',job['seed']],root/(name+'.driver.log'))
  pair=json.loads((out/'run.json').read_text())['predictions'][0];structure=out/pair['structure'];pae=out/pair['pae']
 else:
  run([r/'env/esmfold2/bin/python','-m','tasks.antibody.tools.complex_screening.esmfold2','--input',inputs/f"{job['case']}-fv.json",'--out',out,'--model-dir',r/'cache/esmfold2-69869f73','--gpu',gpu,'--seed',job['seed']],root/(name+'.driver.log'))
  structure=out/'model.cif';pae=out/'ipsae-confidence.json'
  run([py,'-m','tasks.antibody.tools.complex_screening.esmfold2_pae',out/'native-confidence.npz',structure,pae],out/'conversion.log')
 run([py,'-m','tasks.antibody.tools.complex_screening','--out',out/'ipsae','ipsae','--python',py,'--script',r/'env/sources/IPSAE/ipsae.py','--pae',pae,'--structure',structure],out/'score-driver.log')
 if job['case'].endswith('positive'):
  run([py,'-m','tasks.antibody.tools.complex_screening.pose_audit',reference,structure,'--out',out/'native-pose-audit.json'],out/'pose-driver.log')
 print('DONE',name,round(time.monotonic()-start,1),flush=True)
def queue(tasks,gpu):
 for job in tasks:execute(job,gpu)
boltz=[j for j in jobs if j['model']=='boltz2'];esm=[j for j in jobs if j['model']=='esmfold2']
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 futures=[pool.submit(queue,boltz[::2],6),pool.submit(queue,boltz[1::2],7),pool.submit(queue,esm,2)]
 for f in futures:f.result()
(root/'completed.json').write_text(json.dumps(dict(status='succeeded',jobs=len(jobs)))+'\n');print('ALL 24 PREDICTIONS AND IPSAE FINISHED',flush=True)
