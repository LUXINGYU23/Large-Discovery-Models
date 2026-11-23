import concurrent.futures,json,os,pathlib,subprocess,time
r=pathlib.Path('/Aidata/shared/ldm/antibody-poc');code=r/'tmp/expanded-protein-tools';root=r/'runs/her2-complex-calibration-001';py=r/'env/boltzgen/bin/python';plan=json.loads((root/'plan.json').read_text());start=time.monotonic()
print('Waiting for declared PD1 matrix to release GPUs',flush=True)
while not (r/'runs/complex-construct-calibration-001/completed.json').is_file():
 if time.monotonic()-start>1800:raise TimeoutError('PD1 prerequisite did not finish')
 time.sleep(10)
def run(command,log):
 with log.open('w') as f:subprocess.run([str(x) for x in command],cwd=code,env=dict(os.environ,OMP_NUM_THREADS='4'),stdout=f,stderr=subprocess.STDOUT,check=True)
def execute(model,case,seed,gpu):
 name=f'{model}-{case}-seed{seed}';out=root/name;print('START',name,'GPU',gpu,flush=True)
 if model=='boltz2':
  run([py,'-m','tasks.antibody.tools.complex_screening.boltz','--input',root/'inputs'/(case+'.yaml'),'--out',out,'--executable',r/'env/boltz2/bin/boltz','--cache',r/'cache/boltz','--gpu',gpu,'--seed',seed],root/(name+'.driver.log'))
  n=json.loads((out/'run.json').read_text());pair=n['predictions'][0];structure=out/pair['structure'];pae=out/pair['pae']
 else:
  run([r/'env/esmfold2/bin/python','-m','tasks.antibody.tools.complex_screening.esmfold2','--input',root/'inputs'/(case+'.json'),'--out',out,'--model-dir',r/'cache/esmfold2-69869f73','--gpu',gpu,'--seed',seed],root/(name+'.driver.log'))
  structure=out/'model.cif';pae=out/'ipsae-confidence.json'
  run([py,'-m','tasks.antibody.tools.complex_screening.esmfold2_pae',out/'native-confidence.npz',structure,pae],out/'conversion.log')
 run([py,'-m','tasks.antibody.tools.complex_screening','--out',out/'ipsae','ipsae','--python',py,'--script',r/'env/sources/IPSAE/ipsae.py','--pae',pae,'--structure',structure],out/'score-driver.log')
 if case.endswith('positive'):
  run([py,'-m','tasks.antibody.tools.complex_screening.pose_audit',plan['reference'],structure,'--reference-chains','B','A','C','--out',out/'native-pose-audit.json'],out/'pose-driver.log')
 print('DONE',name,flush=True)
def queue(model,tasks,gpu):
 for case,seed in tasks:execute(model,case,seed,gpu)
tasks=[(c['name'],seed) for c in plan['cases'] for seed in plan['seeds']]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 futures=[pool.submit(queue,'boltz2',tasks,6),pool.submit(queue,'esmfold2',tasks[::2],2),pool.submit(queue,'esmfold2',tasks[1::2],7)]
 for f in futures:f.result()
(root/'completed.json').write_text(json.dumps(dict(status='succeeded',jobs=12))+'\n');print('ALL 12 HER2 PREDICTIONS AND IPSAE COMPLETED',flush=True)
