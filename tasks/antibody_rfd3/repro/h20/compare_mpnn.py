import json,subprocess,hashlib,time
from pathlib import Path
from tasks.antibody.tools.structure_design.common import extract_h3
from tasks.antibody.core.engine_adapters import _sequence_violations
from tasks.antibody.core.absolut_oracle import AbsolutOracle
r=Path('/Aidata/shared/ldm/antibody-poc');code=r/'tmp/protein-tools-attempt'
source=r/'runs/antibody_rfd3/soluble-llm-001/her2/design_jobs/job-00000'
generation=source/'output/generation.json';base=json.load(open(source/'config.json'))
out=r/'runs/mpnn-paired-comparison-001';out.mkdir(exist_ok=False)
plan={'seeds':[101,202,303],'samples_per_seed':16,'temperature':0.5,'generation':str(generation),'generation_sha256':hashlib.sha256(generation.read_bytes()).hexdigest(),'targets':['HER2','PD-1'],'note':'Same generated backbone, same 16 attempts per seed per method; report admission and duplicate failures without replacement. One-backbone pilot, not a universal default decision.'}
(out/'plan.json').write_text(json.dumps(plan,indent=2));report=[]
for seed in plan['seeds']:
 for method in ['vanilla','soluble']:
  job=out/f'{method}-{seed}';config=dict(base,seed=seed,proteinmpnn_variant=method,sequences_per_design=16,sampling_temperature=.5)
  cfg=out/f'{method}-{seed}.json';cfg.write_text(json.dumps(config))
  cmd=[str(r/'env/rfd3/bin/python'),'-m','tasks.antibody.tools.structure_design','sequence','--config',str(cfg),'--generation',str(generation),'--out-dir',str(job),'--device','cpu','--extract-h3']
  with (out/f'{method}-{seed}.log').open('w') as log:subprocess.run(cmd,cwd=code,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
  designs=json.load(open(job/'designs.json'));seqs=[extract_h3(x,11) for x in designs['candidates']];assert len(seqs)==16
  valid=[s for s in seqs if not _sequence_violations(s,11)];unique=list(dict.fromkeys(valid));row={'method':method,'seed':seed,'attempts':16,'admitted':len(valid),'unique_admitted':len(unique),'sequences':seqs,'scores':{}}
  for target,dirs in [('HER2',[r/'env/sources/Absolut',r/'runs/absolut-her2-check-fUwLve']),('PD-1',[r/'runs/absolut-pd1-5GGR-Y'])]:
   if unique:
    result=AbsolutOracle(r/'env/sources/Absolut',dirs,out/'oracle',timeout=900).evaluate(unique,target);row['scores'][target]=result
  report.append(row);(out/'results.json').write_text(json.dumps(report,indent=2));print(method,seed,'admitted',len(valid),'unique',len(unique),flush=True)
summary=[]
for row in report:
 summary.append({'method':row['method'],'seed':row['seed'],'admitted':row['admitted'],'unique':row['unique_admitted'],'best':{t:min(x['absolut_energy'] for x in v['results']) for t,v in row['scores'].items()}})
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
