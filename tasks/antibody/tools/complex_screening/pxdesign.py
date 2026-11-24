"""Experimental PXDesign raw binder generation, independent of 11-residue CDRH3 campaigns.

Runs the official raw API with prepared local resources. Does not run the optional
AF2/Protenix evaluation/selection pipeline or download its unused checkpoints.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def design(spec, output, source_root, checkpoint_dir, ccd_dir, gpu, seed=19, steps=200):
    import yaml
    spec,output,source_root,checkpoint_dir,ccd_dir=map(lambda x:Path(x).absolute(),(spec,output,source_root,checkpoint_dir,ccd_dir))
    data=yaml.safe_load(spec.read_text())
    if not isinstance(data,dict) or not isinstance(data.get('binder_length'),int) or data['binder_length']<1:
        raise ValueError('Native target YAML with positive binder_length required')
    target=Path(data['target']['file'])
    if not target.is_absolute(): target=spec.parent/target
    if not target.is_file() or not isinstance(data['target'].get('chains'),dict) or len(data['target']['chains'])!=1:
        raise ValueError('A local target structure and exactly one explicit target chain are required')
    if gpu<0 or not 0<=seed<2**32 or steps<1:
        raise ValueError('Invalid GPU or inference parameters')
    if importlib.metadata.version('pxdesign')!='0.1.0' or importlib.metadata.version('pxdbench')!='0.1.2':
        raise ValueError('Validated API requires pxdesign==0.1.0 and pxdbench==0.1.2')
    if not (source_root/'pxdesign/configs/configs_base.py').is_file():
        raise FileNotFoundError('Official PXDesign source checkout required (wheel omits namespace configs)')
    files=[checkpoint_dir/'pxdesign_v0.1.0.pt']+[ccd_dir/n for n in ('components.v20240608.cif','components.v20240608.cif.rdkit_mol.pkl','clusters-by-entity-40.txt')]
    for path in files:
        if not path.is_file(): raise FileNotFoundError(path)
    if output.exists() and any(output.iterdir()): raise ValueError('Output must be empty')
    (output/'inputs').mkdir(parents=True,exist_ok=True)
    snapshot=output/'inputs'/('target'+target.suffix);snapshot.write_bytes(target.read_bytes())
    data['target']['file']=str(snapshot)
    native=output/'inputs/spec.yaml';native.write_text(yaml.safe_dump(data,sort_keys=False))
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),PROTENIX_DATA_ROOT_DIR=str(ccd_dir))
    sys.path.insert(0,str(source_root))
    record=dict(tool='pxdesign_raw',status='running',candidate_kind='protein_binder',seed=seed,
        binder_length=data['binder_length'],samples=1,diffusion_steps=steps,msa=False,
        cdrh3_campaign_compatible=False,full_selection_pipeline_validated=False,hard_filter_eligible=False,
        spec_sha256=digest(spec),target_sha256=digest(target),source_root=str(source_root),
        resources={str(p):digest(p) for p in files})
    start=time.monotonic()
    try:
        from pxdesign.utils.infer import get_configs,convert_to_bioassembly_dict
        from pxdesign.utils.inputs import process_input_file
        from pxdesign.runner.inference import InferenceRunner
        from protenix.config import save_config
        from protenix.utils.seed import seed_everything
        args=['--input_json_path',str(native),'--dump_dir',str(output),'--load_checkpoint_dir',str(checkpoint_dir),
              '--N_sample','1','--N_step',str(steps),'--seeds',str(seed),'--dtype','bf16','--use_msa','False',
              '--use_fast_ln','False','--use_deepspeed_evo_attention','False']
        configs=get_configs(args)
        configs.input_json_path=process_input_file(configs.input_json_path,out_dir=configs.dump_dir)
        inputs=json.loads(Path(configs.input_json_path).read_text())
        for item in inputs:convert_to_bioassembly_dict(item,configs.dump_dir)
        configs.input_json_path=str(output/'input_tasks.json')
        Path(configs.input_json_path).write_text(json.dumps(inputs,indent=2)+'\n')
        save_config(configs,str(output/'config.yaml'))
        runner=InferenceRunner(configs);seed_everything(seed=seed,deterministic=False);runner._inference(seed)
        structures=sorted(output.glob('**/predictions/*.cif'))
        if len(structures)!=1:raise RuntimeError(f'Expected one actual generated CIF, got {len(structures)}')
        from Bio.PDB.MMCIF2Dict import MMCIF2Dict
        atoms=MMCIF2Dict(str(structures[0]));chains={}
        for chain,residue,name in zip(atoms['_atom_site.label_asym_id'],atoms['_atom_site.label_seq_id'],atoms['_atom_site.label_atom_id']):
            if name=='CA':chains.setdefault(chain,[]).append(residue)
        if set(chains) != {'A0','B0'} or len(set(chains['B0'])) != data['binder_length']:
            raise RuntimeError('Generated structure lacks target/binder chains with requested length')
        record.update(status='succeeded',structures=[str(p.relative_to(output)) for p in structures],
            chain_residue_counts={k:len(set(v)) for k,v in chains.items()})
    except Exception as exc:
        record.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        record['seconds']=time.monotonic()-start
        (output/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('spec','out','source-root','checkpoint-dir','ccd-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--seed',type=int,default=19);p.add_argument('--steps',type=int,default=200)
    a=p.parse_args(); print(json.dumps(design(a.spec,a.out,a.source_root,a.checkpoint_dir,a.ccd_dir,a.gpu,a.seed,a.steps),indent=2))
