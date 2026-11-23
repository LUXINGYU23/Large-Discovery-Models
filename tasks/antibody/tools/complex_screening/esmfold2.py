"""Local official ESMFold2 inference. Auxiliary confidence, without MSA submission."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time


def predict(input_path, output, model_dir, gpu, seed=19, loops=20, steps=200):
    source, output, model_dir = map(lambda x: Path(x).absolute(), (input_path, output, model_dir))
    cases = json.loads(source.read_text())
    if not isinstance(cases, list) or len(cases) != 1:
        raise ValueError('Provide exactly one native protein complex in a JSON list')
    proteins=[]
    for entry in cases[0]['sequences']:
        if set(entry) != {'proteinChain'}:
            raise ValueError('Only protein chains are supported')
        chain = entry['proteinChain']; ids = chain.get('id', [])
        seq = chain['sequence']
        if not isinstance(ids,list) or len(ids)!=1 or not isinstance(ids[0],str) or len(ids[0])!=1:
            raise ValueError('Provide one explicit one-character ID per chain')
        if not isinstance(seq,str) or not seq or set(seq)-set('ACDEFGHIKLMNPQRSTVWY'):
            raise ValueError('Canonical complete protein sequences required')
        proteins.append((ids[0],seq))
    if len(proteins)<2 or len(set(p[0] for p in proteins)) != len(proteins):
        raise ValueError('At least two uniquely identified chains required')
    if gpu<0 or not 0<=seed<2**32 or loops<1 or steps<1:
        raise ValueError('Invalid GPU or inference parameters')
    for name in ('config.json','model.safetensors.index.json','ccd.pkl'):
        if not (model_dir/name).is_file():
            raise FileNotFoundError(model_dir/name)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be empty')
    if importlib.metadata.version('esm') != '3.5.0':
        raise ValueError('Validated native API requires esm==3.5.0')
    output.mkdir(parents=True,exist_ok=True)
    (output/'input.json').write_text(json.dumps(cases,indent=2)+'\n')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), ESMCFOLD_CCD_PATH=str(model_dir/'ccd.pkl'), HF_HUB_OFFLINE='1')
    record=dict(tool='esmfold2',status='running',seed=seed,loops=loops,sampling_steps=steps,
        samples=1,msa='none',kernel_backend=None,hard_filter_eligible=False,
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),model_dir=str(model_dir),
        model_index_sha256=hashlib.sha256((model_dir/'model.safetensors.index.json').read_bytes()).hexdigest())
    start=time.monotonic()
    try:
        import torch
        import numpy as np
        from esm.models.esmfold2 import EsmFold2Model, ESMFold2InputBuilder, ProteinInput, StructurePredictionInput
        model=EsmFold2Model.from_pretrained(str(model_dir),local_files_only=True).to('cuda').eval()
        model.set_kernel_backend(None)
        spi=StructurePredictionInput(sequences=[ProteinInput(id=i,sequence=s) for i,s in proteins])
        with torch.inference_mode():
            result=ESMFold2InputBuilder().fold(model,spi,num_loops=loops,num_sampling_steps=steps,num_diffusion_samples=1,seed=seed)
        (output/'model.cif').write_text(result.complex.to_mmcif())
        np.savez_compressed(output/'native-confidence.npz',pae=result.pae.detach().cpu().numpy(),
            plddt=result.plddt.detach().cpu().numpy(),token_to_atoms=result.complex.token_to_atoms,
            token_chain_ids=result.complex.chain_id)
        record.update(status='succeeded',ptm=float(result.ptm),iptm=float(result.iptm),
            pae_shape=list(result.pae.shape),plddt_shape=list(result.plddt.shape),
            artifacts=['model.cif','native-confidence.npz'])
    except Exception as exc:
        record.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        record['seconds']=time.monotonic()-start
        (output/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('input','out','model-dir'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True); p.add_argument('--seed',type=int,default=19)
    p.add_argument('--loops',type=int,default=20); p.add_argument('--steps',type=int,default=200)
    a=p.parse_args();print(json.dumps(predict(a.input,a.out,a.model_dir,a.gpu,a.seed,a.loops,a.steps),indent=2))
