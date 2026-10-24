"""Serve the pinned Enformer model through a host-only serial queue."""
import argparse,json,os
from pathlib import Path
import torch
from ldm_tts.engine.run_store import CampaignRuntime, atomic_json_write
from tasks.nucleobench.core.oracles.official import load_prepared_case,load_official_case
from tasks.nucleobench.core.oracles.queued import OracleQueue,oracle_identity

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-dir',type=Path,required=True);p.add_argument('--prepared-dir',type=Path,required=True);p.add_argument('--socket',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('Queued Enformer service requires its assigned GPU')
    prepared=load_prepared_case(a.prepared_dir,case_id='enformer_muscle_not_liver',start_index=62)
    identity=oracle_identity(prepared)
    runtime=CampaignRuntime.open(a.run_dir,task='nucleobench_oracle',config={'identity':identity,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'batch_size':4},resume=(a.run_dir/'campaign.json').exists())
    api=load_official_case(a.source_dir,prepared,runtime)
    with OracleQueue(a.socket,api.model,identity,a.run_dir/'receipts') as server:
        atomic_json_write(a.run_dir/'ready.json',{'pid':os.getpid(),'identity':identity,'socket':str(a.socket),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'device':torch.cuda.get_device_name(0),'max_batch_size':4})
        try:server.serve_forever()
        finally:a.socket.unlink(missing_ok=True)

if __name__=='__main__':main()
