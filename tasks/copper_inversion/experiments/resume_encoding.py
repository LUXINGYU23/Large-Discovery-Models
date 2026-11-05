"""Resume an unchanged campaign on a different physical GPU; retain original receipts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from encoding_ablation import save, read, audit_run

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--name',required=True)
p.add_argument('--gpu',required=True,choices=['0','1','2','3']);p.add_argument('--detach',action='store_true')
p.add_argument('--attempt',type=int,default=1)
a=p.parse_args();base=a.root.resolve();key=sys.stdin.read().strip()
if not key:raise ValueError('protected key required on stdin')
suffix='' if a.attempt==1 else f'-attempt{a.attempt}'
record_path=base/f'recovery-{a.name}{suffix}.json'
if record_path.exists():raise ValueError('recovery already dispatched; inspect record')
if a.detach:
    with (base/f'recovery-{a.name}{suffix}.log').open('x') as log:
        child=subprocess.Popen([sys.executable,__file__,'--root',str(base),'--name',a.name,'--gpu',a.gpu,'--attempt',str(a.attempt)],
            stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        child.stdin.write(key.encode());child.stdin.close()
    print(json.dumps({'pid':child.pid,'record':str(record_path)}));sys.exit(0)
status=base/'status'/f'full_{a.name}.json';original=read(status)
if Path(f'/proc/{original["pid"]}').exists():raise ValueError('original process still alive')
run=Path(original['run']);before=read(run/'checkpoint.json')['state']['observations']
save(base/f'pre-recovery-{a.name}{suffix}.json',{'status':original,'budget':read(run/'budget.json'),
    'observations':[o['candidate']['candidate_id'] for o in before]})
env=dict(os.environ,LLM_API_KEY=key,PYTHONPATH=str(base/'source'),TMPDIR='/Aidata/shared/ldm/tmp',
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES=a.gpu)
command=original['command']+['--set','args.resume=true']
child=subprocess.Popen(command,cwd=base/'source',env=env)
record=dict(pid=child.pid,controller_pid=os.getpid(),started_at=time.time(),status='running',
            physical_gpu=a.gpu,logical_device='cuda:0',command=command,original_status=original,
            reason='CUDA OOM on shared GPU 0; unchanged runtime code/config/checkpoint values; inspect checkpoint-order-repair.json if present')
save(record_path,record);save(status,{**original,'pid':child.pid,'status':'running','recovery_record':str(record_path)})
code=child.wait();record.update(returncode=code,finished_at=time.time(),status='completed' if code==0 else 'failed')
if code==0:
    after=read(run/'checkpoint.json')['state']['observations']
    assert after[:len(before)]==before,'historical observations changed'
    record['verification']=audit_run(run,20)
save(record_path,record);save(status,{**read(status),'status':record['status'],'returncode':code,'finished_at':time.time()})
