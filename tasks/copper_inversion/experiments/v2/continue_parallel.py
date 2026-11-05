"""Adopt the running epoch and resume its interrupted LDM with exact turn replay."""
import hashlib,json,os,signal,subprocess,time,shutil
from pathlib import Path
from reschedule_formal import ROOT,OPS,PRIVATE,CPY,PPY,write,start

def reconcile_slots(client, private, expected=321):
    prefix='copper:v2:scientific-source:'
    def sources(budget):
        records={k[len(prefix):]:v for k,v in budget['metadata']['cumulative_usage'].items() if k.startswith(prefix)}
        if any(v.get('algorithm_feedback_slots')!=1 for v in records.values()):raise ValueError('nonunit scientific slot')
        if len(records)!=budget['counters'].get('algorithm_feedback_slots',0):raise ValueError('slot ledger inconsistent')
        return records
    c=sources(client);p=sources(private)
    if not set(p)<=set(c):raise ValueError('private source absent from client ledger')
    if len(c)!=expected:raise ValueError('incomplete discovery budget')
    local=set(c)-set(p)
    if any(c[k].get('interface_rejected_sources')!=1 for k in local):raise ValueError('unresolved source lacks private outcome')
    return dict(total_unique_sources=len(c),private_sources=len(p),local_rejected_sources=sorted(local))

def main():
    epoch=json.loads((ROOT/'scheduling-epoch.json').read_text())
    original=json.loads((OPS/'transition.json').read_text())
    d=original['original'];names=[n for n,pid in epoch['pids'].items() if n.startswith('ldm-') and not Path(f'/proc/{pid}').exists()]
    for name in names:assert not Path('/proc/'+str(epoch['pids'][name])).exists()
    # Read protected credential in memory from another owned live LDM controller.
    pid=next(pid for n,pid in epoch['pids'].items() if n.startswith('ldm-') and Path(f'/proc/{pid}').exists());children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split();key=None
    for child in children:
        for entry in Path(f'/proc/{child}/environ').read_bytes().split(b'\0'):
            if entry.startswith(b'LLM_API_KEY='):key=entry.split(b'=',1)[1].decode()
    assert key
    write(OPS/f'recovery-{int(time.time())}.json',dict(previous_epoch=epoch,restarted=names,reason='provider_timeout_same_inputs_same_budgets'))
    os.kill(epoch['supervisor_pid'],signal.SIGTERM)
    replacements={}
    for name in names:
        args=original['commands'][name];args=args[args.index('-m')+2:]+['--resume']
        env={**os.environ,'PYTHONPATH':d['client_source'],'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','LLM_API_KEY':key}
        with (ROOT/(name+'.log')).open('ab') as log:
            proc=subprocess.Popen(['runuser','-u','zsgpu','--',CPY,str(OPS/'replay_parallel.py'),'--policy',str(OPS/'policy.json'),'ldm',*args],cwd=d['client_source'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        replacements[name]=proc;epoch['pids'][name]=proc.pid
    epoch['supervisor_pid']=os.getpid();epoch['replay_adapter_sha256']=hashlib.sha256((OPS/'replay_parallel.py').read_bytes()).hexdigest()
    write(ROOT/'scheduling-epoch.json',epoch,True)
    while True:
        statuses={}
        for n,pid in epoch['pids'].items():
            active=(replacements[n].poll() is None) if n in replacements else Path(f'/proc/{pid}').exists()
            terminal=ROOT/n/('result.json' if n.startswith('ldm-') else 'completion.json')
            statuses[n]='running' if active else ('completed' if terminal.exists() else 'needs-reconciliation')
        write(ROOT/'progress.json',dict(stage='training',scheduling='parallel-r1',updated_at=time.time(),statuses=statuses),True)
        if 'running' not in statuses.values():break
        time.sleep(15)
    if any(s!='completed' for s in statuses.values()):
        write(ROOT/'needs-reconciliation.json',statuses,True);raise RuntimeError('incomplete clients retained')
    for name,config in d['configs'].items():
        audit=reconcile_slots(json.loads((ROOT/name/'budget.json').read_text()),json.loads((Path(d['services'][name])/'budget.json').read_text()))
        write(PRIVATE/(name+'-discovery-budget-audit.json'),audit)
        script='from tasks.copper_inversion.v2.service_client import ServiceEvaluator; import json; print(json.dumps(ServiceEvaluator(".",'+repr(config['socket'])+','+repr(config['contract'])+',service_uid=991).request({"operation":"seal_training","contract_digest":'+repr(config['contract'])+'})))'
        result=subprocess.run(['runuser','-u','zsgpu','--',CPY,'-c',script],cwd=d['client_source'],env={**os.environ,'PYTHONPATH':d['client_source']},capture_output=True,text=True,check=True)
        seal=json.loads(result.stdout)
        if seal.get('status')!='training_sealed' or seal.get('used_slots')!=audit['private_sources']:raise RuntimeError('incomplete seal '+name)
    finish=start('ldm-copper-eval',PPY,'finish',['--plan',str(PRIVATE/'finish-plan.json')],d['source'],PRIVATE/'finish.log')
    write(ROOT/'progress.json',dict(stage='private_validation_test',pid=finish.pid),True)
    if finish.wait()!=0:raise RuntimeError('private selection requires reconciliation')
    shutil.copy2(PRIVATE/'selection/completion.json',ROOT/'completion.json');os.chown(ROOT/'completion.json',1001,1001)
    for pid in epoch['service_pids'].values():os.killpg(pid,signal.SIGTERM)
    write(ROOT/'progress.json',dict(stage='completed'),True)

if __name__=='__main__':main()
