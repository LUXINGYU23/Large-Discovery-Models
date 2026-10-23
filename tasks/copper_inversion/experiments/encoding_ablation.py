"""Remote-only experiment orchestration. Key arrives via stdin, never a run file."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CASES = ('drillholes_3d','drillholes_outcrops_3d')
MODES = ('ast','qwen','fusion')
EVALUATOR = 'sha256:9c0c855e43120f0000e3e5dcf79002763c65e55839e4908306cac29d1ff71062'
SIDECAR = 'sha256:caffc0f9407f0ac895f8019af3d80f6d2feaa94df38218e800a01af7226a0df4'


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2))
    tmp.replace(path)


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def audit_run(root, rounds):
    result=read(root/'result.json',{})
    events=[json.loads(l) for l in (root/'events.jsonl').read_text().splitlines()]
    selections=[e for e in events if e['event_type']=='candidates_selected']
    assert len(selections)==rounds+1
    assert result['observations']==1+4*rounds
    if rounds==1:assert result['successful']==5
    records={}
    for p in (root/'encodings').glob('*.json'):
        r=read(p);assert r['source_sha256']==p.stem;records[p.stem]=r
    selected=set()
    for i,e in enumerate(selections):
        payload=e['payload'];ids=payload['selected_candidate_ids'];pred=payload['predictions']
        assert len(ids)==(1 if i==0 else 4)
        assert len(pred)==(0 if i==0 else 16)
        assert not selected.intersection(ids)
        selected.update(ids)
        if i:
            assert payload['metadata']['fit_status']=='fitted'
            ranked=sorted(pred,key=lambda p:(-p['acquisition_score'],p['candidate_id']))
            assert ids==[p['candidate_id'] for p in ranked[:4]]
            assert all(p['candidate_id'] in records for p in pred)
            assert all(abs(p['acquisition_score'] + p['mean'][0] - p['std'][0]) < 1e-7 for p in pred)
    requests=0
    for path in root.rglob('*.request.bin'):
        if any(p in ('fork-source','fork-sources','nodes') for p in path.parts):continue
        obj=json.loads(path.read_text())
        assert obj['model']=='deepseek-flash'
        assert obj.get('reasoning',{}).get('effort')=='low'
        requests+=1
    assert requests>0
    for panel_path in (root/'harness/branches').glob('*/panel.json'):
        panel=read(panel_path);assert len(panel['proposals'])==16
        assert len({p['payload']['sha256'] for p in panel['proposals']})==16
        assert len(panel['sessions'])==4
    counters=read(root/'budget.json')['counters']
    assert counters['solver_jobs']==1+4*rounds
    assert counters['interface_jobs']>=2*len(records)
    preflight=read(root/'harness/preflight.json')
    assert preflight['base_url']=='https://api.deepseek.com'
    assert preflight['thinking']=='low'
    checkpoint=read(root/'checkpoint.json')['state']['observations']
    assert {o['candidate']['candidate_id'] for o in checkpoint}==selected
    return {'observations':len(checkpoint),'successful':result['successful'],'requests_inspected':requests,
            'encoding_records':len(records),'selected_count':len(selected),
            'best_loss':result['best']['evaluation']['metrics']['loss_total'],
            'encoding_seconds':sum(r['elapsed_seconds'] for r in records.values()),
            'model_load_seconds':sum(r['model_load_seconds'] for r in records.values()),
            'budget':read(root/'budget.json')['counters']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=('tiny','full'),required=True)
    a=p.parse_args();base=a.root.resolve();source=base/'source'
    key=sys.stdin.read().strip()
    if not key:raise ValueError('protected provider key required on stdin')
    if a.stage=='full':
        gate=read(base/'tiny-verification.json',{})
        if gate.get('status')!='passed':raise ValueError('six tiny runs must pass first')
    if (base/f'{a.stage}-supervisor.json').exists():
        raise ValueError('stage already dispatched; inspect durable status before recovery')
    env=dict(os.environ,LLM_API_KEY=key,PYTHONPATH=str(source),TMPDIR='/Aidata/shared/ldm/tmp',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
             OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0')
    python=str(source/'tasks/copper_inversion/.venv/bin/python')
    started=time.time()
    save(base/f'{a.stage}-supervisor.json',{'pid':os.getpid(),'status':'running','started_at':started,'stage':a.stage})
    def run(pair):
        case,mode=pair;name=f'{case}_{mode}';run=base/a.stage/name
        config=f'config/copper_inversion/encoding_{case}_{mode}_{a.stage}.yaml'
        cmd=[python,'scripts/run_ldm_tts.py',config]
        for v in (f'args.out-dir={run}', 'args.harness-cache-dir=/Aidata/shared/ldm/cache/gondolin',
                  f'args.evaluator-image={EVALUATOR}',f'args.sidecar-image={SIDECAR}'):
            cmd+=['--set',v]
        status=base/'status'/f'{a.stage}_{name}.json'
        run.parent.mkdir(parents=True,exist_ok=True)
        log=base/a.stage/f'{name}.log'
        before=time.time()
        with log.open('w') as stream:
            proc=subprocess.Popen(cmd,cwd=source,env=env,stdout=stream,stderr=subprocess.STDOUT)
            record={'pid':proc.pid,'status':'running','started_at':before,'run':str(run),'command':cmd}
            save(status,record)
            code=proc.wait()
        record.update(status='completed' if code==0 else 'failed',returncode=code,finished_at=time.time(),
                      elapsed_seconds=time.time()-before)
        save(status,record)
        if code: return {'name':name,**record}
        try:
            evidence=audit_run(run,1 if a.stage=='tiny' else 20)
            # Verify completed recovery on the identical source/config, without new calls.
            counters=read(run/'budget.json')['counters']
            with (base/a.stage/f'{name}-resume.log').open('w') as stream:
                recovered=subprocess.run(cmd+['--set','args.resume=true'],cwd=source,env=env,stdout=stream,stderr=subprocess.STDOUT)
            assert recovered.returncode==0
            assert read(run/'budget.json')['counters']==counters
            evidence['completed_resume']='unchanged counters'
            record['verification']='passed'
            save(status,record)
            return {'name':name,**record,'evidence':evidence}
        except Exception as exc:
            record.update(verification='failed',error=f'{type(exc).__name__}: {exc}')
            save(status,record)
            return {'name':name,**record}
    with ThreadPoolExecutor(max_workers=6) as pool:
        results=list(pool.map(run,[(c,m) for c in CASES for m in MODES]))
    passed=all(r.get('verification')=='passed' for r in results)
    save(base/f'{a.stage}-verification.json',{'status':'passed' if passed else 'failed','runs':results,'finished_at':time.time()})
    save(base/f'{a.stage}-supervisor.json',{'pid':os.getpid(),'status':'completed' if passed else 'failed',
         'started_at':started,'finished_at':time.time(),'stage':a.stage})
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
