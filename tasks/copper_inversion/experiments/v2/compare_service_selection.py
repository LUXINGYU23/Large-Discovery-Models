"""Trusted development comparison: freeze every method before any validation."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tasks.copper_inversion.core.execution import digest
from tasks.copper_inversion.v2.selection import run, verify
from tasks.copper_inversion.experiments.v2.freeze_service_selection import freeze_from_service


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--private-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();base=a.private_root;root=a.output
    services={'ldm':base/'runs/service-r3','mcmc':base/'runs/service-mcmc-r1','acq':base/'runs/service-acq-r1'}
    data=base/'datasets/development-suite-r1'
    plan=dict(services={k:str(v) for k,v in services.items()},data=str(data),workers=3,
              phases=['freeze_all','validation_all','test_all'],scope='development-only')
    if a.resume:
        if json.loads((root/'plan.json').read_text())!=plan:raise ValueError('comparison plan differs')
    else:
        root.mkdir(parents=True,exist_ok=False)
        (root/'plan.json').write_text(json.dumps(plan,indent=2))
    for name,service in services.items():
        target=root/name
        if target.exists():verify(target)
        else:freeze_from_service(service,target,data/'id/validation/manifest.json',data/'id/test_id/manifest.json',data/'ood/test_ood/manifest.json')
    print(json.dumps(dict(phase='all_methods_frozen',methods=list(services))),flush=True)
    results={}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for phase in ('validation','test'):
            futures={name:pool.submit(run,root/name,phase) for name in services}
            results[phase]={name:future.result() for name,future in futures.items()}
            print(json.dumps(dict(phase=phase,status='completed')),flush=True)
    # A completed replay must not create/modify any job receipt.
    before={str(p):p.stat().st_mtime_ns for p in root.rglob('receipt.json')}
    for name in services:
        assert run(root/name,'validation')==results['validation'][name]
        assert run(root/name,'test')==results['test'][name]
    after={str(p):p.stat().st_mtime_ns for p in root.rglob('receipt.json')}
    if before!=after:raise ValueError('completed protocol replay modified job receipts')
    report=dict(status='completed',plan_sha256=digest(plan),results=results,
                replay_receipts_unchanged=True,receipt_count=len(after),development_only=True)
    (root/'completion.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(status='completed',receipt_count=len(after))),flush=True)


if __name__=='__main__':main()
