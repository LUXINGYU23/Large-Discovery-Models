"""Private-account barrier: freeze all discoveries, validate all, then test all."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .freeze_service_selection import freeze_from_service
from tasks.copper_inversion.v2.selection import run,verify


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());release=Path(plan['release']);root=Path(plan['selection_root'])
    root.mkdir(parents=True,exist_ok=True)
    for name,service in plan['services'].items():
        target=root/name
        if target.exists():verify(target)
        else:freeze_from_service(service,target,release/'validation/manifest.json',release/'test_id/manifest.json',release/'test_ood/manifest.json')
    results={}
    with ThreadPoolExecutor(max_workers=9) as pool:
        for phase in ('validation','test'):
            futures={name:pool.submit(run,root/name,phase) for name in plan['services']}
            results[phase]={name:future.result() for name,future in futures.items()}
            print(json.dumps(dict(phase=phase,status='completed')),flush=True)
    before={str(p):p.stat().st_mtime_ns for p in root.rglob('receipt.json')}
    for name in plan['services']:
        for phase in ('validation','test'):assert run(root/name,phase)==results[phase][name]
    after={str(p):p.stat().st_mtime_ns for p in root.rglob('receipt.json')}
    if before!=after:raise ValueError('completed replay changed receipts')
    (root/'completion.json').write_text(json.dumps(dict(status='completed',results=results,replay_receipts_unchanged=True,receipt_count=len(after)),indent=2))


if __name__=='__main__':main()
