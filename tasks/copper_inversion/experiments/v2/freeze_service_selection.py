"""Trusted evaluator-account command; no private paths enter the research process."""
import argparse
import json
from pathlib import Path
from tasks.copper_inversion.core.execution import digest
from tasks.copper_inversion.v2.selection import freeze
from tasks.copper_inversion.v2.training_receipts import eligible


def freeze_from_service(service_root, output, validation, test_id, test_ood):
    service_root=Path(service_root)
    identity=json.loads((service_root/'identity.json').read_text())
    sealed=json.loads((service_root/'training-sealed.json').read_text())
    if sealed['contract_digest']!=digest(identity):raise ValueError('training seal differs from service identity')
    config=identity['configuration']
    current=eligible(service_root,config['manifest'])
    frozen=sealed['selected_receipts']
    order=sealed['source_sha256s']
    if (not frozen or len(order)!=len(set(order)) or set(order)!=set(frozen)
            or any(current.get(sha)!=record for sha,record in frozen.items())):
        raise ValueError('training receipts changed after service seal')
    return freeze(output,[frozen[sha]['source_path'] for sha in order],validation,[test_id,test_ood],
        seconds=config['seconds'],image=config['image'],training_run=service_root,training_manifest=config['manifest'])


def main():
    p=argparse.ArgumentParser()
    for name in ('service-root','output','validation','test-id','test-ood'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    result=freeze_from_service(a.service_root,a.output,a.validation,a.test_id,a.test_ood)
    print(json.dumps(dict(status='frozen',source_sha256s=[r['sha256'] for r in result['candidates']])))


if __name__=='__main__':main()
