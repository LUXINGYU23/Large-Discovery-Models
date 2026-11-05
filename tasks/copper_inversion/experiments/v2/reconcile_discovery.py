"""Trusted cap reconciliation across client rejections and private evaluation."""
import json
from pathlib import Path
from ldm_tts.engine.run_store import BudgetLedger
from tasks.copper_inversion.v2.training_receipts import eligible


def charged(ledger):
    prefix = 'copper:v2:scientific-source:'
    records = {key[len(prefix):]: value for key, value in
               ledger.metadata.get('cumulative_usage', {}).items() if key.startswith(prefix)}
    if any(value.get('algorithm_feedback_slots') != 1 for value in records.values()):
        raise ValueError('non-unit source charge')
    if ledger.counters.get('algorithm_feedback_slots', 0) != len(records):
        raise ValueError('source count differs from ledger')
    return records


def reconcile(client_root, service_root, cap=321):
    client_root, service_root = Path(client_root), Path(service_root)
    client = charged(BudgetLedger.load(client_root / 'budget.json'))
    private = charged(BudgetLedger.load(service_root / 'budget.json'))
    if len(client) != cap or not private.keys() <= client.keys():
        raise ValueError('incomplete cap or unaccounted private source')
    local_only = client.keys() - private.keys()
    if any(client[sha].get('interface_rejected_sources') != 1 for sha in local_only):
        raise ValueError('local-only charge lacks rejection evidence')
    identity = json.loads((service_root / 'identity.json').read_text())
    complete = eligible(service_root, identity['configuration']['manifest'])
    private_rejected = {sha for sha, usage in private.items()
                        if usage.get('interface_rejected_sources') == 1}
    if set(private) != set(complete) | private_rejected:
        raise ValueError('private sources include unfinished outcomes')
    return dict(status='passed', cap=cap, client_sources=len(client),
                private_sources=len(private), complete_sources=len(complete),
                local_rejections=len(local_only), private_rejections=len(private_rejected))
