import json
import pytest
from ldm_tts.engine.run_store import BudgetLedger
from tasks.copper_inversion.v2.budget import ScientificSlots
from tasks.copper_inversion.experiments.v2 import reconcile_discovery as module


def test_reconcile_local_rejection_and_private_completion(tmp_path, monkeypatch):
    client, private = tmp_path / 'client', tmp_path / 'private'
    for root in (client, private):
        root.mkdir()
        ledger = BudgetLedger(limits={'algorithm_feedback_slots': 3}, path=root / 'budget.json')
        slots = ScientificSlots(ledger.consume_many)
        slots.reserve('complete')
        slots.reserve('private-rejected', interface_rejected=True)
        if root == client:
            slots.reserve('local-rejected', interface_rejected=True)
    (private / 'identity.json').write_text(json.dumps({'configuration': {'manifest': 'unused'}}))
    monkeypatch.setattr(module, 'eligible', lambda *args: {'complete': {}})
    result = module.reconcile(client, private, cap=3)
    assert result['status'] == 'passed'
    assert result['local_rejections'] == result['private_rejections'] == 1
    monkeypatch.setattr(module, 'eligible', lambda *args: {})
    with pytest.raises(ValueError, match='unfinished'):
        module.reconcile(client, private, cap=3)


def test_reconcile_unfinished_local_charge_is_not_rejection(tmp_path, monkeypatch):
    client, private = tmp_path / 'client', tmp_path / 'private'
    for root in (client, private):
        root.mkdir()
        ledger = BudgetLedger(limits={'algorithm_feedback_slots': 2}, path=root / 'budget.json')
        ScientificSlots(ledger.consume_many).reserve('complete')
        if root == client:
            ScientificSlots(ledger.consume_many).reserve('unfinished')
    with pytest.raises(ValueError, match='lacks rejection'):
        module.reconcile(client, private, cap=2)
