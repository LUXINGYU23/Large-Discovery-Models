import json
from types import SimpleNamespace

import pytest

from tasks.copper_inversion.experiments.rsi import access_audit
from tasks.copper_inversion.tests import rsi_rpc_probe


def test_project_access_audit_and_existing_canary(tmp_path, monkeypatch):
    private = tmp_path / 'private'; private.mkdir()
    release = private / 'manifest.json'; release.write_text('{}')
    monkeypatch.setattr(access_audit.os, 'geteuid', lambda: 0)
    monkeypatch.setattr(access_audit.os, 'fchown', lambda *args: None)
    monkeypatch.setattr(access_audit.pwd, 'getpwuid', lambda uid: SimpleNamespace(pw_uid=991, pw_gid=991, pw_name='eval'))
    monkeypatch.setattr(access_audit.pwd, 'getpwnam', lambda name: SimpleNamespace(pw_uid=1000))
    original = access_audit.Path.lstat
    def lstat(path):
        result = original(path)
        if path.name == 'access-canary.txt':
            return SimpleNamespace(st_uid=991, st_mode=result.st_mode)
        return result
    monkeypatch.setattr(access_audit.Path, 'lstat', lstat)
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0 if args[2] == 'eval' else 1, stderr=b'')
    monkeypatch.setattr(access_audit.subprocess, 'run', run)
    report = access_audit.audit(tmp_path, 'client', release)
    assert report['account_can_read'] and report['research_host_user_direct_read_denied']
    assert len(calls) == 4 and all(row[:2] == ['runuser', '-u'] for row in calls)
    canary = private / 'access-canary.txt'
    assert canary.read_text() == 'private evaluator access probe\n'
    before = canary.stat().st_ino
    access_audit.audit(tmp_path, 'client', release)
    assert canary.stat().st_ino == before
    monkeypatch.setattr(access_audit.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=1, stderr=b'runuser failed'))
    failed = access_audit.audit(tmp_path, 'client', release)
    assert not failed['account_can_read'] and not failed['research_host_user_direct_read_denied']
    outside = tmp_path / 'outside'; outside.write_text('{}')
    with pytest.raises(ValueError, match='inside'):
        access_audit.audit(tmp_path, 'client', outside)


class FakeService:
    def __init__(self, fault=None):
        self.sources = {}; self.slots = 4; self.fault = fault; self.evaluations = 0
    def send(self, path, request):
        if request == {'operation': 'info'}:
            return dict(status='ready', used_slots=len(self.sources), slots=self.slots, contract_digest='contract')
        if request.get('contract_digest') != 'contract':
            return {'status': 'contract_mismatch'}
        if request.get('operation') != 'evaluate' or set(request) != {'operation', 'source', 'contract_digest'}:
            return {'status': 'rejected_request'}
        source = request['source']; self.evaluations += 1
        if source not in self.sources:
            if len(self.sources) == self.slots:
                return {'status': 'budget_exhausted'}
            invalid = 'np.ones' in source
            row = dict(source_sha256=__import__('hashlib').sha256(source.encode()).hexdigest(),
                       status='rejected' if invalid else 'valid', risk=1. if invalid else .5,
                       risk_by_case={'case': 1. if invalid else .5}, legal_rate=0. if invalid else 1.,
                       failed_solves=1 if invalid else 0, seconds_per_solve=1.,
                       message='interface rejected' if invalid else '')
            self.sources[source] = row
        result = dict(self.sources[source])
        if self.fault == 'leak': result['mean_iou'] = .7
        if self.fault == 'guest_failure' and 'np.ones' not in source: result['status'] = 'failed'
        return result


def test_rpc_probe_whitelist_budget_and_replay():
    fake = FakeService()
    result = rsi_rpc_probe.probe('socket', '/private/service', '/private/release', send=fake.send)
    assert result['status'] == 'passed' and result['slots'] == result['final_used_slots'] == 4
    assert all(result['checks'].values())
    assert fake.evaluations == fake.slots + 2  # one replay, one over cap
    assert len(fake.sources) == fake.slots
    assert '/private/service' in next(iter(fake.sources))
    for fault in ('leak', 'guest_failure'):
        fake = FakeService(fault)
        assert rsi_rpc_probe.probe('socket', '/private/service', '/private/release', send=fake.send)['status'] == 'failed'
        assert len(fake.sources) <= 2


@pytest.mark.parametrize('fault', ['info', 'missing_digest', 'wrong_digest', 'override', 'unknown', 'replay', 'invalid', 'cap'])
def test_rpc_probe_rejects_broken_service(fault):
    fake = FakeService()
    def broken(path, payload):
        result = fake.send(path, payload)
        if fault == 'info' and payload == {'operation': 'info'}:
            return {**result, 'used_slots': 1}
        if fault == 'missing_digest' and payload.get('operation') == 'evaluate' and 'contract_digest' not in payload:
            return {'status': 'rejected_request'}
        if fault == 'wrong_digest' and payload.get('contract_digest') == 'wrong':
            return {'status': 'rejected_request'}
        if fault == 'override' and 'manifest' in payload:
            return {'status': 'contract_mismatch'}
        if fault == 'unknown' and payload.get('operation') == 'unknown':
            return {'status': 'contract_mismatch'}
        if fault == 'replay' and fake.evaluations == 2 and payload.get('operation') == 'evaluate':
            return {**result, 'message': 'changed'}
        if fault == 'invalid' and 'np.ones' in payload.get('source', ''):
            return {'status': 'invalid_interface'}
        if fault == 'cap' and result == {'status': 'budget_exhausted'}:
            return {'status': 'rejected_request'}
        return result
    assert rsi_rpc_probe.probe('/socket', '/private/service', '/private/release', send=broken)['status'] == 'failed'
    assert len(fake.sources) <= fake.slots
