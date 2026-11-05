"""Shared response routing keeps pending snapshots without blocking peer events."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from ldm_tts.harness import (HarnessClient, HarnessPoolConfig, HarnessProfile,
    HarnessSubmissionContract, HarnessSubmissionValidation, HarnessSubmissionError, HarnessError)
from ldm_tts.harness.protocol import HARNESS_PROTOCOL_VERSION


def fixture(tmp_path):
    config = HarnessPoolConfig(artifact_root=tmp_path,
        profiles=tuple(HarnessProfile(p, Path('/AGENTS.md'), agents_sha256='a'*64) for p in ('a','b')),
        campaign_id='batch', task_id='test', case_id='test', seed=0,
        submission_contract=HarnessSubmissionContract('test','submit',
            {'type':'object','additionalProperties':False,'properties':{}}))
    client = HarnessClient(('unused',), api_key='fixture', config=config)
    client._protocol_version = HARNESS_PROTOCOL_VERSION
    client._process = SimpleNamespace(stdin=True, poll=lambda: None)
    sent = []
    client._send_frame = sent.append
    return client, sent


def frame(kind, **fields):
    return dict(type=kind, requestId='python-000001', campaignId='batch',
                protocolVersion=HARNESS_PROTOCOL_VERSION, **fields)


def submission(profile, attempt=1):
    value = {'value': profile + str(attempt)}
    encoded = json.dumps({'artifacts':[], 'submission':value},sort_keys=True,separators=(',',':'))
    return frame('submission_validation_requested', profileId=profile, turnId='turn_'+profile,
        attemptIndex=attempt, validationId=profile+str(attempt), submission=value, artifacts=[],
        submissionJson=encoded, submissionDigest=hashlib.sha256(encoded.encode()).hexdigest())


@pytest.mark.parametrize('repair', [False, True])
def test_pending_peers_and_repair_are_routed(tmp_path, repair):
    client, sent = fixture(tmp_path)
    frames = [submission('a'), submission('b')]
    if repair:
        frames.append(submission('a', 2))
    for value in [*frames, frame('turn_committed', turns=[])]:
        client._responses.put(json.dumps(value))
    seen = []
    def validate(requests):
        seen.append(tuple(r.profile_id for r in requests))
        if repair and requests[0].profile_id == 'a' and requests[0].attempt_index == 1:
            return {'a': HarnessSubmissionValidation('retry', (
                HarnessSubmissionError('/value','invalid','repair'),))}
        if len(requests) < 2:
            assert len(sent) == (2 if repair else 1)  # request, and optional retry only
            return {}
        return {r.profile_id: HarnessSubmissionValidation() for r in requests}
    client._request('run_turn', {'turns':[]}, 'turn_committed', submission_batch_validator=validate)
    assert seen == ([('a',), ('b',), ('b','a')] if repair else [('a',), ('a','b')])
    assert [v['decision'] for v in sent[1:]] == (['retry','accept','accept'] if repair else ['accept','accept'])


def test_commit_with_unresolved_snapshot_is_rejected(tmp_path):
    client, _ = fixture(tmp_path)
    for value in [submission('a'), frame('turn_committed', turns=[])]:
        client._responses.put(json.dumps(value))
    with pytest.raises(HarnessError, match='unresolved'):
        client._request('run_turn', {'turns':[]}, 'turn_committed', submission_batch_validator=lambda r: {})


def test_slow_validation_does_not_starve_provider_authorization(tmp_path, monkeypatch):
    import threading
    client, sent = fixture(tmp_path)
    authorized = threading.Event()
    finished = threading.Event()
    def authorize(response, callback):
        assert not finished.is_set()
        authorized.set()
    monkeypatch.setattr(client, '_answer_provider_authorization', authorize)
    def validate(requests):
        assert authorized.wait(2), 'provider request blocked behind admission'
        finished.set()
        return {r.profile_id: HarnessSubmissionValidation() for r in requests}
    for response in [submission('a'), frame('provider_authorization_requested'), frame('turn_committed', turns=[])]:
        client._responses.put(json.dumps(response))
    client._request('run_turn', {'turns':[]}, 'turn_committed', submission_batch_validator=validate,
                    provider_authorizer=lambda _: True)
    assert finished.is_set() and sent[-1]['decision']=='accept'


def test_concurrent_budget_reservations_keep_cap_and_replay(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from ldm_tts.engine.run_store import BudgetLedger, BudgetExceededError
    ledger=BudgetLedger(limits={'slots':32},path=tmp_path/'budget.json')
    def reserve(i):
        try:ledger.consume_many({'slots':1},usage_key=f'source:{i%48}')
        except BudgetExceededError:return False
        return True
    with ThreadPoolExecutor(max_workers=16) as pool:list(pool.map(reserve,range(192)))
    saved=BudgetLedger.load(ledger.path)
    assert saved.counters['slots']==32
    assert len(saved.metadata['cumulative_usage'])==32
    assert saved.snapshot()==ledger.snapshot()


@pytest.mark.parametrize('late_peer', [False, True])
@pytest.mark.parametrize('exhausted', [False, True])
def test_terminal_rejection_releases_batch_barrier(tmp_path, monkeypatch, late_peer, exhausted):
    from ldm_tts.harness import HarnessTurn
    client, _ = fixture(tmp_path)
    turns = tuple(HarnessTurn(profile_id=p, turn_id='turn_'+p, round_index=1,
        history_from_seq=0, history_to_seq=0, history_digest='a'*64, message='test') for p in ('a','b'))
    from dataclasses import replace
    client.config = replace(client.config, submission_contract=replace(client.config.submission_contract, max_validation_attempts=1))
    calls = []
    def validate(requests):
        calls.append(1)
        return {'a': HarnessSubmissionValidation('retry' if exhausted else 'reject_turn', (
            HarnessSubmissionError('/','invalid','terminal'),))}
    def request(*args, **kwargs):
        callback = kwargs['submission_batch_validator']
        a = client._parse_submission_request(submission('a'))
        b = client._parse_submission_request(submission('b'))
        if late_peer:
            assert callback((a,))['a'].decision == 'reject_turn'
            answer = callback((b,))
        else:
            answer = callback((a,b))
        assert answer['b'].decision == 'reject_turn'
        assert answer['b'].errors[0].code == 'peer_rejected'
        raise HarnessError('test complete')
    monkeypatch.setattr(client, '_request', request)
    with pytest.raises(HarnessError, match='test complete'):
        client.run_turn(turns, submission_batch_validator=validate)
    assert len(calls) == 1


def test_host_validation_has_independent_timeout(tmp_path):
    import threading
    import time
    client, sent = fixture(tmp_path)
    def peer():
        time.sleep(.08)
        client._responses.put(json.dumps(submission('a')))
    worker=threading.Thread(target=peer);worker.start()
    def validate(request):
        time.sleep(.08)
        client._responses.put(json.dumps(frame('turn_committed', turns=[])))
        return HarnessSubmissionValidation('accept')
    client._request('run_turn', {'turns':[]}, 'turn_committed', timeout_seconds=.13,
                    submission_validator=validate)
    worker.join()
    assert sent[-1]['decision']=='accept'
