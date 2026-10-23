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
