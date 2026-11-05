import json
import os
import socket
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from ldm_tts.contracts.evaluation import EvaluationPaused
from ldm_tts.engine.run_store import BudgetExceededError
from tasks.copper_inversion.core.candidate import seal
from tasks.copper_inversion.v2.service_client import ServiceEvaluator, validate_metrics


def metrics():
    return dict(mean_iou=.5,loss_total=.5,research_utility=.5,legal_rate=1.,
                parent_count=1,solve_count=2,failed_solves=0,mean_solver_wall_seconds=.2)


@pytest.mark.parametrize('field',['contract_digest','public_digest','seconds','slots'])
def test_service_handshake_binds_observations_and_budget(tmp_path,field):
    client=ServiceEvaluator(tmp_path,'unused','a'*64,service_uid=os.getuid())
    info=dict(status='ready',contract_digest='a'*64,public_digest='b'*64,seconds=10,slots=5,used_slots=0)
    client.request=Mock(return_value=info)
    assert client.verify_contract('b'*64,10,5)==info
    info[field]='wrong'
    with pytest.raises(EvaluationPaused):client.verify_contract('b'*64,10,5)


@pytest.mark.parametrize('change',[
    {'mean_iou':float('nan')},{'mean_iou':2},{'loss_total':.1},
    {'solve_count':1.5},{'failed_solves':3},{'legal_rate':True},
    {'extra':1},{'mean_solver_wall_seconds':-1},
])
def test_invalid_metrics_pause(change):
    with pytest.raises(EvaluationPaused):validate_metrics(metrics()|change)


def test_replay_requires_contract_and_source_receipt(tmp_path):
    payload=seal(tmp_path,b'def solve(problem,budget,seed): return None\n')
    candidate=SimpleNamespace(payload=payload,canonical_key=payload['sha256'],candidate_id=payload['sha256'])
    client=ServiceEvaluator(tmp_path,'unused','a'*64,service_uid=os.getuid())
    client.request=Mock(return_value=dict(status='succeeded',source_sha256=payload['sha256'],metrics=metrics()))
    first=client.evaluate(candidate)
    assert client.evaluate(candidate)==first
    assert client.request.call_count==1
    assert client.request.call_args.args[0]['contract_digest']=='a'*64
    path=tmp_path/'evaluations'/client.evaluation_attempt_usage_key(candidate)/'result.json'
    saved=json.loads(path.read_text());saved['metadata']['service_contract_digest']='b'*64
    path.write_text(json.dumps(saved))
    with pytest.raises(EvaluationPaused):client.evaluate(candidate)


@pytest.mark.parametrize('status,exception',[
    ('budget_exhausted',BudgetExceededError),('paused',EvaluationPaused),
    ('contract_mismatch',EvaluationPaused),('infrastructure_error',EvaluationPaused),
])
def test_unavailable_result_never_becomes_score(tmp_path,status,exception):
    payload=seal(tmp_path,b'def solve(problem,budget,seed): return None\n')
    candidate=SimpleNamespace(payload=payload,canonical_key=payload['sha256'],candidate_id=payload['sha256'])
    client=ServiceEvaluator(tmp_path,'unused','a'*64,service_uid=os.getuid())
    client.request=Mock(return_value=dict(status=status))
    with pytest.raises(exception):client.evaluate(candidate)
    assert not (tmp_path/'evaluations').exists()


def test_real_unix_peer_and_disconnect(tmp_path):
    path=str(tmp_path/'socket')
    server=socket.socket(socket.AF_UNIX);server.bind(path);server.listen()
    def respond():
        with server.accept()[0] as connection:
            connection.recv(4096)
            connection.sendall(b'{"status":"ready"}\n')
    worker=threading.Thread(target=respond);worker.start()
    client=ServiceEvaluator(tmp_path,path,'a'*64,service_uid=os.getuid(),timeout=2)
    assert client.request({'operation':'info'})=={'status':'ready'}
    worker.join(3);assert not worker.is_alive()
    wrong=ServiceEvaluator(tmp_path,path,'a'*64,service_uid=os.getuid()+1,timeout=2)
    with pytest.raises(EvaluationPaused,match='peer identity'):wrong.request({'operation':'info'})
    server.close()
    with pytest.raises(EvaluationPaused,match='unavailable'):client.request({'operation':'info'})


def test_private_interface_failure_is_charged_zero_and_replayed(tmp_path):
    from ldm_tts.engine.run_store import BudgetLedger
    from tasks.copper_inversion.v2.budget import ScientificSlots
    ledger=BudgetLedger(limits={'algorithm_feedback_slots':1},path=tmp_path/'budget.json')
    client=ServiceEvaluator(tmp_path,'unused','a'*64,service_uid=os.getuid(),slots=ScientificSlots(ledger.consume_many))
    payload=seal(tmp_path,b'def solve(problem,budget,seed): return None\n')
    candidate=SimpleNamespace(payload=payload,canonical_key=payload['sha256'],candidate_id=payload['sha256'])
    failure=metrics()|dict(mean_iou=0.,loss_total=1.,research_utility=0.,legal_rate=0.,failed_solves=2,mean_solver_wall_seconds=0.)
    client.request=Mock(return_value=dict(status='invalid_interface',source_sha256=payload['sha256'],metrics=failure))
    result=client.evaluate(candidate)
    assert result.succeeded and result.metrics['loss_total']==1
    assert result.metadata['service_outcome']=='invalid_interface'
    assert client.evaluate(candidate)==result
    assert client.request.call_count==1 and ledger.counters['algorithm_feedback_slots']==1


def test_corrupt_receipt_pauses_without_resubmission(tmp_path):
    payload=seal(tmp_path,b'def solve(problem,budget,seed): return None\n')
    candidate=SimpleNamespace(payload=payload,canonical_key=payload['sha256'],candidate_id=payload['sha256'])
    client=ServiceEvaluator(tmp_path,'unused','a'*64,service_uid=os.getuid())
    path=tmp_path/'evaluations'/client.evaluation_attempt_usage_key(candidate)/'result.json'
    path.parent.mkdir(parents=True);path.write_text('{')
    client.request=Mock()
    with pytest.raises(EvaluationPaused):client.evaluate(candidate)
    client.request.assert_not_called()
