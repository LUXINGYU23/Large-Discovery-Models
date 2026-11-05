from tasks.copper_inversion.core.execution import digest
from types import SimpleNamespace
from unittest.mock import Mock
from ldm_tts.engine.run_store import BudgetLedger
from tasks.copper_inversion.core.candidate import Domain
from tasks.copper_inversion.v2.budget import ScientificSlots
from tasks.copper_inversion.v2.service import EvaluatorService


def service(root):
    backend=EvaluatorService.__new__(EvaluatorService)
    backend.root=root;backend.identity={"scope":"test"}
    backend.ledger=BudgetLedger(limits={"algorithm_feedback_slots":1},path=root/"budget.json")
    backend.slots=ScientificSlots(backend.ledger.consume_many)
    backend.domain=Domain(root)
    backend.evaluator=SimpleNamespace(evaluate=Mock(return_value=SimpleNamespace(metrics={"mean_iou":.5})))
    return backend


def test_fixed_wire_schema_cannot_select_paths_or_budget(tmp_path):
    backend=service(tmp_path)
    for request in ({"operation":"test"},{"operation":"evaluate","source":"x","manifest":"/secret"},
                    {"operation":"evaluate","source":"x","seconds":999},["evaluate"],{"operation":"shell","command":"id"}):
        assert backend.dispatch(request)=={"status":"rejected_request"}
    backend.evaluator.evaluate.assert_not_called()
    assert backend.ledger.counters=={}


def test_source_feedback_cap_and_private_error_redaction(tmp_path):
    backend=service(tmp_path)
    source="def solve(problem,budget,seed):\n    return None\n"
    assert backend.dispatch({"operation":"evaluate","source":source,"contract_digest":digest(backend.identity)})["status"]=="succeeded"
    assert backend.dispatch({"operation":"evaluate","source":source+"# new\n","contract_digest":digest(backend.identity)})=={"status":"budget_exhausted"}
    assert backend.evaluator.evaluate.call_count==1
    backend.evaluator.evaluate.side_effect=OSError("private dataset /secret/truth.npy")
    result=backend.dispatch({"operation":"evaluate","source":source,"contract_digest":digest(backend.identity)})
    assert result=={"status":"infrastructure_error"} and "/secret" not in str(result)
    assert backend.ledger.counters["algorithm_feedback_slots"]==1


def test_wrong_contract_has_no_feedback_or_execution(tmp_path):
    backend=service(tmp_path)
    result=backend.dispatch(dict(operation="evaluate",source="def solve(p,b,s): return None",contract_digest="0"*64))
    assert result=={"status":"contract_mismatch"}
    assert backend.ledger.counters=={}
    backend.evaluator.evaluate.assert_not_called()


def test_private_interface_rejection_preserves_denominator_and_slot(tmp_path):
    from ldm_tts.contracts import CandidateRejection
    backend=service(tmp_path)
    backend.interface_failure_metrics=dict(mean_iou=0.,loss_total=1.,research_utility=0.,legal_rate=0.,
        parent_count=8,solve_count=16,failed_solves=16,mean_solver_wall_seconds=0.)
    backend.domain.admit=Mock(return_value=CandidateRejection('invalid_interface','fixture failed','private_service'))
    request=dict(operation='evaluate',source='def solve(p,b,s): return None',contract_digest=digest(backend.identity))
    first=backend.dispatch(request)
    assert first['status']=='invalid_interface'
    assert first['metrics']['failed_solves']==16
    assert backend.dispatch(request)==first
    assert backend.ledger.counters['algorithm_feedback_slots']==1
    backend.evaluator.evaluate.assert_not_called()


def test_seal_selects_three_from_private_receipts_and_ends_training(tmp_path,monkeypatch):
    backend=service(tmp_path);backend.manifest_path=tmp_path/'private-manifest.json'
    records={str(i)*64:dict(source_path=f'/private/candidate-{i}',metrics=dict(
        mean_iou=(5-i)/10,loss_total=1-i/10,legal_rate=1.,mean_solver_wall_seconds=1.)) for i in range(1,5)}
    monkeypatch.setattr('tasks.copper_inversion.v2.service.eligible',lambda *args:records)
    request=dict(operation='seal_training',contract_digest=digest(backend.identity))
    reply=backend.dispatch(request)
    assert reply['source_sha256s']==['4'*64,'3'*64,'2'*64]
    assert '/private' not in str(reply)
    assert backend.dispatch(request)==reply
    assert backend.dispatch(dict(operation='evaluate',source='def solve(p,b,s): return None',
        contract_digest=digest(backend.identity)))=={'status':'training_sealed'}
    backend.evaluator.evaluate.assert_not_called()
    assert backend.ledger.counters=={}
