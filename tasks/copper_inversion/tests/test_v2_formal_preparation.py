"""Budget and scientific qualification checks, executed remotely."""
import json
from pathlib import Path
import pytest
from tasks.copper_inversion.v2.workflow import parse_args
from tasks.copper_inversion.experiments.v2.calibrate_contract import check


REQUIRED=['--public-catalog','/unused','--evaluation-socket','/unused','--evaluation-contract','a'*64,'--evaluation-service-uid','991','--out-dir','/unused','--embedding-model-path','/unused','--harness-cache-dir','/unused']


def test_full_budget_and_main_history_defaults():
    for seed in (42,43,3407):
        a=parse_args(REQUIRED+['--seed',str(seed)])
        assert a.seed==seed
        assert a.iterations==20 and a.evaluations_per_round==16
        assert 4*a.candidates_per_researcher==32
        assert a._research_strategy.fork_policy.identity()['name']=='main_history_1plus3'
    for extra in (['--iterations','21'],['--seed','-1'],['--candidates-per-researcher','3']):
        with pytest.raises(SystemExit):parse_args(REQUIRED+extra)


def test_declared_analytic_calibration():
    result=check()
    assert result['status']=='passed'
    assert len(result['rows'])==18
    assert all(row['forbidden_positive_rejected'] for row in result['rows'])


def test_task_registration_uses_same_feedback_count():
    data=json.loads((Path(__file__).parents[1]/'experiment.json').read_text())
    assert data['budget']['external_evaluations']==321
    a=data['profiles']['fusion_full']['locked_args']
    assert 1+a['iterations']*a['evaluations-per-round']==321


def test_formal_release_rejects_missing_and_changed_evidence(tmp_path):
    import os
    from tasks.copper_inversion.v2.qualification import verify_release
    from tasks.copper_inversion.v2.suite import file_hash
    from tasks.copper_inversion.v2.fixtures import fixtures
    from tasks.copper_inversion.v2.data import save_problem
    split=tmp_path/'train';split.mkdir();manifest=split/'manifest.json'
    from tasks.copper_inversion.v2.geological_policy import public_priors,POLICY_VERSION
    problem=next(fixtures());problem['priors']=public_priors(problem['domain_mask']&~problem['forbidden_mask'],problem['spacing'])
    save_problem(problem,split/'public')
    manifest.write_text(json.dumps({'cases':[{'case_id':problem['case_id'],'public':'public'}]}))
    with pytest.raises(FileNotFoundError):verify_release(manifest)
    evidence=tmp_path/'evidence';evidence.mkdir()
    for name in ('calibration','population','historical_disjointness','confirmation','isolation','geological_constraints'):
        (evidence/(name+'.json')).write_text('{"status":"passed"}')
    (evidence/'geological_constraints.json').write_text(json.dumps(dict(status='passed',policy=POLICY_VERSION,calibration_scope='independent_development',used_evaluation_truth=False)))
    record=dict(schema='copper-qualified-synthetic-v1',status='qualified',evaluator_uid=os.geteuid(),manifest_sha256={'train/manifest.json':file_hash(manifest)},evidence_sha256={str(p.relative_to(tmp_path)):file_hash(p) for p in evidence.iterdir()})
    (tmp_path/'qualification.json').write_text(json.dumps(record))
    assert verify_release(manifest)==record
    (evidence/'isolation.json').write_text('{"status":"failed"}')
    with pytest.raises(ValueError,match='evidence changed'):verify_release(manifest)


def test_disabled_geological_priors_cannot_be_formal(tmp_path):
    from tasks.copper_inversion.v2.fixtures import fixtures
    from tasks.copper_inversion.v2.data import save_problem
    from tasks.copper_inversion.v2.qualification import verify_geological_constraints
    problem=next(fixtures());problem['priors'].update(area_weight=0.,hard_volume_bounds=None,max_components=None)
    save_problem(problem,tmp_path/'public')
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'cases':[{'case_id':problem['case_id'],'public':'public'}]}))
    with pytest.raises(ValueError,match='no enabled geological prior'):
        verify_geological_constraints(manifest)
