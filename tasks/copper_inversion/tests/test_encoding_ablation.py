import json
from dataclasses import replace
import numpy as np
import pytest
from tasks.copper_inversion.core import encoding
from tasks.copper_inversion.core.candidate import seal, Domain
from ldm_tts.contracts import RawProposal
from tasks.copper_inversion.core.workflow import main


def candidate(root, body):
    return Domain(root).admit(RawProposal(seal(root, body.encode()), 'test'))


def test_normalization_preserves_names_values_order():
    a = b'# comment\ndef solve( p,b,s ):\n return p + 2\n'
    b = b'def solve(p, b, s): return p+2\n'
    assert encoding.normalized_code(a) == encoding.normalized_code(b)
    assert encoding.normalized_code(a) != encoding.normalized_code(b.replace(b'+2', b'+3'))
    assert encoding.normalized_code(a) != encoding.normalized_code(b.replace(b'p', b'q'))


@pytest.mark.parametrize('mode,dimension', [('ast',256),('qwen',1024),('fusion',1280)])
def test_fusion_block_weights_cache_and_metadata_independence(tmp_path, monkeypatch, mode, dimension):
    calls = []
    class FakeQwen:
        def __init__(self,*args): pass
        def encode(self,code):
            calls.append(code)
            return encoding.unit(np.arange(1,1025)), 30
    monkeypatch.setattr(encoding,'QwenEmbedding',FakeQwen)
    c = candidate(tmp_path, 'def solve(p,b,s): return p\n')
    encoder = encoding.CodeEncoder(tmp_path,mode,'fake')
    v = encoder.encode(c)
    assert len(v.values) == dimension
    assert np.linalg.norm(v.values) == pytest.approx(1.)
    record = json.loads((tmp_path/'encodings'/f'{c.candidate_id}.json').read_text())
    if mode=='fusion':
        assert np.linalg.norm(v.values[:256]) == pytest.approx(np.sqrt(.5))
        assert np.linalg.norm(v.values[256:]) == pytest.approx(np.sqrt(.5))
        assert np.allclose(v.values[:256],np.array(record['ast_values'])/np.sqrt(2))
    assert encoder.encode(replace(c,metadata={'loss':-1e9,'rationale':'best'})) == v
    assert len(calls)==int(mode!='ast')
    record['version']='tampered'
    (tmp_path/'encodings'/f'{c.candidate_id}.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='identity'):
        encoder.encode(c)


def test_no_silent_token_truncation():
    q=object.__new__(encoding.QwenEmbedding)
    class Model:
        max_seq_length=4
        def tokenizer(self,*args,**kwargs):
            assert kwargs['truncation'] is False
            return {'input_ids':[0]*5}
        def encode(self,*args,**kwargs):raise AssertionError('must reject before inference')
    q.model=Model()
    with pytest.raises(ValueError,match='no truncation'):q.encode('code')


def test_sixteen_proposals_four_selected_and_resume(tmp_path):
    root=tmp_path/'run'
    args=['--mock','--search-method','ldm_harness','--iterations','3','--candidates-per-researcher','4',
          '--evaluations-per-round','4','--out-dir',str(root)]
    assert main(args)==0
    result=json.loads((root/'result.json').read_text())
    assert result['observations']==13
    events=[json.loads(l) for l in (root/'events.jsonl').read_text().splitlines()]
    selected=[e['payload'] for e in events if e['event_type']=='candidates_selected']
    assert [len(e['predictions']) for e in selected]==[0,16,16,16]
    assert [len(e['selected_candidate_ids']) for e in selected]==[1,4,4,4]
    assert len(list((root/'encodings').glob('*.json')))==49
    panels=[json.loads(p.read_text()) for p in sorted((root/'harness/branches').glob('*/panel.json'))]
    assert [p['history_count'] for p in panels]==[1,5,9]
    assert all(len(p['proposals'])==16 for p in panels)
    before=json.loads((root/'budget.json').read_text())['counters']
    assert main(args+['--resume'])==0
    assert json.loads((root/'budget.json').read_text())['counters']==before


def test_public_instructions_use_turn_candidate_count():
    from tasks.copper_inversion.core.data import RESOURCES
    for name in ('TASK.md', 'harness/profiles/researcher/AGENTS.md', 'harness/tools/copper.mjs'):
        text=(RESOURCES/name).read_text()
        assert 'candidates_required' in text
        assert 'Submit exactly one candidate' not in text
        assert 'Submit exactly one entry' not in text


def test_submission_repair_budget_scales_with_candidate_count():
    from tasks.copper_inversion.core.harness import submission_contract
    assert submission_contract(1).max_validation_attempts == 3
    assert submission_contract(4).max_validation_attempts == 12


def test_snapshot_error_identifies_actual_candidate(tmp_path):
    import hashlib
    from types import SimpleNamespace
    from tasks.copper_inversion.core.harness import parse_snapshot, SolverSubmissionError, submission_error
    rows, artifacts = [], []
    for i in range(4):
        body = f'def solve(p,b,s): return {i}\n'.encode()
        (tmp_path/f'{i}.py').write_bytes(body)
        rows.append(dict(artifact_path=f'candidates/{i}/solver.py', parent_sha256=None,
                         change_summary='test', rationale='test'))
        artifacts.append(SimpleNamespace(snapshot_path=f'{i}.py', relative_path=rows[-1]['artifact_path'],
            path_pointer=f'/candidates/{i}/artifact_path', sha256=hashlib.sha256(body).hexdigest(), size_bytes=len(body)))
    with pytest.raises(SolverSubmissionError) as caught:
        parse_snapshot({'candidates':rows}, artifacts, tmp_path, {artifacts[2].sha256}, 4)
    assert submission_error(caught.value).path == '/candidates/2/artifact_path'
    assert 'already evaluated' in str(caught.value)


def test_interface_error_reports_native_failure_without_supervisor_envelope(tmp_path):
    from types import SimpleNamespace
    from tasks.copper_inversion.core.interface_checks import InterfaceChecks
    job = tmp_path/'job'
    job.mkdir()
    (job/'log.txt').write_text('COPPER_RESOURCE_V1 private-envelope\nstd::system_error: Resource temporarily unavailable\n')
    execution = SimpleNamespace(run=lambda *a, **k: (job, dict(exit_code=134, oom=False, reason=None)))
    with pytest.raises(ValueError) as caught:
        InterfaceChecks(tmp_path, execution, 'drillholes_3d')({'sha256':'a'*64, 'source_path':'candidate/solver.py'})
    assert 'Resource temporarily unavailable' in str(caught.value)
    assert 'contacts=False' in str(caught.value)
    assert 'private-envelope' not in str(caught.value)
