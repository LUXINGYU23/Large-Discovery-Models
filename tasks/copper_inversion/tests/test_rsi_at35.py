"""G3-bound analysis settings and rehearsal labeling tools."""
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tasks.copper_inversion.experiments.rsi.analysis import at35, tagger, bundle
from tasks.copper_inversion.experiments.rsi import launch
from tasks.copper_inversion.rsi import protocol, intervention


def analysis():
    return {'freeze_prefixes':{'M1':['full',5,10], 'E1':['full',12], 'E2':['full']},
            'checkpoint_experiment':'M1','checkpoint_round':10,
            'tagger':{'mode':'tags','rules':tagger.DRAFT_RULES,'cluster_reference':None}}


def test_schema_and_prefix_binding():
    value = json.loads((protocol.RESOURCES/'rsi/protocol.example.json').read_text())
    value['analysis'] = analysis(); protocol.validate(value)
    other = json.loads(json.dumps(value)); other['analysis']['checkpoint_round'] = 11
    assert protocol.protocol_id(value) != protocol.protocol_id(other)
    launch.check_frozen_prefixes({'experiment':'E1','prefixes':['full',12]}, value['analysis'])
    with pytest.raises(launch.PlanError): launch.check_frozen_prefixes({'experiment':'E1','prefixes':['full']}, value['analysis'])
    with pytest.raises(launch.PlanError): launch.check_frozen_prefixes({'experiment':'M1','checkpoint_round':9,'prefixes':['full',5,10]}, value['analysis'])
    other['analysis']['tagger']['mode'] = 'clusters'
    with pytest.raises(ValueError, match='cluster reference'): protocol.validate(other)
    with pytest.raises(ValueError, match='requires protocol analysis'): protocol.require_analysis({})


def test_at35_packet_labels_and_sweep(tmp_path, monkeypatch):
    runs = [tmp_path/'a', tmp_path/'b']
    for run in runs: run.mkdir()
    ids = []
    for i in range(12):
        body = f'def solve(problem,budget,seed):\n    # hidden comment\n    fast_march_{i} = {i}\n    return fast_march_{i}\n'.encode()
        sha = hashlib.sha256(body).hexdigest(); ids.append(sha)
        p = runs[i%2]/'candidates'/sha/'solver.py'; p.parent.mkdir(parents=True); p.write_bytes(body)
    rules = {'version':'test','rules':[{'name':'fast','description':'fast marching','patterns':['fast_march']}]}
    rules_path = tmp_path/'rules.json'; rules_path.write_text(json.dumps(rules))
    out = tmp_path/'packet'; at35.sample(runs, 17, 10, out, rules_path)
    assert len(list(out.glob('*.py'))) == 10
    assert all('hidden comment' not in p.read_text() for p in out.glob('*.py'))
    assert 'fast marching' in (out/'definitions.json').read_text()
    with (out/'labels.csv').open(newline='') as f: rows = list(csv.DictReader(f))
    labels = tmp_path/'human.csv'
    with labels.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['program_id','fast']); w.writeheader()
        for i,row in enumerate(rows): w.writerow({'program_id':row['program_id'],'fast':int(i<5)})
    a=at35.agreement({str(i):{'x':int(i<5)} for i in range(10)}, {str(i):{'x':int(i<5)} for i in range(10)}, ['x'], [str(i) for i in range(10)])
    assert a['primary_kappa'] == 1
    opposite = at35.agreement({'a':{'x':0}}, {'a':{'x':1}}, ['x'], ['a'])
    assert opposite['primary_kappa'] is None and opposite['excluded_constant'] == ['x']
    with pytest.raises(ValueError, match='incomplete|invalid'): at35.read_labels(out/'labels.csv', set(ids), ['fast'])
    assert at35.sweep(runs)['status']=='passed'
    # Use nondegenerate fixed vectors, to exercise cluster calibration and pinning.
    monkeypatch.setattr(at35, 'Run', lambda p: SimpleNamespace(proposals={1:[{'program_id':x} for x in ids]}, source=lambda sha: next(body for _,body in at35.sources(runs).values() if hashlib.sha256(body).hexdigest()==sha)))
    monkeypatch.setattr(at35, 'reference_vectors', lambda run, keys, version: ({sha:np.array([1.,(i+1)/20]) for i,sha in enumerate(keys)},'test-v1'))
    monkeypatch.setattr(at35, 'calibrate_threshold', lambda _: (.2,{.2:2.}))
    monkeypatch.setattr(at35, 'cluster_labels', lambda vectors, threshold: ['c0' if i%2 else 'c1' for i in range(len(vectors))])
    report = at35.score(out, labels, rules_path, runs, 'model:claude-fable-5-1', labels)
    assert report['mode']=='clusters'
    assert report['rater_id']=='model:claude-fable-5-1' and report['agreement_label']=='model-rater kappa'
    assert report['comparisons']['rater_vs_rater2']['primary_kappa']==1
    ref=report['cluster_reference']; assert hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest()==ref['sha256']
    assert tagger.load_cluster_reference(ref, rules)['feature_version']=='test-v1'


def test_protocol_rules_are_used_for_intervention(tmp_path):
    rules={'version':'x','rules':[{'name':'chosen','patterns':['chosen']}]}
    p={'release':{'name':'rsi-dev'},'analysis':{**analysis(),'tagger':{'mode':'tags','rules':rules,'cluster_reference':None}}}
    srcs=[b'def solve(problem,budget,seed):\n    chosen = 1\n    return chosen\n',b'def solve(problem,budget,seed):\n    return 2\n']
    ids=[]
    for body in srcs:
        sha=hashlib.sha256(body).hexdigest(); ids.append(sha)
        f=tmp_path/'candidates'/sha/'solver.py'; f.parent.mkdir(parents=True); f.write_bytes(body)
    (tmp_path/'proposals').mkdir()
    for n in (1,2): (tmp_path/f'proposals/round_{n:02d}.jsonl').write_text(''.join(json.dumps({'program_id':i})+'\n' for i in ids))
    ck=SimpleNamespace(snapshot={},path=tmp_path,round=2,sha256='a'*64,
        source_manifest={'protocol_id':protocol.protocol_id(p)},records=[{'status':'valid','risk':.1,'program_id':ids[0]}])
    r=intervention.select_frozen(ck,p); assert r['target_tag']=='chosen'
    note=intervention.from_checkpoint(ck,p)
    assert intervention.identity(note)['tagger_rules_sha256']==tagger.rules_sha256(rules)
    p['analysis']['tagger']['rules']={'version':'other','rules':rules['rules']}
    with pytest.raises(ValueError, match='frozen tagger'): intervention.from_checkpoint(ck,p)


def test_cluster_target_and_shared_membership(tmp_path, monkeypatch):
    from tasks.copper_inversion.experiments.rsi.analysis import measures
    rules={'version':'x','rules':[{'name':'alpha','patterns':['alpha']},{'name':'beta','patterns':['beta']}]}
    ref={'schema':'rsi-cluster-reference-v1','threshold':.2,'feature_version':'fake',
         'vectors':[[1,0],[1,.01],[0,1],[.01,1]],'labels':['c0','c0','c1','c1'],
         'tags':[['alpha'],['alpha'],['beta'],['beta']], 'rules_sha256':tagger.rules_sha256(rules)}
    path=tmp_path/'ref.json'; path.write_text(json.dumps(ref))
    spec={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    p={'release':{'name':'rsi-dev'},'analysis':{**analysis(),'tagger':{'mode':'clusters','rules':rules,'cluster_reference':spec}}}
    ids=[]
    for i in range(2):
        body=f'def solve(problem,budget,seed):\n    return {i}\n'.encode()
        sha=hashlib.sha256(body).hexdigest(); ids.append(sha)
        f=tmp_path/'candidates'/sha/'solver.py'; f.parent.mkdir(parents=True); f.write_bytes(body)
    (tmp_path/'proposals').mkdir()
    for n in (1,2): (tmp_path/f'proposals/round_{n:02d}.jsonl').write_text(''.join(json.dumps({'program_id':i})+'\n' for i in ids))
    ck=SimpleNamespace(snapshot={},path=tmp_path,round=2,sha256='a'*64,
                       source_manifest={'protocol_id':protocol.protocol_id(p)},records=[{'status':'valid','risk':.1,'program_id':ids[0]}])
    monkeypatch.setattr(measures, 'run_vectors', lambda run, keys: ({k:np.array([1.,0.]) if k==ids[0] else np.array([0.,1.]) for k in keys},'fake'))
    record=intervention.select_frozen(ck,p)
    assert record['target_tag']=='c0' and record['note_tag']=='alpha' and record['shares']['c0']==.5
    note=intervention.from_checkpoint(ck,p)
    assert 'alpha' in note['note'] and intervention.identity(note)['target_tag']=='c0'
    assert tagger.assign_clusters([[1,0],[0,1]],ref['vectors'],ref['labels'],ref['threshold']) == ['c0','c1']
    measured = measures.RunMeasures.__new__(measures.RunMeasures)
    measured.family_config = {'mode':'cluster'}
    measured.programs = {ids[0]:{'family':'c0'}, ids[1]:{'family':'c1'}}
    measured.run = SimpleNamespace(proposals={3:[{'program_id':pid} for pid in ids]})
    assert measured.tag_share(record['target_tag'], 3) == .5
    path.write_text('{}')
    with pytest.raises(ValueError,match='hash changed'): intervention.from_checkpoint(ck,p)


def test_formal_analysis_has_no_default_override(tmp_path):
    value=json.loads((protocol.RESOURCES/'rsi/protocol.example.json').read_text())
    value['release']['name']='rsi-r1'
    original=value.pop('analysis')
    with pytest.raises(ValueError,match='requires protocol analysis'):
        protocol.require_analysis(value)
    value['analysis']=original
    needed={k:v for k,v in bundle.DEFAULTS.items() if k not in {'freeze_prefixes','checkpoint_experiment','checkpoint_round','tagger_rules','family'}}
    path=tmp_path/'config.json'; path.write_text(json.dumps(needed))
    config=bundle.load_config(path,value)
    assert config['freeze_prefixes']==value['analysis']['freeze_prefixes']
    needed['checkpoint_round']=9; path.write_text(json.dumps(needed))
    with pytest.raises(ValueError,match='differs from frozen protocol'):
        bundle.load_config(path,value)
    with pytest.raises(ValueError,match='explicit statistical configuration'):
        bundle.load_config(None,value)


def test_formal_entrypoints_reject_missing_analysis(tmp_path):
    from tasks.copper_inversion.rsi import workflow
    p=json.loads((protocol.RESOURCES/'rsi/protocol.example.json').read_text())
    p['release']['name']='rsi-r1'; p.pop('analysis')
    path=tmp_path/'protocol.json'; path.write_text(json.dumps(p))
    batch={'experiment':'M1','tier':'S','runs':[{'arm':'ldm4geo','judge':'standard','seed':101}]}
    with pytest.raises(ValueError,match='requires protocol analysis'):
        launch.check_batch(batch,p,{})
    args=workflow.parse_args(['--protocol',str(path),'--experiment','M1','--arm','ldm4geo','--tier','S','--seed','101',
        '--out-dir',str(tmp_path/'ldm4geo-s101'),'--public-catalog',str(tmp_path/'catalog'),
        '--evaluation-socket',str(tmp_path/'socket'),'--evaluation-contract','a'*64,'--evaluation-service-uid','991',
        '--embedding-model-path',str(tmp_path/'model'),'--sidecar-image','unused','--harness-cache-dir',str(tmp_path/'cache'),
        '--api-key-file',str(tmp_path/'key')])
    with pytest.raises(ValueError,match='requires protocol analysis'):
        workflow.prepare(args)


def test_score_cli_requires_rater_identity():
    with pytest.raises(SystemExit) as exc:
        at35.main(['score','--packet','p','--labels','l','--rules','r','--run','a','--run','b'])
    assert exc.value.code == 2
