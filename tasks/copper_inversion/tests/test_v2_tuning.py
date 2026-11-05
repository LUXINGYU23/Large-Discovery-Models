import ast
import json
import sys
from collections import Counter
from types import SimpleNamespace
from pathlib import Path
import pytest
from tasks.copper_inversion.experiments.v2.tune_stochastic import configurations,source_for


@pytest.mark.parametrize("method",["mcmc_v2","acq_v2"])
def test_frozen_search_space_and_source_variants(method):
    rows=configurations(method,321,47)
    assert len(rows)==321 and rows==configurations(method,321,47)
    assert rows[1:]!=configurations(method,321,48)[1:]
    assert rows[0]["temperature"]==.1 and rows[0]["initializer"]=="nearest"
    assert {r["initializer"] for r in rows}=={"nearest","random_field"}
    template=(Path(__file__).parents[1]/"experiments/v2/stochastic_solver.py").read_text()
    sources=[source_for(template,r) for r in rows]
    assert len(set(sources))==321
    for params,source in zip(rows,sources):
        tree=ast.parse(source)
        config=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="CONFIG" for t in n.targets))
        assert ast.literal_eval(config)==params
        assert 1e-3<=params["temperature"]<=10 and .5<=params["step_cells"]<=4
        assert .125<=params["range_scale"]<=2


def test_no_extra_feedback_slots():
    with pytest.raises(ValueError):configurations("mcmc_v2",322,47)
    assert len(configurations("mcmc_v2",1,47))==1


def test_source_identity_survives_plan_serialization():
    template=(Path(__file__).parents[1]/"experiments/v2/stochastic_solver.py").read_text()
    for params in configurations("acq_v2",17,42):
        restored=json.loads(json.dumps(params,sort_keys=True))
        assert source_for(template,params)==source_for(template,restored)


def test_search_retains_subvoxel_motion_and_rejects_illegal_geometry():
    import numpy as np
    import skfmm
    from tasks.copper_inversion.v2.geometry import canonicalize
    from tasks.copper_inversion.v2.fixtures import fixtures
    # Import only the numerical helper; the full executable requires /baseline
    # inside its solver container and must not execute on the controller.
    tree=ast.parse((Path(__file__).parents[1]/"experiments/v2/stochastic_solver.py").read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="search_distance")
    namespace=dict(np=np,skfmm=skfmm,canonicalize=canonicalize)
    exec(compile(ast.Module(body=[node],type_ignores=[]),"solver-helper","exec"),namespace)
    p=next(fixtures());p['forbidden_mask']=np.zeros(p['shape'],bool)
    p['open_boundary_faces']=['x-','x+','y-','y+','z-','z+']
    field=np.broadcast_to((np.arange(p['shape'][0])-3.5)[:,None,None],p['shape']).copy()*p['spacing'][0]
    initial=field>0;quantized=canonicalize(p,field)[1]
    for _ in range(10):
        field=namespace['search_distance'](p,field-.1*p['spacing'][0])
        quantized=canonicalize(p,quantized-.1*p['spacing'][0])[1]
    assert np.any((field>0)!=initial)
    np.testing.assert_array_equal(quantized>0,initial)
    p['forbidden_mask'][-1]=True
    with pytest.raises(ValueError,match='feasible'):
        namespace['search_distance'](p,field)


def test_interrupted_tuning_resume_keeps_feedback_cap(tmp_path,monkeypatch):
    from tasks.copper_inversion.experiments.v2 import tune_stochastic as tuning
    from tasks.copper_inversion.tests.test_v2_selection import inputs
    from ldm_tts.contracts.evaluation import EvaluationPaused
    dataset=tmp_path/"data";dataset.mkdir();inputs(dataset)
    from tasks.copper_inversion.experiments.v2.publish_training import publish
    catalog=publish(dataset/'train/manifest.json',tmp_path/'public')
    root=tmp_path/"run";image="sha256:"+"a"*64
    monkeypatch.setattr(tuning,"docker",lambda *args:SimpleNamespace(stdout=image))
    monkeypatch.setattr(tuning,"Execution",lambda path,img,account:SimpleNamespace(image=img))
    monkeypatch.setattr(tuning,"InterfaceChecks",lambda *args:lambda payload:True)
    calls=[]
    class Evaluator:
        def __init__(self,root,socket,contract,*,service_uid,slots):self.slots=slots
        def verify_contract(self,*args):pass
        def evaluate(self,candidate):
            self.slots.reserve(candidate.canonical_key);calls.append(candidate.canonical_key)
            if len(calls)==2:raise EvaluationPaused("injected interruption")
            return SimpleNamespace(status="succeeded",metrics={"mean_iou":.2})
    monkeypatch.setattr(tuning,"ServiceEvaluator",Evaluator)
    argv=["tune","--public-catalog",str(catalog),"--evaluation-socket","/unused", "--evaluation-contract","a"*64,
        "--evaluation-service-uid","991","--output",str(root),"--method","mcmc_v2","--slots","2","--seed","47","--seconds","10"]
    monkeypatch.setattr(sys,"argv",argv)
    with pytest.raises(EvaluationPaused):tuning.main()
    first=(root/"tuning-receipts/000.json").read_bytes()
    monkeypatch.setattr(sys,"argv",[*argv,"--resume"]);tuning.main()
    assert calls[1]==calls[2] and calls[0]!=calls[1]
    assert (root/"tuning-receipts/000.json").read_bytes()==first
    assert json.loads((root/"completion.json").read_text())["scientific_feedback_slots"]==2
    before=len(calls);tuning.main();assert len(calls)==before


def test_reinitialization_rejects_real_near_zero_sign_drift():
    import numpy as np
    import skfmm
    from scipy.ndimage import gaussian_filter
    from tasks.copper_inversion.v2.geometry import canonicalize
    from tasks.copper_inversion.v2.fixtures import fixtures
    tree=ast.parse((Path(__file__).parents[1]/"experiments/v2/stochastic_solver.py").read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="search_distance")
    namespace=dict(np=np,skfmm=skfmm,canonicalize=canonicalize)
    exec(compile(ast.Module(body=[node],type_ignores=[]),"solver-helper","exec"),namespace)
    rng=np.random.default_rng(731)
    for _ in range(15):
        field=gaussian_filter(rng.normal(size=(12,12,12)),1.5)
        field=np.where(field>0,np.maximum(field,1e-15)*10.**(-int(rng.integers(0,18))),field)
    p=next(fixtures())
    p.update(shape=[12,12,12],spacing=[1.,1.,1.],domain_mask=np.ones((12,12,12),bool),
             forbidden_mask=np.zeros((12,12,12),bool),open_boundary_faces=['x-','x+','y-','y+','z-','z+'])
    p['priors'].update(hard_volume_bounds=None,max_components=4)
    for observation in p['observations']:
        observation['points']=np.broadcast_to(np.asarray(p['origin'])+6.,observation['points'].shape).copy()
    assert not np.array_equal(skfmm.distance(field)>0,field>0)
    with pytest.raises(ValueError,match='changed validated occupancy'):
        namespace['search_distance'](p,field)


def test_slow_proposal_returns_legal_incumbent_before_hard_budget(tmp_path):
    import signal
    import time
    import numpy as np
    tree=ast.parse((Path(__file__).parents[1]/'experiments/v2/stochastic_solver.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    initial=np.ones((2,2,2))
    def slow_gaussian(*args):
        time.sleep(2)
        raise AssertionError('soft deadline failed to interrupt proposal')
    namespace=dict(time=time,signal=signal,np=np,json=json,Counter=Counter,Path=lambda _:tmp_path/'search.json',
        CONFIG=dict(initializer='random_field',range_scale=1.),
        initialize=lambda p:initial.copy(),gaussian=slow_gaussian,
        default_config=lambda:dict(range_x=[1,2],range_y=[1,2],range_z=[1,2]))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'solver-deadline','exec'),namespace)
    problem=dict(spacing=[1.,1.,1.],origin=[0.,0.,0.],shape=[2,2,2],
                 domain_mask=np.ones((2,2,2),bool),forbidden_mask=np.zeros((2,2,2),bool))
    old=signal.getsignal(signal.SIGALRM);start=time.monotonic()
    result=namespace['solve'](problem,dict(wall_time_seconds=.2),47)
    assert time.monotonic()-start<1.
    np.testing.assert_array_equal(result,initial)
    assert signal.getsignal(signal.SIGALRM)==old
    assert signal.getitimer(signal.ITIMER_REAL)==(0.,0.)


def test_anisotropic_development_sign_regression():
    """Default second order flips a nonzero voxel in real development inputs."""
    import numpy as np
    import skfmm
    from tasks.copper_inversion.v2.geometry import canonicalize
    from tasks.copper_inversion.v2.fixtures import fixtures
    root=Path(__file__).parents[3]
    tree=ast.parse((Path(__file__).parents[1]/'experiments/v2/stochastic_solver.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='search_distance')
    ns=dict(np=np,skfmm=skfmm,canonicalize=canonicalize)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'helper','exec'),ns)
    captured=sorted((root/'docs/copper-baseline-audit-20261027/numeric-fixtures').glob('drift-*.npz'))
    assert len(captured)==4
    for f in captured:
        a=np.load(f);field=a['field'];spacing=a['spacing']
        assert np.any((skfmm.distance(field,dx=tuple(spacing),order=2)>0)!=(field>0))
        for scale in [.001,1.,1000.]:
            p=next(fixtures());p.update(shape=list(field.shape),origin=[0.,0.,0.],spacing=(spacing*scale).tolist(),
                domain_mask=np.ones(field.shape,bool),forbidden_mask=np.zeros(field.shape,bool),
                open_boundary_faces=['x-','x+','y-','y+','z-','z+'])
            p['priors'].update(hard_volume_bounds=None,max_components=None)
            for obs in p['observations']:
                obs['points']=np.broadcast_to(np.asarray(field.shape)*spacing*scale*.5,obs['points'].shape).copy()
            result=ns['search_distance'](p,field*scale)
            np.testing.assert_array_equal(result>0,field>0)
            assert np.isfinite(result).all()


def test_projected_proposal_preserves_legal_motion_and_hard_priors():
    import numpy as np
    from scipy import ndimage
    from tasks.copper_inversion.v2.baselines import constrained_initial
    from tasks.copper_inversion.v2.geometry import canonicalize
    from tasks.copper_inversion.v2.fixtures import fixtures
    tree=ast.parse((Path(__file__).parents[1]/'experiments/v2/stochastic_solver.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='project_proposal')
    ns=dict(np=np,ndimage=ndimage,constrained_initial=constrained_initial)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'projector','exec'),ns)
    p=next(fixtures());p['priors'].update(max_components=2,hard_volume_bounds=None)
    field=np.full(p['shape'],-2.)
    for q in [(1,1,1),(3,1,1),(5,1,1),(7,1,1),(1,5,1)]:field[q]=.37
    result,changed=ns['project_proposal'](p,field)
    assert changed==3 and ndimage.label(result>0)[1]==2
    canonicalize(p,result)
    same,n=ns['project_proposal'](p,result)
    assert n==0 and same is result
    cell=float(np.prod(p['spacing']));p['priors']['hard_volume_bounds']=[10*cell,12*cell]
    grown,n=ns['project_proposal'](p,result)
    assert n>0 and 10<=np.count_nonzero(grown>0)<=12
    canonicalize(p,grown)
