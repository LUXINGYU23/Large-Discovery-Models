"""Optional tool integration preserves the streaming ledger and role boundaries."""
from contextlib import contextmanager
import http.client
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

from tasks.antibody_rfd3.core import harness
from tasks.antibody_rfd3.core.optional_tools import active_tools, optional_host_flags
from tasks.antibody_rfd3.core.worker import DesignWorker
from tasks.antibody_rfd3.core.workflow import parse_args
from tasks.antibody_rfd3.core.synchronous.worker import DesignWorker as SynchronousWorker
from tasks.antibody_rfd3.core.synchronous.workflow import parse_args as sync_args


@pytest.fixture(params=['streaming', 'synchronous'])
def worker(request, tmp_path):
    synchronous = request.param == 'synchronous'
    parse = sync_args if synchronous else parse_args
    args = parse(['--method', 'harness' if synchronous else 'ldm_harness', *([] if synchronous else ['--mock']),
                  '--enable-protein-tools', '--enable-boltzgen', '--samples-per-call', '2',
                  '--sequences-per-round', '16', '--backbones-per-round', '4', '--pool-size', '2',
                  '--root', str(tmp_path)])
    args.mock = True  # Worker-only deterministic fixture; preserve the public synchronous CLI restriction.
    worker_type = SynchronousWorker if synchronous else DesignWorker
    w = worker_type(tmp_path/'run', args)
    role = 'cdrh3_research' if synchronous else 'explorer'
    initial = w.execute('design_h3', 2, .5, profile_id=role, call_id='initial')
    w.round = 1
    return w, role, initial


def test_optional_calls_preserve_idempotency_and_shared_budgets(worker):
    w, role, initial = worker
    params = dict(action='design_h3_boltzgen', count=3, profile_id=role, call_id='native')
    rows = w.execute(**params)
    assert w.remaining() == {'backbones': 3, 'sequences': 13}
    assert all(r['backbone_id'].endswith('/boltzgen') and r['generation_action']=='design_h3_boltzgen' for r in rows)
    assert w.execute(**params) == rows
    assert len(list(w.records())) == 2
    assert w.remaining() == {'backbones': 3, 'sequences': 13}
    with pytest.raises(RuntimeError):
        w.execute(**{**params, 'count': 4})
    if role == 'explorer':
        assert w.role_remaining(role) == {'backbones': 1, 'sequences': 5}
        with pytest.raises(ValueError, match='Role generation budget'):
            w.execute('design_h3_boltzgen', 1, profile_id='evidence_reviewer')
    w.execute('sequence_h3_soluble', 2, .5, backbone_id=initial[0]['backbone_id'],
              profile_id=role, call_id='soluble')
    assert w.remaining() == {'backbones': 3, 'sequences': 11}
    for action in ('sequence_h3', 'sequence_h3_soluble'):
        with pytest.raises(ValueError, match='not compatible'):
            w.execute(action, 1, .5, backbone_id=rows[0]['backbone_id'], profile_id=role)
    w.args.enable_boltzgen = False
    with pytest.raises(ValueError, match='enabled'):
        w.execute('design_h3_boltzgen', 1, profile_id=role)
    w.args.enable_protein_tools = False
    with pytest.raises(ValueError, match='enabled'):
        w.execute('sequence_h3_soluble', 1, .5, backbone_id=initial[0]['backbone_id'], profile_id=role)


def test_soluble_weights_do_not_inherit_vanilla_cache(worker, monkeypatch):
    w, _, _ = worker
    root = w.args.root
    (root/'tmp').mkdir()
    w.args.proteinmpnn_checkpoint = root/'vanilla.pt'
    if isinstance(w, SynchronousWorker):
        w.runtime_cache = root/'cache'
        w.runtime_cache.mkdir()
        (w.runtime_cache/'manifest.json').write_text(json.dumps({k:{'file': k+'.bin'}
            for k in ('input_structure','rfd3_checkpoint','proteinmpnn_checkpoint')}))
    monkeypatch.setattr('os.sched_getaffinity', lambda pid: {0}, raising=False)
    def intercepted(*args, **kwargs):
        raise RuntimeError('Intercepted model launch')
    monkeypatch.setattr('subprocess.Popen', intercepted)
    for explicit in (None, root/'soluble.pt'):
        w.args.solublempnn_checkpoint = explicit
        job = w.jobs/('explicit' if explicit else 'default')
        job.mkdir()
        with pytest.raises(RuntimeError, match='Intercepted model launch'):
            w.run_models(job, 'backbone', 1, .5, 19, soluble=True)
        config = json.loads((job/'config.json').read_text())
        assert config['proteinmpnn_variant'] == 'soluble'
        assert config.get('proteinmpnn_checkpoint') == (str(explicit) if explicit else None)


def test_default_registry_and_opt_in_flags(tmp_path):
    baseline = parse_args(['--mock'])
    assert active_tools(baseline, harness.TOOLS) == harness.TOOLS
    assert all(p.name!='cdrh3-protein-tools' for p in harness.profile().skill_dirs)
    optional_host_flags(SimpleNamespace(args=baseline), tmp_path)
    assert not list(tmp_path.iterdir())
    enabled = parse_args(['--method','ldm_harness','--mock','--enable-protein-tools','--enable-boltzgen'])
    assert set(active_tools(enabled,harness.TOOLS)) == set(harness.TOOLS) | {
        'sequence_h3_soluble','score_interface','design_h3_boltzgen'}
    assert any(p.name=='cdrh3-protein-tools' for p in harness.profile(args=enabled).skill_dirs)
    optional_host_flags(SimpleNamespace(args=enabled), tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {'protein-tools.enabled','boltzgen.enabled'}
    for parse in (parse_args,sync_args):
        with pytest.raises(SystemExit):parse(['--method','fixed','--enable-boltzgen'])


class UnixConnection(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__('localhost', timeout=5)
        self.path = str(path)
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)


def test_streaming_socket_keeps_catalog_replay_and_auxiliary_scores_separate(tmp_path, monkeypatch):
    args = parse_args(['--method','ldm_harness','--mock','--enable-protein-tools','--enable-boltzgen',
                      '--research-candidates-per-role','1','--pool-size','2',
                      '--backbones-per-round','4','--sequences-per-round','32'])
    w = DesignWorker(tmp_path/'run',args);w.round = 1
    workspaces = {r:tmp_path/r for r in harness.PANEL}
    monkeypatch.setattr(harness,'score_interface',lambda worker, artifact_id: {'status':'succeeded','id':artifact_id})
    def request(directory,payload):
        c=UnixConnection(directory/'design.sock')
        try:
            c.request('POST','/',json.dumps(payload),{'Content-Type':'application/json'})
            response=c.getresponse();return response.status,json.loads(response.read())
        finally:c.close()
    with harness.worker_socket(w,workspaces=workspaces) as directory:
        assert (directory/'boltzgen.enabled').is_file()
        call=dict(action='design_h3_boltzgen',count=4,call_id='one',profile_id='explorer')
        status,data=request(directory,call);assert status==200
        assert data['candidate_catalog']=='/workspace/generated/g000002.json'
        catalog=json.loads((workspaces['explorer']/'generated/g000002.json').read_text())
        assert catalog['candidates']
        assert all(row['backbone_id'].endswith('/boltzgen') for row in catalog['candidates'])
        assert request(directory,call)==(status,data)
        before=w.remaining()
        status,data=request(directory,dict(action='score_interface',artifact_id='example',call_id='score',profile_id='evidence_reviewer'))
        assert status==200 and data['interface_score']['status']=='succeeded'
        assert w.remaining()==before and len(list(w.records()))==1
        w.args.enable_protein_tools=False
        assert request(directory,dict(action='score_interface',artifact_id='example',call_id='denied',profile_id='explorer'))[0]==400


def test_reviewer_cannot_spend_new_generation_tool_budgets(tmp_path, monkeypatch):
    from ldm_tts.harness import HarnessHistoryConfig
    from ldm_tts.harness.history import HistoryLog
    clients=[]
    class Probe:
        def __init__(self,**kwargs):pass
        def preflight(self):pass
    class Client:
        def __init__(self,command,*,api_key,config,history,response_timeout_seconds):
            self.config=config;self.history=history;clients.append(self)
        def __enter__(self):return self
        def __exit__(self,*args):pass
    @contextmanager
    def socket_context(worker,*,workspaces):yield tmp_path
    monkeypatch.setattr(harness,'OpenAICompatibleProposalClient',Probe)
    monkeypatch.setattr(harness,'HarnessClient',Client)
    monkeypatch.setattr(harness,'worker_socket',socket_context)
    monkeypatch.setattr(harness,'pi_sidecar_command',lambda *args,**kwargs:('fake',))
    for key,value in {'LLM_BASE_URL':'https://example.invalid/v1/responses','LLM_MODEL_NAME':'fixture','LLM_API_KEY':'fixture'}.items():monkeypatch.setenv(key,value)
    args=parse_args(['--method','ldm_harness','--mock','--enable-protein-tools','--enable-boltzgen'])
    w=DesignWorker(tmp_path/'run',args)
    service=SimpleNamespace(config=HarnessHistoryConfig(),log=HistoryLog(),lineage_of=lambda role:role)
    bridge=SimpleNamespace(service=service,add_client=lambda client:None)
    with harness.open_agent(args,tmp_path,w,bridge,'fixture'):
        reviewer=next(c for c in clients if c.config.profiles[0].profile_id=='evidence_reviewer')
        budgets=reviewer.config.limits.tool_call_budgets
        assert budgets['sequence_h3_soluble']==budgets['design_h3_boltzgen']==0
        assert budgets['score_interface']==4


def test_boltzgen_host_runner_keeps_gpu_lock_and_override(worker, monkeypatch):
    w, _, _ = worker
    root=w.args.root;(root/'tmp').mkdir()
    w.args.antigen='1N8Z_C';w.args.share_gpu=True;w.args.gpu=2
    if isinstance(w,SynchronousWorker):w.execution_gpu=6
    gpu=getattr(w,'execution_gpu',w.args.gpu)
    for key in ('BOLTZGEN_PYTHON','BOLTZGEN_SCAFFOLD','BOLTZGEN_CHECKPOINT','BOLTZGEN_INVERSE_FOLD_CHECKPOINT','BOLTZGEN_MOLDIR','BOLTZGEN_SPEC_HER2'):
        monkeypatch.setenv(key,str(root/key))
    monkeypatch.setattr('os.sched_getaffinity',lambda pid:{0},raising=False)
    import fcntl
    lock_name='antibody-rfd3-worker.lock' if isinstance(w,SynchronousWorker) else f'antibody-rfd3-worker-gpu{gpu}.lock'
    job=w.jobs/'host-test';job.mkdir()
    class Process:
        def __init__(self,command,**kwargs):
            assert command[command.index('--gpu')+1]==str(gpu)
            assert command[command.index('--memory-limit-gib')+1]=='24'
            assert command[command.index('--spec')+1]==str(root/'BOLTZGEN_SPEC_HER2')
            assert kwargs['pass_fds']
        def wait(self,timeout):
            with (root/'tmp'/lock_name).open('a') as lock:
                with pytest.raises(BlockingIOError):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            output=job/'output';output.mkdir()
            (output/'run.json').write_text(json.dumps({'status':'succeeded','candidates':[{'h3':'ASTYDGSPTAN'}]}))
            return 0
    monkeypatch.setattr('subprocess.Popen',Process)
    assert w.run_boltzgen(job,1,19)==['ASTYDGSPTAN']
