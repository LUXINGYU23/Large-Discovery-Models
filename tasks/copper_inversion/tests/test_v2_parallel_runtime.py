import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from ldm_tts.engine.run_store import BudgetLedger
from tasks.copper_inversion.v2.budget import ScientificSlots
from tasks.copper_inversion.experiments.v2.parallel_runtime import JobQueue,ParallelEvaluator,ParallelService,service
import pytest
from tasks.copper_inversion.experiments.v2.parallel_runtime import execution,limit_container_startup


def test_infrastructure_fault_stops_future_dispatch(tmp_path):
    (tmp_path/'000.lock').touch()
    queue=JobQueue(tmp_path,1,4)
    with pytest.raises(execution.EvaluationPaused):
        with queue.acquire():raise execution.EvaluationPaused('unknown start outcome')
    with pytest.raises(execution.EvaluationPaused,match='shared scheduler paused'):
        with queue.acquire():pytest.fail('must not dispatch')
    assert (tmp_path/'PAUSED.json').exists()


def test_startup_limit_and_fail_closed(tmp_path,monkeypatch):
    (tmp_path/'startup').mkdir()
    for i in range(4):(tmp_path/'startup'/f'{i:03}.lock').touch()
    queue=JobQueue(tmp_path,180,32)
    active=peak=0;guard=threading.Lock()
    def docker(command,*args,**kwargs):
        nonlocal active,peak
        if args==('fail',):raise execution.EvaluationPaused('unknown start outcome')
        with guard:active+=1;peak=max(peak,active)
        time.sleep(.02)
        with guard:active-=1
    monkeypatch.setattr(execution,'docker',docker)
    limit_container_startup(queue,4)
    with ThreadPoolExecutor(max_workers=32) as pool:list(pool.map(lambda _:execution.docker('start','ok'),range(64)))
    assert 1<=peak<=4
    with pytest.raises(execution.EvaluationPaused):execution.docker('start','fail')
    with pytest.raises(execution.EvaluationPaused,match='shared scheduler paused'):execution.docker('create','ok')


def test_32_workers_keep_321_unique_charges_and_replays(tmp_path):
    ledger=BudgetLedger(limits={'algorithm_feedback_slots':321},path=tmp_path/'budget.json')
    slots=ScientificSlots(ledger.consume_many)
    def reserve(index):
        source=f'{index:064x}'
        slots.reserve(source,interface_rejected=index%7==0)
        slots.reserve(source,interface_rejected=index%7==0)
        ledger.consume_many({'solver_jobs':1},usage_key=f'solver:{source}')
        ledger.consume_many({'solver_jobs':1},usage_key=f'solver:{source}')
    with ThreadPoolExecutor(max_workers=32) as pool:list(pool.map(reserve,range(321)))
    restored=BudgetLedger.load(ledger.path)
    assert restored.counters['algorithm_feedback_slots']==321
    assert restored.counters['solver_jobs']==321
    assert restored.counters['interface_rejected_sources']==46


@pytest.mark.parametrize('slots',[128,144,180])
def test_queue_enforces_32_local_jobs(tmp_path,slots):
    for i in range(slots):(tmp_path/f'{i:03}.lock').touch()
    queue=JobQueue(tmp_path,slots,32);queue.interface_slots=8
    guard=threading.Lock();active=0;peak=0
    barrier=threading.Barrier(32)
    def work(index):
        nonlocal active,peak
        with queue.acquire():
            with guard:active+=1;peak=max(peak,active)
            barrier.wait(timeout=10)
            time.sleep(.01)
            with guard:active-=1
    with ThreadPoolExecutor(max_workers=64) as pool:list(pool.map(work,range(64)))
    assert peak==32 and active==0


def test_queue_bounds_and_releases_after_exception(tmp_path):
    for i in range(2):(tmp_path/f'{i:03}.lock').touch()
    queue=JobQueue(tmp_path,2,8);guard=threading.Lock();active=0;peak=0
    def work(i):
        nonlocal active,peak
        try:
            with queue.acquire():
                with guard:active+=1;peak=max(peak,active)
                time.sleep(.02)
                with guard:active-=1
                if i==0:raise ValueError('fixture')
        except ValueError:pass
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(work,range(12)))
    assert peak==2 and active==0
    with queue.acquire():pass


def test_batch_is_ordered_and_slots_idempotent(tmp_path):
    ledger=BudgetLedger(limits={'algorithm_feedback_slots':4},path=tmp_path/'budget.json')
    ev=ParallelEvaluator(tmp_path,'/unused','a'*64,service_uid=991,slots=ScientificSlots(ledger.consume_many))
    seen=[];guard=threading.Lock();active=0;peak=0
    def evaluate(c):
        nonlocal active,peak
        with guard:active+=1;peak=max(peak,active)
        time.sleep(.04*(4-c.index))
        with guard:active-=1;seen.append(c.index)
        return c.index
    ev.evaluate=evaluate
    candidates=[SimpleNamespace(canonical_key=str(i)*64,index=i) for i in range(4)]
    assert ev.evaluate_batch(candidates)==(0,1,2,3)
    assert seen!=[0,1,2,3] and peak>1
    assert ev.evaluate_batch(candidates)==(0,1,2,3)
    assert ledger.counters['algorithm_feedback_slots']==4


def test_duplicate_sources_serialized_and_seal_waits(tmp_path,monkeypatch):
    def init(self,config):
        self.ledger=BudgetLedger(limits={'algorithm_feedback_slots':3},path=tmp_path/'budget.json')
        self.slots=ScientificSlots(self.ledger.consume_many)
        self.domain=SimpleNamespace(admit=lambda p:p)
        self.evaluator=SimpleNamespace(execution=SimpleNamespace(account=None))
    monkeypatch.setattr(service.EvaluatorService,'__init__',init)
    calls=[];running=set();guard=threading.Lock()
    def dispatch(self,request):
        if request['operation']=='seal_training':
            assert not running;calls.append('seal');return 'sealed'
        s=request['source']
        with guard:assert s not in running;running.add(s)
        self.slots.reserve(s*64);time.sleep(.04)
        with guard:running.remove(s);calls.append(s)
        return s
    monkeypatch.setattr(service.EvaluatorService,'dispatch',dispatch)
    backend=ParallelService({})
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(backend.dispatch,dict(operation='evaluate',source=s)) for s in ('a','a','b')]
        time.sleep(.01);sealed=pool.submit(backend.dispatch,dict(operation='seal_training'))
        assert [f.result() for f in futures]==['a','a','b']
        assert sealed.result()=='sealed'
    assert calls[-1]=='seal' and backend.ledger.counters['algorithm_feedback_slots']==2
