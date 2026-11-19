"""Audited scheduling layer for immutable scientific runtimes.

Load this file by absolute path with PYTHONPATH pointing at the frozen runtime.
Science, inputs, candidate identities, receipts and budgets are not rewritten.
The separately hashed scheduling policy is an additional run identity component.

Policy fields (parallelism.md §6): queue_root, global_jobs, interface_slots,
startup_slots, jobs_per_process, case_workers ({tier: n} or n), batch_workers,
start_retry_seconds, max_concurrent_runs, max_agent_sessions,
load_qualification_sha256, runtime_sha256, policy_sha256. Only the first five
are required; older policies keep the prototype defaults.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import socketserver
import sys
import threading
import time

from ldm_tts.engine.run_store import atomic_json_write, BudgetLedger
from ldm_tts.contracts import RawProposal,CandidateRejection
from tasks.copper_inversion.core import execution
from tasks.copper_inversion.v2 import service, service_client


class JobQueue:
    retryable=()

    def __init__(self, root, slots, local_slots):
        self.root=Path(root);self.slots=slots
        self.local=threading.BoundedSemaphore(local_slots)
        self._thread=threading.local()

    def in_use(self, held=None):
        """Machine-wide slots held now, by probing every slot lock (held counts as busy)."""
        busy=0
        for index in range(self.slots):
            if index==held:busy+=1;continue
            fd=os.open(self.root/f'{index:03}.lock',os.O_RDWR)
            try:
                try:fcntl.flock(fd,fcntl.LOCK_SH|fcntl.LOCK_NB)
                except BlockingIOError:busy+=1
                else:fcntl.flock(fd,fcntl.LOCK_UN)
            finally:os.close(fd)
        return busy

    @property
    def slots_in_use_at_start(self):
        return getattr(self._thread,'in_use',None)

    def check(self):
        if (self.root/'PAUSED.json').exists():
            raise execution.EvaluationPaused('shared scheduler paused; reconcile infrastructure before dispatch')

    def pause(self, error):
        # First fault wins; publication is atomic across evaluator/client UIDs.
        path=self.root/f'fault-{os.getpid()}-{threading.get_ident()}.json'
        atomic_json_write(path,dict(time=time.time(),pid=os.getpid(),error=str(error)))
        path.chmod(0o644)
        try:os.link(path,self.root/'PAUSED.json')
        except FileExistsError:pass

    @contextmanager
    def acquire(self, *, interface=False):
        start=time.monotonic()
        with self.local:
            reserved=getattr(self,"interface_slots",0)
            indices=range(reserved) if interface and reserved else range(reserved,self.slots)
            order=list(indices);random.SystemRandom().shuffle(order)
            while True:
                self.check()
                for index in order:
                    fd=os.open(self.root/f'{index:03}.lock',os.O_RDWR)
                    try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:os.close(fd)
                    else:
                        try:
                            self.check()
                            # Load covariate for the analysis (parallelism.md §5 rule 5).
                            self._thread.in_use=self.in_use(held=index)
                            yield time.monotonic()-start
                        except self.retryable:raise
                        except execution.EvaluationPaused as exc:
                            self.pause(exc)
                            raise
                        finally:fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)
                        return
                time.sleep(.1)


def limit_container_startup(queue, slots=4):
    """Bound daemon mutations independently from long-running worker capacity.

    A create/start timeout is left to Execution's bounded retry; it pauses the
    scheduler only once the retries are spent (core/execution.py)."""
    startup=JobQueue(queue.root/'startup',slots,slots)
    startup.in_use=lambda held=None:None
    startup.retryable=(execution.DockerTimeout,)
    original=execution.docker
    def docker(*args,**kwargs):
        if args[0] not in ('create','start'):return original(*args,**kwargs)
        queue.check()
        with startup.acquire():
            queue.check()
            try:return original(*args,**kwargs)
            except execution.DockerTimeout:raise
            except execution.EvaluationPaused as exc:
                queue.pause(exc)
                raise
    execution.docker=docker


class ParallelEvaluator(service_client.ServiceEvaluator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._budget_lock=threading.RLock()
        if self.slots is not None:
            account=self.slots.account
            def synchronized(*args,**kwargs):
                with self._budget_lock:return account(*args,**kwargs)
            self.slots.account=synchronized

    def evaluate_batch(self,candidates):
        if not candidates:return ()
        # Reserve in selection order before concurrent transport. Replays are idempotent.
        if self.slots is not None:
            for candidate in candidates:self.slots.reserve(candidate.canonical_key)
        with ThreadPoolExecutor(max_workers=16) as pool:
            return tuple(pool.map(self.evaluate,candidates))


class Concurrent:
    """Concurrent evaluations: one at a time per source, seal waits for all.

    Admission (interface checks) runs concurrently too; InterfaceChecks writes its
    shared fixture inputs once under its own lock and only reads them after.
    """
    def _concurrent(self):
        self._budget_lock=threading.RLock()
        self._condition=threading.Condition()
        self._source_locks={};self._active=0;self._closing=False
        account=self.ledger.consume_many
        def synchronized(*args,**kwargs):
            with self._budget_lock:return account(*args,**kwargs)
        self.slots.account=synchronized
        self.evaluator.execution.account=synchronized

    def dispatch(self,request):
        if isinstance(request,dict) and request.get('operation')=='seal_training':
            with self._condition:
                self._closing=True
                while self._active:self._condition.wait()
                result=super().dispatch(request)
                self._closing=False;self._condition.notify_all()
                return result
        if not isinstance(request,dict) or request.get('operation')!='evaluate':
            with self._budget_lock:return super().dispatch(request)
        source=request.get('source')
        if not isinstance(source,str):return super().dispatch(request)
        key=hashlib.sha256(source.encode()).hexdigest()
        with self._condition:
            while self._closing:self._condition.wait()
            lock=self._source_locks.setdefault(key,threading.Lock());self._active+=1
        try:
            with lock:return super().dispatch(request)
        finally:
            with self._condition:self._active-=1;self._condition.notify_all()


class ParallelService(Concurrent,service.EvaluatorService):
    def __init__(self,configuration):
        super().__init__(configuration)
        self._concurrent()


def risk_service(configuration):
    """Threaded risk-only service (rsi/service.py) for the batch launcher."""
    from tasks.copper_inversion.rsi.service import RiskOnlyService
    class ParallelRiskService(Concurrent,RiskOnlyService):
        def __init__(self,configuration):
            super().__init__(configuration)
            self._concurrent()
    return ParallelRiskService(configuration)


class ThreadedServer(socketserver.ThreadingMixIn,socketserver.UnixStreamServer):
    daemon_threads=False
    request_queue_size=128


def case_workers(policy, tier=None):
    """Cases of one evaluation solved at once; a scheduling value, never in the key."""
    value=policy.get('case_workers',4)
    if isinstance(value,dict):value=value.get(tier,max(value.values()))
    if type(value) is not int or value<1:raise ValueError('invalid case concurrency')
    return value


def container_labels(policy):
    return {'ldm4geo.managed':'1','ldm4geo.queue':execution.digest(str(Path(policy['queue_root']).resolve()))[:16]}


def install(policy, tier=None):
    from tasks.copper_inversion.v2.suite import SuiteEvaluator
    queue=JobQueue(policy['queue_root'],policy['global_jobs'],policy['jobs_per_process'])
    queue.interface_slots=policy.get('interface_slots',0)
    limit_container_startup(queue,policy['startup_slots'])
    execution.Execution.start_retry_seconds=tuple(policy.get('start_retry_seconds',(5,15,45)))
    execution.Execution.labels=container_labels(policy)
    SuiteEvaluator.case_workers=case_workers(policy,tier)
    original=execution.Execution.run
    def queued(self,*args,**kwargs):
        with queue.acquire(interface=kwargs.get('phase')=='interface') as seconds:
            in_use=queue.slots_in_use_at_start
            result=original(self,*args,**kwargs)
        # Separate operational record; immutable scientific job receipts stay intact.
        record=self.root/'scheduling'/f'{result[0].name}.json'
        if not record.exists():
            job=result[0]/'job.json'
            retries=json.loads(job.read_text()).get('start_retries',[]) if job.exists() else []
            atomic_json_write(record,dict(queue_wait_seconds=seconds,slots_in_use_at_start=in_use,
                global_jobs=policy['global_jobs'],phase=kwargs.get('phase'),start_retries=retries,
                policy=policy['policy_sha256']))
        return result
    execution.Execution.run=queued
    return queue


def tune(run):
    """Resume the prespecified parameter table, ordered publication, parallel execution."""
    from tasks.copper_inversion.experiments.v2.tune_stochastic import source_for,configurations
    from tasks.copper_inversion.core.candidate import seal
    from tasks.copper_inversion.v2.domain import CheckedDomain
    from tasks.copper_inversion.v2.interface_checks import InterfaceChecks
    from tasks.copper_inversion.v2.budget import ScientificSlots
    root=Path(run);plan=json.loads((root/'plan.json').read_text())
    if plan['implementation_sha256']!=execution.source_identity():raise ValueError('frozen science identity changed')
    if configurations(plan['method'],plan['slots'],plan['discovery_seed'])!=plan['parameters']:raise ValueError('parameter table changed')
    from tasks.copper_inversion.experiments.v2 import tune_stochastic
    template=Path(tune_stochastic.__file__).with_name('stochastic_solver.py').read_text()
    if execution.digest(template)!=plan['template_sha256']:raise ValueError('baseline template changed')
    ledger=BudgetLedger.load(root/'budget.json');budget_lock=threading.RLock();admission_lock=threading.Lock()
    def account(*args,**kwargs):
        with budget_lock:return ledger.consume_many(*args,**kwargs)
    slots=ScientificSlots(account);executor=execution.Execution(root/'jobs',plan['image'],account)
    domain=CheckedDomain(root,InterfaceChecks(root,executor,slots))
    evaluator=ParallelEvaluator(root,plan['evaluation_socket'],plan['evaluation_contract'],service_uid=plan['evaluation_service_uid'],slots=slots)
    evaluator.verify_contract(plan['suite_sha256'],plan['seconds'],plan['slots'])
    def evaluate(item):
        index,parameters=item;payload=seal(root,source_for(template,parameters).encode())
        path=root/'tuning-receipts'/f'{index:03}.json'
        if path.exists():
            row=json.loads(path.read_text())
            if row['source_sha256']!=payload['sha256']:raise ValueError('receipt source changed')
            return row
        with admission_lock:candidate=domain.admit(RawProposal(payload,'fixed_sobol_parameter_search'))
        if isinstance(candidate,CandidateRejection):row=dict(index=index,source_sha256=payload['sha256'],status='interface_rejected',reason=str(candidate))
        else:
            result=evaluator.evaluate(candidate)
            row=dict(index=index,source_sha256=payload['sha256'],status=result.status,metrics=dict(result.metrics))
        atomic_json_write(path,row);return row
    rows=[]
    with ThreadPoolExecutor(max_workers=16) as pool:
        # Initialization, then 20 batches of 16. Reserve before concurrent work,
        # in original order; neither admission timing nor completion reorders cost.
        items=list(enumerate(plan['parameters']))
        batches=[items[:1]]+[items[i:i+16] for i in range(1,len(items),16)]
        for batch in batches:
            for _,parameters in batch:
                payload=seal(root,source_for(template,parameters).encode())
                slots.reserve(payload['sha256'])
            for row in pool.map(evaluate,batch):
                rows.append(row);atomic_json_write(root/'results.json',dict(development_only=False,results=rows))
                print(json.dumps(row),flush=True)
    atomic_json_write(root/'completion.json',dict(status='completed',plan_sha256=execution.digest(plan),evaluated=len(rows),scientific_feedback_slots=ledger.counters.get('algorithm_feedback_slots',0)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--tier',help='tier whose case concurrency applies (service modes read it from the service)')
    p.add_argument('mode',choices=['service','ldm','tune','finish','rsi-service','rsi-freeze','rsi-q1','rsi']);a,rest=p.parse_known_args()
    policy=json.loads(a.policy.read_text())
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=policy['runtime_sha256']:raise ValueError('scheduling runtime changed')
    expected=execution.digest({k:v for k,v in policy.items() if k!='policy_sha256'})
    if expected!=policy['policy_sha256']:raise ValueError('scheduling policy changed')
    if a.mode=='rsi':
        # The controller starts no containers; it may read batch_workers from this policy.
        os.environ['LDM4GEO_SCHEDULING_POLICY']=str(a.policy.resolve())
        from tasks.copper_inversion.rsi import workflow
        # This parser also knows --tier and took it from the controller's arguments; hand it back.
        raise SystemExit(workflow.main(rest+(['--tier',a.tier] if a.tier is not None else [])))
    install(policy,a.tier)
    if a.mode=='rsi-q1':
        from tasks.copper_inversion.experiments.rsi import q1
        raise SystemExit(q1.main(rest, policy=policy))
    if a.mode=='rsi-service':
        q=argparse.ArgumentParser();q.add_argument('--config',type=Path,required=True);b=q.parse_args(rest)
        config=json.loads(b.config.read_text());backend=risk_service(config)
        tier=a.tier or (backend.tier or {}).get('name')
        backend.evaluator.case_workers=case_workers(policy,tier)
        with ThreadedServer(config['socket'],service.Handler) as server:
            os.chmod(config['socket'],0o600);server.backend=backend
            # Same startup receipt as rsi/service.py, plus the policy.
            print(json.dumps({"status":"listening","pid":os.getpid(),"contract_digest":backend.public_contract,
                              "policy":policy['policy_sha256'],"case_workers":backend.evaluator.case_workers}),flush=True)
            server.serve_forever()
    elif a.mode=='rsi-freeze':
        from tasks.copper_inversion.experiments.rsi import freeze
        raise SystemExit(freeze.main(rest))
    elif a.mode=='service':
        q=argparse.ArgumentParser();q.add_argument('--config',type=Path,required=True);b=q.parse_args(rest)
        config=json.loads(b.config.read_text());backend=ParallelService(config)
        with ThreadedServer(config['socket'],service.Handler) as server:
            os.chmod(config['socket'],0o600);server.backend=backend
            print(json.dumps(dict(status='listening',pid=os.getpid(),policy=policy['policy_sha256'])),flush=True)
            server.serve_forever()
    elif a.mode=='ldm':
        from tasks.copper_inversion.v2 import workflow
        workflow.ServiceEvaluator=ParallelEvaluator
        raise SystemExit(workflow.main(rest))
    elif a.mode=='tune':
        q=argparse.ArgumentParser();q.add_argument('--run',type=Path,required=True);b=q.parse_args(rest);tune(b.run)
    else:
        from tasks.copper_inversion.experiments.v2 import finish_formal
        sys.argv=[sys.argv[0],*rest];finish_formal.main()


if __name__=='__main__':main()
