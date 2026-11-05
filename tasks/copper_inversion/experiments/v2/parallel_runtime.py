"""Audited scheduling layer for immutable scientific runtimes.

Load this file by absolute path with PYTHONPATH pointing at the frozen runtime.
Science, inputs, candidate identities, receipts and budgets are not rewritten.
The separately hashed scheduling policy is an additional run identity component.
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
    def __init__(self, root, slots, local_slots):
        self.root=Path(root);self.slots=slots
        self.local=threading.BoundedSemaphore(local_slots)

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
                            yield time.monotonic()-start
                        except execution.EvaluationPaused as exc:
                            self.pause(exc)
                            raise
                        finally:fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)
                        return
                time.sleep(.1)


def limit_container_startup(queue, slots=4):
    """Bound daemon mutations independently from long-running worker capacity."""
    startup=JobQueue(queue.root/'startup',slots,slots)
    original=execution.docker
    def docker(*args,**kwargs):
        if args[0] not in ('create','start'):return original(*args,**kwargs)
        queue.check()
        with startup.acquire():
            queue.check()
            try:return original(*args,**kwargs)
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


class ParallelService(service.EvaluatorService):
    def __init__(self,configuration):
        super().__init__(configuration)
        self._budget_lock=threading.RLock()
        self._admission_lock=threading.RLock()
        self._condition=threading.Condition()
        self._source_locks={};self._active=0;self._closing=False
        account=self.ledger.consume_many
        def synchronized(*args,**kwargs):
            with self._budget_lock:return account(*args,**kwargs)
        self.slots.account=synchronized
        self.evaluator.execution.account=synchronized
        # Fixed fixture inputs share paths; admit sequentially, run suites concurrently.
        admit=self.domain.admit
        def checked(proposal):
            with self._admission_lock:return admit(proposal)
        self.domain.admit=checked

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


class ThreadedServer(socketserver.ThreadingMixIn,socketserver.UnixStreamServer):
    daemon_threads=False
    request_queue_size=128


def install(policy):
    queue=JobQueue(policy['queue_root'],policy['global_jobs'],policy['jobs_per_process'])
    queue.interface_slots=policy.get('interface_slots',0)
    limit_container_startup(queue,policy['startup_slots'])
    original=execution.Execution.run
    def queued(self,*args,**kwargs):
        with queue.acquire(interface=kwargs.get('phase')=='interface') as seconds:
            result=original(self,*args,**kwargs)
        # Separate operational record; immutable scientific job receipts stay intact.
        record=self.root/'scheduling'/f'{result[0].name}.json'
        if not record.exists():atomic_json_write(record,dict(queue_wait_seconds=seconds,policy=policy['policy_sha256']))
        return result
    execution.Execution.run=queued


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
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True);p.add_argument('mode',choices=['service','ldm','tune','finish']);a,rest=p.parse_known_args()
    policy=json.loads(a.policy.read_text())
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=policy['runtime_sha256']:raise ValueError('scheduling runtime changed')
    expected=execution.digest({k:v for k,v in policy.items() if k!='policy_sha256'})
    if expected!=policy['policy_sha256']:raise ValueError('scheduling policy changed')
    install(policy)
    if a.mode=='service':
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
