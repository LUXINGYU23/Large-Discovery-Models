"""Saturated real-container acceptance, on public fixtures only.

    Six independent solver controllers share 24 solver slots; two interface controllers
share eight reserved slots. Every controller stays below thirty-two active jobs. Background work uses one
CPU thread, matching measured MCMC/ACQ consumption; container caps stay at two CPUs.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

from tasks.copper_inversion.core.execution import Execution, digest, source_identity
from tasks.copper_inversion.core.candidate import seal
from tasks.copper_inversion.core.geometry import read_model
from tasks.copper_inversion.v2.data import save_problem
from tasks.copper_inversion.v2.fixtures import fixtures
from tasks.copper_inversion.v2.geometry import canonicalize
from tasks.copper_inversion.experiments.v2.parallel_runtime import JobQueue, limit_container_startup
from tasks.copper_inversion.core.execution import EvaluationPaused
from tasks.copper_inversion.experiments.v2.tune_stochastic import configurations, source_for

IMAGE='sha256:f6856385a0da9444b1e5edb5048b2f6081593fe568c3473c2fda524d0c4489f5'
BACKGROUND='''import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tasks.copper_inversion.v2.baselines import initialize
def solve(problem,budget,seed):
    started=time.monotonic()
    result=initialize(problem)
    deadline=started+max(0.,budget['wall_time_seconds']-8.)
    def busy(i):
        x=np.linspace(.1,1.,200000)
        while time.monotonic()<deadline: x=np.sin(x)+.1
    with ThreadPoolExecutor(max_workers=1) as pool:list(pool.map(busy,range(1)))
    return result
'''


def controller(args):
    root, group, interface, jobs, small = args
    root=Path(root);work=root/group;work.mkdir()
    queue=JobQueue(root/'queue',32,32);queue.interface_slots=8
    limit_container_startup(queue,1)
    execution=Execution(work/'jobs',IMAGE)
    problems=list(fixtures())
    template=Path(__file__).with_name('stochastic_solver.py').read_text()
    # SciPy's first Sobol direction-table initialization is not thread-safe.
    parameters={(method,seed):configurations(method,17,seed)
                for method in ('mcmc_v2','acq_v2') for seed in (42,43,3407)}
    def one(index):
        problem=problems[index%6]
        jobroot=work/f'input-{index:03}';save_problem(problem,jobroot)
        seconds=5 if interface else 120
        (jobroot/'budget.json').write_text(json.dumps(dict(seed=47+index%3,wall_time_seconds=seconds,
            cpu_count=1 if interface else 2,memory_mb=2048 if interface else 4096)))
        if interface:
            method='acq_v2' if group.endswith('1') else 'mcmc_v2'
            config=parameters[method,(42,43,3407)[(index//102)%3]][(index//6)%17]
            body=source_for(template,config)
        else:body=BACKGROUND
        payload=seal(work,body.encode())
        with queue.acquire(interface=interface) as wait:
            acquired=time.time()
            try:
                folder,receipt=execution.run(dict(group=group,index=index,source=payload['sha256']),
                    phase='interface' if interface else 'solver',inputs=jobroot,
                    timeout=seconds,cpus=1 if interface else 2,memory_mb=2048 if interface else 4096,
                    source=(work/payload['source_path']).parent,science_module='v2')
                legal=False
                if receipt['exit_code']==0 and receipt['reason'] is None:
                    canonicalize(problem,read_model(folder/'output/model.npy',problem['shape']));legal=True
                row=dict(index=index,interface=interface,acquired=acquired,released=time.time(),
                    queue_seconds=wait,receipt=receipt,legal=legal)
            except Exception as exc:
                if isinstance(exc,EvaluationPaused):queue.pause(exc)
                row=dict(index=index,interface=interface,acquired=acquired,released=time.time(),
                         queue_seconds=wait,legal=False,error=type(exc).__name__+': '+str(exc))
        (work/f'result-{index:03}.json').write_text(json.dumps(row,indent=2))
        return row
    # Exercise the actual authenticated asynchronous wire protocol while the
    # same global container queue is saturated. Each source is a unique job.
    import threading
    from tasks.copper_inversion.v2.async_transport import AsyncServiceMixin, evaluate_request
    from tasks.copper_inversion.v2.service import Handler
    from tasks.copper_inversion.v2.service_client import ServiceEvaluator
    from tasks.copper_inversion.experiments.v2.parallel_runtime import ThreadedServer
    class Backend:
        def __init__(self):
            self.root=work;self.identity=dict(group=group);self.slot_limit=jobs
            self.caller_uid=os.getuid()
        def dispatch(self,request):
            row=one(int(request['source']))
            self.task_progress()
            return row
    class AsyncBackend(AsyncServiceMixin,Backend):pass
    backend=AsyncBackend();sock=str(work/'rpc.sock')
    client=ServiceEvaluator(work,sock,digest(backend.identity),service_uid=os.getuid(),timeout=30)
    def remote_one(index):
        payload=dict(operation='evaluate',source=str(index),contract_digest=digest(backend.identity))
        return evaluate_request(client,payload,client.request,poll_seconds=.2,stall_seconds=1800)
    with ThreadedServer(sock,Handler) as server:
        server.backend=backend
        thread=threading.Thread(target=server.serve_forever);thread.start()
        try:
            with ThreadPoolExecutor(max_workers=21 if not interface else 8) as pool:
                result=list(pool.map(remote_one,range(jobs)))
            assert len(backend._tasks)==jobs
            return result
        finally:
            server.shutdown();thread.join();backend._task_pool.shutdown(wait=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--small',action='store_true');a=p.parse_args()
    root=a.root.resolve();root.mkdir(parents=True,exist_ok=False);(root/'queue').mkdir()
    for i in range(32):(root/'queue'/f'{i:03}.lock').touch()
    (root/'queue/startup').mkdir()
    for i in range(4):(root/'queue/startup'/f'{i:03}.lock').touch()
    plan=dict(implementation=source_identity(),image=IMAGE,small=a.small,
              global_jobs=32,interface_slots=8,jobs_per_process=32,startup_slots=1,background_threads=1,solver_jobs=0 if a.small else 696,
              interface_jobs=12 if a.small else 612,template_sha256=hashlib.sha256(Path(__file__).with_name('stochastic_solver.py').read_bytes()).hexdigest())
    (root/'plan.json').write_text(json.dumps(plan,indent=2))
    with ProcessPoolExecutor(max_workers=8) as pool:
        tasks=[]
        if not a.small:
            tasks=[pool.submit(controller,(str(root),f'solver-{i}',False,116,False)) for i in range(6)]
            time.sleep(30)
        tasks += [pool.submit(controller,(str(root),f'interface-{i}',True,6 if a.small else 306,a.small)) for i in range(2)]
        rows=[row for future in tasks for row in future.result()]
    events=sorted([(r['acquired'],1) for r in rows]+[(r['released'],-1) for r in rows])
    active=peak=0
    for _,delta in events:active+=delta;peak=max(peak,active)
    passed=all(r['legal'] for r in rows) and (a.small or peak==32)
    result=dict(status='passed' if passed else 'failed',plan=plan,jobs=len(rows),peak_active=peak,
                failures=[r for r in rows if not r['legal']],rows=rows)
    (root/'completion.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ('status','jobs','peak_active')}),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
