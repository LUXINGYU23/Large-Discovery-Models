import fcntl
import multiprocessing as mp
import os
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import pytest
from fair_queue_20261028 import JobQueue, prepare
from tasks.copper_inversion.core.execution import EvaluationPaused


def setup(root, slots=1):
    prepare(root)
    for i in range(slots): (root/f'{i:03}.lock').touch()


def until(fn):
    end=time.monotonic()+10
    while not fn():
        assert time.monotonic()<end
        time.sleep(.01)


def child(root, label, output, interface=False):
    q=JobQueue(root,2 if interface else 1,1)
    if interface:q.interface_slots=1
    with q.acquire(interface=interface):
        output.put(label)
        time.sleep(.03)


def test_cross_process_fifo_and_dead_waiter(tmp_path):
    setup(tmp_path)
    hold=os.open(tmp_path/'000.lock',os.O_RDWR);fcntl.flock(hold,fcntl.LOCK_EX)
    ctx=mp.get_context("spawn");out=ctx.Queue();workers=[]
    for i in range(12):
        p=ctx.Process(target=child,args=(tmp_path,i,out));p.start();workers.append(p)
        until(lambda:len(list((tmp_path/'solver').iterdir()))==i+1)
    workers[0].kill();workers[0].join()
    os.close(hold)
    assert [out.get(timeout=10) for _ in range(11)]==list(range(1,12))
    for p in workers[1:]:p.join(10);assert p.exitcode==0
    assert not list((tmp_path/'solver').iterdir())


def test_local_fifo_tail_cannot_be_overtaken(tmp_path):
    setup(tmp_path);q=JobQueue(tmp_path,1,1);order=[]
    def one(i):
        with q.acquire():order.append(i);time.sleep(.005)
    with q.acquire():
        with ThreadPoolExecutor(1) as bootstrap:
            # Start threads explicitly so their arrival is observable while holder runs.
            threads=[]
            for i in range(30):
                t=threading.Thread(target=one,args=(i,));t.start();threads.append(t)
                until(lambda:len(q._waiting)==i+1)
    for t in threads:t.join(10);assert not t.is_alive()
    assert order==list(range(30))


def test_reserved_interface_not_blocked_by_solver(tmp_path):
    setup(tmp_path,2)
    q=JobQueue(tmp_path,2,32);q.interface_slots=1
    with q.acquire():
        with q.acquire(interface=True):pass


def test_pause_cleans_tickets_and_local_waiters(tmp_path):
    setup(tmp_path);q=JobQueue(tmp_path,1,2)
    hold=os.open(tmp_path/'000.lock',os.O_RDWR);fcntl.flock(hold,fcntl.LOCK_EX)
    def wait():
        with pytest.raises(EvaluationPaused):
            with q.acquire():pass
    threads=[threading.Thread(target=wait) for _ in range(4)]
    for t in threads:t.start()
    until(lambda:len(list((tmp_path/'solver').iterdir()))==2)
    q.pause('development test')
    for t in threads:t.join(10);assert not t.is_alive()
    os.close(hold)
    assert q._active==0 and not q._waiting and not list((tmp_path/'solver').iterdir())


def test_exception_releases_slot(tmp_path):
    setup(tmp_path);q=JobQueue(tmp_path,1,1)
    with pytest.raises(ValueError):
        with q.acquire():raise ValueError('fixture')
    with q.acquire():pass


def docker_child(root, label, output):
    import subprocess
    q=JobQueue(root,1,1)
    with q.acquire() as waited:
        admitted=time.monotonic()
        r=subprocess.run(['docker','run','--rm','--network','none','--cpus','2','--memory','4g',
            '--entrypoint','python3','sha256:f6856385a0da9444b1e5edb5048b2f6081593fe568c3473c2fda524d0c4489f5',
            '-c','import time; time.sleep(1)'],capture_output=True,timeout=40)
        output.put((label,r.returncode,waited,admitted,time.monotonic()))


def test_real_docker_asymmetric_tail(tmp_path):
    setup(tmp_path)
    hold=os.open(tmp_path/'000.lock',os.O_RDWR);fcntl.flock(hold,fcntl.LOCK_EX)
    ctx=mp.get_context('spawn');out=ctx.Queue();workers=[]
    # A single tail arrives behind one busy-group task, ahead of ten more.
    labels=['busy-0','tail']+[f'busy-{i}' for i in range(1,11)]
    try:
        for i,label in enumerate(labels):
            p=ctx.Process(target=docker_child,args=(tmp_path,label,out));p.start();workers.append(p)
            until(lambda:len(list((tmp_path/'solver').iterdir()))==i+1)
        os.close(hold);hold=None
        rows=[out.get(timeout=60) for _ in labels]
        assert [r[0] for r in rows]==labels
        assert all(r[1]==0 for r in rows)
        assert rows[1][3]-rows[0][3]<15
        for p in workers:p.join(10);assert p.exitcode==0
        assert not list((tmp_path/'solver').iterdir())
    finally:
        if hold is not None:os.close(hold)
        for p in workers:
            if p.is_alive():p.terminate();p.join()
