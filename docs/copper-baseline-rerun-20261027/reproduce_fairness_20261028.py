"""Deterministic late-arrival overtaking regression; development queues only."""
import json,tempfile,threading,time
from pathlib import Path
from tasks.copper_inversion.experiments.v2.parallel_runtime import JobQueue as OldQueue
from fair_queue_20261028 import JobQueue,prepare
original_sleep=time.sleep

def run(cls):
    with tempfile.TemporaryDirectory(prefix='fair-overtake-') as temp:
        root=Path(temp);prepare(root);(root/'000.lock').touch()
        q=cls(root,1,32);blocked=threading.Event();release=threading.Event();late_done=threading.Event();order=[]
        def sleep(seconds):
            if threading.current_thread().name=='tail' and not blocked.is_set():
                blocked.set();assert release.wait(10)
            original_sleep(seconds)
        def one(name):
            with q.acquire():order.append(name)
            if name=='late':late_done.set()
        time.sleep=sleep
        tail=threading.Thread(target=one,args=('tail',),name='tail')
        late=threading.Thread(target=one,args=('late',),name='late')
        try:
            with q.acquire():
                tail.start();assert blocked.wait(10)
            late.start()
            overtaken=late_done.wait(.3)
            release.set()
            for t in (tail,late):t.join(10);assert not t.is_alive()
            return dict(order=order,late_overtook=overtaken)
        finally:
            release.set();time.sleep=original_sleep

old=run(OldQueue);new=run(JobQueue)
assert old==dict(order=['late','tail'],late_overtook=True)
assert new==dict(order=['tail','late'],late_overtook=False)
print(json.dumps(dict(old=old,fifo=new,status='passed')))
