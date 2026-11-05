import hashlib
import threading
from types import SimpleNamespace
import pytest
from tasks.copper_inversion.v2.async_transport import AsyncServiceMixin, evaluate_request
from tasks.copper_inversion.core.execution import digest
from ldm_tts.contracts.evaluation import EvaluationPaused


class Backend:
    def __init__(self, root):
        self.root = root
        self.identity = {"test": 1}
        self.slot_limit = 3
        self.calls = 0
        self.release = threading.Event()
    def dispatch(self, request):
        if request['operation'] == 'evaluate':
            self.calls += 1
            assert self.release.wait(3)
            return {'status': 'succeeded', 'source_sha256': hashlib.sha256(request['source'].encode()).hexdigest()}
        return {'status': 'sealed'}


class Service(AsyncServiceMixin, Backend):
    pass


def test_duplicate_submit_disconnect_and_seal(tmp_path):
    s = Service(tmp_path)
    request = dict(operation='submit', source='same bytes', contract_digest=digest(s.identity))
    try:
        first = s.dispatch(request)
        for _ in range(20):
            assert s.dispatch(request)['source_sha256'] == first['source_sha256']
        assert s.dispatch(dict(operation='seal_training', contract_digest=digest(s.identity)))['status'] == 'training_active'
        assert s.dispatch(request | {'contract_digest':'wrong'})['status'] == 'contract_mismatch'
        s.release.set()
        s._task_pool.shutdown(wait=True)
        assert s.calls == 1
        assert s.dispatch(request)['result']['status'] == 'succeeded'
        assert s.dispatch(request | {'operation':'evaluate'})['status'] == 'rejected_request'
    finally:
        s.release.set()
        s._task_pool.shutdown(wait=True)


def test_long_progress_and_lost_reply():
    payload=dict(operation='evaluate', source='x', contract_digest='a'*64)
    key=hashlib.sha256(b'x').hexdigest()
    calls=[]
    def request(p):
        calls.append(p)
        if len(calls)==1:raise EvaluationPaused('private evaluator outcome unavailable; replay same source after reconciliation')
        if len(calls)<8:return dict(status='pending',source_sha256=key,progress=len(calls))
        return dict(status='completed',source_sha256=key,result={'status':'succeeded'})
    assert evaluate_request(None,payload,request,poll_seconds=.01,stall_seconds=.025)['status']=='succeeded'
    assert calls[0]==calls[1]
    assert all(x.get('source','x')=='x' for x in calls)


def test_stalled_service_does_not_wait_forever():
    payload=dict(operation='evaluate',source='x',contract_digest='a'*64)
    key=hashlib.sha256(b'x').hexdigest()
    with pytest.raises(EvaluationPaused,match='no completed-job progress'):
        evaluate_request(None,payload,lambda p:dict(status='pending',source_sha256=key,progress=0),poll_seconds=.005,stall_seconds=.02)


@pytest.mark.parametrize('error',['private evaluator peer identity differs','private evaluator response is incomplete'])
def test_identity_and_protocol_errors_are_not_retried(error):
    def request(p):raise EvaluationPaused(error)
    with pytest.raises(EvaluationPaused,match=error):
        evaluate_request(None,dict(operation='evaluate',source='x'),request,poll_seconds=0)


def test_source_substitution_is_rejected():
    with pytest.raises(EvaluationPaused,match='source identity'):
        evaluate_request(None,dict(operation='evaluate',source='x'),lambda p:dict(status='pending',source_sha256='0'*64,progress=1),poll_seconds=0)


def test_real_unix_parallel_clients_and_peer(tmp_path):
    import os
    import socketserver
    from concurrent.futures import ThreadPoolExecutor
    from tasks.copper_inversion.v2.service import Handler
    from tasks.copper_inversion.v2.service_client import ServiceEvaluator
    class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        pass
    s=Service(tmp_path)
    s.slot_limit=16
    s.caller_uid=os.getuid()
    path=str(tmp_path/'sock')
    with Server(path,Handler) as server:
        server.backend=s
        thread=threading.Thread(target=server.serve_forever);thread.start()
        client=ServiceEvaluator(tmp_path,path,digest(s.identity),service_uid=os.getuid(),timeout=1)
        request=dict(operation='evaluate',source='x',contract_digest=digest(s.identity))
        try:
            # An abandoned connection must not own or cancel the submitted task.
            assert client.request(dict(request,operation='submit'))['status']=='pending'
            s.release.set()
            with ThreadPoolExecutor(max_workers=16) as pool:
                results=list(pool.map(lambda _:evaluate_request(client,request,client.request,poll_seconds=.005,stall_seconds=2),range(32)))
            assert all(x['status']=='succeeded' for x in results)
            assert s.calls==1
            wrong=ServiceEvaluator(tmp_path,path,digest(s.identity),service_uid=os.getuid()+1,timeout=1)
            with pytest.raises(EvaluationPaused,match='peer identity'):
                evaluate_request(wrong,request,wrong.request,poll_seconds=0)
        finally:
            s.release.set();s._task_pool.shutdown(wait=True)
            server.shutdown();thread.join(3)
