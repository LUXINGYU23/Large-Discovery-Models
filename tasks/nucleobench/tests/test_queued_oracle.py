from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
import pytest
from ldm_tts.contracts.evaluation import EvaluationPaused
from tasks.nucleobench.core.oracles.queued import OracleQueue,QueuedOracle,request_identity


def test_queue_serializes_clients_and_replays_only_within_campaign(tmp_path):
    calls=[];active=0;peak=0
    def model(sequences):
        nonlocal active,peak
        active+=1;peak=max(peak,active);time.sleep(.01)
        calls.append(list(sequences));active-=1
        return [len(s) for s in sequences]
    socket=tmp_path/'oracle.sock'
    with OracleQueue(socket,model,'identity',tmp_path/'receipts') as server:
        thread=threading.Thread(target=server.serve_forever);thread.start()
        try:
            with ThreadPoolExecutor(max_workers=15) as pool:
                outputs=list(pool.map(lambda i:QueuedOracle(socket,str(i),'identity')(['ACGT']),range(15)))
            assert outputs==[[4.0]]*15 and peak==1 and len(calls)==15
            assert QueuedOracle(socket,'0','identity')(['ACGT'])==[4.0]
            assert len(calls)==15
            with pytest.raises(EvaluationPaused,match='identity mismatch'):
                QueuedOracle(socket,'0','wrong')(['ACGT'])
            with pytest.raises(EvaluationPaused,match='minibatch'):
                QueuedOracle(socket,'0','identity')(['A']*5)
            assert len(calls)==15
        finally:server.shutdown();thread.join()


def test_queue_unknown_dispatch_never_repeats_model(tmp_path):
    socket=tmp_path/'oracle.sock';receipts=tmp_path/'receipts';receipts.mkdir()
    key=request_identity('run','identity',['AC'])
    (receipts/f'{key}.json').write_text(json.dumps({'status':'dispatched'}))
    def model(_):raise AssertionError('must not dispatch')
    with OracleQueue(socket,model,'identity',receipts) as server:
        thread=threading.Thread(target=server.serve_forever);thread.start()
        try:
            with pytest.raises(EvaluationPaused,match='unknown outcome'):
                QueuedOracle(socket,'run','identity')(['AC'])
        finally:server.shutdown();thread.join()
