import json
import socket
from pathlib import Path
from types import SimpleNamespace
import pytest
from tasks.antibody_rfd3.core.worker import DesignWorker
from tasks.antibody_rfd3.core.harness import worker_socket, TOOLS
from tasks.antibody_rfd3.core.interface import score_interface


def test_soluble_shared_budget(tmp_path):
    args=SimpleNamespace(backbones_per_round=2,sequences_per_round=4,seed=42,mock=True,temperature=.5,samples_per_call=1,initialization_run=None)
    worker=DesignWorker(tmp_path,args)
    first=worker.execute('design_h3',1,.5)
    worker.round=1
    worker.execute('sequence_h3_soluble',2,.5,first[0]['backbone_id'])
    assert worker.remaining()=={'backbones':2,'sequences':2}
    assert list(worker.records())[-1]['action']=='sequence_h3_soluble'
    assert 'sequence_h3_soluble' in TOOLS and 'score_interface' in TOOLS
    with pytest.raises(ValueError): worker.execute('sequence_h3_soluble',1,.5,'../escape')


def test_interface_rejects_paths(tmp_path):
    worker=SimpleNamespace(jobs=tmp_path/'design_jobs')
    with pytest.raises(ValueError): score_interface(worker,'../secret')
    inputs=tmp_path/'interface_inputs';inputs.mkdir()
    (inputs/'escape.json').write_text(json.dumps({'pae':'../../secret','structure':'model.cif'}))
    with pytest.raises(ValueError): score_interface(worker,'escape')
