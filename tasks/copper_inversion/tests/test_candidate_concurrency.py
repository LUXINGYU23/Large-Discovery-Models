from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import pytest
from tasks.copper_inversion.core.candidate import seal

SOURCE=b'def solve(problem,budget,seed):\n    return None\n'+b'# payload\n'*20000


def write_same(root):
    return seal(root,SOURCE)


def test_concurrent_seal_publishes_complete_identical_snapshot(tmp_path):
    for executor in (ThreadPoolExecutor,ProcessPoolExecutor):
        root=tmp_path/executor.__name__
        with executor(max_workers=16) as pool:
            results=list(pool.map(write_same,[root]*64))
        assert all(row==results[0] for row in results)
        path=root/results[0]['source_path']
        assert path.read_bytes()==SOURCE
        assert path.stat().st_mode&0o222==0
        assert list(path.parent.iterdir())==[path]


def test_seal_does_not_overwrite_corrupted_snapshot(tmp_path):
    result=seal(tmp_path,SOURCE);path=tmp_path/result['source_path']
    path.chmod(0o644);path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='snapshot changed'):seal(tmp_path,SOURCE)
    assert path.read_bytes()==b'changed'
