import json
from types import SimpleNamespace

import pytest

from tasks.copper_inversion.experiments.rsi import isolation_evidence as iso


def test_isolation_evidence_rejects_each_bad_input(tmp_path, monkeypatch):
    monkeypatch.setattr(iso.os, 'geteuid', lambda: 0)
    monkeypatch.setattr(iso.pwd, 'getpwuid', lambda uid: SimpleNamespace(pw_name='eval'))
    monkeypatch.setattr(iso.pwd, 'getpwnam', lambda name: SimpleNamespace(pw_uid=1000))
    monkeypatch.setattr(iso.grp, 'getgrnam', lambda name: SimpleNamespace(gr_gid=123))
    monkeypatch.setattr(iso.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=1, stderr=b''))
    # Simulate lstat ownership, while retaining real evidence bytes.
    original = iso.Path.lstat
    def lstat(path):
        st = original(path)
        return SimpleNamespace(st_mode=st.st_mode, st_uid=991, st_gid=123)
    monkeypatch.setattr(iso.Path, 'lstat', lstat)
    rows = {
        'guest': dict(status='passed', own_root_paths_absent=True, sibling_paths_absent=True,
                      private_paths_absent=True, host_credentials_absent=True, http_egress_probe_failed=True,
                      read_tool_escape_denied={str(i): True for i in range(5)}),
        'audit': dict(uid=991, account='eval', account_can_read=True, research_host_user_direct_read_denied=True),
        'rpc': dict(status='passed', replay_identical=True, private_mounts_absent=True,
                    request_path_override_rejected=True, invalid_output_rejected=True, feedback_cap_enforced=True, final_used_slots=2),
        'positive': dict(status='control_detected', marker_hits=dict(sentinel=1, private_value=1), iou_hits_by_kind=dict(unexplained=1)),
    }
    paths = {}
    for name, value in rows.items():
        paths[name] = tmp_path / name
        paths[name].write_text(json.dumps(value))
    private = tmp_path / 'private'; private.mkdir(mode=0o700)
    socket = tmp_path / 'socket'; socket.mkdir(mode=0o750)
    release = private / 'release'; release.write_text('private')
    args = dict(**paths, private_roots=[private], release_file=release, socket_dirs=[socket],
                client='client', client_group='client', operator='operator', root=tmp_path)
    result = iso.assemble(**args)
    assert result['status'] == 'passed'
    assert all(result['checks'][n]['sha256'] for n in ('guest_probe', 'access_audit', 'rpc_probe', 'positive_control'))
    for name in rows:
        paths[name].write_text('{}')
        assert iso.assemble(**args)['status'] == 'failed'
        paths[name].write_text(json.dumps(rows[name]))
    private.chmod(0o755)
    assert iso.assemble(**args)['status'] == 'failed'
    private.chmod(0o700); socket.chmod(0o700)
    assert iso.assemble(**args)['status'] == 'failed'
    socket.chmod(0o750)
    monkeypatch.setattr(iso.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stderr=b''))
    assert iso.assemble(**args)['status'] == 'failed'
