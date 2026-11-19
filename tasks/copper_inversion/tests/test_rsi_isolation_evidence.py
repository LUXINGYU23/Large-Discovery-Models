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
        'audit': dict(uid=991, account='eval', base=str(tmp_path / 'private'), client='client',
                      canary=str(tmp_path / 'private/access-canary.txt'), release_file=str(tmp_path / 'private/release'),
                      account_can_read=True, research_host_user_direct_read_denied=True,
                      checks={f'{role}_{name}': {'returncode': rc, 'stderr_empty': True}
                              for role, rc in (('evaluator', 0), ('client', 1)) for name in ('canary', 'release_file')}),
        'rpc': dict(schema='rsi-rpc-probe-v1', status='passed',
                    checks={name: True for name in __import__('tasks.copper_inversion.tests.rsi_rpc_probe', fromlist=['CHECKS']).CHECKS},
                    slots=13, final_used_slots=13),
        'positive': dict(control='passed', scan_rc=1, error='blind audit: sentinel, private_value', values_echoed=False),
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
    import copy
    bad_rows = []
    for field, value in [('control', 'failed'), ('scan_rc', 0), ('error', 'sentinel only'), ('values_echoed', True)]:
        row = copy.deepcopy(rows['positive']); row[field] = value; bad_rows.append(('positive', row))
    for field, value in [('status', 'failed'), ('slots', 12), ('final_used_slots', 12), ('schema', 'wrong')]:
        row = copy.deepcopy(rows['rpc']); row[field] = value; bad_rows.append(('rpc', row))
    for check in rows['rpc']['checks']:
        row = copy.deepcopy(rows['rpc']); row['checks'][check] = False; bad_rows.append(('rpc', row))
    for field, value in [('client', 'another'), ('release_file', '/other'), ('account_can_read', False),
                         ('research_host_user_direct_read_denied', False)]:
        row = copy.deepcopy(rows['audit']); row[field] = value; bad_rows.append(('audit', row))
    for check in rows['audit']['checks']:
        row = copy.deepcopy(rows['audit']); row['checks'][check]['returncode'] = 2; bad_rows.append(('audit', row))
    for name, row in bad_rows:
        paths[name].write_text(json.dumps(row))
        assert iso.assemble(**args)['status'] == 'failed', (name, row)
        paths[name].write_text(json.dumps(rows[name]))
    private.chmod(0o755)
    assert iso.assemble(**args)['status'] == 'failed'
    private.chmod(0o700); socket.chmod(0o700)
    assert iso.assemble(**args)['status'] == 'failed'
    socket.chmod(0o750)
    monkeypatch.setattr(iso.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stderr=b''))
    assert iso.assemble(**args)['status'] == 'failed'
