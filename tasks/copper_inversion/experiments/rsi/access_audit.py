"""Audit evaluator/client access to a project-private canary and release file."""
import argparse
import json
import os
from pathlib import Path
import pwd
import stat
import subprocess


def audit(root, client, release_file):
    if os.geteuid() != 0:
        raise PermissionError('run access audit as root')
    base = Path(root).resolve() / 'private'
    evaluator = pwd.getpwuid(991)
    client_entry = pwd.getpwnam(client)
    if client_entry.pw_uid == 991:
        raise ValueError('client and evaluator must be different accounts')
    if not base.is_dir() or base.is_symlink():
        raise ValueError('project private directory is missing or a symlink')
    release = Path(release_file).resolve(strict=True)
    if not release.is_file() or not release.is_relative_to(base):
        raise ValueError('release file must be inside the project private directory')
    canary = base / 'access-canary.txt'
    try:
        fd = os.open(canary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(b'private evaluator access probe\n')
            os.fchown(stream.fileno(), evaluator.pw_uid, evaluator.pw_gid)
            os.fchmod(stream.fileno(), 0o400)
    st = canary.lstat()
    if not stat.S_ISREG(st.st_mode) or st.st_uid != 991 or stat.S_IMODE(st.st_mode) != 0o400:
        raise ValueError('existing canary must be a regular file owned by uid 991 with mode 0400')
    checks = {}
    for label, path in (('canary', canary), ('release_file', release)):
        for role, name in (('evaluator', evaluator.pw_name), ('client', client)):
            result = subprocess.run(['runuser', '-u', name, '--', 'test', '-r', str(path)], capture_output=True)
            checks[f'{role}_{label}'] = {'returncode': result.returncode,
                                         'stderr_empty': not result.stderr}
    allowed = all(checks[f'evaluator_{name}'] == {'returncode': 0, 'stderr_empty': True}
                  for name in ('canary', 'release_file'))
    denied = all(checks[f'client_{name}'] == {'returncode': 1, 'stderr_empty': True}
                 for name in ('canary', 'release_file'))
    return {'account': evaluator.pw_name, 'uid': evaluator.pw_uid, 'base': str(base), 'client': client,
            'canary': str(canary), 'release_file': str(release), 'account_can_read': allowed,
            'research_host_user_direct_read_denied': denied, 'checks': checks}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'client', 'release-file', 'output'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args(argv)
    result = audit(a.root, a.client, a.release_file)
    with Path(a.output).open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True); f.write('\n')
    print(json.dumps(result, sort_keys=True))
    return 0 if result['account_can_read'] and result['research_host_user_direct_read_denied'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
