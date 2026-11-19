"""Exercise a fresh RSI risk-only service over its client socket (real evaluations)."""
import argparse
import json
from pathlib import Path
import socket

from tasks.copper_inversion.rsi.exposure import FIELDS, validate_feedback

from tasks.copper_inversion.experiments.rsi.isolation_evidence import RPC_CHECKS as CHECKS


def request(path, payload):
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(1800)
        connection.connect(str(path))
        connection.sendall(json.dumps(payload).encode() + b'\n')
        with connection.makefile('rb') as stream:
            return json.loads(stream.readline(1024 * 1024))


def probe(socket_path, private_root, release_dir, *, send=request):
    checks = {key: False for key in CHECKS}
    report = {'schema': 'rsi-rpc-probe-v1', 'status': 'failed', 'checks': checks,
              'socket': str(socket_path), 'private_root': str(private_root), 'release_dir': str(release_dir),
              'slots': None, 'final_used_slots': None,
              'scope': 'Risk-only RPC and solver mount probe; not an adversarial guest attack suite.'}
    def call(payload):
        return send(socket_path, payload)
    def info():
        return call({'operation': 'info'})
    def valid_feedback(value):
        try:
            validate_feedback(value)
            return set(value) == FIELDS
        except (ValueError, TypeError, KeyError):
            return False
    try:
        if any(not Path(p).is_absolute() or str(Path(p)) == '/' for p in (private_root, release_dir)):
            raise ValueError('private root and release directory must be absolute non-root paths')
        initial = info(); slots = initial.get('slots'); digest = initial.get('contract_digest')
        report['slots'] = slots
        checks['info_ready'] = (initial.get('status') == 'ready' and initial.get('used_slots') == 0
                                and type(slots) is int and slots >= 2 and isinstance(digest, str) and bool(digest))
        if not checks['info_ready']:
            raise ValueError('service must be fresh and have at least two slots')
        source = ('import os\nfrom tasks.copper_inversion.v2.baselines import initialize\n'
                  'def solve(problem,budget,seed):\n'
                  f'    for path in {tuple(map(str, ("/truth", "/var/run/docker.sock", private_root, release_dir)))!r}:\n'
                  '        assert not os.path.exists(path), "unexpected privileged mount"\n'
                  '    return initialize(problem)\n')
        evaluate = lambda src: {'operation': 'evaluate', 'source': src, 'contract_digest': digest}
        checks['missing_digest_rejected'] = call({'operation': 'evaluate', 'source': source}) == {'status': 'contract_mismatch'}
        checks['wrong_digest_rejected'] = call({**evaluate(source), 'contract_digest': 'wrong'}) == {'status': 'contract_mismatch'}
        checks['request_path_override_rejected'] = call({**evaluate(source), 'manifest': '/truth/truth.npy'}) == {'status': 'rejected_request'}
        checks['unknown_operation_rejected'] = call({'operation': 'unknown', 'contract_digest': digest}) == {'status': 'rejected_request'}
        checks['rejected_requests_free'] = info().get('used_slots') == 0
        if not all(checks[k] for k in ('missing_digest_rejected', 'wrong_digest_rejected',
                                        'request_path_override_rejected', 'unknown_operation_rejected', 'rejected_requests_free')):
            raise ValueError('request gate failed; stopping before evaluations')
        first = call(evaluate(source))
        checks['private_mounts_absent'] = first.get('status') == 'valid' and first.get('failed_solves') == 0
        checks['feedback_whitelist'] = valid_feedback(first)
        used_after_first = info().get('used_slots')
        replay = call(evaluate(source))
        checks['replay_identical'] = replay == first
        checks['replay_free'] = used_after_first == 1 and info().get('used_slots') == 1
        invalid = call(evaluate("import numpy as np\ndef solve(problem,budget,seed):\n    return np.ones(problem['shape'])\n"))
        checks['invalid_output_rejected'] = valid_feedback(invalid) and invalid.get('status') == 'rejected' and invalid.get('message') == 'interface rejected'
        # Stop if an unexpected response makes the scientific slot state ambiguous.
        if not all(checks[k] for k in ('private_mounts_absent', 'feedback_whitelist', 'replay_identical',
                                       'replay_free', 'invalid_output_rejected')) or info().get('used_slots') != 2:
            raise ValueError('initial evaluations failed or slot count differs')
        for index in range(slots - 2):
            result = call(evaluate(source + f'# probe candidate {index}\n'))
            if not valid_feedback(result) or result.get('status') != 'valid':
                raise ValueError('slot-fill evaluation failed')
        extra = call(evaluate(source + '# over budget\n'))
        final = info(); report['final_used_slots'] = final.get('used_slots')
        checks['feedback_cap_enforced'] = (extra == {'status': 'budget_exhausted'} and
                                           report['final_used_slots'] == slots)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        report['error'] = str(exc)
    report['status'] = 'passed' if all(checks.values()) else 'failed'
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('socket', 'private-root', 'release-dir', 'output'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args(argv)
    result = probe(a.socket, a.private_root, a.release_dir)
    with Path(a.output).open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True); f.write('\n')
    print(json.dumps(result, sort_keys=True))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
