"""Explicit versioned restart: preserve old runs, gate new full runs on six preflights.

Never edits checkpoints, budget receipts, closed turns or scientific identities.
The provider credential is passed to existing stage supervisors through pipes only.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from encoding_ablation import save, read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    base, previous = args.root.resolve(), args.previous.resolve()
    key = sys.stdin.read().strip()
    if not key:
        raise ValueError('protected provider credential required on stdin')
    if (base/'restart.json').exists():
        raise ValueError('restart already dispatched; inspect its durable state')
    if args.detach:
        with (base/'restart.log').open('x') as log:
            child = subprocess.Popen([sys.executable, __file__, '--root', str(base), '--previous', str(previous)],
                stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            child.stdin.write(key.encode())
            child.stdin.close()
        print(json.dumps(dict(pid=child.pid, root=str(base))))
        return
    record = dict(pid=os.getpid(), started_at=time.time(), previous=str(previous), status='stopping_previous')
    save(base/'restart.json', record)
    try:
        # Check exact owned command before signalling its detached process group.
        supervisor = read(previous/'full-supervisor.json')
        pid = supervisor['pid']
        command_path = Path(f'/proc/{pid}/cmdline')
        if command_path.exists():
            command = command_path.read_bytes().split(b'\0')
            if str(previous).encode() not in command or os.getpgid(pid) != pid:
                raise ValueError('previous supervisor ownership mismatch')
            save(previous/'progress-at-restart.json',
                 __import__('progress_encoding').snapshot(previous))
            os.killpg(pid, signal.SIGINT)
            time.sleep(15)
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        # Clean only stopped campaign containers; no shared caches or other tasks.
        identifiers = subprocess.check_output(['docker','ps','-q'], text=True).split()
        for cid in identifiers:
            info = json.loads(subprocess.check_output(['docker','inspect',cid], text=True))[0]
            if any(str(m.get('Source','')).startswith(str(previous/'full')+'/') for m in info['Mounts']):
                subprocess.run(['docker','stop','-t','10',cid], check=True, stdout=subprocess.DEVNULL)
        save(previous/'restart-receipt.json', dict(at=time.time(), replacement=str(base),
            reason='indexed repair feedback, per-candidate retry allowance and terminal-peer barrier fix; old identities retained'))
        for stage in ('tiny','full'):
            record.update(status='running', stage=stage)
            save(base/'restart.json', record)
            script=base/'source/tasks/copper_inversion/experiments/encoding_ablation.py'
            with (base/f'{stage}-supervisor.log').open('x') as log:
                child=subprocess.Popen([sys.executable,str(script),'--root',str(base),'--stage',stage],
                    stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
                record['stage_pid']=child.pid
                save(base/'restart.json',record)
                child.communicate(key.encode())
            if child.returncode:
                raise RuntimeError(f'{stage} failed verification; inspect {stage}-verification.json and logs')
        record.update(status='completed', finished_at=time.time())
    except BaseException as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}', finished_at=time.time())
        raise
    finally:
        save(base/'restart.json', record)


if __name__ == '__main__':
    main()
