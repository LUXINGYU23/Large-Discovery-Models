"""Start one independent private evaluator as a different local account.

Run as root on the authorized host. The manifest and all private data must
already be readable by the service account and inaccessible to research guests.
The launcher creates a new private run and an empty client socket directory.
"""

import argparse
import json
import os
import pwd
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--private-root", required=True, type=Path)
    parser.add_argument("--socket-dir", required=True, type=Path)
    parser.add_argument("--service-user", default="ldm-copper-eval")
    parser.add_argument("--caller-uid", required=True, type=int)
    parser.add_argument("--caller-gid", required=True, type=int)
    parser.add_argument("--image", required=True)
    parser.add_argument("--seconds", required=True, type=int)
    parser.add_argument("--slots", required=True, type=int)
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise PermissionError("starting an isolated evaluator requires the trusted operator")
    account = pwd.getpwnam(args.service_user)
    if account.pw_uid == args.caller_uid:
        raise ValueError("research and evaluator accounts must differ")
    if not args.image.startswith("sha256:") or len(args.image) != 71:
        raise ValueError("pin the evaluator image digest")
    private, socket_dir = args.private_root.resolve(), args.socket_dir.resolve()
    socket_path = socket_dir / "e.sock"
    if len(os.fsencode(socket_path)) >= 104:
        raise ValueError("Unix socket path is too long")
    if not args.manifest.resolve().is_file():
        raise ValueError("private manifest is missing")
    private.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chown(private, account.pw_uid, account.pw_gid)
    private.chmod(0o700)
    socket_dir.mkdir(parents=True, mode=0o750, exist_ok=False)
    os.chown(socket_dir, account.pw_uid, args.caller_gid)
    socket_dir.chmod(0o750)
    configuration = {"manifest": str(args.manifest.resolve()), "root": str(private),
                     "image": args.image, "seconds": args.seconds, "slots": args.slots,
                     "caller_uid": args.caller_uid, "socket": str(socket_path)}
    config_path = private / "service-config.json"
    config_path.write_text(json.dumps(configuration, sort_keys=True))
    os.chown(config_path, account.pw_uid, account.pw_gid)
    config_path.chmod(0o600)
    temporary = private / "tmp"
    temporary.mkdir(mode=0o700)
    os.chown(temporary, account.pw_uid, account.pw_gid)
    docker_config = temporary / "docker"
    docker_config.mkdir(mode=0o700)
    os.chown(docker_config, account.pw_uid, account.pw_gid)
    source_root = Path(__file__).resolve().parents[3]
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(source_root),
           "TMPDIR": str(temporary), "DOCKER_CONFIG": str(docker_config),
           "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    log_path = private / "service.log"
    with log_path.open("ab") as log:
        os.chown(log_path, account.pw_uid, account.pw_gid)
        log_path.chmod(0o600)
        process = subprocess.Popen([sys.executable, "-m", "tasks.copper_inversion.rsi.service",
                                    "--config", str(config_path)], cwd=source_root, env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   user=account.pw_uid, group=account.pw_gid,
                                   extra_groups=os.getgrouplist(account.pw_name, account.pw_gid),
                                   start_new_session=True)
    deadline = time.monotonic() + 45
    ready = None
    while ready is None:
        if socket_path.exists():
            lines = log_path.read_text().splitlines()
            ready = next((json.loads(line) for line in lines if line.startswith('{"status": "listening"')), None)
            if ready:
                break
        if process.poll() is not None or time.monotonic() >= deadline:
            if process.poll() is None:
                process.terminate()
            raise RuntimeError("private evaluator did not start; inspect its private service.log")
        time.sleep(.2)
    os.chown(socket_path, account.pw_uid, args.caller_gid)
    socket_path.chmod(0o660)
    if ready["pid"] != process.pid:
        process.terminate()
        raise RuntimeError("private evaluator did not provide a startup receipt")
    deployment = {"socket": str(socket_path), "contract_digest": ready["contract_digest"],
                  "service_uid": account.pw_uid, "pid": process.pid}
    path = socket_dir / "deployment.json"
    path.write_text(json.dumps(deployment, sort_keys=True) + "\n")
    os.chown(path, account.pw_uid, args.caller_gid)
    path.chmod(0o640)
    print(json.dumps(deployment, sort_keys=True))


if __name__ == "__main__":
    main()
