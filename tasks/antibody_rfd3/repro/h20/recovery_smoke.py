"""Kill an offline PoC controller while work is pending, then resume the same run."""
import argparse
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

from audit_async_smoke import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    directory = root / "run"
    repo = Path(__file__).resolve().parents[4]
    command = [sys.executable, "-c", "from tasks.antibody_rfd3.core.workflow import parse_args, run; "
               "raise SystemExit(run(parse_args()))", "--mock",
               "--method", "fixed", "--iterations", "12", "--concurrency", "2", "--out-dir", str(directory)]
    env = dict(os.environ, PYTHONPATH=str(repo), CUDA_VISIBLE_DEVICES="")
    first = second = None
    service = None
    report = {"run_dir": str(directory)}
    try:
        with (root / "before.log").open("w") as log:
            first = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline and first.poll() is None:
                database = directory / "controller.sqlite3"
                state_path = directory / "experiment-service-process.json"
                if database.exists() and state_path.exists():
                    try:
                        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
                            pending = db.execute("SELECT COUNT(*) FROM experiments WHERE pending=1 AND submit_state='ACKED'").fetchone()[0]
                            count = db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
                        if pending:
                            service = json.loads(state_path.read_text())
                            # Stop only the controller, leaving its independent HTTP service alive.
                            os.kill(first.pid, signal.SIGKILL)
                            first.wait()
                            report.update(observations_before=count, pending_before=pending,
                                          service_before=service, killed_controller=first.pid)
                            os.kill(service["pid"], 0)
                            break
                    except sqlite3.OperationalError:
                        pass
                time.sleep(0.005)
            else:
                raise RuntimeError("Did not observe in-flight work before controller completion")
        with (root / "after.log").open("w") as log:
            second = subprocess.Popen(command + ["--resume-run", str(directory)], cwd=repo, env=env,
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            code = second.wait(timeout=120)
        if code:
            raise RuntimeError(f"Resumed controller exited {code}; see after.log")
        reused = json.loads((directory / "experiment-service-process.json").read_text())
        if reused != service:
            raise RuntimeError("Resume replaced the live experiment service")
        report.update(audit(directory), reused_live_service=True)
        if report["problems"]:
            raise RuntimeError("Post-recovery audit failed")
        report["passed"] = True
    except Exception as exc:
        report.update(passed=False, error=f"{type(exc).__name__}: {exc}")
    finally:
        for process in (first, second):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
        if service is not None:
            from tasks.antibody_rfd3.core.campaign import ServiceProcess
            ServiceProcess(service["pid"], directory / "experiment-service.json").stop()
        (root / "recovery-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "report": str(root / "recovery-audit.json")}))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
