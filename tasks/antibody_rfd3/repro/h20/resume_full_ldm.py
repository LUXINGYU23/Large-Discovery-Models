"""Resume a stopped campaign, preserving its launch and pre-recovery databases."""
from __future__ import annotations

import argparse
import copy
from contextlib import closing
import json
import os
import shutil
from pathlib import Path
import sqlite3
import subprocess

from tasks.antibody_rfd3.core.campaign import RunLock
from tasks.antibody_rfd3.repro.h20.launch_utils import (
    ACTIVE_ROOT, REPO, now, private_environment_values, run_case, save_summary,
)
from tasks.antibody_rfd3.repro.h20.run_full_ldm import digest


def backup_database(source: Path, target: Path):
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as original:
        with closing(sqlite3.connect(target)) as backup:
            original.backup(backup)


def resume_case(manifest, campaign: Path, recovery: Path):
    case = copy.deepcopy(manifest["case"])
    for field in ("exit_code", "elapsed_seconds", "started_at", "finished_at"):
        case.pop(field, None)
    # Always build from the initial launch; a second recovery must not accumulate overrides.
    command = manifest.get("initial_command", case["command"])
    case["command"] = [*command, "--set", "args.resume-run=" + json.dumps(str(campaign))]
    case["effective_args"]["resume_run"] = str(campaign)
    case.update(status="preparing", log=str(recovery / "campaign.log"))
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--timeout", type=int, default=172800)
    args = parser.parse_args()
    root = args.run_root.resolve()
    if ACTIVE_ROOT not in root.parents or args.timeout < 1:
        parser.error("Use an existing run under /Aidata/shared/ldm and a positive timeout")
    campaign = root / "campaign"
    manifest = json.loads((root / "launch.json").read_text())
    if Path(manifest["checkout"]).resolve() != REPO:
        parser.error("Resume from the original checkout path to preserve frozen runtime paths")
    if manifest["status"] == "completed":
        parser.error("The run already completed")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip():
        parser.error("Commit the checkout before recovery")
    for name in ("LLM_BASE_URL", "LLM_MODEL_NAME", "LLM_API_KEY"):
        if not os.environ.get(name):
            parser.error(f"Missing environment variable: {name}")
    if os.environ.get("LLM_MODEL_NAME") != manifest["model"] or os.environ.get("LLM_REASONING_EFFORT") != manifest["thinking"]:
        parser.error("Recovery must preserve the model and reasoning effort")
    with RunLock(root / "recovery.lock"):
        # Refuse to snapshot an active campaign. Its own process reacquires this lock on launch.
        with RunLock(campaign / "campaign.lock"):
            recovery = root / "recoveries" / f"attempt-{len(manifest.get('recoveries', [])) + 1:03d}"
            recovery.mkdir(parents=True, exist_ok=False)
            save_summary(recovery / "launch-before.json", manifest)
            for name in ("controller.sqlite3", "experiment-service.sqlite3"):
                backup_database(campaign / name, recovery / name)
            for name in ("summary.json", "status.json", "budget.json"):
                if (campaign / name).is_file():
                    shutil.copyfile(campaign / name, recovery / name)
            case = resume_case(manifest, campaign, recovery)
            entry = {"started_at": now(), "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(), "directory": str(recovery),
                "status": "running", "case": case,
                "database_sha256": {name: digest(recovery / name) for name in
                                    ("controller.sqlite3", "experiment-service.sqlite3")}}
            manifest.setdefault("initial_command", list(manifest["case"]["command"]))
            manifest.setdefault("recoveries", []).append(entry)
            manifest.update(status="running", active_source_commit=entry["source_commit"], case=case)
            manifest.pop("finished_at", None)
        save = lambda: save_summary(root / "launch.json", manifest)
        save()
        env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
                   HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        env["NO_PROXY"] = ",".join(filter(None, (env.get("NO_PROXY"), "localhost,127.0.0.1,172.17.0.1")))
        try:
            run_case(case, env, args.timeout, private_environment_values(env, env["LLM_API_KEY"]), save)
            manifest["status"] = entry["status"] = "completed"
        except BaseException:
            manifest["status"] = entry["status"] = "failed"
            raise
        finally:
            manifest["finished_at"] = entry["finished_at"] = now()
            save_summary(recovery / "recovery.json", entry)
            save()


if __name__ == "__main__":
    main()
