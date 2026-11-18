#!/usr/bin/env python3
"""Prepare an isolated Absolut directory and run a configured real research panel.

Credentials are loaded by the caller on H20 and never written to the manifest.
Run from a clean, committed checkout. Every invocation requires a new run root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import yaml

from tasks.antibody_rfd3.repro.h20.async_smoke import (
    ACTIVE_ROOT, CAMPAIGN_PYTHON, REPO, now, private_environment_values,
    run_case, save_summary,
)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=REPO / "config/antibody_rfd3/her2_ldm_research.yaml")
    parser.add_argument("--rounds", type=int,
                        help="Override batch research rounds and set evaluation budget to rounds * batch-size")
    parser.add_argument("--evaluations", type=int, help="Override the streaming evaluation budget")
    parser.add_argument("--cpu-set", help="Explicit comma-separated CPU IDs, matching configured cpu-limit")
    parser.add_argument("--absolut-source", type=Path, required=True,
                        help="Prepared HER2 cache directory with a sibling launch.json asset manifest")
    parser.add_argument("--timeout", type=int, default=18000)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    source = args.absolut_source.resolve()
    if ACTIVE_ROOT not in root.parents or ACTIVE_ROOT not in source.parents:
        parser.error("Both directories must be under /Aidata/shared/ldm")
    if root.exists():
        parser.error("Use a new output directory; existing runs are never overwritten")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip():
        parser.error("Commit the checkout before launching the reproducible run")
    for name in ("LLM_BASE_URL", "LLM_MODEL_NAME", "LLM_API_KEY"):
        if not os.environ.get(name):
            parser.error(f"Missing H20 environment variable: {name}")
    if os.environ.get("LLM_REASONING_EFFORT") != "low":
        parser.error("LLM_REASONING_EFFORT must be low")
    source_manifest = json.loads((source.parent / "launch.json").read_text())
    asset_records = source_manifest["assets"]
    if source_manifest.get("antigen", source_manifest.get("case", {}).get("effective_args", {}).get("antigen")) != "1N8Z_C":
        parser.error("Asset manifest must identify HER2 (1N8Z_C)")
    if not (source / "src/bin/Absolut").is_file():
        parser.error("Absolut source must contain src/bin/Absolut")
    config = args.config.resolve()
    effective = {key.replace("-", "_"): value
                 for key, value in yaml.safe_load(config.read_text())["args"].items()}
    overrides = ["out_dir", "absolut_path"]
    if args.evaluations is not None:
        if args.evaluations < 1 or args.rounds is not None or effective.get("pacing_mode") != "streaming":
            parser.error("--evaluations requires streaming mode, a positive count and no --rounds")
        effective["iterations"] = args.evaluations
        overrides.append("iterations")
    if args.rounds is not None:
        if args.rounds < 1 or effective.get("batch_size", 1) <= 1:
            parser.error("--rounds requires a positive count and a configured batch-size greater than one")
        effective.update(research_rounds=args.rounds, iterations=args.rounds * effective["batch_size"])
        overrides.extend(("research_rounds", "iterations"))
    effective.update(out_dir=str(root / "campaign"), absolut_path=str(root / "absolut"))
    available = os.sched_getaffinity(0)
    if args.cpu_set is not None:
        try:
            selected = [int(value) for value in args.cpu_set.split(",")]
        except ValueError:
            parser.error("--cpu-set must contain comma-separated integer CPU IDs")
        if len(selected) != len(set(selected)) or len(selected) != effective["cpu_limit"] or not set(selected) <= available:
            parser.error("--cpu-set must contain cpu-limit distinct available CPU IDs")
    else:
        selected = sorted(available)[:effective["cpu_limit"]]
    cpus = ",".join(map(str, selected))
    command = ["taskset", "-c", cpus, str(CAMPAIGN_PYTHON), "scripts/run_ldm_tts.py", str(config)]
    for key in overrides:
        command.extend(("--set", f"args.{key.replace('_', '-')}={json.dumps(effective[key])}"))
    image = effective["sidecar_image"]
    image_id = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True).strip()
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    case = {"method": "ldm_harness", "effective_args": effective, "command": command,
            "status": "preparing", "log": str(root / "logs/ldm_harness.log")}
    manifest = {"status": "preparing", "started_at": now(), "checkout": str(REPO),
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                "config_sha256": digest(config), "model": os.environ["LLM_MODEL_NAME"],
                "thinking": os.environ["LLM_REASONING_EFFORT"], "sidecar_image": image,
                "cpu_affinity": selected,
                "sidecar_image_id": image_id, "absolut_source": str(source), "assets": [], "case": case}
    save = lambda: save_summary(root / "launch.json", manifest)
    save()
    try:
        target = root / "absolut"
        target.mkdir()
        # Copy only immutable scoring inputs, not prior result files or lock files.
        for record in asset_records:
            relative = Path(record["name"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Asset names must be relative paths inside the scoring directory")
            path = source / relative
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["cp", "--reflink=auto", "--dereference", str(path), str(destination)], check=True)
            checksum = digest(destination)
            if checksum != record["sha256"]:
                raise RuntimeError(f"Copy verification failed: {relative}")
            manifest["assets"].append({"name": str(relative), "bytes": destination.stat().st_size,
                                       "sha256": checksum})
            destination.chmod(0o555 if relative == Path("src/bin/Absolut") else 0o444)
        save_summary(target / "assets.json", {"schema_version": 1, "antigen": effective["antigen"],
                                              "assets": manifest["assets"]})
        (target / "assets.json").chmod(0o444)
        save()
        env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
                   HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        env["NO_PROXY"] = ",".join(filter(None, (env.get("NO_PROXY"), "localhost,127.0.0.1,172.17.0.1")))
        private = private_environment_values(env, env["LLM_API_KEY"])
        manifest["status"] = "running"
        run_case(case, env, args.timeout, private, save)
        manifest["status"] = "completed"
    except BaseException:
        manifest["status"] = "failed"
        raise
    finally:
        manifest["finished_at"] = now()
        save()


if __name__ == "__main__":
    main()
