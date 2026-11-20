"""Run and audit the H20 HER2 comparison using a common initialization bank."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import sqlite3
import subprocess

from ldm_tts.harness import file_sha256
from ldm_tts.optimization.tk_ard import TK_ARD_VERSION
from tasks.antibody.tools.structure_design.common import extract_h3
from tasks.antibody_rfd3.repro.h20.launch_utils import (
    ACTIVE_ROOT, CAMPAIGN_PYTHON, REPO, ROOT, now, private_environment_values, run_case, save_summary,
)

ANCHOR = ROOT / "runs/her2-stream-640-d8808f4d"
METHODS = ("ldm_harness", "fixed", "harness", "antbo")
MAPPING = {"ldm_harness": ("8-15", 0), "fixed": ("16,17", 1),
           "harness": ("18,19", 2), "antbo": ("20,21", 0)}
SEED_COUNT = 16
IMAGE = "ldm-copper-pi:antibody-async-20261118"
GUEST_CACHE = ROOT / "cache/gondolin"
ASSETS = ANCHOR / "absolut"
ANTIGEN = "1N8Z_C"


def query(path, sql):
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        return db.execute(sql).fetchall()


def snapshot_backbone(source, dest):
    document = json.loads(source.read_text())
    if document["status"] != "success" or len(document["candidates"]) != 1:
        raise ValueError("Expected a successful, single-backbone source manifest")
    candidate = document["candidates"][0]
    dest.mkdir(parents=True, exist_ok=False)
    copied = {}
    for field, name in (("backbone_path", "backbone.pdb"), ("structure_path", "structure.cif.gz")):
        original = Path(candidate[field]).resolve(strict=True)
        if not original.is_relative_to(ANCHOR.resolve()):
            raise ValueError("Seed structure is outside its completed source run")
        target = dest / name
        shutil.copyfile(original, target)
        checksum = file_sha256(target)
        if checksum != file_sha256(original):
            raise ValueError("Copied seed structure differs from its source")
        target.chmod(0o444)
        copied[field] = checksum
        candidate[field] = str(target)
    manifest = dest / "generation.json"
    save_summary(manifest, document)
    manifest.chmod(0o444)
    return {"manifest": str(manifest.relative_to(dest.parent.parent)),
            "manifest_sha256": file_sha256(manifest), "pdb_sha256": copied["backbone_path"],
            "cif_sha256": copied["structure_path"], "source_generation": str(source),
            "source_generation_sha256": file_sha256(source)}


def seed_bank(path):
    ledger = ANCHOR / "campaign/controller.sqlite3"
    rows = query(ledger, "SELECT observation_json,status FROM observations ORDER BY ingest_seq LIMIT 16")
    if len(rows) != SEED_COUNT or any(status != "succeeded" for _, status in rows):
        raise ValueError("The common initial observations are missing")
    initial = [json.loads(row) for row, _ in rows]
    candidates, source_backbones = [], {}
    for o in initial:
        payload = o["candidate"]["payload"]
        seq, backbone_id = o["candidate"]["canonical_key"], payload["backbone_id"]
        design = (ledger.parent / payload["artifact"]).resolve(strict=True)
        if not design.is_relative_to(ledger.parent):
            raise ValueError("Seed design is outside the source campaign")
        design_rows = json.loads(design.read_text())["candidates"]
        if extract_h3(design_rows[payload["index"]], 11) != seq:
            raise ValueError("Seed sequence differs from its archived design")
        source = (ledger.parent / "design_jobs" / backbone_id / "output/generation.json").resolve(strict=True)
        if not source.is_relative_to(ledger.parent):
            raise ValueError("Seed backbone is outside the source campaign")
        key = "seed-" + backbone_id
        source_backbones[key] = source
        candidates.append({"sequence": seq, "backbone_id": key})
    bank = {"antigen": ANTIGEN, "oracle": "absolut", "source_ledger": str(ledger), "candidates": candidates,
            "expected_scores": [o["evaluation"]["metrics"]["absolut_energy"] for o in initial]}
    if path.is_file():
        existing = json.loads(path.read_text())
        if {k: v for k, v in existing.items() if k != "backbones"} != bank:
            raise ValueError("The common initialization bank changed")
        return existing
    bank["backbones"] = {key: snapshot_backbone(source, path.parent / "seed-backbones" / key)
                         for key, source in sorted(source_backbones.items())}
    save_summary(path, bank)
    path.chmod(0o444)
    return bank


def make_case(method, phase, output, seed_path):
    evaluations = 24 if phase == "pilot" else 640
    cpus, gpu = MAPPING[method]
    python = ROOT / "env/antbo_native/bin/python" if method == "antbo" else CAMPAIGN_PYTHON
    config = "her2_ldm_streaming" if method == "ldm_harness" else f"her2_{method}_real"
    command = ["taskset", "-c", cpus, str(python), "scripts/run_ldm_tts.py", f"config/antibody_rfd3/{config}.yaml"]
    common = {"out-dir": str(output / method), "iterations": evaluations, "antigen": ANTIGEN,
              "seed-bank": str(seed_path), "gpu": gpu, "absolut-path": str(ASSETS), "share-gpu": True}
    if method == "ldm_harness":
        if phase == "pilot":
            common.update({"pool-size": 16, "research-candidates-per-role": 8, "evaluation-window": 2, "concurrency": 2,
                           "backbones-per-round": 8, "sequences-per-round": 128, "samples-per-call": 16,
                           "research-rounds": 8})
    else:
        common.update({"initialization-run": None, "absolut-resources": [str(ASSETS)]})
        if method == "harness":
            common.update({"sidecar-image": IMAGE, "guest-cache": str(GUEST_CACHE)})
    for key, value in common.items():
        command.extend(("--set", f"args.{key}={json.dumps(value)}"))
    return {"method": method, "command": command, "log": str(output / (method + ".log")),
            "status": "pending", "effective_args": {"out_dir": common["out-dir"],
            "guest_cache": str(GUEST_CACHE) if method in ("ldm_harness", "harness") else None,
            "absolut_path": str(ASSETS), "initial_evaluations": SEED_COUNT, "iterations": evaluations}}


def observations(directory, method):
    if method == "ldm_harness":
        return [json.loads(row[0]) for row in query(directory / "controller.sqlite3",
                                                   "SELECT observation_json FROM observations ORDER BY ingest_seq")]
    return [event["payload"] for event in map(json.loads, (directory / "events.jsonl").read_text().splitlines())
            if event["event_type"] == "candidate_evaluated"]


def audit(cases, bank):
    expected = dict(zip((c["sequence"] for c in bank["candidates"]), bank["expected_scores"], strict=True))
    known_backbones = {c["sequence"]: c["backbone_id"] for c in bank["candidates"]}
    arms = {}
    for method, case in cases.items():
        reused = bool(case.get("reused_reference"))
        directory = Path(case["run_dir"])
        rows = observations(directory, method)
        scores = [(o["candidate"]["canonical_key"], o["evaluation"]["metrics"]["absolut_energy"]) for o in rows]
        if (len(rows) != case["effective_args"]["iterations"] or
                any(o["evaluation"]["status"] != "succeeded" for o in rows) or len(dict(scores)) != len(rows)
                or dict(scores[:SEED_COUNT]) != expected):
            raise RuntimeError(f"{method} violates the paired score budget or shared initialization")
        if not reused and any(o["candidate"]["payload"].get("backbone_id") !=
                             known_backbones[o["candidate"]["canonical_key"]] for o in rows[:SEED_COUNT]):
            raise RuntimeError(f"{method} lost shared backbone provenance")
        kernel_records, replay_notes = [], {}
        if method == "ldm_harness":
            if not (directory / "experiment-service.sqlite3").is_file():
                raise RuntimeError("Missing independent LDM experiment service")
            generation_bases = query(directory / "controller.sqlite3",
                                     "SELECT base_ingest_seq,state FROM generations ORDER BY generation")
            if not generation_bases or generation_bases[0][0] != SEED_COUNT:
                raise RuntimeError("LDM started policy research before the seed bank was measured")
            decisions = [json.loads(row[0]) for row in query(directory / "controller.sqlite3",
                         "SELECT snapshot_json FROM decisions ORDER BY decision_id")]
            kernel_records = [d["record"]["selection"]["metadata"]["surrogate"] for d in decisions[SEED_COUNT:]]
            adaptive = decisions[SEED_COUNT:]
            heads = [d["head_ingest_seq"] for d in adaptive]
            if not heads or max(heads) <= SEED_COUNT:
                raise RuntimeError("LDM exhausted the online budget before using new feedback")
            jobs = [json.loads(p.read_text()) for p in (directory / "design_jobs").glob("*/receipt.json")]
            seed_reuse = [j for j in jobs if j.get("requested_backbone_id") in bank["backbones"]
                          and j["action"] == "sequence_h3" and j["status"] == "succeeded"]
            replay_notes = {"online_history_heads": heads,
                            "online_decisions_with_new_feedback": sum(h > SEED_COUNT for h in heads),
                            "online_decisions_with_pending": sum(bool(d["inputs"]["pending"]) for d in adaptive),
                            "successful_seed_backbone_calls": len(seed_reuse)}
            if any("plan" in d for d in decisions):
                raise RuntimeError("LDM used a barrier plan in the streaming comparison")
        elif (directory / "controller.sqlite3").exists():
            raise RuntimeError(f"{method} unexpectedly ran through the asynchronous controller")
        if method == "antbo":
            proposals = [o["candidate"]["payload"]["antbo_proposal"] for o in rows[SEED_COUNT:]]
            if any(p["optimizer"] != "bo.optimizer.Optimizer" or p["history_count"] != i
                   for i, p in enumerate(proposals, SEED_COUNT)):
                raise RuntimeError("AntBO did not use its native optimizer with the complete measured history")
        if method == "ldm_harness":
            if not kernel_records or any(record["kernel"] != TK_ARD_VERSION or
                     len(record["position_weights"]) != 11 or not math.isclose(sum(record["position_weights"]), 1)
                     for record in kernel_records):
                raise RuntimeError(f"{method} did not record the shared TK-ARD specification")
        best = math.inf
        curve = []
        for _, energy in scores:
            best = min(best, energy)
            curve.append(best)
        arms[method] = {"successful_scores": len(rows), "seed_best": curve[SEED_COUNT - 1],
                        "best_so_far": curve, "new_best": min(score for _, score in scores[SEED_COUNT:]),
                        "gp_fits_recorded": len(kernel_records), "elapsed_seconds": case["elapsed_seconds"],
                        "reused_reference": reused, "run_dir": str(directory), **replay_notes}
    return arms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--phase", choices=("pilot", "full"), required=True)
    parser.add_argument("--antbo-reference", type=Path,
                        help="Use a completed, matched AntBO pilot as a read-only comparison")
    args = parser.parse_args()
    root = args.root.resolve()
    if ACTIVE_ROOT not in root.parents:
        parser.error("Use a run root below the H20 active data root")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    root.mkdir(parents=True, exist_ok=True)
    seed_path = root / "shared-initialization.json"
    if args.phase == "full":
        acceptance = json.loads((root / "pilot/report.json").read_text())
        if (acceptance["status"] != "passed" or acceptance["commit"] != commit or
                set(acceptance.get("methods", {})) != set(METHODS)):
            parser.error("The pilot must pass on the same code before the full comparison")
    if args.antbo_reference is not None and args.phase != "pilot":
        parser.error("A reused AntBO reference is only allowed for the small comparison")
    bank = seed_bank(seed_path)
    output = root / args.phase
    output.mkdir(exist_ok=False)
    env = dict(os.environ, ANTBO_HOME=str(ROOT / "env/sources/HEBO-77636146/AntBO"),
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
               HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false", TMPDIR=str(ROOT / "tmp"),
               LLM_REASONING_EFFORT="low", NO_PROXY="localhost,127.0.0.1,172.17.0.1",
               no_proxy="localhost,127.0.0.1,172.17.0.1")
    private = private_environment_values(env, env["LLM_API_KEY"])
    methods = ("ldm_harness",) if args.antbo_reference is not None else METHODS
    cases = {method: make_case(method, args.phase, output, seed_path) for method in methods}
    if args.antbo_reference is not None:
        reference = json.loads((args.antbo_reference / "antbo.json").read_text())
        if reference["status"] != "completed":
            parser.error("The AntBO reference must be complete")
        cases["antbo"] = {**reference, "reused_reference": True}
    report = {"status": "running", "phase": args.phase, "commit": commit, "started_at": now(),
              "seed_bank_sha256": hashlib.sha256(seed_path.read_bytes()).hexdigest(),
              "model": env["LLM_MODEL_NAME"]}
    save_summary(output / "report.json", report)

    def run(method):
        case = cases[method]
        path = output / (method + ".json")
        run_case(case, env, 172800 if args.phase == "full" else 9000, private,
                 lambda: save_summary(path, case))

    try:
        with ThreadPoolExecutor(max_workers=4) as workers:
            list(workers.map(run, methods))
        report["methods"] = audit(cases, bank)
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report["finished_at"] = now()
        save_summary(output / "report.json", report)


if __name__ == "__main__":
    main()
