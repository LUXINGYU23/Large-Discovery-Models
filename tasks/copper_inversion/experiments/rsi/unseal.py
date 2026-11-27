"""Evaluator-account unseal: one sealed-body evaluation per experiment (spec §8, gate G6).

The input lists every frozen solver of one experiment. Nothing is evaluated
unless the list covers every started run and every freeze on disk, every
run carries the experiment's protocol, and every run whose blind scan listed
unexplained "iou" hits has a signed-off review of that list. Each execution appends one line to the
unseal log. Repeating the same list only replays the stored results.
"""

import argparse
import fcntl
import json
import os
import pwd
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ldm_tts.engine.run_store import atomic_json_write
from tasks.copper_inversion.core.execution import digest
from tasks.copper_inversion.experiments.rsi.freeze import EXPERIMENTS, layout, parse_prefix, run_manifest
from tasks.copper_inversion.rsi.ledger import read
from tasks.copper_inversion.v2 import selection
from tasks.copper_inversion.v2.suite import file_hash
from tasks.copper_inversion.experiments.rsi.q1 import references as load_references


REVIEW_FILE = "blind-review-required.json"
REFERENCE_EXPERIMENTS = ("REF", "REF2", "REF3")


def check_reviews(root, protocol_id, experiment, run_ids, review=None):
    """Runs whose blind scan listed unexplained "iou" hits need a signed-off review of that
    exact list (rsi-blind-review-v1). A run is never dropped for a hit: anything other than
    "explained" stops the unseal for investigation. Returns the review file's digest."""
    needed = {}
    for run_id in run_ids:
        path = layout(root, protocol_id, experiment, run_id)["service"] / REVIEW_FILE
        if path.is_file():
            needed[run_id] = json.loads(path.read_text())["review_sha256"]
    if not needed:
        return None
    if review is None:
        raise ValueError("blind review required for: " + ", ".join(sorted(needed)))
    body = json.loads(Path(review).read_text())
    if body.get("schema") != "rsi-blind-review-v1" or body.get("experiment") != experiment:
        raise ValueError("invalid blind review file")
    for run_id, sha in sorted(needed.items()):
        entry = body.get("runs", {}).get(run_id) or {}
        if entry.get("review_sha256") != sha or not entry.get("reviewer"):
            raise ValueError(f"run {run_id}: blind review does not cover its current hit list")
        if entry.get("verdict") != "explained":
            raise ValueError(f"run {run_id}: blind review verdict is {entry.get('verdict')!r}; investigate")
    return digest(body)


def load_list(value):
    listing = json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value
    if (not isinstance(listing, dict) or set(listing) != {"experiment", "protocol_id", "solvers"}
            or listing["experiment"] not in EXPERIMENTS or not isinstance(listing["solvers"], list)
            or not listing["solvers"]):
        raise ValueError("invalid frozen solver list")
    pairs = []
    for item in listing["solvers"]:
        if not isinstance(item, dict) or set(item) != {"run_id", "prefix"}:
            raise ValueError("invalid frozen solver entry")
        parse_prefix(item["prefix"])
        pairs.append((item["run_id"], item["prefix"]))
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate frozen solver")
    return {"experiment": listing["experiment"], "protocol_id": listing["protocol_id"],
            "solvers": [{"run_id": r, "prefix": p} for r, p in sorted(pairs)]}


def check(root, listing, expected_runs=None):
    """Refuse an incomplete list or a protocol mismatch; return {(run, prefix): selection dir}."""
    experiment, protocol_id = listing["experiment"], listing["protocol_id"]
    paths = layout(root, protocol_id, experiment)
    by_run = {}
    for item in listing["solvers"]:
        by_run.setdefault(item["run_id"], set()).add(item["prefix"])
    started = {p.name for p in paths["runs"].iterdir() if p.is_dir()} if paths["runs"].is_dir() else set()
    frozen_runs = ({p.name for p in paths["selection"].iterdir() if p.is_dir()}
                   if paths["selection"].is_dir() else set())
    missing = (started | frozen_runs | set(expected_runs or ())) - set(by_run)
    if missing:
        raise ValueError("frozen solver list is missing runs: " + ", ".join(sorted(missing)))
    if set(by_run) - started:
        raise ValueError("frozen solver list names runs that were never started: "
                         + ", ".join(sorted(set(by_run) - started)))
    if len({frozenset(v) for v in by_run.values()}) != 1:
        raise ValueError("every run must list the same freeze prefixes")
    targets = {}
    for run_id, prefixes in sorted(by_run.items()):
        run_paths = layout(root, protocol_id, experiment, run_id)
        run_manifest(run_paths["run"], protocol_id, experiment)
        on_disk = ({p.name for p in run_paths["run_selection"].iterdir() if p.is_dir()}
                   if run_paths["run_selection"].is_dir() else set())
        if on_disk != prefixes:
            raise ValueError(f"run {run_id}: frozen prefixes on disk differ from the list")
        rows = read(run_paths["run"] / "ledger.jsonl")
        for prefix in sorted(prefixes):
            target = run_paths["run_selection"] / prefix
            frozen, state = selection.verify(target)
            record = json.loads((target / "prefix.json").read_text())
            round_index = parse_prefix(prefix)
            used = rows if round_index is None else [r for r in rows if r["round"] <= round_index]
            if (record.get("protocol_id") != protocol_id or record.get("experiment") != experiment
                    or record.get("run_id") != run_id or record.get("prefix") != prefix
                    or record.get("frozen_sha256") != digest(frozen)
                    or record.get("ledger_prefix_sha256") != digest(used)
                    or record.get("candidate_sha256s", [c["sha256"] for c in frozen["candidates"]])
                       != [c["sha256"] for c in frozen["candidates"]]):
                raise ValueError(f"run {run_id} {prefix}: freeze record or protocol differs")
            if state["phase"] not in {"selected", "tested"}:
                raise ValueError(f"run {run_id} {prefix}: validate before unsealing")
            targets[(run_id, prefix)] = target
    return targets


def _receipts(targets):
    return {str(p): p.stat().st_mtime_ns for t in targets for name in ("receipt.json", "result.json",
            "private-case-results.json", "test-result.json") for p in Path(t).rglob(name)}


def _log(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _append(path, entry):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def unseal(root, listing, *, expected_runs=None, operator=None, workers=8, review=None):
    listing = load_list(listing)
    experiment, protocol_id = listing["experiment"], listing["protocol_id"]
    paths = layout(root, protocol_id, experiment)
    targets = check(root, listing, expected_runs)
    review_sha = check_reviews(root, protocol_id, experiment, sorted({r for r, _ in targets}), review)
    list_sha = digest(listing)
    previous = [e for e in _log(paths["unseal_log"]) if e["experiment"] == experiment]
    if any(e["frozen_list_sha256"] != list_sha for e in previous):
        raise ValueError("experiment was already unsealed with a different frozen solver list")
    states = {key: selection.verify(target)[1]["phase"] for key, target in targets.items()}
    tested = [key for key, phase in states.items() if phase == "tested"]
    if not previous and tested:
        raise ValueError("sealed evaluation exists without an unseal log entry")
    mode = "unseal" if not previous else "replay" if len(tested) == len(targets) else "resume"
    before = _receipts(targets[key] for key in tested)
    _append(paths["unseal_log"], {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "experiment": experiment,
        "frozen_list_sha256": list_sha, "protocol_id": protocol_id, "mode": mode,
        "operator": operator or pwd.getpwuid(os.geteuid()).pw_name, "uid": os.geteuid(),
        "solvers": len(targets), "blind_review_sha256": review_sha})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {key: pool.submit(selection.run, target, "test") for key, target in targets.items()}
        results = {key: future.result() for key, future in futures.items()}
    after = _receipts(targets[key] for key in tested)
    if before != after:
        raise ValueError("replay changed sealed evaluation receipts")
    completion = {"schema": "rsi-unseal-v1", "experiment": experiment, "protocol_id": protocol_id,
                  "frozen_list_sha256": list_sha, "solvers": listing["solvers"],
                  "results_sha256": {f"{r}/{p}": digest(results[(r, p)]) for r, p in sorted(results)}}
    path = paths["unseal"] / f"{experiment}.json"
    if path.exists() and json.loads(path.read_text()) != completion:
        raise ValueError("unsealed results changed")
    if not path.exists():
        atomic_json_write(path, completion)
    return {"status": "unsealed", "mode": mode, "experiment": experiment, "solvers": len(targets),
            "frozen_list_sha256": list_sha}


def unseal_references(root, protocol_id, refs_path, selections_path, *,
                      operator=None, experiment="REF", workers=1):
    """Unseal precommitted domain references once under the given reference experiment label."""
    if experiment not in REFERENCE_EXPERIMENTS:
        raise ValueError(f"unknown reference experiment: {experiment!r}")
    paths = layout(root, protocol_id, "Q1")
    refs = load_references(refs_path)
    mapping = json.loads(Path(selections_path).read_text())
    if not isinstance(mapping, dict) or set(mapping) != set(refs):
        raise ValueError("reference selections do not match precommitted sources")
    targets, frozen_hashes, selection_hashes = {}, {}, {}
    for name, ref in refs.items():
        target = Path(mapping[name]).resolve()
        frozen, state = selection.verify(target)
        if (len(frozen["candidates"]) != 1 or frozen["candidates"][0]["sha256"] != ref["sha256"]
                or state["phase"] != "selected"):
            raise ValueError("reference must have one matching, validated, untested solver")
        report = json.loads((target / "selection.json").read_text())
        if digest(report) != state["selection_sha256"] or report["selected_sha256"] != ref["sha256"]:
            raise ValueError("reference validation changed")
        targets[name] = target
        frozen_hashes[name] = digest(frozen)
        selection_hashes[name] = state["selection_sha256"]
    paths["unseal_log"].parent.mkdir(parents=True, exist_ok=True)
    lockpath = paths["unseal_log"].parent / f"{experiment}.lock"
    with lockpath.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if any(e.get("experiment") == experiment for e in _log(paths["unseal_log"])):
            raise ValueError(f"{experiment} has already been unsealed or started")
        common = {"experiment": experiment, "protocol_id": protocol_id, "references_sha256": file_hash(refs_path),
                  "reference_kinds": {n: ("precomputed" if "fields" in r else "source") for n, r in sorted(refs.items())},
                  "selection_roots": {n: str(p) for n, p in targets.items()},
                  "frozen_sha256": frozen_hashes, "selection_sha256": selection_hashes,
                  "operator": operator or pwd.getpwuid(os.geteuid()).pw_name, "uid": os.geteuid()}
        _append(paths["unseal_log"], {**common, "mode": "start", "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        if workers <= 1:
            results = {name: selection.run(target, "test") for name, target in targets.items()}
        else:
            with ThreadPoolExecutor(max_workers=min(int(workers), 8)) as pool:
                futures = {name: pool.submit(selection.run, target, "test") for name, target in targets.items()}
                results = {name: fut.result() for name, fut in futures.items()}
        entry = {**common, "mode": "completed", "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "results_sha256": {n: digest(result) for n, result in results.items()}}
        _append(paths["unseal_log"], entry)
    return {"status": "unsealed", "experiment": experiment, "references": len(targets)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="active area, e.g. /Aidata/shared/xingyulu/rsi4geo")
    parser.add_argument("--list", type=Path,
                        help='{"experiment", "protocol_id", "solvers": [{"run_id", "prefix"}, ...]}')
    parser.add_argument("--expected-runs", type=Path, help="JSON list of run ids the experiment must contain")
    parser.add_argument("--operator")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ref-workers", type=int, default=1,
                        help="Parallel selection roots for reference unseal (default 1).")
    parser.add_argument("--blind-review", type=Path,
                        help='{"schema": "rsi-blind-review-v1", "experiment", "runs": {run_id: '
                             '{"review_sha256", "verdict": "explained", "reviewer", "note"}}}')
    parser.add_argument("--experiment", choices=REFERENCE_EXPERIMENTS)
    parser.add_argument("--protocol-id")
    parser.add_argument("--references", type=Path)
    parser.add_argument("--selections", type=Path)
    args = parser.parse_args(argv)
    if args.experiment in REFERENCE_EXPERIMENTS:
        if not all((args.protocol_id, args.references, args.selections)) or args.list:
            parser.error(f"{args.experiment} requires --protocol-id, --references and --selections")
        print(json.dumps(unseal_references(args.root, args.protocol_id, args.references, args.selections,
                                           operator=args.operator, experiment=args.experiment,
                                           workers=args.ref_workers), sort_keys=True))
        return 0
    if args.list is None:
        parser.error("--list is required for discovery experiments")
    expected = json.loads(args.expected_runs.read_text()) if args.expected_runs else None
    print(json.dumps(unseal(args.root, args.list, expected_runs=expected, operator=args.operator,
                            workers=args.workers, review=args.blind_review), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
