"""Evaluator-account freeze of a finished discovery run, or of a round prefix of it.

Finalists are ranked by the private training receipts. The public ledger only
says which programs belong to rounds 0..t, and it must agree with the receipts.
Each prefix has its own selection directory, so a prefix freeze can never
change the full-run freeze. Sealed evaluation happens only in unseal.py.
"""

import argparse
import json
import math
import re
from pathlib import Path

from ldm_tts.engine.run_store import atomic_json_write
from tasks.copper_inversion.core.execution import digest
from tasks.copper_inversion.rsi.ledger import read
from tasks.copper_inversion.v2 import selection
from tasks.copper_inversion.v2.training_receipts import eligible

EXPERIMENTS = frozenset({"smoke", "rehearsal", "Q1", "M1", "E1", "E2", "E3", "X1", "F1"})
FULL = "full"
FINALISTS = 3


def prefix_name(round_index=None):
    if round_index is None:
        return FULL
    if type(round_index) is not int or round_index < 1:
        raise ValueError("a prefix freeze needs a positive round")
    return f"round_{round_index:02d}"


def parse_prefix(name):
    if name == FULL:
        return None
    match = re.fullmatch(r"round_(\d{2,})", name)
    if match is None or int(match.group(1)) < 1 or prefix_name(int(match.group(1))) != name:
        raise ValueError("invalid freeze prefix name")
    return int(match.group(1))


def layout(root, protocol_id, experiment, run_id=None):
    """contracts §1.1: public run, private service and private selection directories."""
    if experiment not in EXPERIMENTS:
        raise ValueError("unknown experiment")
    for value in (protocol_id, run_id):
        if value is not None and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value) is None:
            raise ValueError("invalid protocol or run identity")
    root = Path(root)
    paths = {"runs": root / "runs" / protocol_id / experiment,
             "services": root / "private" / protocol_id / "services" / experiment,
             "selection": root / "private" / protocol_id / "selection" / experiment,
             "unseal_log": root / "private" / protocol_id / "unseal-log.jsonl",
             "unseal": root / "private" / protocol_id / "unseal"}
    if run_id is not None:
        paths.update(run=paths["runs"] / run_id, service=paths["services"] / run_id,
                     run_selection=paths["selection"] / run_id)
    return paths


def run_manifest(run_dir, protocol_id, experiment):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if (manifest.get("run_id") != run_dir.name or manifest.get("experiment") != experiment
            or manifest.get("protocol_id") != protocol_id):
        raise ValueError("run manifest identity or protocol differs")
    if not (run_dir / "result.json").is_file():
        raise ValueError("freeze only a finished run")
    return manifest


def _key(sha, metrics):
    return (metrics["loss_total"], -metrics["legal_rate"], metrics["mean_solver_wall_seconds"], sha)


def finalists(run_dir, service_root, round_index=None, count=FINALISTS):
    """Pick the lowest-training-risk valid programs among records of rounds <= round_index."""
    prefix_name(round_index)
    run_dir, service_root = Path(run_dir), Path(service_root)
    rows = read(run_dir / "ledger.jsonl")
    if not rows or any(a["round"] > b["round"] for a, b in zip(rows, rows[1:])):
        raise ValueError("ledger is empty or out of round order")
    identity = json.loads((service_root / "identity.json").read_text())
    sealed_path = service_root / "training-sealed.json"
    if not sealed_path.is_file():
        raise ValueError("seal the training service before freezing")
    sealed = json.loads(sealed_path.read_text())
    if sealed["contract_digest"] != digest(identity):
        raise ValueError("training seal differs from service identity")
    if sealed["used_slots"] != len(rows):
        raise ValueError("public records and private charged slots differ")
    private = eligible(service_root, identity["configuration"]["manifest"])
    # The public side is writable by the research account: reconcile every row.
    listed = {row["program_id"] for row in rows}
    if not set(private) <= listed:
        raise ValueError("a privately evaluated program has no public record")
    for row in rows:
        record = private.get(row["program_id"])
        if row["status"] == "rejected":
            if record is not None:
                raise ValueError("an interface-rejected record has a training receipt")
            continue
        if (record is None
                or not math.isclose(record["metrics"]["loss_total"], row["risk"], abs_tol=1e-12)
                or not math.isclose(record["metrics"]["legal_rate"], row["legal_rate"], abs_tol=1e-12)
                or (row["status"] == "valid") != (record["metrics"]["legal_rate"] > 0)):
            raise ValueError("public record differs from its private training receipt")
    prefix = rows if round_index is None else [r for r in rows if r["round"] <= round_index]
    valid = [r["program_id"] for r in prefix if r["status"] == "valid"]
    chosen = sorted(valid, key=lambda sha: _key(sha, private[sha]["metrics"]))[:count]
    if not chosen:
        raise ValueError("no valid training candidate can be frozen")
    if round_index is None and chosen != sealed["source_sha256s"]:
        raise ValueError("full-run finalists differ from the sealed training service")
    return {"source_sha256s": chosen, "sources": [private[sha]["source_path"] for sha in chosen],
            "records": len(prefix), "last_round": prefix[-1]["round"],
            "ledger_prefix_sha256": digest(prefix), "configuration": identity["configuration"]}


def held_out(release, manifest=None, protocol=None):
    """Validation and sealed manifests of the run's tier and judge (contracts §2.1)."""
    release = Path(release)
    if protocol is None:
        return (release / "validation/manifest.json",
                [release / "test_id/manifest.json", release / "test_ood/manifest.json"])
    tier = protocol["tiers"][manifest["tier"]]
    judge = protocol["judges"][manifest["judge"]]
    return (release / judge.get("validation_manifest", tier["validation_manifest"]),
            [release / path for path in tier["sealed_manifests"]])


def judge_objective(manifest, protocol, configuration):
    """spec §3.3: the judge also scores validation; it must match the training service."""
    service = configuration.get("objective", "confirmation")
    if protocol is None:
        return service
    objective = protocol["judges"][manifest["judge"]].get("objective", "confirmation")
    if objective != service:
        raise ValueError("run judge objective differs from its training service")
    return objective


def freeze_run(root, protocol_id, experiment, run_id, round_index=None, *, release, protocol=None):
    paths = layout(root, protocol_id, experiment, run_id)
    manifest = run_manifest(paths["run"], protocol_id, experiment)
    name = prefix_name(round_index)
    chosen = finalists(paths["run"], paths["service"], round_index)
    target = paths["run_selection"] / name
    record = {"schema": "rsi-prefix-freeze-v1", "experiment": experiment, "run_id": run_id,
              "protocol_id": protocol_id, "prefix": name, "round": round_index,
              **{k: chosen[k] for k in ("records", "last_round", "ledger_prefix_sha256", "source_sha256s")}}
    if target.exists():
        frozen, _ = selection.verify(target)
        if [c["sha256"] for c in frozen["candidates"]] != chosen["source_sha256s"]:
            raise ValueError("existing freeze differs from the record prefix")
    else:
        validation, tests = held_out(release, manifest, protocol)
        config = chosen["configuration"]
        frozen = selection.freeze(target, chosen["sources"], validation, tests, seconds=config["seconds"],
                                  image=config["image"], training_run=paths["service"],
                                  training_manifest=config["manifest"],
                                  objective=judge_objective(manifest, protocol, config))
    record["selection_objective"] = frozen.get("selection_objective", "confirmation")
    record["frozen_sha256"] = digest(frozen)
    path = target / "prefix.json"
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise ValueError("prefix freeze record changed")
    else:
        atomic_json_write(path, record)
    return record


def validate_run(root, protocol_id, experiment, run_id, round_index=None):
    target = layout(root, protocol_id, experiment, run_id)["run_selection"] / prefix_name(round_index)
    return selection.run(target, "validation")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="active area, e.g. /Aidata/shared/xingyulu/rsi4geo")
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--run", action="append", required=True, dest="runs")
    parser.add_argument("--prefix", action="append", default=None,
                        help="'full' (default) or a round number; repeatable")
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, help="protocol.json; picks manifests by tier and judge")
    parser.add_argument("--no-validation", action="store_true")
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text()) if args.protocol else None
    rounds = [None if p == FULL else int(p) for p in (args.prefix or [FULL])]
    for run_id in args.runs:
        for round_index in rounds:
            record = freeze_run(args.root, args.protocol_id, args.experiment, run_id, round_index,
                                release=args.release, protocol=protocol)
            status = {"run_id": run_id, "prefix": record["prefix"], "status": "frozen",
                      "finalists": len(record["source_sha256s"])}
            if not args.no_validation:
                status.update(status="selected", selected_sha256=validate_run(
                    args.root, args.protocol_id, args.experiment, run_id, round_index)["selected_sha256"])
            print(json.dumps(status, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
