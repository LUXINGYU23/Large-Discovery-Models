"""Select a reproducible stratified Q1 sample from rehearsal ledgers."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import random
from pathlib import Path

from tasks.copper_inversion.rsi.ledger import read
from tasks.copper_inversion.core.candidate import validate_source
from tasks.copper_inversion.core.execution import digest
from tasks.copper_inversion.v2.service import EvaluatorService
from tasks.copper_inversion.v2.suite import file_hash, load_manifest
from tasks.copper_inversion.v2 import selection



def select(runs, output, seed):
    """Sample six distinct evaluated programs per risk decile, then twelve repeats.

    Identical program IDs across runs are counted once, using the occurrence in
    the lexicographically first resolved run path. Rejected programs have no
    complete training evaluation and are excluded.
    """
    paths = sorted({Path(run).resolve() for run in runs}, key=str)
    if not paths or type(seed) is not int or seed < 0:
        raise ValueError("run directories and a nonnegative integer seed are required")
    found, hashes = {}, {}
    for run in paths:
        ledger = run / "ledger.jsonl"
        body = ledger.read_bytes()
        hashes[str(run)] = hashlib.sha256(body).hexdigest()
        for row in read(ledger):
            if row["status"] != "rejected":
                found.setdefault(row["program_id"], {"program_id": row["program_id"],
                                                    "source_run": str(run), "risk": row["risk"]})
    ordered = sorted(found.values(), key=lambda r: (r["risk"], r["program_id"]))
    if len(ordered) < 60:
        raise ValueError("Q1 requires at least 60 distinct evaluated programs")
    rng = random.Random(seed)
    selected = []
    for decile in range(10):
        group = ordered[len(ordered) * decile // 10:len(ordered) * (decile + 1) // 10]
        selected.extend({**row, "decile": decile + 1} for row in rng.sample(group, 6))
    selected.sort(key=lambda row: (row["decile"], row["program_id"]))
    repeats = sorted(rng.sample([row["program_id"] for row in selected], 12))
    result = {"schema": "rsi-q1-selection-v1", "seed": seed, "input_run_hashes": hashes,
              "programs": selected, "repeat_program_ids": repeats}
    data = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def references(path):
    """Load precommitted source files, checking their exact byte hashes."""
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not data:
        raise ValueError("references must be a nonempty name to source mapping")
    out = {}
    for name, item in sorted(data.items()):
        if not isinstance(name, str) or not name or not isinstance(item, dict) or set(item) != {"source", "sha256"}:
            raise ValueError("invalid reference entry")
        source = Path(item["source"])
        if not source.is_absolute():
            source = path.parent / source
        source = source.resolve()
        if file_hash(source) != item["sha256"] or validate_source(source.read_bytes()) != item["sha256"]:
            raise ValueError("reference source hash differs")
        out[name] = {"source": str(source), "sha256": item["sha256"]}
    return out


def release_config(release):
    release = Path(release).resolve()
    path = next((release / n for n in ("qualification.json", "development-receipt.json")
                 if (release / n).is_file()), None)
    if path is None:
        raise ValueError("release record missing")
    record = json.loads(path.read_text())
    return release, record


def evaluate_sources(service_root, manifest, seconds, slots, image, sources, *, tier=None, purpose=None, policy=None):
    """Submit pinned sources to the v2 training service in the trusted process."""
    root, manifest = Path(service_root).resolve(), Path(manifest).resolve()
    if load_manifest(manifest)["split"] != "train":
        raise ValueError("Q1 requires a training manifest")
    config = {"manifest": str(manifest), "root": str(root), "image": image, "seconds": seconds,
              "slots": slots, "caller_uid": 0, "socket": str(root / "unused.sock")}
    if tier is not None:
        config["tier"] = tier
    if purpose is not None:
        config["purpose"] = purpose
    if policy is None:
        service = EvaluatorService(config)
    else:
        from tasks.copper_inversion.experiments.v2.parallel_runtime import ParallelService, case_workers
        service = ParallelService(config)
        service.evaluator.case_workers = case_workers(policy, tier or "S")
    contract = digest(service.identity)
    def submit(item):
        source, sha = item
        body = Path(source).read_bytes()
        if validate_source(body) != sha:
            raise ValueError("selected source changed")
        reply = service.dispatch({"operation": "evaluate", "source": body.decode(), "contract_digest": contract})
        if reply.get("status") != "succeeded" or reply.get("source_sha256") != sha:
            raise ValueError("training evaluation failed or rejected: " + str(reply.get("status")))
    if policy is None:
        for item in sources:
            submit(item)
    else:
        with ThreadPoolExecutor(max_workers=policy["batch_workers"]) as pool:
            list(pool.map(submit, sources))
    return root


def evaluate(selection_path, release, output, image, *, policy=None):
    selection_path = Path(selection_path)
    chosen = json.loads(selection_path.read_text())
    if chosen.get("schema") != "rsi-q1-selection-v1":
        raise ValueError("invalid Q1 selection")
    rows = chosen["programs"]
    ids = [r["program_id"] for r in rows]
    repeat_ids = chosen["repeat_program_ids"]
    if len(ids) != 60 or len(set(ids)) != 60 or len(repeat_ids) != 12 or len(set(repeat_ids)) != 12 or not set(repeat_ids) <= set(ids):
        raise ValueError("invalid Q1 sample sizes")
    for run, sha in chosen["input_run_hashes"].items():
        if file_hash(Path(run) / "ledger.jsonl") != sha:
            raise ValueError("rehearsal ledger changed")
    sources = [(str(Path(r["source_run"]) / "candidates" / r["program_id"] / "solver.py"), r["program_id"]) for r in rows]
    repeat = [(source, sha) for source, sha in sources if sha in set(repeat_ids)]
    release, record = release_config(release)
    tiers, q = record["tiers"], record["q1"]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    mapping_path = output / "q1.json"
    layouts = {
        "short_12x1": (tiers["S"]["train_manifest"], tiers["S"]["solver_seconds"], tiers["S"]["slots"], "S", None, sources),
        "full_24x1": (tiers["F"]["train_manifest"], tiers["F"]["solver_seconds"], tiers["F"]["slots"], "F", None, sources),
        "short_24x2": (q["both_packs_manifest"], q["solver_seconds"], len(sources), None, "q1", sources),
        "repeats": (q["execution_seeds_manifest"], q["solver_seconds"], len(repeat), None, "q1", repeat),
    }
    mapping = {}
    for name, (manifest, seconds, slots, tier, purpose, entries) in layouts.items():
        if not manifest:
            raise ValueError("release lacks Q1 manifest: " + name)
        root_path = output / name
        identity = {"layout": name, "selection_sha256": file_hash(selection_path), "manifest": str((release / manifest).resolve()),
                    "manifest_sha256": file_hash(release / manifest), "seconds": seconds, "tier": tier, "purpose": purpose,
                    "program_ids": sorted(sha for _, sha in entries)}
        identity_path = root_path / "q1-layout.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("Q1 layout identity changed")
        root_path.mkdir(parents=True, exist_ok=True)
        from ldm_tts.engine.run_store import atomic_json_write
        atomic_json_write(identity_path, identity)
        kwargs = dict(tier=tier, purpose=purpose)
        if policy is not None:
            kwargs["policy"] = policy
        root = evaluate_sources(root_path, release / manifest, seconds, slots, image, entries, **kwargs)
        mapping[name] = str(root)
    if mapping_path.exists() and json.loads(mapping_path.read_text()) != mapping:
        raise ValueError("mapping changed")
    from ldm_tts.engine.run_store import atomic_json_write
    atomic_json_write(mapping_path, mapping)
    return mapping


def build_references(refs_path, release, output, image, *, policy=None):
    refs = references(refs_path)
    release, record = release_config(release)
    tier = record["tiers"]["S"]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    mapping_path = output / "references.json"
    kwargs = {"tier": "S"}
    if policy is not None:
        kwargs["policy"] = policy
    service = evaluate_sources(output / "training", release / tier["train_manifest"], tier["solver_seconds"],
                               tier["slots"], image, [(r["source"], r["sha256"]) for r in refs.values()], **kwargs)
    mapping = {}
    def validate(item):
        name, ref = item
        target = output / "selection" / hashlib.sha256(name.encode()).hexdigest()
        if not target.exists():
            selection.freeze(target, [ref["source"]], release / tier["validation_manifest"],
                             [release / p for p in tier["sealed_manifests"]], seconds=tier["solver_seconds"],
                             image=image, training_run=service, training_manifest=release / tier["train_manifest"])
        frozen, _ = selection.verify(target)
        if len(frozen["candidates"]) != 1 or frozen["candidates"][0]["sha256"] != ref["sha256"]:
            raise ValueError("reference selection differs")
        selection.run(target, "validation")
        return name, str(target)
    if policy is None:
        mapping = dict(map(validate, refs.items()))
    else:
        with ThreadPoolExecutor(max_workers=policy["batch_workers"]) as pool:
            mapping = dict(pool.map(validate, refs.items()))
    if mapping_path.exists() and json.loads(mapping_path.read_text()) != mapping:
        raise ValueError("mapping changed")
    from ldm_tts.engine.run_store import atomic_json_write
    atomic_json_write(mapping_path, mapping)
    return mapping


def main(argv=None, *, policy=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    choice = sub.add_parser("select", help="sample evaluated rehearsal programs")
    choice.add_argument("--run", action="append", required=True, dest="runs")
    choice.add_argument("--seed", type=int, required=True)
    choice.add_argument("--output", type=Path, required=True)
    measure = sub.add_parser("evaluate", help="evaluate the Q1 sample on release training suites")
    measure.add_argument("--selection", type=Path, required=True)
    for command in (measure,):
        command.add_argument("--release", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--image", required=True)
    ref = sub.add_parser("references", help="train, freeze and validate domain references")
    ref.add_argument("--references", type=Path, required=True)
    ref.add_argument("--release", type=Path, required=True)
    ref.add_argument("--output", type=Path, required=True)
    ref.add_argument("--image", required=True)
    args = parser.parse_args(argv)
    if args.command == "select":
        print(select(args.runs, args.output, args.seed))
    elif args.command == "evaluate":
        print(json.dumps(evaluate(args.selection, args.release, args.output, args.image, policy=policy), sort_keys=True))
    else:
        print(json.dumps(build_references(args.references, args.release, args.output, args.image, policy=policy), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
