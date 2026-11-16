"""Make a private, one-layout development release from preassigned GemPy data.

This command does not qualify a formal release or publish any held-out data.
"""

import argparse
import copy
import json
import os
import shutil
from pathlib import Path

import numpy as np

from tasks.copper_inversion.experiments.rsi.measurements import assign_modalities, audit_holes
from tasks.copper_inversion.v2.data import read_problem
from tasks.copper_inversion.v2.suite import file_hash, load_manifest


COUNTS = {"train": 8, "validation": 4, "test_id": 4, "test_ood": 4}


def release(dataset, output):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("release destination already exists")
    if not (dataset / "confirmation-complete.json").is_file():
        raise ValueError("confirmation generation is incomplete")
    prepared, plans, audits = [], {}, []
    for part in ("id", "ood"):
        base = dataset / part
        if not (base / "completion.json").is_file():
            raise ValueError("parent generation is incomplete")
        if json.loads((base / "split-audit.json").read_text()).get("status") != "passed":
            raise ValueError("generated population failed its split audit")
        plan_path = base / "plan.json"
        plan = json.loads(plan_path.read_text())
        if ("modality_allocation" not in plan or plan.get("confirmation_policy", {}).get("rule") !=
                "first of 16 presampled streams with no overlapping whole holes"):
            raise ValueError("observation and confirmation policies were not fixed before generation")
        replay = assign_modalities(copy.deepcopy(plan), plan["modality_allocation"]["seed"])
        if replay != plan:
            raise ValueError("observation allocation does not reproduce from the fixed plan")
        plans[part] = file_hash(plan_path)
        entries = {f"{row['split']}-{row['index']:02}": row for row in plan["parents"]}
        accepted = [row for row in json.loads((base / "progress.json").read_text())["records"]
                    if row["status"] == "accepted"]
        if {row["parent"] for row in accepted} != set(entries):
            raise ValueError("generation does not cover the allocation plan")
        for item in accepted:
            name = item["parent"]
            entry = entries[name]
            parent = base / item["parent_directory"]
            if plan_path.stat().st_mtime_ns > (parent / "geology.npz").stat().st_mtime_ns:
                raise ValueError("allocation plan postdates geology generation")
            pack = base / "packs" / name
            manifest = json.loads((pack / "private-manifest.json").read_text())
            confirmation = base / "confirmation" / name
            hidden = json.loads((confirmation / "private-manifest.json").read_text())
            if manifest["parent_digest"] != hidden["parent_digest"]:
                raise ValueError("confirmation parent differs")
            attempt_receipt = json.loads((base / "confirmation-audits" / (name + ".json")).read_text())
            attempts = attempt_receipt["attempts"]
            if (not attempts or len(attempts) > 16 or attempts[-1]["status"] != "passed"
                    or any(row["status"] != "failed" for row in attempts[:-1])
                    or [row["seed"] for row in attempts] != entry["confirmation_seeds"][:len(attempts)]
                    or hidden["observation_seed"] != attempts[-1]["seed"]):
                raise ValueError("confirmation does not match the preassigned separation policy")
            modality = entry["selected_modality"]
            if modality not in {"drill", "drill_outcrop"}:
                raise ValueError("unknown preassigned modality")
            case = next(row for row in manifest["cases"] if row["modality"] == modality)
            hidden_case = next(row for row in hidden["cases"] if row["modality"] == modality)
            public_path, hidden_path = pack / case["case_id"], confirmation / hidden_case["case_id"]
            public, confirm = read_problem(public_path), read_problem(hidden_path)
            if any(public[key] != confirm[key] for key in ("shape", "origin", "spacing", "priors")):
                raise ValueError("confirmation geometry or priors differ")
            if any(not np.array_equal(public[key], confirm[key]) for key in ("domain_mask", "forbidden_mask")):
                raise ValueError("confirmation masks differ")
            hole_audit = audit_holes(public, confirm)
            audits.append({"parent": name, **hole_audit})
            if hole_audit["status"] != "passed":
                raise ValueError(f"confirmation hole overlap: {name}; release not created")
            prepared.append((entry, manifest, case, public_path, hidden_path, pack / case["truth"]))
    counts = {split: sum(entry["split"] == split for entry, *_ in prepared) for split in COUNTS}
    if counts != COUNTS:
        raise ValueError("development parent counts differ from plan")
    for split in COUNTS:
        modalities = [entry["selected_modality"] for entry, *_ in prepared if entry["split"] == split]
        if modalities.count("drill") != modalities.count("drill_outcrop"):
            raise ValueError("development split has unbalanced observation layouts")
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    manifests = {split: {"schema": "copper-suite-v2", "stage": "development", "split": split,
                         "seeds": [47], "cases": []} for split in COUNTS}
    for entry, manifest, case, public_path, hidden_path, truth_path in prepared:
        split, case_id = entry["split"], case["case_id"]
        root = output / split
        public, truth = root / "public" / case_id, root / "private" / case_id
        shutil.copytree(public_path, public)
        truth.mkdir(parents=True)
        shutil.copy2(truth_path, truth / "truth.npy")
        shutil.copytree(hidden_path, truth / "confirmation")
        hashes = {name: file_hash((public if name in ("problem.json", "problem.npz") else truth) / name)
                  for name in ("problem.json", "problem.npz", "truth.npy", "confirmation/problem.json",
                               "confirmation/problem.npz")}
        manifests[split]["cases"].append({"case_id": case_id, "parent_id": manifest["parent_digest"],
                                          "modality": case["modality"], "public": str(public.relative_to(root)),
                                          "truth": str(truth.relative_to(root)), "sha256": hashes})
    manifest_hashes = {}
    for split, manifest in manifests.items():
        path = output / split / "manifest.json"
        path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        load_manifest(path)
        manifest_hashes[split] = file_hash(path)
    (output / "development-receipt.json").write_text(json.dumps({
        "status": "development_only", "scope": "synthetic GemPy geology; not field validation or formal release",
        "counts": counts, "allocation_plan_sha256": plans, "manifest_sha256": manifest_hashes,
        "confirmation_hole_audits": audits}, sort_keys=True, indent=2) + "\n")
    for path in output.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o444)
    output.chmod(0o700)
    return {"status": "development_only", "counts": counts, "confirmation_hole_audits": len(audits)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.geteuid() != 991:
        raise PermissionError("development release requires the isolated evaluator account")
    print(json.dumps(release(args.dataset, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
