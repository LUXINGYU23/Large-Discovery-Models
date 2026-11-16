"""Freeze, validate, or test a completed run on the private development release.

Development only: its test phase writes no unseal log. Formal freezes, prefix
freezes and sealed evaluation go through experiments/rsi/{freeze,unseal}.py.
"""

import argparse
import json
import os
from pathlib import Path

from tasks.copper_inversion.experiments.v2.freeze_service_selection import freeze_from_service
from tasks.copper_inversion.v2.selection import run
from tasks.copper_inversion.v2.suite import file_hash, load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "validation", "test"))
    parser.add_argument("--service-root", required=True, type=Path)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.geteuid() != 991:
        raise PermissionError("development selection requires the isolated evaluator account")
    release = args.release.resolve()
    receipt = json.loads((release / "development-receipt.json").read_text())
    if receipt.get("status") != "development_only":
        raise ValueError("a development-only release is required")
    for split in ("train", "validation", "test_id", "test_ood"):
        if file_hash(release / split / "manifest.json") != receipt["manifest_sha256"][split]:
            raise ValueError("development release manifest changed")
        if load_manifest(release / split / "manifest.json")["stage"] != "development":
            raise ValueError("development selection cannot evaluate a formal release")
    identity = json.loads((args.service_root / "identity.json").read_text())
    if Path(identity["configuration"]["manifest"]).resolve() != release / "train/manifest.json":
        raise ValueError("training service belongs to a different release")
    if args.phase == "freeze":
        result = freeze_from_service(args.service_root, args.output, release / "validation/manifest.json",
                                     release / "test_id/manifest.json", release / "test_ood/manifest.json")
        print(json.dumps({"scope": "development_only", "status": "frozen",
                          "finalists": len(result["candidates"])}))
    else:
        result = run(args.output, args.phase)
        print(json.dumps({"scope": "development_only", "status": args.phase + "_complete",
                          "selected_sha256": result["selected_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
