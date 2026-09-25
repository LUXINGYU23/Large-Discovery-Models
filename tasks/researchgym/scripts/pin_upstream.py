"""Regenerate resources/upstream_contract.json from a ResearchGym Git checkout.

Digests are read from Git objects at the pinned commit, never from the working
tree, so local edits in the checkout cannot leak into the contract.

    python tasks/researchgym/scripts/pin_upstream.py --upstream-root ~/ResearchGym
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://github.com/Anikethh/ResearchGym"
COMMIT = "0adc08e93754b606d8a76055ed2f1c9574504312"


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True).stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--commit", default=COMMIT)
    args = parser.parse_args(argv)
    root = args.upstream_root.expanduser().resolve()
    catalog = json.loads((TASK_ROOT / "resources/cases/catalog.json").read_text())
    cases = {}
    for case in catalog["cases"]:
        prefix = case["upstream_task"]
        names = _git(root, "ls-tree", "-r", "--name-only", args.commit, prefix).decode().splitlines()
        files = {}
        for name in sorted(names):
            blob = _git(root, "show", f"{args.commit}:{name}")
            files[name[len(prefix) + 1:]] = hashlib.sha256(blob).hexdigest()
        tree = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        cases[case["case_id"]] = {"task_path": prefix, "tree_sha256": tree, "files": files}
    contract = {"schema_version": 1, "source_url": SOURCE_URL, "commit": args.commit, "cases": cases}
    path = TASK_ROOT / "resources/upstream_contract.json"
    path.write_text(json.dumps(contract, indent=1, sort_keys=True) + "\n")
    print(json.dumps({case: {"files": len(v["files"]), "tree_sha256": v["tree_sha256"]}
                      for case, v in cases.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
