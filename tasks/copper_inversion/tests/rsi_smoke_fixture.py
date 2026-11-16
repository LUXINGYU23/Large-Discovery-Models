"""Build a tiny synthetic service-integration fixture; never a benchmark release."""

import argparse
import json
from pathlib import Path

import numpy as np
from tasks.copper_inversion.experiments.v2.publish_training import publish
from tasks.copper_inversion.tests.test_v2_science import fixture, model
from tasks.copper_inversion.v2.data import save_problem
from tasks.copper_inversion.v2.suite import file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    private = root / "private-data"
    private.mkdir()
    truth = private / "truth"
    truth.mkdir()
    np.save(truth / "truth.npy", model(fixture()) > 0)
    confirmation = fixture()
    confirmation["observations"][0]["points"][:, 0] += .01
    save_problem(confirmation, truth / "confirmation")
    rows = []
    for modality in ("drill",):
        problem = fixture()
        problem["case_id"] = "synthetic-" + modality
        public = private / modality
        save_problem(problem, public)
        names = ("problem.json", "problem.npz", "truth.npy", "confirmation/problem.json",
                 "confirmation/problem.npz")
        rows.append({"case_id": problem["case_id"], "parent_id": "synthetic-parent", "modality": modality,
                     "public": modality, "truth": "truth",
                     "sha256": {name: file_hash((truth if name == "truth.npy" or name.startswith("confirmation/")
                                                  else public) / name) for name in names}})
    manifest = private / "manifest.json"
    manifest.write_text(json.dumps({"schema": "copper-suite-v2", "stage": "development", "split": "train",
                                    "seeds": [47], "cases": rows}, sort_keys=True))
    publish(manifest, root / "public-training")
    (root / "FIXTURE_ONLY").write_text("Synthetic integration fixture; no scientific result.\n")
    print(json.dumps({"scope": "synthetic_fixture_only", "cases": len(rows)}))


if __name__ == "__main__":
    main()
