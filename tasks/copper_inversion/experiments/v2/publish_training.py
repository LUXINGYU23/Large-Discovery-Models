"""Trusted one-way publication of training observations, never truth/split lineage."""
import argparse
import json
import shutil
from pathlib import Path
from tasks.copper_inversion.v2.suite import load_manifest
from tasks.copper_inversion.v2.data import read_problem, save_problem
from tasks.copper_inversion.v2.public_catalog import SCHEMA, file_hash, load_catalog


def publish(manifest_path, destination):
    manifest_path = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_path)
    if manifest["split"] != "train":
        raise ValueError("only training observations may be published")
    destination = Path(destination).resolve()
    # Publication is immutable and never replaces an existing release.
    destination.mkdir(parents=True, exist_ok=False)
    try:
        rows = []
        for index, row in enumerate(manifest["cases"]):
            relative = f"cases/{index:04d}"
            folder = destination / relative
            save_problem(read_problem(manifest_path.parent / row["public"]), folder)
            rows.append(dict(case_id=row["case_id"], directory=relative,
                             sha256={name: file_hash(folder/name) for name in ("problem.json", "problem.npz")}))
        catalog = dict(schema=SCHEMA, stage=manifest["stage"], split="train", cases=rows)
        path = destination / "catalog.json"
        path.write_text(json.dumps(catalog, sort_keys=True, indent=2))
        load_catalog(path)
        for file in destination.rglob("*"):
            if file.is_file():file.chmod(0o444)
        return path
    except Exception:
        shutil.rmtree(destination)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(publish(args.manifest, args.destination))


if __name__ == "__main__":
    main()
