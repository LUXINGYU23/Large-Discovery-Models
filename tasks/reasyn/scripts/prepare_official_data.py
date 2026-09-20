#!/usr/bin/env python3
"""Inventory official ReaSyn assets without downloads or upstream mutations."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream-root", type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    root = args.upstream_root.resolve()
    names = [
        "data/test_zinc250k.txt",
        "data/building_blocks/building_blocks_zinc250k.txt",
        "data/enamine_smiles_1k.txt",
        "data/chembl_filtered_1k.txt",
        "data/trained_model/nv-reasyn-ar-166m-v2.ckpt",
        "data/trained_model/nv-reasyn-eb-174m-v2.ckpt",
        "data/processed/comp_2048/fpindex.pkl",
        "data/processed/comp_2048/matrix.pkl",
        "data/processed/zinc250k_2048/fpindex.pkl",
        "data/rxn_templates/comprehensive.txt",
    ]
    items = []
    for name in names:
        path = root / name
        entry = {"path": name, "present": path.is_file()}
        if path.is_file():
            h = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1048576), b""):
                    h.update(chunk)
            entry.update(bytes=path.stat().st_size, sha256=h.hexdigest())
            if path.suffix == ".txt":
                lines = [s.strip() for s in path.read_text().splitlines() if s.strip()]
                entry["rows_excluding_SMILES_header"] = len(lines) - bool(
                    lines and lines[0].lower() == "smiles"
                )
        items.append(entry)
    data = {
        "task": "reasyn",
        "upstream_root": str(root),
        "downloaded": False,
        "assets": items,
        "setup_instructions": "Follow upstream README Data Preparation/Training checkpoint download. Preserve supplied building-block list unchanged.",
    }
    text = json.dumps(data, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0 if all(item["present"] for item in items) else 1


if __name__ == "__main__":
    raise SystemExit(main())
