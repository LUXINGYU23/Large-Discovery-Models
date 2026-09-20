"""Source-preserving dataset preparation; public prompts and private labels are separate."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

PAPER_ACTIONS = (
    "change_atom_action",
    "remove_atom_action",
    "add_atom_action",
    "move_atom_action",
    "move_towards_atom_action",
    "insert_between_atoms_action",
    "swap_atoms_action",
    "delete_below_atom_action",
    "rotate_around_atom_action",
    "super_cell_action",
)
EXTENSION_ACTIONS = (
    "delete_around_atom_action",
    "move_around_atom_action",
    "move_selected_atoms_action",
    "rotate_whole_action",
    "move_all_action",
)
TASK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM = TASK_ROOT.parents[2] / "atomworld-main"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_released_rows(data_dir: Path, action: str) -> tuple[list[dict], list[Path]]:
    """Mirror upstream v1 CSV+HDF5 priority and v2 JSON without rounding CIFs."""
    if any(data_dir.glob("*.csv")):
        import h5py

        csv_path = data_dir / f"{action}.csv"
        input_path = data_dir / "input_cifs.hdf5"
        output_path = data_dir / f"{action}.hdf5"
        rows = []
        with (
            csv_path.open(newline="", encoding="utf-8") as handle,
            h5py.File(input_path, "r") as inputs,
            h5py.File(output_path, "r") as outputs,
        ):
            for index, row in enumerate(csv.DictReader(handle)):
                input_name = Path(row["input_cif"]).name
                output_name = Path(row["output_cif"]).name

                def content(source, key):
                    value = source[key][()]
                    return (
                        value.decode("utf-8")
                        if isinstance(value, bytes)
                        else str(value)
                    )

                rows.append(
                    {
                        "source_row": index,
                        "source_input": input_name,
                        "source_output": output_name,
                        "input_cif": content(inputs, input_name),
                        "action_prompt": row["action_prompt"],
                        "target_cif": content(outputs, output_name),
                    }
                )
        return rows, [csv_path, input_path, output_path]
    path = data_dir / f"{action}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "source_row": i,
            "source_input": "",
            "source_output": "",
            "input_cif": row["input"],
            "action_prompt": row["action_prompt"],
            "target_cif": row["output"],
        }
        for i, row in enumerate(raw)
    ], [path]


def prepare_dataset(
    data_dir: Path,
    destination: Path,
    *,
    actions=PAPER_ACTIONS,
    per_action: int = 1,
    seed: int = 0,
) -> dict:
    if per_action < 0:
        raise ValueError("per_action must be nonnegative (0 means every released row)")
    if (destination / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite prepared dataset {destination}")
    if not actions or len(set(actions)) != len(actions):
        raise ValueError("Select a nonempty set of distinct actions")
    public, private, selected, files, counts = [], [], [], {}, {}
    for action in actions:
        if action not in PAPER_ACTIONS + EXTENSION_ACTIONS:
            raise ValueError(f"Unknown action: {action}")
        rows, paths = load_released_rows(data_dir, action)
        counts[action] = len(rows)
        if not rows:
            raise ValueError(f"No rows for action {action}")
        for path in paths:
            files[path.name] = sha256_file(path)
        indices = list(range(len(rows)))
        random.Random(f"{seed}:{action}").shuffle(indices)
        indices = sorted(indices[:per_action]) if per_action else indices
        for index in indices:
            row = rows[index]
            if not all(
                isinstance(row[k], str) and row[k].strip()
                for k in ("input_cif", "action_prompt", "target_cif")
            ):
                raise ValueError(f"Empty or invalid source row: {action}:{index}")
            sample_id = f"{action}:{index:06d}"
            public.append(
                {
                    "sample_id": sample_id,
                    "action_name": action,
                    "input_cif": row["input_cif"],
                    "action_prompt": row["action_prompt"],
                }
            )
            private.append({"sample_id": sample_id, "target_cif": row["target_cif"]})
            selected.append(
                {
                    "sample_id": sample_id,
                    "source_row": row["source_row"],
                    "source_input": row["source_input"],
                    "source_output": row["source_output"],
                    "input_sha256": hashlib.sha256(
                        row["input_cif"].encode()
                    ).hexdigest(),
                    "target_sha256": hashlib.sha256(
                        row["target_cif"].encode()
                    ).hexdigest(),
                }
            )
    write_jsonl(destination / "public.jsonl", public)
    write_jsonl(destination / "private.jsonl", private)
    (destination / "private.jsonl").chmod(0o600)
    manifest = {
        "schema_version": 1,
        "dataset_kind": "local_released_subset",
        "paper_split_verified": False,
        "split_note": "Released local rows are not identified as the published 250 common structures. No paper-score comparability is claimed.",
        "seed": seed,
        "per_action": per_action,
        "actions": list(actions),
        "available_counts": counts,
        "sample_count": len(public),
        "selected": selected,
        "source_files_sha256": files,
        "public_sha256": sha256_file(destination / "public.jsonl"),
        "private_sha256": sha256_file(destination / "private.jsonl"),
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def load_prepared(data_dir: Path) -> tuple[list[dict], dict[str, str], dict]:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    for name in ("public", "private"):
        if sha256_file(data_dir / f"{name}.jsonl") != manifest[f"{name}_sha256"]:
            raise ValueError(f"Prepared {name} data hash does not match manifest")
    public = read_jsonl(data_dir / "public.jsonl")
    targets = read_jsonl(data_dir / "private.jsonl")
    ids = [row["sample_id"] for row in public]
    if len(set(ids)) != len(ids) or len(public) != manifest["sample_count"]:
        raise ValueError("Duplicate sample IDs or inconsistent sample count")
    if any(
        set(row) != {"sample_id", "action_name", "input_cif", "action_prompt"}
        for row in public
    ):
        raise ValueError(
            "Public data may contain only sample_id, action_name, input_cif, action_prompt"
        )
    if len(targets) != len(ids) or {row["sample_id"] for row in targets} != set(ids):
        raise ValueError("Public/private sample IDs do not match exactly")
    return public, {row["sample_id"]: row["target_cif"] for row in targets}, manifest
