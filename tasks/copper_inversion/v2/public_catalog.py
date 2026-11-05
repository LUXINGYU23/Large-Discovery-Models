"""Strict public training catalog, independently readable without private data."""
import hashlib
import json
from pathlib import Path
from .data import read_problem
from ..core.execution import digest
from ..core.runner import input_digest

SCHEMA = "copper-public-training-v2"


def observation_digest(stage, problems):
    """Bind scientific inputs, independent of archive paths and NPZ timestamps."""
    return digest(dict(stage=stage, split="train", problems=[input_digest(p) for p in problems]))


def catalog_digest(path):
    path = Path(path).resolve()
    catalog = load_catalog(path)
    return observation_digest(catalog["stage"],
                              (read_problem(path.parent/row["directory"]) for row in catalog["cases"]))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_catalog(path):
    path = Path(path).resolve()
    catalog = json.loads(path.read_text())
    if (set(catalog) != {"schema", "stage", "split", "cases"}
            or catalog["schema"] != SCHEMA or catalog["split"] != "train"
            or catalog["stage"] not in {"development", "formal"}):
        raise ValueError("invalid public training catalog")
    if not isinstance(catalog["cases"], list) or not catalog["cases"]:
        raise ValueError("empty public training catalog")
    seen = set()
    for row in catalog["cases"]:
        if set(row) != {"case_id", "directory", "sha256"}:
            raise ValueError("private or unexpected catalog member fields")
        name = row["case_id"]
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("invalid or duplicate public case")
        seen.add(name)
        relative = Path(row["directory"])
        folder = path.parent / relative
        if (relative.is_absolute() or ".." in relative.parts
                or not folder.resolve().is_relative_to(path.parent)):
            raise ValueError("public case path escapes catalog")
        if set(row["sha256"]) != {"problem.json", "problem.npz"}:
            raise ValueError("invalid public file allowlist")
        for filename, expected in row["sha256"].items():
            file = folder / filename
            if file.is_symlink() or not file.resolve().is_relative_to(path.parent):
                raise ValueError("public file path escapes catalog")
            if file_hash(file) != expected:
                raise ValueError("public file digest mismatch")
        if read_problem(folder)["case_id"] != name:
            raise ValueError("public case identity differs")
    return catalog
