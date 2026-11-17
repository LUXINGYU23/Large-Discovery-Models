"""Small, dependency-free contracts shared by the CLI and model workers."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

AA3 = dict(zip(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
    "ARNDCQEGHILKMFPSTWYV"))
AA = frozenset(AA3.values())


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    indent = None if path.suffix == ".jsonl" else 2
    temporary.write_text(json.dumps(value, indent=indent, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def residue_key(value):
    """Insertion codes are part of identity, never discarded during mapping."""
    chain = value.get("chain")
    if not isinstance(chain, str) or not chain.strip():
        raise ValueError("Residue requires an explicit nonempty chain")
    number = value.get("number")
    if isinstance(number, bool) or not isinstance(number, int):
        raise ValueError("Residue number must be an integer")
    insertion = value.get("insertion_code", "")
    if not isinstance(insertion, str) or len(insertion) > 1:
        raise ValueError("Invalid insertion_code")
    return chain, number, insertion.strip()


def residue(chain, number, insertion_code=""):
    return {"chain": str(chain), "number": int(number), "insertion_code": str(insertion_code).strip()}


def validate_config(config):
    if config.get("device", "cuda") not in ("cuda", "cpu"):
        raise ValueError("device must be cuda or cpu")
    threads = config.get("cpu_threads", 8)
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("cpu_threads must be a positive integer")
    if not config.get("input_structure") or not config.get("heavy_chain"):
        raise ValueError("input_structure and heavy_chain are required")
    h3 = config.get("h3_residues")
    if not isinstance(h3, list) or not h3:
        raise ValueError("Explicit h3_residues mapping is required")
    h3_keys = [residue_key(r) for r in h3]
    if len(set(h3_keys)) != len(h3_keys):
        raise ValueError("Duplicate H3 residue identity")
    if any(k[0] != config["heavy_chain"] for k in h3_keys):
        raise ValueError("H3 residues must belong to heavy_chain")
    required = config.get("required_h3_length")
    if required is not None and (isinstance(required, bool) or not isinstance(required, int) or required < 1):
        raise ValueError("required_h3_length must be a positive integer or null")
    if required is not None and len(h3) != required:
        raise ValueError(f"H3 mapping length is {len(h3)}, required {required}; no trimming/padding")
    for key in ["num_designs", "sequences_per_design", "num_timesteps"]:
        val = config.get(key, {"num_designs": 1, "sequences_per_design": 1, "num_timesteps": 200}[key])
        if isinstance(val, bool) or not isinstance(val, int) or val < 1:
            raise ValueError(f"{key} must be a positive integer")
    if isinstance(config.get("seed", 42), bool) or not isinstance(config.get("seed", 42), int) or config.get("seed", 42) <= 0:
        raise ValueError("seed must be a positive integer (ProteinMPNN treats zero as random)")
    for key, default in [("sampling_temperature", 0.1), ("fixed_backbone_tolerance_angstrom", 0.1),
                         ("fixed_backbone_rmsd_tolerance_angstrom", 0.05)]:
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    wait = config.get("gpu_wait_seconds", 600)
    if isinstance(wait, bool) or not isinstance(wait, (int, float)) or not math.isfinite(wait) or wait < 0:
        raise ValueError("gpu_wait_seconds must be nonnegative and finite")
    if config.get("allow_gpu_sharing", False):
        limit = config.get("gpu_memory_limit_gib")
        if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit <= 0:
            raise ValueError("GPU sharing requires a positive finite gpu_memory_limit_gib")
    designed = [residue_key(r) for r in config.get("design_residues", h3)]
    fixed = [residue_key(r) for r in config.get("fixed_residues", [])]
    if len(designed) != len(set(designed)) or len(fixed) != len(set(fixed)):
        raise ValueError("Duplicate design/fixed residue identity")
    if set(designed) & set(fixed):
        raise ValueError("A residue cannot be both fixed and designed")
    return config


def extract_h3(candidate, required_length=None):
    """Extract from sequence positions linked to source identities, not a motif search."""
    mapping = candidate.get("residue_mapping", [])
    requested = candidate.get("h3_residues", [])
    heavy = candidate.get("heavy_chain")
    if not requested or not heavy:
        raise ValueError("Candidate lacks explicit heavy-chain / H3 mapping")
    indexed = {}
    for item in mapping:
        key = residue_key(item["source"])
        if key in indexed:
            raise ValueError("Ambiguous duplicate source mapping")
        indexed[key] = item
    if len({residue_key(r) for r in requested}) != len(requested):
        raise ValueError("Duplicate H3 residue identity")
    letters, positions = [], []
    for ref in requested:
        key = residue_key(ref)
        if key[0] != heavy or key not in indexed:
            raise ValueError(f"Missing heavy-chain H3 mapping: {key}")
        item = indexed[key]
        chain, position = item["sequence_chain"], item["sequence_position"]
        if isinstance(position, bool) or not isinstance(position, int) or position < 1:
            raise ValueError("Invalid sequence position")
        sequence = candidate["full_sequences"].get(chain, "")
        if position > len(sequence):
            raise ValueError("Mapped position exceeds sequence length")
        letters.append(sequence[position - 1])
        positions.append((chain, position))
    if len(set(positions)) != len(positions) or len({p[0] for p in positions}) != 1:
        raise ValueError("H3 mapping must address distinct positions on one chain")
    if any(b[1] != a[1] + 1 for a, b in zip(positions, positions[1:])):
        raise ValueError("H3 mapping must preserve contiguous residue order")
    sequence = "".join(letters)
    if any(letter not in AA for letter in sequence):
        raise ValueError("H3 contains nonstandard amino acids")
    if required_length is not None and len(sequence) != required_length:
        raise ValueError(f"H3 length is {len(sequence)}, required {required_length}; no trimming/padding")
    return sequence
