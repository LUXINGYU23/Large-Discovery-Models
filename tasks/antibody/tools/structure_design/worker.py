"""Structure operations run only inside the modern RFD3 environment."""
from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

# Executed by absolute file path so the worker need not install the legacy task.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np
import biotite.structure as struc
from biotite.structure.io import pdb, pdbx

from tasks.antibody.tools.structure_design.common import (
    AA3, extract_h3, load_json, residue, residue_key, save_json, validate_config,
)


def read_structure(path):
    path = Path(path)
    if path.name.endswith((".cif", ".cif.gz", ".mmcif", ".mmcif.gz")):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as stream:
            array = pdbx.get_structure(pdbx.CIFFile.read(stream), model=1, use_author_fields=True)
    elif path.suffix.lower() == ".pdb":
        array = pdb.PDBFile.read(path).get_structure(model=1)
    else:
        raise ValueError("Expected PDB or CIF(.gz) input")
    if not np.isfinite(array.coord).all():
        raise ValueError("Structure has non-finite coordinates")
    return array


def residues(array):
    groups = []
    for chunk in struc.residue_iter(array):
        key = (str(chunk.chain_id[0]), int(chunk.res_id[0]), str(chunk.ins_code[0]).strip())
        names = set(chunk.res_name)
        if len(names) != 1 or chunk.res_name[0] not in AA3:
            raise ValueError(f"Nonstandard/ambiguous protein residue: {key}, {names}")
        for atom in ("N", "CA", "C", "O"):
            if np.count_nonzero(chunk.atom_name == atom) != 1:
                raise ValueError(f"Expected one {atom} backbone atom at {key}")
        groups.append((key, AA3[chunk.res_name[0]], chunk))
    return groups


def write_pdb(array, path):
    file = pdb.PDBFile()
    file.set_structure(array)
    file.write(path)


def prepare(config, directory):
    validate_config(config)
    array = read_structure(config["input_structure"])
    # Select configured protein regions before rejecting other chains/ligands.
    regions = config.get("regions")
    if regions:
        mask = np.zeros(len(array), dtype=bool)
        for region in regions:
            mask |= ((array.chain_id == region["chain"]) & (array.res_id >= region["start"]) &
                     (array.res_id <= region["end"]))
        array = array[mask & ~array.hetero]
    else:
        array = array[~array.hetero]
    groups = residues(array)
    if not groups:
        raise ValueError("No residues selected")
    source_keys = [g[0] for g in groups]
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("Duplicate source residue identifiers")
    h3_keys = [residue_key(r) for r in config["h3_residues"]]
    designed = set(residue_key(r) for r in config.get("design_residues", config["h3_residues"]))
    fixed = set(residue_key(r) for r in config.get("fixed_residues", []))
    if not (set(h3_keys) | designed | fixed) <= set(source_keys):
        raise ValueError("H3/design/fixed mapping references missing input residues")
    positions = [source_keys.index(k) for k in h3_keys]
    if positions != list(range(positions[0], positions[0] + len(positions))):
        raise ValueError("H3 residues must be in contiguous structural order")
    chain_ids = list(dict.fromkeys(k[0] for k in source_keys))
    if len(chain_ids) > 26:
        raise ValueError("At most 26 protein chains supported")
    chunks, mapping, layout, contigs = [], [], [], []
    for chain_index, chain in enumerate(chain_ids):
        # Normalize identifiers, including insertion codes, before passing to RFD3.
        normalized = chr(ord("A") + chain_index)
        chain_groups = [g for g in groups if g[0][0] == chain]
        layout.append({"chain": normalized, "length": len(chain_groups), "source_chain": chain})
        parts, current, is_design = [], [], None
        for i, (key, aa, atoms) in enumerate(chain_groups, 1):
            active = key in designed
            if current and active != is_design:
                parts.append(str(len(current)) if is_design else f"{normalized}{current[0]}-{current[-1]}")
                current = []
            current.append(i)
            is_design = active
            copy = atoms.copy()
            copy.chain_id[:] = normalized
            copy.res_id[:] = i
            copy.ins_code[:] = ""
            chunks.append(copy)
            mapping.append({"source": residue(*key), "sequence_chain": normalized,
                            "sequence_position": i, "input_aa": aa, "designed": active})
        parts.append(str(len(current)) if is_design else f"{normalized}{current[0]}-{current[-1]}")
        contigs.append(",".join(parts))
    prepared = struc.concatenate(chunks)
    write_pdb(prepared, directory / "prepared.pdb")
    candidate = {"candidate_id": "input", "heavy_chain": config["heavy_chain"],
                 "h3_residues": config["h3_residues"], "residue_mapping": mapping,
                 "full_sequences": {c["chain"]: "".join(m["input_aa"] for m in mapping
                                                        if m["sequence_chain"] == c["chain"]) for c in layout},
                 "backbone_path": str(directory / "prepared.pdb"), "layout": layout}
    native_h3 = extract_h3(candidate, config.get("required_h3_length"))
    if config.get("expected_input_h3") and native_h3 != config["expected_input_h3"]:
        raise ValueError("Input H3 does not match the documented example sequence")
    candidate["input_h3"] = native_h3
    save_json(directory / "input.json", candidate)
    save_json(directory / "specification.json", {"design": {"input": str(directory / "prepared.pdb"),
                                                            "contig": ",/0,".join(contigs)}})


def fixed_backbone_alignment(reference, moving):
    """Kabsch fit on fixed backbone atoms; coordinates may have a new global frame."""
    x, y = np.asarray(reference), np.asarray(moving)
    if len(x) < 3:
        raise ValueError("Need at least three fixed backbone atoms for alignment")
    x0, y0 = x - x.mean(0), y - y.mean(0)
    u, _, vt = np.linalg.svd(y0.T @ x0)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    aligned = y0 @ (u @ correction @ vt)
    distances = np.linalg.norm(aligned - x0, axis=1)
    return {"fixed_backbone_rmsd_angstrom": float(np.sqrt(np.mean(distances**2))),
            "fixed_backbone_max_error_angstrom": float(distances.max())}


def convert(config, input_manifest, structure, directory):
    candidate = load_json(input_manifest)
    groups = residues(read_structure(structure))
    output_chains = list(dict.fromkeys(k[0] for k, _, _ in groups))
    layout = candidate["layout"]
    if len(output_chains) != len(layout):
        raise ValueError("RFD3 output chain count differs from explicit contig layout")
    reference = residues(read_structure(Path(input_manifest).parent / "prepared.pdb"))
    ref_by_slot = {(key[0], key[1]): atoms for key, _, atoms in reference}
    metadata_path = Path(str(structure).removesuffix(".gz")).with_suffix(".json")
    index_map = None
    if metadata_path.exists():
        index_map = load_json(metadata_path).get("diffused_index_map")
        if not isinstance(index_map, dict):
            raise ValueError("RFD3 metadata lacks diffused_index_map")
    mapping, sequences, chunks, ref_coords, output_coords = [], {}, [], [], []
    for expected, output_chain in zip(layout, output_chains):
        chain_groups = [g for g in groups if g[0][0] == output_chain]
        if len(chain_groups) != expected["length"]:
            raise ValueError("Output length differs from explicit contig slots")
        chain = expected["chain"]
        slots = [m for m in candidate["residue_mapping"] if m["sequence_chain"] == chain]
        sequences[chain] = "".join(aa for _, aa, _ in chain_groups)
        for position, ((key, aa, atoms), slot) in enumerate(zip(chain_groups, slots), 1):
            if not slot["designed"]:
                if aa != slot["input_aa"]:
                    raise ValueError(f"RFD3 changed fixed sequence at {slot['source']}")
                if index_map is not None and index_map.get(f"{chain}{position}") != f"{key[0]}{key[1]}{key[2]}":
                    raise ValueError("RFD3 index map contradicts an explicit fixed contig slot")
                original = ref_by_slot[(chain, position)]
                for name in ["N", "CA", "C", "O"]:
                    ref_coords.append(original.coord[original.atom_name == name][0])
                    output_coords.append(atoms.coord[atoms.atom_name == name][0])
            item = dict(slot, generated=residue(*key))
            mapping.append(item)
            # MPNN gets backbone only; the original all-atom CIF is retained separately.
            copy = atoms[np.isin(atoms.atom_name, ["N", "CA", "C", "O"])].copy()
            copy.chain_id[:] = chain
            copy.res_id[:] = position
            copy.ins_code[:] = ""
            chunks.append(copy)
    validation = fixed_backbone_alignment(ref_coords, output_coords)
    tolerance = config.get("fixed_backbone_tolerance_angstrom", 0.1)
    rmsd_tolerance = config.get("fixed_backbone_rmsd_tolerance_angstrom", 0.05)
    if (validation["fixed_backbone_max_error_angstrom"] > tolerance or
            validation["fixed_backbone_rmsd_angstrom"] > rmsd_tolerance):
        raise ValueError(f"Fixed backbone exceeds max-error {tolerance} / RMSD {rmsd_tolerance} Angstrom: {validation}")
    validation.update(fixed_backbone_max_error_limit_angstrom=tolerance,
                      fixed_backbone_rmsd_limit_angstrom=rmsd_tolerance)
    write_pdb(struc.concatenate(chunks), directory / "backbone.pdb")
    candidate.update(full_sequences=sequences, residue_mapping=mapping,
                     structure_path=str(Path(structure).resolve()), backbone_path=str(directory / "backbone.pdb"),
                     mapping_method="explicit input contig slots; chain lengths and fixed sequences verified",
                     validation=validation)
    extract_h3(candidate, config.get("required_h3_length"))
    save_json(directory / "candidate.json", candidate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "convert"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--input-manifest")
    parser.add_argument("--structure")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = load_json(args.config)
    if args.command == "prepare":
        prepare(config, args.out_dir)
    else:
        convert(config, args.input_manifest, args.structure, args.out_dir)


if __name__ == "__main__":
    main()
