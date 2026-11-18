"""Use synthetic structures to verify identity and geometric constraints."""
import numpy as np
import pytest

pytest.importorskip("biotite")
import biotite.structure as struc

from tasks.antibody.tools.structure_design.common import extract_h3, load_json, residue, save_json
from tasks.antibody.tools.structure_design.worker import prepare, convert, read_structure, write_pdb


def setup_input(tmp_path):
    array = struc.AtomArray(20)
    array.chain_id[:] = "B"
    array.res_id[:] = np.repeat([98, 99, 100, 100, 110], 4)
    array.ins_code[:] = np.repeat(["", "", "A", "B", ""], 4)
    array.res_name[:] = np.repeat(["ALA", "GLY", "SER", "ASP", "TRP"], 4)
    array.atom_name[:] = ["N", "CA", "C", "O"] * 5
    array.element[:] = ["N", "C", "C", "O"] * 5
    array.hetero[:] = False
    array.coord[:] = [[i, np.sin(i), np.cos(i)] for i in range(20)]
    path = tmp_path / "source.pdb"
    write_pdb(array, path)
    config = {"input_structure": str(path), "heavy_chain": "B",
              "h3_residues": [residue("B", 99), residue("B", 100, "A"), residue("B", 100, "B")],
              "expected_input_h3": "GSD", "required_h3_length": 3}
    inputs = tmp_path / "input"
    inputs.mkdir()
    prepare(config, inputs)
    return config, inputs


def generated(inputs, tmp_path):
    array = read_structure(inputs / "prepared.pdb")
    array.chain_id[:] = "Z"
    array.res_id[:] += 10
    # A rigid transform must not be confused with changing a fixed framework.
    array.coord[:] = array.coord @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) + [3, 4, 5]
    target = tmp_path / "generated.pdb"
    write_pdb(array, target)
    return target


def test_prepare_uses_explicit_slots_and_preserves_insertion_codes(tmp_path):
    config, inputs = setup_input(tmp_path)
    candidate = load_json(inputs / "input.json")
    assert extract_h3(candidate, 3) == "GSD"
    assert candidate["residue_mapping"][2]["source"] == residue("B", 100, "A")
    assert load_json(inputs / "specification.json")["design"]["contig"] == "A1-1,3,A5-5"


def test_rigid_transform_and_chain_rename_are_valid(tmp_path):
    config, inputs = setup_input(tmp_path)
    target = generated(inputs, tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    convert(config, inputs / "input.json", target, output)
    candidate = load_json(output / "candidate.json")
    assert extract_h3(candidate, 3) == "GSD"
    assert candidate["residue_mapping"][1]["generated"]["chain"] == "Z"
    assert candidate["validation"]["fixed_backbone_max_error_angstrom"] < 0.01


@pytest.mark.parametrize("mutation", ["fixed_sequence", "fixed_coordinates", "missing_backbone", "wrong_length", "nonfinite"])
def test_invalid_generated_structures_rejected(tmp_path, mutation):
    config, inputs = setup_input(tmp_path)
    target = generated(inputs, tmp_path)
    array = read_structure(target)
    if mutation == "fixed_sequence":
        array.res_name[:4] = "LYS"
    elif mutation == "fixed_coordinates":
        array.coord[0] += [0, 1, 0]
    elif mutation == "missing_backbone":
        array = array[1:]
    elif mutation == "wrong_length":
        array = array[:-4]
    else:
        # Biotite's writer itself rejects NaNs; inject one into serialized input
        # to exercise our reader's admission check rather than the writer.
        lines = target.read_text().splitlines()
        index = next(i for i, line in enumerate(lines) if line.startswith("ATOM"))
        lines[index] = lines[index][:30] + "     nan" + lines[index][38:]
        target.write_text("\n".join(lines) + "\n")
    if mutation != "nonfinite":
        write_pdb(array, target)
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError):
        convert(config, inputs / "input.json", target, output)


def test_missing_input_h3_is_rejected(tmp_path):
    config, inputs = setup_input(tmp_path)
    config["h3_residues"][-1] = residue("B", 100, "C")
    with pytest.raises(ValueError, match="missing input residues"):
        prepare(config, inputs)


@pytest.mark.parametrize("wrong_map", [False, True])
def test_upstream_index_map_must_match_fixed_contig_anchors(tmp_path, wrong_map):
    config, inputs = setup_input(tmp_path)
    target = generated(inputs, tmp_path)
    save_json(target.with_suffix(".json"), {"diffused_index_map": {
        "A1": "Z12" if wrong_map else "Z11", "A5": "Z15"}})
    output = tmp_path / "out"
    output.mkdir()
    if wrong_map:
        with pytest.raises(ValueError, match="index map contradicts"):
            convert(config, inputs / "input.json", target, output)
    else:
        convert(config, inputs / "input.json", target, output)
