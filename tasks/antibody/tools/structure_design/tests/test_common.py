"""Meaningful admission and mapping checks; no model downloads or GPU needed."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tasks.antibody.tools.structure_design.common import extract_h3, residue, save_json, validate_config
from tasks.antibody.tools.structure_design.__main__ import main, resolve_config, model_command


def candidate():
    refs = [residue("H", 100, "A"), residue("H", 100, "B"), residue("H", 101)]
    return {"candidate_id": "test", "heavy_chain": "H", "h3_residues": refs,
            "full_sequences": {"Z": "AGSDW"},
            "residue_mapping": [{"source": r, "sequence_chain": "Z", "sequence_position": i + 2}
                                for i, r in enumerate(refs)]}


def config():
    value = candidate()
    return {"input_structure": "example.pdb", "heavy_chain": "H", "h3_residues": value["h3_residues"],
            "required_h3_length": 3}


def test_chain_rename_and_insertions_preserved():
    assert extract_h3(candidate(), 3) == "GSD"


@pytest.mark.parametrize("change", ["missing", "duplicate_source", "wrong_chain", "bad_position",
                                     "out_of_range", "duplicate_position", "order", "alphabet", "length"])
def test_invalid_mapping_rejected(change):
    value = candidate()
    if change == "missing":
        value["residue_mapping"].pop()
    elif change == "duplicate_source":
        value["residue_mapping"].append(value["residue_mapping"][0])
    elif change == "wrong_chain":
        value["heavy_chain"] = "B"
    elif change == "bad_position":
        value["residue_mapping"][0]["sequence_position"] = 0
    elif change == "out_of_range":
        value["residue_mapping"][0]["sequence_position"] = 20
    elif change == "duplicate_position":
        value["residue_mapping"][0]["sequence_position"] = 3
    elif change == "order":
        value["h3_residues"].reverse()
    elif change == "alphabet":
        value["full_sequences"]["Z"] = "AXSDW"
    with pytest.raises(ValueError):
        extract_h3(value, 11 if change == "length" else 3)


def test_length_is_optional_and_never_trims():
    assert extract_h3(candidate()) == "GSD"
    with pytest.raises(ValueError, match="no trimming/padding"):
        extract_h3(candidate(), 2)


def test_duplicate_and_conflicting_config_rejected():
    value = config()
    value["design_residues"] = value["h3_residues"]
    value["fixed_residues"] = [value["h3_residues"][0]]
    with pytest.raises(ValueError, match="both fixed and designed"):
        validate_config(value)
    value = config()
    value["h3_residues"].append(value["h3_residues"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_config(value)


@pytest.mark.parametrize("key,value", [("seed", 0), ("num_designs", 0), ("sequences_per_design", -1),
                                        ("num_timesteps", 0), ("required_h3_length", 11), ("seed", True),
                                        ("sampling_temperature", float("nan")),
                                        ("fixed_backbone_tolerance_angstrom", float("inf")),
                                        ("gpu_wait_seconds", -1)])
def test_invalid_parameters(key, value):
    cfg = config()
    cfg[key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_extract_cli_requires_no_model_or_gpu(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config()))
    designs = tmp_path / "designs.json"
    designs.write_text(json.dumps({"status": "success", "candidates": [candidate()]}))
    output = tmp_path / "out"
    assert main(["extract-h3", "--config", str(cfg), "--out-dir", str(output), "--designs", str(designs)]) == 0
    assert json.loads((output / "h3.json").read_text())["candidates"][0]["h3"] == "GSD"


def test_failed_extraction_not_reported_as_success(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config()))
    designs = tmp_path / "designs.json"
    designs.write_text(json.dumps({"status": "failed", "candidates": [candidate()]}))
    output = tmp_path / "out"
    assert main(["extract-h3", "--config", str(cfg), "--out-dir", str(output), "--designs", str(designs)]) == 1
    assert json.loads((output / "run.json").read_text())["status"] == "failed"
    assert not (output / "h3.json").exists()


def test_venv_interpreter_symlink_is_not_resolved_to_system_python(tmp_path):
    interpreter = tmp_path / "env/rfd3/bin/python"
    interpreter.parent.mkdir(parents=True)
    try:
        interpreter.symlink_to(sys.executable)
    except OSError:
        pytest.skip("Creating symlinks is not permitted on this host")
    cfg = config()
    cfg["workspace_root"] = str(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    resolved = resolve_config(path)
    assert Path(resolved["rfd3_python"]) == interpreter
    assert Path(resolved["rfd3_python"]).is_symlink()


def test_cli_does_not_overwrite_an_existing_run(tmp_path):
    original = tmp_path / "run.json"
    original.write_text('{"status":"success"}')
    with pytest.raises(SystemExit):
        main(["pipeline", "--config", "unused.json", "--out-dir", str(tmp_path)])
    assert original.read_text() == '{"status":"success"}'


def test_mpnn_jsonl_constraints_are_one_json_object_per_line(tmp_path):
    path = tmp_path / "fixed.jsonl"
    value = {"backbone": {"A": [1, 2, 3]}}
    save_json(path, value)
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == value


def test_busy_gpu_launcher_is_explicit_and_preserves_arguments():
    command = ["/venv/bin/rfd3", "design", "n_batches=1"]
    assert model_command({}, command, module="rfd3.cli") == command
    cfg = {"allow_gpu_sharing": True, "gpu_memory_limit_gib": 24, "rfd3_python": "/venv/bin/python"}
    wrapped = model_command(cfg, command, module="rfd3.cli")
    assert wrapped[0] == "/venv/bin/python"
    assert wrapped[-3:] == ["--", "design", "n_batches=1"]
    assert wrapped[2:6] == ["--limit-gib", 24, "--module", "rfd3.cli"]


def test_sharing_without_explicit_gpu_does_not_launch_a_job(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config()))
    output = tmp_path / "out"
    assert main(["pipeline", "--config", str(path), "--out-dir", str(output), "--allow-gpu-sharing"]) == 1
    report = json.loads((output / "run.json").read_text())
    assert "explicit --gpu" in report["error"]
    assert not (output / "rfd3.log").exists()


def test_gpu_cap_launcher_preserves_sibling_imports_and_cli_arguments(tmp_path, monkeypatch):
    from tasks.antibody.tools.structure_design import gpu_runner
    calls = []
    fake_cuda = SimpleNamespace(
        get_device_properties=lambda index: SimpleNamespace(total_memory=96 * 1024**3),
        set_per_process_memory_fraction=lambda fraction, device: calls.append((fraction, device)))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setattr(sys, "path", list(sys.path))
    (tmp_path / "neighbor_for_launcher_test.py").write_text("VALUE = 42\n")
    result = tmp_path / "result.json"
    script = tmp_path / "upstream.py"
    script.write_text("import json,sys\nfrom pathlib import Path\nfrom neighbor_for_launcher_test import VALUE\n"
                      f"Path({str(result)!r}).write_text(json.dumps([VALUE,sys.argv[1:]]))\n")
    monkeypatch.setattr(sys, "argv", ["gpu_runner", "--limit-gib", "24", "--script", str(script), "--", "--seed", "42"])
    gpu_runner.main()
    assert calls == [(0.25, 0)]
    assert json.loads(result.read_text()) == [42, ["--seed", "42"]]


def test_explicit_cpu_mode_never_acquires_a_gpu(tmp_path, monkeypatch):
    from tasks.antibody.tools.structure_design import __main__ as cli
    path = tmp_path / "config.json"
    cfg = config()
    cfg["workspace_root"] = str(tmp_path)
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(cli, "provenance", lambda config: {})

    def forbid_gpu(*args, **kwargs):
        pytest.fail("CPU mode must not acquire a GPU")

    def check_cpu(config, directory, env, origin):
        assert env["CUDA_VISIBLE_DEVICES"] == ""
        assert env["OMP_NUM_THREADS"] == "8"

    monkeypatch.setattr(cli, "gpu_slot", forbid_gpu)
    monkeypatch.setattr(cli, "doctor", check_cpu)
    output = tmp_path / "out"
    assert cli.main(["doctor", "--config", str(path), "--out-dir", str(output), "--device", "cpu"]) == 0
    assert json.loads((output / "run.json").read_text())["device"] == "cpu"
    assert not (output / "gpu.json").exists()
