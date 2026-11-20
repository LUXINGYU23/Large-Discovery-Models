import json
from types import SimpleNamespace

import pytest

from ldm_tts.harness import file_sha256
from tasks.antibody_rfd3.core.seed_bank import load_seed_bank
from tasks.antibody_rfd3.core.worker import DesignWorker


def test_seed_candidates_retain_verified_backbones_available_to_sequence_tool(tmp_path, monkeypatch):
    folder = tmp_path / "seed-backbones" / "seed-job-00000"
    folder.mkdir(parents=True)
    pdb, cif, manifest = (folder / name for name in ("backbone.pdb", "structure.cif.gz", "generation.json"))
    pdb.write_text("input coordinates")
    cif.write_text("source coordinates")
    manifest.write_text(json.dumps({"status": "success", "candidates": [
        {"backbone_path": str(pdb), "structure_path": str(cif)}]}))
    key = folder.name
    bank = tmp_path / "seed.json"
    bank.write_text(json.dumps({"antigen": "1N8Z_C", "oracle": "absolut",
        "candidates": [{"sequence": s, "backbone_id": key} for s in ("SNGDWGSPGTV", "GSNSGSSAGAA")],
        "backbones": {key: {"manifest": str(manifest.relative_to(tmp_path)),
                            "manifest_sha256": file_sha256(manifest), "pdb_sha256": file_sha256(pdb),
                            "cif_sha256": file_sha256(cif)}}}))
    args = SimpleNamespace(seed_bank=bank, initialization_run=None, mock=False, antigen="1N8Z_C",
                           oracle="absolut", iterations=4, method="fixed", backbones_per_round=1,
                           sequences_per_round=8, seed=42)
    seeds, backbones = load_seed_bank(args)
    assert [row.payload["backbone_id"] for row in seeds] == [key, key]
    assert backbones[key] == manifest
    worker = DesignWorker(tmp_path / "run", args, seed_backbones=backbones)
    worker.round = 1
    calls = []

    def sequence_tool(_job, backbone, count, _temp, _seed):
        calls.append(backbone)
        return ["GSGSGTGTGAA"] * count

    monkeypatch.setattr(worker, "run_models", sequence_tool)
    result, = worker.execute("sequence_h3", 1, 0.5, backbone_id=key)
    assert calls == [key]
    assert result["backbone_id"] == key

    pdb.write_text("tampered")
    with pytest.raises(ValueError, match="Seed backbone files differ"):
        load_seed_bank(args)
