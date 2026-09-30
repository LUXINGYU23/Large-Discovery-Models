from dataclasses import replace
import json

import pytest

from ldm_tts.contracts.evaluation import EvaluationPaused
from tasks.alphabench.core.gateway import OracleGateway
from tasks.alphabench.core.generator import Generator
from tasks.alphabench.core.initialization import parse_seeds
from tasks.alphabench.core.protocol import T3Protocol
from tasks.alphabench.ldm_task.procedure import main


@pytest.mark.parametrize("text,suffix,pool", [
    ('{"alpha": "$close"}', ".json", False),
    ('[{"name":"alpha", "qlib_expression_default":"$close", "metrics":{"ic":999}}]', ".json", False),
    ('# comment\n$close\n', ".txt", False),
    ('{"name":"alpha", "expression":"$close", "val_metrics":{"ic":999}}\n', ".jsonl", True),
    ('{"final_pool":[{"expression":"$close", "test":{"ic":999}}]}', ".json", True),
])
def test_source_formats_discard_external_outcomes(text, suffix, pool):
    seeds = parse_seeds(text, suffix, pool=pool)
    assert len(seeds) == 1 and seeds[0]["expression"] == "$close"
    assert set(seeds[0]) == {"name", "expression"}


def test_import_records_duplicates_rejections_and_changed_sources(tmp_path):
    path, run = tmp_path / "pool.jsonl", tmp_path / "run"
    path.write_text('\n'.join(json.dumps(item) for item in [
        {"expression": "$close", "metrics": {"ic": 999}}, {"expression": " $close "},
        {"expression": "Ref($close,-1)"}, {"missing_expression": True}, {"expression": "$volume"}]))
    assert main(["--mock", "--init-mode", "import_pool", "--import-pool", str(path), "--out-dir", str(run)]) == 0
    manifest = json.loads((run / "initialization/seed_manifest.json").read_text())
    assert [row["status"] for row in manifest["admissions"]] == ["admitted", "duplicate", "rejected", "rejected", "admitted"]
    assert len(manifest["observations"]) == 2
    assert all(item["evaluation"]["metrics"]["mock_ic"] != 999 for item in manifest["observations"])
    report = json.loads((run / "result.json").read_text())
    assert report["search"]["attempts"] == 4
    assert report["initialization"]["source_records"] == 5
    assert report["initialization"]["source_dispositions"] == {"admitted": 2, "duplicate": 1, "rejected": 2}
    path.write_text('{"expression":"$open"}')
    with pytest.raises(ValueError, match="source changed"):
        main(["--mock", "--protocol-file", str(run / "protocol.json"), "--resume-run", str(run), "--import-pool", str(path)])


def test_empty_initialization_uses_no_model_or_evaluation_budget(tmp_path):
    protocol = replace(T3Protocol(), cold_seed_count=0,
                       budgets={**T3Protocol().budgets, "initialization_evaluations": 0})
    path, run = tmp_path / "protocol.json", tmp_path / "run"
    path.write_text(json.dumps(protocol.to_dict()))
    assert main(["--mock", "--protocol-file", str(path), "--out-dir", str(run)]) == 0
    report = json.loads((run / "result.json").read_text())
    assert report["initialization"]["seed_count"] == 0
    counters = report["initialization"]["budget"]["counters"]
    assert all(counters[name] == 0 for name in ("model_requests", "initialization_evaluations", "validation_evaluations"))
    assert report["search"]["attempts"] == 4


def test_initialization_pause_resumes_frozen_source_before_search_exists(tmp_path, monkeypatch):
    source, run = tmp_path / "seeds.txt", tmp_path / "run"
    source.write_text("$close\n$volume\n$open\n")
    evaluate, send = OracleGateway.evaluate, OracleGateway._send
    paused = []
    def pause_once(self, candidate, **kwargs):
        if kwargs["phase"] == "initialization" and kwargs["position"] == 1 and not paused:
            paused.append(True)
            raise EvaluationPaused("injected pause before oracle dispatch")
        return evaluate(self, candidate, **kwargs)
    def failed_seed(self, request):
        result = send(self, request)
        if request.get("phase") == "initialization" and request.get("expression") == "$open":
            result.update(success=False, metrics={"ic": None, "diagnostic": "not numerical"}, error="injected invalid seed")
        return result
    monkeypatch.setattr(OracleGateway, "evaluate", pause_once)
    monkeypatch.setattr(OracleGateway, "_send", failed_seed)
    assert main(["--mock", "--init-mode", "file", "--seed-file", str(source), "--out-dir", str(run)]) == 2
    assert not (run / "campaign.json").exists()
    assert json.loads((run / "initialization/status.json").read_text())["status"].startswith("paused_")
    source.unlink()
    args = ["--mock", "--protocol-file", str(run / "protocol.json"), "--resume-run", str(run)]
    assert main(args) == 0
    manifest = json.loads((run / "initialization/seed_manifest.json").read_text())
    assert len(manifest["observations"]) == 3
    assert [item["evaluation"]["status"] for item in manifest["observations"]] == ["succeeded", "succeeded", "invalid"]
    assert manifest["observations"][-1]["evaluation"]["metrics"] == {}
    before = {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    assert main(args) == 0
    assert {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()} == before


def test_distinct_runs_have_distinct_physical_initialization_requests(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(replace(T3Protocol(), cold_seed_count=1).to_dict()))
    identities = []
    for parent in ("first", "second"):
        run = tmp_path / parent / "same-name"
        assert main(["--mock", "--protocol-file", str(protocol), "--out-dir", str(run)]) == 0
        identities.append({json.loads(path.read_text())["request"]["request_id"]
                           for path in (run / "initialization/private/oracle").glob("*.json")})
    assert all(identities) and identities[0].isdisjoint(identities[1])


def test_cold_generation_pause_is_durable_before_engine_starts(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise EvaluationPaused("provider unavailable", status="paused_provider")
    monkeypatch.setattr(Generator, "request", unavailable)
    run = tmp_path / "run"
    assert main(["--mock", "--out-dir", str(run)]) == 2
    assert json.loads((run / "initialization/status.json").read_text())["status"] == "paused_provider"
    assert not (run / "initialization/seed_manifest.json").exists()
