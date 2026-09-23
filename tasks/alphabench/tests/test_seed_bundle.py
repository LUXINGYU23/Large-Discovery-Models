from dataclasses import replace
import json

import pytest

from ldm_tts.engine.run_store import CampaignRuntime
from tasks.alphabench.core.generator import Generator
from tasks.alphabench.core.protocol import T3Protocol, digest
from tasks.alphabench.core.seed_bundle import verify_seed_bundle
from tasks.alphabench.ldm_task.procedure import main


def make_source(tmp_path):
    protocol = replace(T3Protocol(), cold_seed_count=3)
    path, run = tmp_path / "protocol.json", tmp_path / "source"
    path.write_text(json.dumps(protocol.to_dict()))
    assert main(["--mock", "--protocol-file", str(path), "--out-dir", str(run)]) == 0
    return protocol, run / "initialization"


def test_matched_bundle_shares_observations_without_repeating_creation_cost(tmp_path, monkeypatch):
    protocol, source = make_source(tmp_path)
    target_protocol = replace(protocol, method="ldm", budgets={**protocol.budgets, "initialization_evaluations": 0})
    path, target = tmp_path / "target-protocol.json", tmp_path / "target"
    path.write_text(json.dumps(target_protocol.to_dict()))
    generate = Generator.generate
    def search_only(self, **kwargs):
        assert kwargs["identity"] != "initialization", "shared initialization generated new factors"
        return generate(self, **kwargs)
    monkeypatch.setattr(Generator, "generate", search_only)
    assert main(["--mock", "--protocol-file", str(path), "--initialization-bundle", str(source), "--out-dir", str(target)]) == 0
    original = json.loads((source / "seed_manifest.json").read_text())
    imported = json.loads((target / "initialization/seed_manifest.json").read_text())
    assert imported["observations"] == original["observations"]
    assert imported["pool"] == original["pool"]
    report = json.loads((target / "result.json").read_text())
    assert report["initialization"]["public_information_digest"] == original["public_information_digest"]
    assert all(value == 0 for value in report["initialization"]["budget"]["counters"].values())
    assert report["initialization"]["budget"]["metadata"]["shared_seed_creation"]["budget"]["counters"]["initialization_evaluations"] == 3
    assert report["including_initialization"]["initialization_cost_basis"] == "shared_source_creation"
    assert report["including_initialization"]["generation_cost"]["total_attempts"] == report["search"]["generation_cost"]["total_attempts"] + 1
    assert report["search"]["attempts"] == 4
    before = {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()}
    assert main(["--mock", "--protocol-file", str(path), "--resume-run", str(target)]) == 0
    assert {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("corruption", ["response_hash", "substituted_measurement", "unknown_receipt"])
def test_bundle_cannot_reuse_wrong_or_unfinished_receipts(tmp_path, corruption):
    protocol, source = make_source(tmp_path)
    for path in (source / "private/oracle").glob("*.json"):
        record = json.loads(path.read_text())
        if record["request"]["phase"] == "initialization":
            break
    if corruption == "unknown_receipt":
        record["state"] = "dispatch_intent"
    else:
        record["response"]["metrics"]["rank_ic"] = .999
        if corruption == "substituted_measurement":
            record["response_digest"] = digest(record["response"])
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="receipt"):
        verify_seed_bundle(source, protocol, True)


def test_bundle_rejects_different_science_and_mock_identity(tmp_path):
    protocol, source = make_source(tmp_path)
    with pytest.raises(ValueError, match="contract"):
        verify_seed_bundle(source, replace(protocol, data_digest="different"), True)
    with pytest.raises(ValueError, match="contract"):
        verify_seed_bundle(source, protocol, False)


@pytest.mark.parametrize("method,query,capabilities", [
    ("harness", False, ("prior_mean@1", "ldm_weights@1")),
    ("ldm_harness", True, ("prior_mean@1", "ldm_weights@1")),
    ("ldm_harness_compiled", True, ("prior_mean@1",)),
])
def test_bundle_shares_across_agent_search_options(tmp_path, method, query, capabilities):
    protocol, source = make_source(tmp_path)
    target = replace(protocol, method=method, harness_surrogate_query=query,
                     policy_capabilities=capabilities,
                     budgets={**protocol.budgets, **({"surrogate_queries": 5} if query else {})})
    assert verify_seed_bundle(source, target, True)["public_information_digest"] == json.loads(
        (source / "seed_manifest.json").read_text())["public_information_digest"]


def test_shared_import_recovers_interruption_before_finish(tmp_path, monkeypatch):
    protocol, source = make_source(tmp_path)
    target, path = tmp_path / "target", tmp_path / "protocol.json"
    finish, failures = CampaignRuntime.finish, []
    def interrupt_once(self, summary, **kwargs):
        if summary.get("imported") and not failures:
            failures.append(True)
            raise OSError("injected stop before import finish")
        return finish(self, summary, **kwargs)
    monkeypatch.setattr(CampaignRuntime, "finish", interrupt_once)
    with pytest.raises(OSError):
        main(["--mock", "--protocol-file", str(path), "--initialization-bundle", str(source), "--out-dir", str(target)])
    assert main(["--mock", "--protocol-file", str(path), "--resume-run", str(target)]) == 0
    events = [json.loads(line) for line in (target / "initialization/events.jsonl").read_text().splitlines()]
    assert sum(event["event_type"] == "shared_seeds_imported" for event in events) == 1
    report = json.loads((target / "result.json").read_text())
    assert report["search"]["attempts"] == 4
    assert all(value == 0 for value in report["initialization"]["budget"]["counters"].values())
