import copy
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tasks.copper_inversion.rsi import protocol as protocols


EXAMPLE = Path(protocols.RESOURCES) / "rsi/protocol.example.json"
REPO = protocols.REPO
NOW = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)
SIDECAR, EVALUATOR = "sha256:" + "a" * 64, "sha256:" + "b" * 64


def example():
    return json.loads(EXAMPLE.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True))
    return path


@pytest.fixture
def sources(tmp_path, monkeypatch):
    """Analysis code, tagger and embedding provenance for a complete freeze."""
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    (analysis / "metrics.py").write_text("X = 1\n")
    (analysis / "tagger.py").write_text("RULES = ()\n")
    model = tmp_path / "model"
    model.mkdir()
    (model / "provenance.json").write_text('{"model": "fixture"}\n')
    monkeypatch.setattr(protocols, "ANALYSIS", analysis)
    monkeypatch.setattr(protocols, "TAGGER", analysis / "tagger.py")
    return {"embedding_model_path": model, "sidecar_image": SIDECAR, "evaluator_image": EVALUATOR}


def test_example_protocol_validates_and_has_a_hash_suffixed_id():
    value = protocols.load(EXAMPLE)
    sha = protocols.protocol_sha256(value)
    assert protocols.protocol_id(value) == "rsi-dev-" + sha[:16]
    assert value["release"]["name"] == "rsi-dev" and value["token_cap_input"] == 0
    assert set(value["hashes"]) == set(protocols.HASH_FIELDS)
    assert set(value["arms"]) == {"ldm4geo", "accumulated", "restart_best", "independent_sampling",
                                  "single_conversation", "separate_runs", "program_evolution"}


def test_hash_is_canonical_and_covers_every_value():
    value = example()
    shuffled = json.loads(json.dumps(value, sort_keys=False), object_pairs_hook=lambda p: dict(reversed(p)))
    assert protocols.protocol_sha256(shuffled) == protocols.protocol_sha256(value)
    changed = copy.deepcopy(value)
    changed["loop"]["harness_seconds"] += 1
    assert protocols.protocol_sha256(changed) != protocols.protocol_sha256(value)
    changed = copy.deepcopy(value)
    changed["hashes"]["tool"] = "c" * 64
    assert protocols.protocol_id(changed) != protocols.protocol_id(value)
    assert protocols.content_sha256(changed) == protocols.content_sha256(value)


@pytest.mark.parametrize("change", [
    lambda p: p.pop("model"),
    lambda p: p.update(protocol_version=2),
    lambda p: p["release"].update(name="RSI R1"),
    lambda p: p["release"].update(qualification_sha256="abc"),
    lambda p: p["tiers"]["S"].update(slots=321),
    lambda p: p["tiers"]["S"].update(train_manifest="../train/manifest.json"),
    lambda p: p["tiers"]["S"].pop("solver_seconds"),
    lambda p: p["loop"].update(batch=3),
    lambda p: p["loop"].update(rounds=0),
    lambda p: p["judges"].update(oracle={"objective": "confirmation"}),
    lambda p: p["judges"].update(q25={"objective": "oracle_iou"}),
    lambda p: p["judges"].pop("standard"),
    lambda p: p["model"].update(thinking="extreme"),
    lambda p: p["model"].update(api_key="secret"),
    lambda p: p.update(token_cap_input=-1),
    lambda p: p["arms"]["ldm4geo"].update(clear="sometimes"),
    lambda p: p["arms"]["ldm4geo"].pop("budget_split"),
    lambda p: p["arms"]["ldm4geo"].update(extra=1),
    lambda p: p["seeds"].update(E3=[101]),
    lambda p: p["hashes"].update(tool="xyz"),
    lambda p: p["hashes"].update(sidecar_image="a" * 64),
    lambda p: p["hashes"].update(unknown=None),
    lambda p: p["decisions"].update(mde_bound_relative=1.5),
])
def test_validation_rejects_malformed_protocols(change):
    value = example()
    change(value)
    with pytest.raises(ValueError, match="invalid protocol"):
        protocols.validate(value)


def test_load_rejects_non_finite_numbers(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text(EXAMPLE.read_text().replace('"q1_spearman_min": 0.8', '"q1_spearman_min": NaN'))
    with pytest.raises(ValueError, match="NaN"):
        protocols.load(path)


def test_legacy_protocol_without_tier_table_validates():
    value = example()
    value.pop("tiers")
    protocols.validate(value)


@pytest.mark.parametrize("strategy", [
    {"name": "main_forks"},
    {"name": "main_forks", "fork_policy": {"name": "main_history_1plus3"}},
    {"name": "independent", "measurement_policy": {"factory": "x:y"}},
    {"factory": "tasks.copper_inversion.rsi.arms:policy"},
])
def test_protocol_refuses_fork_strategies(strategy):
    value = example()
    value["research_strategy"] = strategy
    with pytest.raises(ValueError, match="fork strategies are refused"):
        protocols.validate(value)
    value = example()
    value["arms"]["ldm4geo"]["fork_policy"] = {"name": "latest"}
    with pytest.raises(ValueError, match="fork strategies are refused"):
        protocols.validate(value)
    value = example()
    value["research_strategy"] = {"name": "independent"}
    protocols.validate(value)


def test_freeze_fills_hashes_and_writes_a_timestamped_record(tmp_path, sources):
    path = write(tmp_path / "protocol.json", example())
    with pytest.raises(ValueError, match="cannot compute evaluator_image, sidecar_image"):
        protocols.freeze(path, now=NOW, embedding_model_path=sources["embedding_model_path"])
    assert not list(tmp_path.glob("protocol.json.freeze-*"))
    partial = protocols.freeze(path, allow_pending=True, now=NOW,
                               embedding_model_path=sources["embedding_model_path"])
    assert partial["pending"] == ["evaluator_image", "sidecar_image"] and not partial["complete"]
    frozen = protocols.load(path)
    current = protocols.local_hashes(embedding_model_path=sources["embedding_model_path"])
    for name in ("task_brief", "renderer", "tool", "implementation", "embedding_model", "analysis", "tagger"):
        assert frozen["hashes"][name] == current[name] is not None
    assert frozen["hashes"]["task_brief"] == protocols._file(protocols.RESOURCES / "rsi/TASK.md")
    with pytest.raises(ValueError, match="no complete freeze record"):
        protocols.require_frozen(path, frozen)
    # Completed on the run host with the image digests.
    later = NOW + timedelta(minutes=5)
    record = protocols.freeze(path, now=later, **sources)
    assert record["complete"] and record["pending"] == []
    assert record["previous_protocol_sha256"] == partial["protocol_sha256"]
    assert record["content_sha256"] == partial["content_sha256"]
    assert record["frozen_at"] == later.isoformat(timespec="seconds")
    saved = tmp_path / "protocol.json.freeze-20261115T120500Z.json"
    assert json.loads(saved.read_text()) == record
    frozen = protocols.load(path)
    assert protocols.require_frozen(path, frozen) == record
    assert record["protocol_id"] == protocols.protocol_id(frozen)
    assert frozen["hashes"]["sidecar_image"] == SIDECAR
    with pytest.raises(ValueError, match="already frozen"):
        protocols.freeze(path, now=later + timedelta(minutes=1), **sources)


def test_partial_freeze_cannot_be_completed_after_an_edit(tmp_path, sources):
    path = write(tmp_path / "protocol.json", example())
    protocols.freeze(path, allow_pending=True, now=NOW, embedding_model_path=sources["embedding_model_path"])
    value = protocols.load(path)
    value["loop"]["harness_seconds"] = 900
    write(path, value)
    with pytest.raises(ValueError, match="changed since its partial freeze"):
        protocols.freeze(path, now=NOW + timedelta(minutes=1), **sources)
    with pytest.raises(ValueError, match="differs from its freeze record|no complete freeze"):
        protocols.require_frozen(path, value)


def test_formal_freeze_requires_cap_qualification_and_tiers(tmp_path, sources):
    value = example()
    value["release"]["name"] = "rsi-r1"
    path = write(tmp_path / "protocol.json", value)
    with pytest.raises(ValueError, match="token_cap_input"):
        protocols.freeze(path, now=NOW, **sources)
    value["token_cap_input"] = 50_000_000
    write(path, value)
    with pytest.raises(ValueError, match="qualification"):
        protocols.freeze(path, now=NOW, **sources)
    value["release"]["qualification_sha256"] = "d" * 64
    write(path, value)
    assert protocols.freeze(path, now=NOW, **sources)["protocol_id"].startswith("rsi-r1-")


def test_verify_lists_changed_frozen_components(tmp_path, sources):
    path = write(tmp_path / "protocol.json", example())
    protocols.freeze(path, now=NOW, **sources)
    frozen = protocols.load(path)
    resources = tmp_path / "resources"
    shutil.copytree(protocols.RESOURCES / "rsi", resources / "rsi")
    current = protocols.local_hashes(resources=resources, **sources)
    protocols.verify(frozen, current)
    task = resources / "rsi/TASK.md"
    body = bytearray(task.read_bytes())
    body[0] ^= 1
    task.write_bytes(bytes(body))
    current = protocols.local_hashes(resources=resources, **sources)
    with pytest.raises(ValueError, match="no longer match: task_brief$"):
        protocols.verify(frozen, current)
    assert protocols.mismatches(frozen, {**current, "analysis": None}) == ["analysis", "task_brief"]
    assert protocols.mismatches(frozen, current, skip=("task_brief",)) == []


def test_unfrozen_fields_are_not_checked():
    value = example()
    assert protocols.mismatches(value, {name: None for name in protocols.HASH_FIELDS}) == []


def test_implementation_hash_ignores_protocol_files_in_the_tree():
    identity = protocols.implementation_identity()
    assert identity == protocols.implementation_identity()
    assert len(identity) == 64


def test_image_digest_accepts_only_immutable_digests():
    assert protocols.image_digest(SIDECAR) == SIDECAR
    assert protocols.image_digest(None) is None
    with pytest.raises(ValueError):
        protocols.image_digest("sha256:short")


# Entry point ---------------------------------------------------------------

workflow = pytest.importorskip("tasks.copper_inversion.rsi.workflow")


def _argv(tmp_path, protocol_path, *, experiment="smoke", arm="ldm4geo", seed=101, tier="S", judge=None,
          out=None, extra=()):
    argv = ["--protocol", str(protocol_path), "--experiment", experiment, "--arm", arm, "--seed", str(seed),
            "--out-dir", str(out or tmp_path / "runs" / workflow.run_id(experiment, arm, judge or "standard", seed)),
            "--public-catalog", str(tmp_path / "catalog.json"), "--evaluation-socket", str(tmp_path / "e.sock"),
            "--evaluation-contract", "c" * 64, "--evaluation-service-uid", "991",
            "--embedding-model-path", str(tmp_path / "model"), "--sidecar-image", "sidecar:dev",
            "--harness-cache-dir", str(tmp_path / "cache"), "--api-key-file", str(tmp_path / "key")]
    if tier is not None:
        argv += ["--tier", tier]
    if judge is not None:
        argv += ["--judge", judge]
    return argv + list(extra)


def _prepare(*args, **kwargs):
    return workflow.prepare(workflow.parse_args(_argv(*args, **kwargs)))


def _strategy_files():
    yield from sorted((REPO / "config/research_strategies").glob("*.json"))
    yield protocols.TASK / "resources/harness/research_coverage.json"


@pytest.mark.parametrize("config", [{"name": "main_forks"},
                                    {"name": "main_forks", "fork_policy": {"name": "coverage"}}])
def test_entry_point_refuses_main_forks_strategy_at15(tmp_path, config):
    """AT-15: starting the new entry point with a main_forks strategy is refused."""
    path = tmp_path / "strategy.json"
    path.write_text(json.dumps(config))
    argv = _argv(tmp_path, EXAMPLE, extra=["--harness-research-strategy-config", str(path)])
    with pytest.raises(ValueError, match="fork strategies are refused"):
        workflow.main(argv)
    assert not (tmp_path / "runs").exists()


def test_entry_point_refuses_every_shipped_fork_strategy(tmp_path):
    files = [p for p in _strategy_files() if p.is_file() and json.loads(p.read_text())["name"] == "main_forks"]
    assert len(files) >= 3
    for path in files:
        with pytest.raises(ValueError, match="fork strategies are refused"):
            _prepare(tmp_path, EXAMPLE, extra=["--harness-research-strategy-config", str(path)])
    independent = tmp_path / "independent.json"
    independent.write_text('{"name": "independent"}')
    _prepare(tmp_path, EXAMPLE, extra=["--harness-research-strategy-config", str(independent)])


def test_entry_point_refuses_a_run_after_a_one_byte_brief_change_at39(tmp_path, sources, monkeypatch):
    """AT-39: after the protocol freeze, a one-byte change to the task brief refuses the run."""
    path = write(tmp_path / "protocol.json", example())
    protocols.freeze(path, now=NOW, **sources)
    resources = tmp_path / "resources"
    shutil.copytree(protocols.RESOURCES / "rsi", resources / "rsi")
    monkeypatch.setattr(workflow, "RESOURCES", resources)
    plan = _prepare(tmp_path, path, experiment="M1")
    assert plan["manifest"]["protocol_id"] == protocols.protocol_id(protocols.load(path))
    task = resources / "rsi/TASK.md"
    task.write_bytes(task.read_bytes() + b" ")
    with pytest.raises(ValueError, match="no longer match: task_brief"):
        _prepare(tmp_path, path, experiment="M1")
    with pytest.raises(ValueError, match="no longer match: task_brief"):
        workflow.main(_argv(tmp_path, path, experiment="M1"))
    assert not (tmp_path / "runs").exists()


def test_formal_experiments_require_a_complete_freeze(tmp_path, sources):
    path = write(tmp_path / "protocol.json", example())
    with pytest.raises(ValueError, match="no complete freeze record"):
        _prepare(tmp_path, path, experiment="M1")
    protocols.freeze(path, allow_pending=True, now=NOW, embedding_model_path=sources["embedding_model_path"])
    with pytest.raises(ValueError, match="no complete freeze record"):
        _prepare(tmp_path, path, experiment="M1")
    _prepare(tmp_path, path, experiment="rehearsal")


def test_manifest_has_every_contract_field(tmp_path, sources):
    path = write(tmp_path / "protocol.json", example())
    protocols.freeze(path, now=NOW, **sources)
    plan = _prepare(tmp_path, path, experiment="E3", arm="accumulated", seed=203, judge="oracle")
    manifest = plan["manifest"]
    assert {"run_id", "experiment", "protocol_id", "arm", "tier", "judge", "seed", "blind", "continuation",
            "evaluation", "code_commit", "started_at", "host"} <= set(manifest)
    value = protocols.load(path)
    assert manifest["run_id"] == "accumulated-oracle-s203" and manifest["experiment"] == "E3"
    assert manifest["protocol_id"] == protocols.protocol_id(value)
    assert (manifest["arm"], manifest["tier"], manifest["judge"], manifest["seed"]) == ("accumulated", "S", "oracle", 203)
    assert manifest["blind"] is False and manifest["objective"] == "oracle_iou"
    assert manifest["continuation"] is None and "intervention" not in manifest
    assert manifest["evaluation"] == {"socket": str(tmp_path / "e.sock"), "contract_digest": "c" * 64,
                                      "service_uid": 991}
    assert manifest["solver_seconds"] == 10 and manifest["model"] == value["model"]
    assert datetime.fromisoformat(manifest["started_at"]).tzinfo is not None
    standard = _prepare(tmp_path, EXAMPLE, tier="F")["manifest"]
    assert standard["blind"] is True and standard["run_id"] == "ldm4geo-s101" and standard["solver_seconds"] == 120


def test_resume_keeps_the_original_start(tmp_path):
    out = tmp_path / "runs" / "ldm4geo-s101"
    out.mkdir(parents=True)
    (out / "manifest.json").write_text(json.dumps({"started_at": "2026-11-01T00:00:00+00:00", "host": "h20"}))
    manifest = _prepare(tmp_path, EXAMPLE, extra=["--resume"])["manifest"]
    assert (manifest["started_at"], manifest["host"]) == ("2026-11-01T00:00:00+00:00", "h20")


@pytest.mark.parametrize("kwargs, message", [
    ({"arm": "tree_search"}, "arm is not defined"),
    ({"judge": "q25"}, "judge is not defined"),
    ({"experiment": "M1", "judge": "oracle"}, "only in E3"),
    ({"experiment": "M1", "seed": 7}, "seed table"),
    ({"tier": "X"}, "tier is not defined"),
    ({"tier": None}, "tier is not defined"),
    ({"out": "wrong-name"}, "must be named ldm4geo-s101"),
    ({"extra": ["--solver-seconds", "30"]}, "tier table"),
])
def test_entry_point_refuses_inconsistent_requests(tmp_path, kwargs, message):
    if kwargs.get("out"):
        kwargs["out"] = tmp_path / kwargs["out"]
    with pytest.raises(ValueError, match=message):
        _prepare(tmp_path, EXAMPLE, **kwargs)


def test_reserve_seeds_are_allowed_for_their_experiment(tmp_path, sources):
    path = write(tmp_path / "protocol.json", example())
    protocols.freeze(path, now=NOW, **sources)
    assert _prepare(tmp_path, path, experiment="M1", seed=113)["manifest"]["seed"] == 113


def test_legacy_release_keeps_the_old_full_tier_limits(tmp_path):
    value = example()
    value.pop("tiers")
    path = write(tmp_path / "protocol.json", value)
    plan = _prepare(tmp_path, path, tier=None)
    assert plan["manifest"]["tier"] == "F" and plan["solver_seconds"] == 120
    assert _prepare(tmp_path, path, tier=None, extra=["--solver-seconds", "10"])["solver_seconds"] == 10
    with pytest.raises(ValueError, match="between 1 and 120"):
        _prepare(tmp_path, path, tier=None, extra=["--solver-seconds", "200"])
    with pytest.raises(ValueError, match="legacy full tier"):
        _prepare(tmp_path, path, tier="S")


def test_controller_settings_come_from_the_protocol(tmp_path):
    value = protocols.load(EXAMPLE)
    config = workflow.build_config(value, "ldm4geo", "S", 101)
    loop = value["loop"]
    assert (config.arm, config.seed, config.rounds, config.proposals_per_branch, config.batch,
            config.reset_period, config.harness_seconds) == (
        "ldm4geo", 101, loop["rounds"], loop["proposals_per_branch"], loop["batch"],
        loop["reset_period"], loop["harness_seconds"])
    assert config.slots == value["tiers"]["S"]["slots"]
    if not hasattr(workflow.Config, "from_protocol"):
        assert config.token_cap_input == workflow.NO_TOKEN_CAP
        capped = copy.deepcopy(value)
        capped["token_cap_input"] = 1000
        assert workflow.build_config(capped, "ldm4geo", "S", 101).token_cap_input == 1000
        for arm in ("single_conversation", "separate_runs", "program_evolution"):
            with pytest.raises(ValueError, match="cannot run the protocol's definition"):
                workflow.build_config(value, arm, "S", 101)
        changed = copy.deepcopy(value)
        changed["arms"]["ldm4geo"]["clear"] = "never"
        with pytest.raises(ValueError, match="cannot run the protocol's definition"):
            workflow.build_config(changed, "ldm4geo", "S", 101)


def test_model_settings_reach_the_agent_or_are_refused():
    value = protocols.load(EXAMPLE)

    class Legacy:
        def __init__(self, *, public_catalog, api_key, sidecar_image, cache_dir, arm, reset_period):
            pass

    class Configurable:
        def __init__(self, *, public_catalog, api_key, sidecar_image, cache_dir, arm, reset_period,
                     base_url, model, thinking, bash_calls):
            pass

    assert workflow.agent_settings(value, Legacy) == {}
    other = copy.deepcopy(value)
    other["model"].update(name="other-model", thinking="high")
    other["loop"]["bash_calls"] = 4
    with pytest.raises(ValueError, match="bash_calls, model, thinking"):
        workflow.agent_settings(other, Legacy)
    assert workflow.agent_settings(other, Configurable) == {
        "base_url": value["model"]["base_url"], "model": "other-model", "thinking": "high", "bash_calls": 4}


def test_evaluation_concurrency_comes_from_the_scheduling_policy(tmp_path):
    from tasks.copper_inversion.rsi.workflow import evaluation_concurrency
    assert evaluation_concurrency({}) == 1
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"batch_workers": 16}))
    assert evaluation_concurrency({"LDM4GEO_SCHEDULING_POLICY": str(policy)}) == 16
    policy.write_text(json.dumps({"batch_workers": 0}))
    with pytest.raises(ValueError):
        evaluation_concurrency({"LDM4GEO_SCHEDULING_POLICY": str(policy)})
