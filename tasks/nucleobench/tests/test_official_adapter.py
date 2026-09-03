from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ldm_tts.data import DataCollectionSink
from ldm_tts.engine import LDMEngine
from ldm_tts.engine.run_store import CampaignRuntime, atomic_json_write
from tasks.nucleobench.core.candidate import MutationContext, NucleoBenchCandidateDomain
from tasks.nucleobench.core.cases import get_case
from tasks.nucleobench.core.designer import (
    NucleoBenchDesigner,
    initialize_designer_state,
)
from tasks.nucleobench.core.evaluator import NucleoBenchEvaluator
from tasks.nucleobench.core.mock import (
    MOCK_CONTEXT,
    MOCK_START_SEQUENCE,
    build_mock_expander,
    build_mock_task_spec,
)
from tasks.nucleobench.core.oracles import official as official_module
from tasks.nucleobench.core.oracles.official import (
    PreparedCase,
    RecordingSequenceModel,
    load_official_case,
    load_prepared_case,
)
from tasks.nucleobench.core.source import require_clean_revision
from tasks.nucleobench.core.workflow import (
    _finish_completed_resume,
    build_official_runner_args,
    run_official_driver,
)
from tasks.nucleobench.scripts import prepare_official_data as prepare_module
from tasks.nucleobench.scripts.prepare_official_data import prepare_case_data


def _start_sequences() -> list[str]:
    alphabet = "ACGT"
    sequences = []
    for index in range(100):
        value = index
        suffix = []
        for _ in range(4):
            suffix.append(alphabet[value % 4])
            value //= 4
        sequences.append("A" * 196 + "".join(reversed(suffix)))
    return sequences


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    starts = tmp_path / "starts.json"
    starts.write_text(json.dumps(_start_sequences()), encoding="utf-8")
    model = tmp_path / "malinois.tar.gz"
    model.write_bytes(b"source-pinned-model")
    positions = tmp_path / "positions.json"
    positions.write_text(json.dumps(list(range(200))), encoding="utf-8")
    return starts, model, positions


def _start_set_sha256(sequences: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(sequences, separators=(",", ":")).encode()
    ).hexdigest()


def _set_test_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    model: Path,
    starts: list[str],
    source: Path | None = None,
) -> None:
    artifacts = {
        "model": {
            "bytes": model.stat().st_size,
            "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        }
    }
    start_source = None
    if source is not None:
        artifacts["starts"] = {
            "bytes": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
        start_source = {
            "kind": "csv_block",
            "artifact": "starts",
            "sequence_column": "0",
            "first_index": 1300,
            "last_index": 1399,
        }
    contract = tmp_path / "upstream_contract.json"
    contract.write_text(
        json.dumps(
            {
                "artifacts": artifacts,
                "case_preparation": {
                    "malinois_k562": {
                        "start_source": start_source,
                        "start_set_sha256": _start_set_sha256(starts),
                        "model_artifact": "model",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(prepare_module, "CONTRACT_PATH", contract)


def test_prepare_case_data_writes_a_digest_bound_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    starts, model, positions = _write_inputs(tmp_path)
    _set_test_contract(
        monkeypatch,
        tmp_path,
        model=model,
        starts=_start_sequences(),
    )
    output = tmp_path / "prepared"
    manifest = prepare_case_data(
        case_id="malinois_k562",
        starts_path=starts,
        model_artifact=model,
        output_dir=output,
        editable_positions_path=positions,
        bending_factor=1.0,
    )

    assert manifest["case_id"] == "malinois_k562"
    assert manifest["start_set"]["count"] == 100
    assert manifest["start_set"]["sha256"] == _start_set_sha256(_start_sequences())
    assert manifest["start_set"]["source_file"] == {
        "name": "starts.json",
        "bytes": starts.stat().st_size,
        "sha256": hashlib.sha256(starts.read_bytes()).hexdigest(),
    }
    assert manifest["editable_positions"]["count"] == 200
    assert (
        manifest["model_artifact"]["sha256"]
        == hashlib.sha256(model.read_bytes()).hexdigest()
    )
    assert manifest["model_init_args"] == {
        "a_max": 6.0,
        "a_min": -2.0,
        "bending_factor": 1.0,
        "flank_length": 200,
        "target_alpha": 1.0,
        "target_feature": 0,
    }
    assert json.loads((output / "prepared_manifest.json").read_text()) == manifest
    assert json.loads((output / "starts.json").read_text()) == _start_sequences()
    assert json.loads((output / "editable_positions.json").read_text()) == list(
        range(200)
    )
    prepared = load_prepared_case(
        output,
        case_id="malinois_k562",
        start_index=7,
    )
    assert prepared.context.start_sequence == _start_sequences()[7]
    assert prepared.context.editable_positions == tuple(range(200))
    assert prepared.model_artifact == model.resolve()


def test_prepare_case_data_extracts_the_official_malinois_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, model, positions = _write_inputs(tmp_path)
    starts = _start_sequences()
    source = tmp_path / "start_sequences_df.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "0"])
        writer.writerow([1299, "A" * 3_000])
        writer.writerows(
            (source_index, sequence)
            for source_index, sequence in enumerate(starts, start=1300)
        )
    _set_test_contract(
        monkeypatch,
        tmp_path,
        model=model,
        starts=starts,
        source=source,
    )

    output = tmp_path / "prepared-csv"
    manifest = prepare_case_data(
        case_id="malinois_k562",
        starts_path=source,
        model_artifact=model,
        output_dir=output,
        editable_positions_path=positions,
        bending_factor=1.0,
    )

    assert json.loads((output / "starts.json").read_text()) == starts
    assert manifest["start_set"]["source_file"]["name"] == source.name
    assert (
        manifest["start_set"]["source_file"]["sha256"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )


def test_prepare_case_data_rejects_unverified_or_invalid_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    starts, model, positions = _write_inputs(tmp_path)
    _set_test_contract(
        monkeypatch,
        tmp_path,
        model=model,
        starts=_start_sequences(),
    )

    model.write_bytes(b"modified-after-contract")
    with pytest.raises(ValueError, match="model artifact size"):
        prepare_case_data(
            case_id="malinois_k562",
            starts_path=starts,
            model_artifact=model,
            output_dir=tmp_path / "bad-model",
            editable_positions_path=positions,
            bending_factor=1.0,
        )
    model.write_bytes(b"source-pinned-model")

    duplicate_starts = tmp_path / "duplicate-starts.json"
    duplicate_starts.write_text(json.dumps(["A" * 200] * 100), encoding="utf-8")
    with pytest.raises(ValueError, match="100 unique start sequences"):
        prepare_case_data(
            case_id="malinois_k562",
            starts_path=duplicate_starts,
            model_artifact=model,
            output_dir=tmp_path / "bad-starts",
            editable_positions_path=positions,
            bending_factor=1.0,
        )

    invalid_positions = tmp_path / "invalid-positions.json"
    invalid_positions.write_text(json.dumps([*range(199), 200]), encoding="utf-8")
    with pytest.raises(ValueError, match="outside the sequence"):
        prepare_case_data(
            case_id="malinois_k562",
            starts_path=starts,
            model_artifact=model,
            output_dir=tmp_path / "bad-positions",
            editable_positions_path=invalid_positions,
            bending_factor=1.0,
        )


def test_prepared_case_selects_the_paired_editable_mask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alphabet = "ACGT"
    starts = []
    for index in range(100):
        value = index
        suffix = []
        for _ in range(4):
            suffix.append(alphabet[value % 4])
            value //= 4
        starts.append("AAAA" + "".join(reversed(suffix)))
    positions = [[index % 7, index % 7 + 1] for index in range(100)]
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    prepared_dir = tmp_path / "prepared-paired"
    prepared_dir.mkdir()
    starts_path = prepared_dir / "starts.json"
    positions_path = prepared_dir / "editable_positions.json"
    starts_path.write_text(json.dumps(starts), encoding="utf-8")
    positions_path.write_text(json.dumps(positions), encoding="utf-8")
    starts_digest = _start_set_sha256(starts)
    manifest = {
        "case_id": "fixture",
        "benchmark_source": {"revision": official_module.UPSTREAM_COMMIT},
        "start_set": {
            "path": starts_path.name,
            "sha256": starts_digest,
            "file_sha256": hashlib.sha256(starts_path.read_bytes()).hexdigest(),
        },
        "editable_positions": {
            "path": positions_path.name,
            "file_sha256": hashlib.sha256(positions_path.read_bytes()).hexdigest(),
        },
        "model_artifact": {
            "path": str(model),
            "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        },
        "model_init_args": {},
    }
    (prepared_dir / "prepared_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    monkeypatch.setattr(
        official_module,
        "get_case",
        lambda _case_id: SimpleNamespace(
            case_id="fixture",
            sequence_length=8,
            editable_position_count=2,
        ),
    )

    prepared = load_prepared_case(prepared_dir, case_id="fixture", start_index=11)

    assert prepared.context.start_sequence == starts[11]
    assert prepared.context.editable_positions == tuple(positions[11])


@pytest.mark.parametrize(
    "case_id",
    [
        "malinois_hepg2",
        "bpnet_ctcf",
        "rinalmo_mrl",
        "enformer_muscle_not_liver",
    ],
)
def test_official_loader_dispatches_every_model_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_id: str,
) -> None:
    case = get_case(case_id)
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"model")
    context = MutationContext(
        case=case,
        start_set_digest="0" * 64,
        start_index=0,
        start_sequence="A" * case.sequence_length,
        editable_positions=tuple(range(case.editable_position_count)),
    )
    init_args = dict(case.model_selector)
    if case.model_family == "malinois":
        init_args.update(
            bending_factor=1.0,
            a_min=-2.0,
            a_max=6.0,
            target_alpha=1.0,
            flank_length=200,
        )
    elif case.model_family == "enformer":
        init_args.update(spatial_bins_to_aggregate=None, run_sanity_checks=True)
    prepared = PreparedCase(context, artifact, init_args)
    constructed: list[dict[str, object]] = []

    class FakeModel:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

        def __call__(self, sequences):
            return [0.0] * len(sequences)

    def official_resolver(_source_dir: Path, name: str):
        if name == "docker_entrypoint":
            return SimpleNamespace(run_loop=Mock())
        if name == "nucleobench.common.argparse_lib":
            return SimpleNamespace(ParsedArgs=SimpleNamespace)
        if name.endswith("bpnet.load_model"):
            return SimpleNamespace(
                CountWrapper=lambda value: ("count", value),
                ControlWrapper=lambda value: ("control", value),
            )
        class_name = {
            "malinois": "Malinois",
            "bpnet": "BPNet",
            "rinalmo": "RinalmoMRL",
            "enformer": "Enformer",
        }[case.model_family]
        return SimpleNamespace(**{class_name: FakeModel})

    fake_torch = SimpleNamespace(
        load=lambda *_args, **_kwargs: "raw-model",
        device=lambda value: value,
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    fake_lightning = SimpleNamespace(
        LightningModel=SimpleNamespace(
            load_from_checkpoint=lambda *_args, **_kwargs: "checkpoint"
        )
    )
    real_import = official_module.importlib.import_module
    monkeypatch.setattr(official_module, "require_clean_revision", Mock())
    monkeypatch.setattr(official_module, "_official_module", official_resolver)
    monkeypatch.setattr(
        official_module.importlib,
        "import_module",
        lambda name: (
            fake_torch
            if name == "torch"
            else fake_lightning
            if name == "grelu.lightning"
            else real_import(name)
        ),
    )
    runtime = CampaignRuntime.open(tmp_path / "run", task="nucleobench")

    loaded = load_official_case(tmp_path / "source", prepared, runtime)

    assert isinstance(loaded.model, RecordingSequenceModel)
    assert len(constructed) == 1
    if case.model_family == "bpnet":
        assert constructed[0]["protein"] == case.target
        assert constructed[0]["override_model"][0] == "count"
    elif case.model_family == "rinalmo":
        assert constructed[0] == {"override_weights_local_path": str(artifact)}
    elif case.model_family == "enformer":
        assert constructed[0]["override_model"] == "checkpoint"
    else:
        assert constructed[0]["target_feature"] == 1
        fake_torch.cuda.is_available = lambda: True
        with pytest.raises(RuntimeError, match="requires CPU execution"):
            load_official_case(tmp_path / "source", prepared, runtime)


def test_source_revision_check_rejects_dirty_or_mismatched_checkouts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(
        ["git", "-C", str(source), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source), "config", "user.name", "Test"], check=True
    )
    (source / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-qm", "fixture"], check=True)
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    require_clean_revision(source, revision)
    with pytest.raises(ValueError, match="revision mismatch"):
        require_clean_revision(source, "0" * 40)

    (source / "README.md").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty"):
        require_clean_revision(source, revision)


def _build_designer(tmp_path: Path):
    task_spec = build_mock_task_spec()
    runtime = CampaignRuntime.open(
        tmp_path / "campaign",
        task="nucleobench",
        task_spec=task_spec,
    )

    def score(sequences):
        return [
            -float(sum(index + 1 for index, base in enumerate(sequence) if base != "A"))
            for sequence in sequences
        ]

    model = RecordingSequenceModel(score, runtime)
    evaluator = NucleoBenchEvaluator(MOCK_CONTEXT, model)
    state = initialize_designer_state(MOCK_CONTEXT, evaluator, runtime)
    engine = LDMEngine(
        task_spec=task_spec,
        expander=build_mock_expander(),
        candidate_domain=NucleoBenchCandidateDomain(
            MOCK_CONTEXT,
            sink=DataCollectionSink.disabled(),
        ),
        evaluator=evaluator,
        runtime=runtime,
    )
    designer = NucleoBenchDesigner(
        engine=engine,
        state=state,
        context=MOCK_CONTEXT,
        reservoir_size=4,
        evaluations_per_step=2,
    )
    return designer, model, runtime


def test_designer_advances_exact_steps_and_get_samples_is_read_only(
    tmp_path: Path,
) -> None:
    designer, model, runtime = _build_designer(tmp_path)

    assert designer.get_samples(1) == [MOCK_START_SEQUENCE]
    assert runtime.budget.counters["initialization_evaluations"] == 1
    assert runtime.budget.counters.get("external_evaluations", 0) == 0
    calls_before = model.call_count

    designer.run(2)

    assert designer.active_steps == 2
    assert designer.state.next_round == 3
    assert len(designer.state.observations) == 5
    assert model.call_count == calls_before + 2
    calls_before = model.call_count
    samples = designer.get_samples(3)
    assert len(samples) == 3
    assert len(set(samples)) == 3
    assert all(sequence[1::2] == "AAAA" for sequence in samples)
    assert samples[0] != MOCK_START_SEQUENCE
    assert model.call_count == calls_before
    assert json.loads(runtime.status.path.read_text())["phase"] == (
        "awaiting_external_driver"
    )
    assert [
        event["payload"]["batch_size"]
        for event in runtime.events()
        if event["event_type"] == "official_model_called"
    ] == [1, 2, 2]
    assert designer.is_finished() is False


def test_recording_model_preserves_output_and_records_only_sequence_digests(
    tmp_path: Path,
) -> None:
    runtime = CampaignRuntime.open(tmp_path / "campaign", task="nucleobench")
    raw = [-1.25, 2.5]
    model = RecordingSequenceModel(lambda sequences: raw, runtime)

    assert model(["A" * 8, "C" * 8]) is raw

    event = runtime.events()[-1]
    assert event["event_type"] == "official_model_called"
    assert event["payload"]["energies"] == raw
    assert event["payload"]["utilities"] == [1.25, -2.5]
    assert len(event["payload"]["sequence_sha256"]) == 2
    assert "AAAAAAAA" not in json.dumps(event)


def test_recording_model_continues_trace_counters_after_resume(tmp_path: Path) -> None:
    run_dir = tmp_path / "campaign"
    runtime = CampaignRuntime.open(run_dir, task="nucleobench")
    RecordingSequenceModel(lambda sequences: [1.0] * len(sequences), runtime)(["A" * 8])

    resumed = CampaignRuntime.open(run_dir, task="nucleobench", resume=True)
    RecordingSequenceModel(lambda sequences: [1.0] * len(sequences), resumed)(
        ["C" * 8, "G" * 8]
    )

    calls = [
        event["payload"]
        for event in resumed.events()
        if event["event_type"] == "official_model_called"
    ]
    assert [call["call_index"] for call in calls] == [1, 2]
    assert [call["cumulative_sequence_count"] for call in calls] == [1, 3]


def test_completed_resume_preserves_existing_result_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "campaign"
    runtime = CampaignRuntime.open(run_dir, task="nucleobench")
    summary = {"observation_count": 2, "official_output_count": 3}
    result = {
        "evaluation_count": 2,
        "official_outputs": [{"path": "official/results.parquet"}],
    }
    runtime.finish(summary)
    atomic_json_write(run_dir / "result.json", result)

    resumed = CampaignRuntime.open(run_dir, task="nucleobench", resume=True)
    assert _finish_completed_resume(resumed) == result
    assert json.loads((run_dir / "summary.json").read_text()) == summary
    assert json.loads((run_dir / "result.json").read_text()) == result


def test_official_driver_preserves_raw_outputs_and_finishes_once(
    tmp_path: Path,
) -> None:
    designer, model, runtime = _build_designer(tmp_path)
    output_dir = runtime.run_dir / "official"
    all_args = build_official_runner_args(
        SimpleNamespace,
        model_name="malinois",
        optimization_name="ldm_tts",
        start_sequence=MOCK_START_SEQUENCE,
        positions_to_mutate=list(MOCK_CONTEXT.editable_positions),
        output_path=output_dir,
        proposals_per_round=2,
        max_seconds=60,
        model_init_args={"target_feature": 0},
        optimizer_init_args={"reservoir_size": 4},
    )

    def fake_run_loop(*, model, opt, all_args, ignore_errors):
        assert ignore_errors is False
        output = Path(all_args.main_args.output_path)
        output.mkdir(parents=True)
        (output / "START.txt").write_text("START", encoding="utf-8")
        model(opt.get_samples(all_args.main_args.proposals_per_round))
        opt.run(1)
        model(opt.get_samples(all_args.main_args.proposals_per_round))
        opt.run(1)
        model(opt.get_samples(all_args.main_args.proposals_per_round))
        nested = output / "ldm_tts_malinois" / "fixture"
        nested.mkdir(parents=True)
        (nested / "results.parquet").write_bytes(b"official-raw-output")
        (output / "SUCCESS.txt").write_text("SUCCESS", encoding="utf-8")

    result = run_official_driver(
        run_loop=fake_run_loop,
        model=model,
        designer=designer,
        all_args=all_args,
        runtime=runtime,
        execution={
            "execution_profile": "qualification",
            "termination_kind": "rounds",
            "total_rounds": 3,
            "active_optimization_rounds": 2,
            "initialization_evaluations": 1,
            "benchmark_comparable": False,
            "case_id": "mock_dna",
            "start_index": 0,
            "start_set_digest": MOCK_CONTEXT.start_set_digest,
            "optimization_seed": 0,
            "hardware_profile": "test-cpu",
            "search_method": "bo",
            "method_preset_sha256": "1" * 64,
            "max_seconds": None,
        },
        oracle_manifest={"schema_version": 1, "oracle": "fixture"},
        expected_active_steps=2,
    )

    raw_output = next(
        item
        for item in result["official_outputs"]
        if item["path"].endswith("results.parquet")
    )
    assert raw_output["sha256"] == hashlib.sha256(b"official-raw-output").hexdigest()
    assert json.loads((runtime.run_dir / "result.json").read_text()) == result
    assert json.loads(runtime.status.path.read_text())["status"] == "completed"
    assert designer.active_steps == 2
    assert model.call_count == 6
    assert MOCK_START_SEQUENCE not in runtime.event_path.read_text()
    assert (
        sum(event["event_type"] == "campaign_finished" for event in runtime.events())
        == 1
    )


def test_official_runner_args_require_exactly_one_termination_mode(
    tmp_path: Path,
) -> None:
    common = {
        "parsed_args_type": SimpleNamespace,
        "model_name": "malinois",
        "optimization_name": "ldm_tts",
        "start_sequence": MOCK_START_SEQUENCE,
        "positions_to_mutate": list(MOCK_CONTEXT.editable_positions),
        "output_path": tmp_path,
        "proposals_per_round": 2,
    }
    with pytest.raises(ValueError, match="exactly one termination mode"):
        build_official_runner_args(**common)
    with pytest.raises(ValueError, match="exactly one termination mode"):
        build_official_runner_args(
            **common,
            max_seconds=60,
            max_number_of_rounds=2,
        )

    args = build_official_runner_args(**common, max_number_of_rounds=2)
    assert isinstance(args.main_args, Namespace)
    assert args.main_args.max_seconds is None
    assert args.main_args.max_number_of_rounds == 2
