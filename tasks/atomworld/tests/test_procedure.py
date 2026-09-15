from __future__ import annotations
import json
import pytest

from ldm_tts.contracts import Candidate, CandidateRejection, RawProposal
from ldm_tts.data import validate_ir_record
from ldm_tts.data.rendering import render_prose
from ldm_tts.engine.expansion import ExpansionRequest
from ldm_tts.transport import CallableProposalClient
from tasks.atomworld.core.data import (
    TASK_ROOT,
    load_prepared,
    prepare_dataset,
    write_json,
)
from tasks.atomworld.core.proposals import (
    AtomWorldDomain,
    BlindRefinementExpander,
    official_prompt,
)
from tasks.atomworld.core.workflow import parse_args, run


def fixture():
    return json.loads((TASK_ROOT / "resources/mock_fixture.json").read_text())


def test_shared_campaign_collection_and_last_answer(tmp_path, monkeypatch):
    monkeypatch.setenv("LDM_DATA_COLLECTION_ENABLED", "1")
    result = run(
        parse_args(
            ["--mock", "--attempts-per-sample", "4", "--out-dir", str(tmp_path / "run")]
        )
    )
    folder = result.runtime.run_dir
    for name in (
        "events.jsonl",
        "checkpoint.json",
        "summary.json",
        "status.json",
        "budget.json",
        "result.json",
        "trajectory.csv",
        "experiment_contract.json",
    ):
        assert (folder / name).is_file(), name
    assert result.projected["one_shot_accuracy"] == 0
    assert result.projected["extended_final_accuracy"] == 1
    assert len(list((folder / "attempts").glob("*.json"))) == 4
    assert (
        len(result.engine.state.observations) == 2
    )  # Repeated revisions are deduplicated.
    budget = json.loads((folder / "budget.json").read_text())
    assert budget["counters"]["mock_model_requests"] == 4
    assert budget["counters"]["llm_requests"] == 0
    assert budget["counters"]["expensive_evaluation_attempts"] == 2
    assert budget["counters"]["benchmark_jobs"] == 2
    files = list((folder / "ldm_data").rglob("*.jsonl"))
    assert files
    records = [
        json.loads(line) for path in files for line in path.read_text().splitlines()
    ]
    records = [
        record.get("ir", record)
        for record in records
        if record.get("ir", record).get("schema_version") == "ldm-2.0"
    ]
    assert records
    for record in records:
        validate_ir_record(record)
        rendered = render_prose(record)
        assert "candidate_id" not in rendered
        assert "target_cif" not in rendered
        assert "oracle" not in rendered


def test_refinement_never_reads_oracle_state(tmp_path):
    data = fixture()
    public = data["public"]
    output = data["mock_outputs"][public[0]["sample_id"]][0]
    prompts = []
    client = CallableProposalClient(
        lambda request: prompts.append(request.messages) or output
    )
    expander = BlindRefinementExpander(
        public, client, attempts_per_sample=2, run_dir=tmp_path, mock=True
    )
    expander.expand(ExpansionRequest(0, 1))

    class Poison:
        def __getattribute__(self, name):
            raise AssertionError("Read private judge state")

    expander.expand(
        ExpansionRequest(
            1,
            1,
            observations=(Poison(),),
            parent=Poison(),
            acquisition_feedback={"secret_target": "LEAK"},
        )
    )
    assert len(prompts[0]) == 1
    assert len(prompts[1]) == 3
    assert "LEAK" not in json.dumps(prompts)
    assert "secret_target" not in json.dumps(prompts)
    assert "public" in prompts[1][-1]["content"].lower()


def test_invalid_cif_is_scored_but_invalid_record_rejected():
    sample = fixture()["public"][0]
    domain = AtomWorldDomain([sample], mock=True)
    payload = {
        "sample_id": sample["sample_id"],
        "action_name": sample["action_name"],
        "generated_output": "no CIF here",
    }
    assert isinstance(domain.admit(RawProposal(payload, "test")), Candidate)
    assert isinstance(
        domain.admit(RawProposal({**payload, "target_cif": "secret"}, "test")),
        CandidateRejection,
    )
    assert isinstance(
        domain.admit(
            RawProposal({**payload, "generated_output": "x" * 1000001}, "test")
        ),
        CandidateRejection,
    )


def test_parser_rejects_invalid_budgets_and_extra_body():
    with pytest.raises(SystemExit):
        parse_args(["--mock", "--attempts-per-sample", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--mock", "--llm-extra-body-json", "[]"])


def test_resume_never_repeats_calls(tmp_path):
    args = parse_args(
        ["--mock", "--attempts-per-sample", "2", "--out-dir", str(tmp_path)]
    )
    first = run(args)
    args.out_dir = first.runtime.run_dir
    args.resume = True
    second = run(
        args,
        client=CallableProposalClient(
            lambda request: pytest.fail("repeated completed request")
        ),
    )
    assert first.projected["one_shot_accuracy"] == second.projected["one_shot_accuracy"]
    args.attempts_per_sample = 3
    with pytest.raises(ValueError, match="Resume requires"):
        run(args)


def test_prepare_hashes_and_public_boundary(tmp_path):
    sample = fixture()["public"][0]
    source = tmp_path / "source"
    source.mkdir()
    (source / "move_atom_action.json").write_text(
        json.dumps(
            [
                {
                    "input": sample["input_cif"],
                    "action_prompt": sample["action_prompt"],
                    "output": "SECRET_TARGET",
                }
            ]
        )
    )
    prepared = tmp_path / "prepared"
    manifest = prepare_dataset(
        source, prepared, actions=("move_atom_action",), per_action=1
    )
    public, private, loaded = load_prepared(prepared)
    assert "SECRET_TARGET" not in (prepared / "public.jsonl").read_text()
    assert list(private.values()) == ["SECRET_TARGET"]
    assert manifest == loaded
    assert manifest["dataset_kind"] == "local_released_subset"
    assert not manifest["paper_split_verified"]
    with pytest.raises(FileExistsError):
        prepare_dataset(source, prepared, actions=("move_atom_action",))
    (prepared / "public.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="hash"):
        load_prepared(prepared)


def test_official_evaluator_matches_identity_and_format_errors():
    pytest.importorskip("pymatgen")
    from tasks.atomworld.core.data import DEFAULT_UPSTREAM
    from tasks.atomworld.core.evaluator import load_official_evaluator

    if not DEFAULT_UPSTREAM.exists():
        pytest.skip("Optional local upstream source archive absent")
    evaluate = load_official_evaluator(DEFAULT_UPSTREAM)
    cif = fixture()["private"][0]["target_cif"]
    assert evaluate(cif, f"<cif>{cif}</cif>").correct
    assert evaluate(cif, "invalid").wrong_type == "OutputFormatError"
    assert evaluate(cif, "<cif>broken</cif>").wrong_type == "CIFParsingError"
    wrong_species = cif.replace("Cl1 Cl", "F1 F")
    assert (
        evaluate(cif, f"<cif>{wrong_species}</cif>").wrong_type == "AtomCountMismatch"
    )


def test_prompt_is_verbatim_upstream():
    from tasks.atomworld.core.data import DEFAULT_UPSTREAM
    import importlib.util

    path = DEFAULT_UPSTREAM / "src/prompts/cif_action_prompt.py"
    if not path.exists():
        pytest.skip("Optional upstream source archive absent")
    spec = importlib.util.spec_from_file_location("upstream_prompt_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sample = fixture()["public"][0]
    assert official_prompt(sample) == module.cif_action_prompt(
        sample["input_cif"], sample["action_prompt"]
    )


def test_final_answer_is_not_best_or_last_valid(tmp_path):
    data = fixture()
    sid = data["public"][0]["sample_id"]
    responses = iter([data["mock_outputs"][sid][1], "I cannot produce a CIF"])
    args = parse_args(
        [
            "--mock",
            "--attempts-per-sample",
            "2",
            "--out-dir",
            str(tmp_path / "campaign"),
        ]
    )
    result = run(args, client=CallableProposalClient(lambda request: next(responses)))
    assert result.projected["one_shot_accuracy"] == 1
    assert result.projected["extended_final_accuracy"] == 0
    assert result.projected["oracle_any_attempt_accuracy_diagnostic"] == 1


def test_official_judge_failure_is_not_reported_as_wrong_answer(tmp_path, monkeypatch):
    data = fixture()
    sample = data["public"][0]
    sample_id = sample["sample_id"]
    targets = {sample_id: data["private"][0]["target_cif"]}
    manifest = {
        "dataset_kind": "local_released_subset",
        "paper_split_verified": False,
    }
    monkeypatch.setattr(
        "tasks.atomworld.core.workflow.load_prepared",
        lambda _data_dir: ([sample], targets, manifest),
    )

    def failing_official(*_args, **_kwargs):
        raise RuntimeError("judge runtime unavailable")

    monkeypatch.setattr(
        "tasks.atomworld.core.workflow.load_official_evaluator",
        lambda _upstream: failing_official,
    )
    args = parse_args(
        [
            "--data-dir",
            str(tmp_path / "data"),
            "--attempts-per-sample",
            "1",
            "--out-dir",
            str(tmp_path / "campaign"),
        ]
    )

    with pytest.raises(RuntimeError, match="judge runtime unavailable"):
        run(
            args,
            client=CallableProposalClient(
                lambda _request: data["mock_outputs"][sample_id][0]
            ),
        )

    status = json.loads((args.out_dir / "status.json").read_text())
    assert status["status"] == "failed"
    assert "judge runtime unavailable" in status["message"]
    assert not (args.out_dir / "result.json").exists()
    evaluation = json.loads((args.out_dir / "checkpoint.json").read_text())["state"][
        "observations"
    ][0]["evaluation"]
    assert evaluation["status"] == "failed"
    assert evaluation["error"] == "judge runtime unavailable"


def test_operations_real_tool_and_official_judge(tmp_path):
    pytest.importorskip("ase")
    pytest.importorskip("pymatgen")
    from tasks.atomworld.core.data import DEFAULT_UPSTREAM

    if not DEFAULT_UPSTREAM.exists():
        pytest.skip("Optional official upstream archive absent")
    fixture_data = fixture()
    public = fixture_data["public"][0]
    source = tmp_path / "source"
    source.mkdir()
    write_json(
        source / "move_atom_action.json",
        [
            {
                "input": public["input_cif"],
                "action_prompt": public["action_prompt"],
                "output": fixture_data["private"][0]["target_cif"],
            }
        ],
    )
    data_dir = tmp_path / "data"
    prepare_dataset(source, data_dir, actions=("move_atom_action",))
    args = parse_args(
        [
            "--data-dir",
            str(data_dir),
            "--proposal-format",
            "operations",
            "--attempts-per-sample",
            "2",
            "--out-dir",
            str(tmp_path / "campaign"),
        ]
    )
    seen = []

    def respond(request):
        seen.append(request.messages)
        return (
            '[{"op":"move","index":999,"d_pos":[1,0,0]}]'
            if len(seen) == 1
            else '[{"op":"move","index":0,"d_pos":[1,0,0]}]'
        )

    result = run(args, client=CallableProposalClient(respond))
    assert result.projected["proposal_format"] == "operations"
    assert result.projected["one_shot_accuracy"] == 0
    assert result.projected["extended_final_accuracy"] == 1
    assert "Tool execution error" in seen[1][-1]["content"]
    assert "target_cif" not in json.dumps(seen)
    assert result.runtime.budget.counters["geometry_tool_calls"] == 2
    assert result.runtime.budget.counters["benchmark_jobs"] == 2


def test_endpoint_failure_pauses_and_resumes_with_fixed_answer_budget(tmp_path):
    pytest.importorskip("pymatgen")
    from tasks.atomworld.core.data import DEFAULT_UPSTREAM
    from ldm_tts.transport.openai import EndpointRequestError

    if not DEFAULT_UPSTREAM.exists():
        pytest.skip("Optional upstream source archive absent")
    data = fixture()
    sample = data["public"][0]
    source = tmp_path / "source"
    source.mkdir()
    write_json(
        source / "move_atom_action.json",
        [
            {
                "input": sample["input_cif"],
                "action_prompt": sample["action_prompt"],
                "output": data["private"][0]["target_cif"],
            }
        ],
    )
    data_dir = tmp_path / "data"
    prepare_dataset(source, data_dir, actions=("move_atom_action",))
    args = parse_args(
        [
            "--data-dir",
            str(data_dir),
            "--attempts-per-sample",
            "2",
            "--out-dir",
            str(tmp_path / "campaign"),
        ]
    )

    def fail_request(request):
        raise EndpointRequestError("synthetic endpoint outage")

    with pytest.raises(EndpointRequestError):
        run(args, client=CallableProposalClient(fail_request))
    assert (
        json.loads((args.out_dir / "status.json").read_text())["status"]
        == "paused_endpoint"
    )
    args.resume = True
    result = run(
        args,
        client=CallableProposalClient(
            lambda request: "<cif>" + data["private"][0]["target_cif"] + "</cif>"
        ),
    )
    assert result.projected["extended_final_accuracy"] == 1
    assert (
        result.runtime.budget.counters["llm_requests"] == 3
    )  # One failed service call + two submitted answers.
    assert result.runtime.budget.counters["proposal_attempts"] == 2
    assert len(list((args.out_dir / "attempts").glob("*.json"))) == 2
