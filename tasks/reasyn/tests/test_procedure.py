import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
from ldm_tts.contracts import Candidate, CandidateRejection, RawProposal
from ldm_tts.data import validate_ir_record
from ldm_tts.data.rendering import render_prose
from ldm_tts.optimization.records import BOObservation
from ldm_tts.transport import ProposalResponse
from tasks.reasyn.core import workflow
from tasks.reasyn.core.candidate import ReaSynDomain
from tasks.reasyn.core.evaluator import TDCOracleEvaluator
from tasks.reasyn.core.metrics import ORACLES, reconstruction_metrics, top_auc
from tasks.reasyn.core.proposals import parse_targets
from tasks.reasyn.core.selection import TanimotoGPSelector
from tasks.reasyn.core.surrogate import MoleculeEncoder


@pytest.mark.parametrize("benchmark", ["reconstruction", "tdc"])
def test_mock_shared_campaign_collection_and_resume(tmp_path, monkeypatch, benchmark):
    monkeypatch.setenv("LDM_DATA_COLLECTION_ENABLED", "1")
    out = tmp_path / benchmark
    argv = [
        "--mock",
        "--benchmark",
        benchmark,
        "--iterations",
        "2",
        "--reservoir-size",
        "4",
        "--evaluations-per-round",
        "2",
        "--out-dir",
        str(out),
    ]
    assert workflow.main(argv) == 0
    for name in (
        "campaign.json",
        "events.jsonl",
        "checkpoint.json",
        "summary.json",
        "status.json",
        "budget.json",
        "result.json",
        "trajectory.csv",
        "search_manifest.json",
        "selection_record.json",
        "experiment_contract.json",
    ):
        assert (out / name).is_file(), name
    counters = json.loads((out / "budget.json").read_text())["counters"]
    assert counters["outer_iterations"] == 2
    assert counters["valid_search_candidates"] == 8
    assert counters["expensive_evaluation_attempts"] == 4
    assert counters["successful_evaluations"] == 4
    assert counters["llm_requests"] == 0
    assert counters["oracle_calls"] == (4 if benchmark == "tdc" else 0)
    assert counters["projection_targets"] == (8 if benchmark == "tdc" else 4)
    ir_files = list((out / "ldm_data").rglob("*.jsonl"))
    assert ir_files
    collected = []
    for path in ir_files:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("schema_version") == "ldm-2.0":
                validate_ir_record(row)
                collected.append(row)
                rendered = render_prose(row)
                assert "synthetic_fixture" not in rendered
                assert "projection_artifact" not in rendered
    assert collected
    assert workflow.main(argv + ["--resume-from", str(out)]) == 0
    after = json.loads((out / "budget.json").read_text())["counters"]
    assert after == counters
    result = json.loads((out / "result.json").read_text())
    if benchmark == "tdc":
        assert result["metrics"]["auc_top10"] is None
        assert result["oracle_calls"] == 4


def test_requested_target_count_rejects_short_dataset(tmp_path):
    targets = tmp_path / "targets.txt"
    targets.write_text("SMILES\nCCO\n")
    args = workflow.parse_args(
        ["--dataset", "custom", "--targets-file", str(targets), "--target-limit", "2"]
    )
    with pytest.raises(ValueError, match="target-limit requests 2 targets"):
        workflow._targets(args)


def test_relative_targets_file_is_resolved_from_upstream_root(tmp_path):
    args = workflow.parse_args(
        [
            "--upstream-root",
            str(tmp_path),
            "--dataset",
            "custom",
            "--targets-file",
            "data/targets.txt",
        ]
    )
    assert args.targets_file == (tmp_path / "data/targets.txt").resolve()


def test_parser_and_trusted_projector_boundary():
    with pytest.raises(ValueError):
        parse_targets(
            '{"candidates":[{"target_smiles":"CCO","score":1}]}', count=1, mock=True
        )
    with pytest.raises(ValueError):
        parse_targets('{"candidates":[{"target_smiles":"bad"}]}', count=1, mock=True)
    domain = ReaSynDomain("tdc", mock=True)
    payload = {
        "smiles": "CCO",
        "synthesis": "CCO",
        "projection_artifact": "result.json",
    }
    assert isinstance(
        domain.admit(RawProposal(payload, "llm", {"pathway_verified": True})),
        CandidateRejection,
    )
    assert isinstance(
        domain.admit(RawProposal(payload, "reasyn_projector")), CandidateRejection
    )
    assert isinstance(
        domain.admit(
            RawProposal(payload, "reasyn_projector", {"pathway_verified": True})
        ),
        Candidate,
    )


def test_reconstruction_full_denominator_and_stereo():
    pytest.importorskip("rdkit")
    rows = [
        {"target": "CCO", "smiles": "OCC", "synthesis": "CCO"},
        {
            "target": "F[C@H](Cl)Br",
            "smiles": "F[C@@H](Cl)Br",
            "synthesis": "F[C@@H](Cl)Br",
        },
    ]
    result = reconstruction_metrics(
        ["CCO", "CCN", "F[C@H](Cl)Br"], rows, diversity=lambda s: 0.2
    )
    assert result["target_count"] == 3
    assert result["reconstruction_rate"] == pytest.approx(2 / 3)
    assert result["mean_similarity"] == pytest.approx(2 / 3)
    assert result["success_rate"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        reconstruction_metrics(
            ["CCO"], [{"target": "CCN", "smiles": "CCN"}], diversity=lambda s: 0
        )


def test_auc_matches_released_source_formula():
    np = pytest.importorskip("numpy")
    upstream = Path(
        os.environ.get("REASYN_ROOT", workflow.REPO_ROOT.parent / "ReaSyn-reasyn_v2")
    )
    source = upstream / "scripts/optimize_tdc.py"
    if not source.exists():
        pytest.skip("supplied upstream archive unavailable")
    tree = ast.parse(source.read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "top_auc"
    )
    namespace = {"np": np}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    for n in (1, 99, 100, 101, 255, 10000):
        values = [((i * 73) % 997) / 997 for i in range(n)]
        buffer = {str(i): [score, i + 1] for i, score in enumerate(values)}
        for finish in (False, True):
            assert top_auc(values, finish=finish) == pytest.approx(
                namespace["top_auc"](buffer, 10, finish, 100, 10000)
            )
    assert top_auc([], finish=True) == 0
    with pytest.raises(ValueError):
        top_auc([1, 1], max_calls=1)
    assert len(ORACLES) == 13


def test_oracle_dedup_and_hard_cap(tmp_path):
    pytest.importorskip("rdkit")
    args = SimpleNamespace(mock=False, oracle="jnk3", max_oracle_calls=1)
    calls = []
    evaluator = TDCOracleEvaluator(
        args, tmp_path, scorer=lambda s: calls.append(s) or 0.7
    )

    def candidate(smiles):
        return Candidate(
            smiles, {"smiles": smiles, "projection_artifact": "projection.json"}, smiles
        )

    assert evaluator.evaluate(candidate("OCC")).metrics["oracle_score"] == 0.7
    assert evaluator.evaluate(candidate("CCO")).metrics["oracle_score"] == 0.7
    assert calls == ["CCO"]
    assert evaluator.evaluate(candidate("CCN")).status == "invalid"
    resumed = TDCOracleEvaluator(
        args, tmp_path, scorer=lambda s: pytest.fail("completed score repeated")
    )
    assert resumed.evaluate(candidate("CCO")).metrics["oracle_score"] == 0.7
    assert len(resumed.entries) == 1


def test_tanimoto_gp_transfers_to_related_heldout_molecules():
    pytest.importorskip("rdkit")
    encoder = MoleculeEncoder()

    def candidate(s):
        return Candidate(s, {"smiles": s}, s)

    positive = candidate("CCO")
    negative = candidate("c1ccccc1")
    history = [
        BOObservation(positive.candidate_id, (0.9,), encoder.encode(positive)),
        BOObservation(negative.candidate_id, (0.1,), encoder.encode(negative)),
    ]
    related = candidate("CCCO")
    aromatic = candidate("Cc1ccccc1")
    selector = TanimotoGPSelector(
        objective_name="oracle_score", beta=0, feature_version=encoder.version
    )
    selector.fit(history)
    result = selector.select(
        [aromatic, related],
        {c.candidate_id: encoder.encode(c) for c in (related, aromatic)},
    )
    assert result.selected_candidate_ids == ("CCCO",)
    predictions = {p.candidate_id: p.scalar_mean for p in result.predictions}
    assert predictions["CCCO"] > predictions["Cc1ccccc1"] + 0.1


def test_completed_resume_preserves_preflight_count_and_keeps_credentials_private(tmp_path, monkeypatch):
    calls = []

    class Client:
        def __init__(self, **kwargs):
            pass

        def preflight(self):
            calls.append("preflight")
            return {"ok": True}

        def propose(self, request):
            calls.append("proposal")
            return ProposalResponse(
                text='{"candidates":[{"target_smiles":"CCO"},{"target_smiles":"CCN"}]}'
            )

    monkeypatch.setattr(workflow, "OpenAICompatibleProposalClient", Client)
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("LLM_MODEL_NAME", "fixture")
    monkeypatch.setenv("LLM_API_KEY", "must-not-appear-in-artifacts")
    argv = [
        "--mock",
        "--proposal-mode",
        "openai",
        "--iterations",
        "1",
        "--reservoir-size",
        "2",
        "--out-dir",
        str(tmp_path / "run"),
    ]
    assert workflow.main(argv) == 0
    assert workflow.main(argv + ["--resume-from", str(tmp_path / "run")]) == 0
    counters = json.loads((tmp_path / "run/budget.json").read_text())["counters"]
    assert counters["endpoint_preflight_requests"] == 1
    assert counters["llm_requests"] == 1
    assert calls == ["preflight", "proposal"]
    for file in (tmp_path / "run").rglob("*"):
        if file.is_file():
            assert "must-not-appear-in-artifacts" not in file.read_text()


def test_malformed_model_action_exhausts_bounded_repair_without_measurement(tmp_path, monkeypatch):
    class Client:
        def __init__(self, **kwargs):
            pass

        def preflight(self):
            return {"ok": True}

        def propose(self, request):
            return ProposalResponse(
                text='{"candidates":[{"target_smiles":"not-valid"}]}'
            )

    monkeypatch.setattr(workflow, "OpenAICompatibleProposalClient", Client)
    argv = [
        "--mock",
        "--proposal-mode",
        "openai",
        "--llm-url",
        "http://localhost:1234/v1",
        "--llm-model",
        "fixture",
        "--iterations",
        "1",
        "--reservoir-size",
        "1",
        "--out-dir",
        str(tmp_path / "run"),
    ]
    assert workflow.main(argv) == 1
    counters = json.loads((tmp_path / "run/budget.json").read_text())["counters"]
    assert counters["llm_requests"] == 5 and counters["proposal_attempts"] == 5
    assert (
        counters["expensive_evaluation_attempts"] == 0
        and counters["projection_targets"] == 0
    )
    assert "invalid_projection_targets" in (tmp_path / "run/events.jsonl").read_text()
    result = json.loads((tmp_path / "run/benchmark_result.json").read_text())
    assert result["target_count"] == 1 and result["reconstruction_rate"] == 0


def test_cached_projection_rechecks_boundary(tmp_path):
    from tasks.reasyn.core.projector import Projector

    args = workflow.parse_args(["--mock"])
    projector = Projector(args, tmp_path)
    rows, artifact = projector.project(["CCO"], sampling_seed=0, identity="query")
    output = tmp_path / artifact
    bad = json.loads(output.read_text())
    bad["rows"][0]["pathway_verified"] = False
    output.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="trusted-pathway"):
        projector.project(["CCO"], sampling_seed=0, identity="query")
