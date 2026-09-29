from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ldm_tts.data import (
    DataCollectionSink,
    LDMDataCollectionError,
    dataset_info_payload,
    make_complete_design_ir,
    make_parameter_edit_ir,
    render_record,
    validate_ir_record,
)
from tasks.small_molecule.core.data_ir import smallmol_ir_from_prompt_response


def _example_ir():
    return make_complete_design_ir(
        task_id="small_molecule",
        domain="molecule",
        task_description="Propose molecules for a two-objective search.",
        objectives=[
            {
                "name": "vina",
                "direction": "minimize",
                "description": "Docking score; lower is better.",
            },
            {
                "name": "activity",
                "direction": "maximize",
                "description": "Predicted activity; higher is better.",
            },
        ],
        design_space_description="Single-component organic SMILES.",
        observations=[
            {
                "design": "CCO",
                "results": {"vina": -2.4, "activity": 5.1},
                "roles": ["recent"],
            }
        ],
        candidates=[
            {"design": "CCN", "rationale": "amine neighbor"},
            {"design": "CCC", "rationale": "alkyl neighbor"},
        ],
        request_description="Generate valid, novel SMILES without scores.",
        num_candidates=2,
        num_evaluated=1,
        do_not_repeat=["CCO"],
    )


def test_make_complete_design_ir_normalizes_task_and_renders_action_json():
    ir = _example_ir()

    validate_ir_record(ir)
    assert ir["task"]["id"] == "smallmol"

    row = render_record(ir)
    assert row["source"] == "smallmol"
    assert row["input"] == ""
    assert "Return a single JSON object" in row["instruction"]
    assert "Do not repeat" in row["instruction"]
    action = json.loads(row["output"])
    assert action["type"] == "propose"
    assert action["payload"]["candidates"][0]["design"] == "CCN"


def test_sink_writes_ir_sft_and_keeps_provenance_out_of_prompt(tmp_path):
    sink = DataCollectionSink(tmp_path)
    ir = _example_ir()

    sink.append(
        ir,
        provenance={"run_id": "private-run-marker"},
        outcome={"selected": ["CCN"]},
    )

    ir_rows = [
        json.loads(line)
        for line in (tmp_path / "ldm_ir.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sft_rows = [
        json.loads(line)
        for line in (tmp_path / "ldm_sft.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dataset_info = json.loads((tmp_path / "dataset_info.json").read_text(encoding="utf-8"))

    assert ir_rows[0]["collection"]["provenance"]["run_id"] == "private-run-marker"
    assert ir_rows[0]["collection"]["outcome"]["selected"] == ["CCN"]
    assert len(sft_rows) == 1
    assert "private-run-marker" not in sft_rows[0]["instruction"]
    assert "collection" not in sft_rows[0]["instruction"]
    assert dataset_info == dataset_info_payload("ldm_sft.jsonl")


def test_json_renderer_keeps_collection_metadata_out_of_prompt():
    ir = _example_ir()
    ir["collection"] = {
        "provenance": {"run_id": "private-run-marker"},
        "augmentation": {"expert": "private-model-marker"},
    }

    row = render_record(ir, mode="json")

    assert "collection" not in row["instruction"]
    assert "private-run-marker" not in row["instruction"]
    assert "private-model-marker" not in row["instruction"]


def test_prose_renderer_includes_model_visible_candidate_pool():
    ir = _example_ir()
    ir["raw_context"] = {"candidate_pool": [{"id": 3, "sequence": "ADGHTKQNPRA"}]}

    row = render_record(ir)

    assert "## Candidate reservoir" in row["instruction"]
    assert "ADGHTKQNPRA" in row["instruction"]


def test_validate_rejects_action_not_allowed():
    ir = _example_ir()
    ir["action"]["type"] = "expand_design_space"

    with pytest.raises(LDMDataCollectionError, match="not in request.allowed_actions"):
        validate_ir_record(ir)


def test_smallmol_prompt_response_adapter_builds_ldm2_ir():
    history = {
        "pareto_front": [{"smiles": "CCO", "scores": [-2.4, 5.1]}],
        "top_low_vina": [{"smiles": "CCC", "scores": [-2.8, 5.0]}],
        "top_high_activity": [],
        "balanced_elites": [],
        "recent_selected": [{"smiles": "CCO", "scores": [-2.4, 5.1]}],
        "avoid_exact_smiles": ["CCO", "CCC"],
        "n_evaluated": 2,
        "recent_diversity_alert": {"instruction": "Use a different parent."},
    }
    context = [
        {
            "smiles": "CCO",
            "history_role": "pareto_front",
            "proposal_lesson": "vary heteroatoms",
        }
    ]
    prompt = "\n".join(
        [
            "Task:",
            "Generate up to 2 valid, unique candidate SMILES. Use compact minified JSON.",
            "Target context:",
            "KRAS target context.",
            "Background:",
            "External scorers decide which candidates are evaluated.",
            "Molecule context table:",
            json.dumps(context),
            "How to use the molecule context:",
            "Use descriptors as evidence.",
            "Generation principles:",
            "- Avoid exact repeats.",
            "SMILES hygiene:",
            "Single-component organic SMILES only.",
            "Generation focus:",
            "history-guided local mutation.",
            "History summary:",
            json.dumps(history),
            "JSON output format:",
            '{"direct_smiles":[{"smiles":"CCN","rationale":"amine"}]}',
        ]
    )
    output = json.dumps(
        {
            "direct_smiles": [
                {"smiles": "CCN", "rationale": "amine neighbor"},
                {"smiles": "CCCl", "rationale": "halogen scan"},
            ]
        }
    )

    ir = smallmol_ir_from_prompt_response(prompt, output, round_idx=7, source_id="m1_strategy_0")

    assert ir is not None
    validate_ir_record(ir)
    assert ir["search_state"]["round"] == 7
    assert ir["search_state"]["num_evaluated"] == 2
    assert ir["search_state"]["do_not_repeat"] == ["CCO", "CCC"]
    assert ir["search_state"]["progress"]["stalled"] is True
    assert ir["action"]["payload"]["candidates"][0]["design"] == "CCN"
    assert ir["raw_context"]["source_id"] == "m1_strategy_0"


def test_parameter_edit_builder_allows_design_space_expansion():
    ir = make_parameter_edit_ir(
        task_id="nanogpt",
        domain="training_program",
        task_description="Search over train.py scalar assignments.",
        objectives=[
            {
                "name": "val_bpb",
                "direction": "minimize",
                "description": "Validation bits per byte.",
            }
        ],
        active_parameters=[
            {
                "name": "HEAD_DIM",
                "type": "choice",
                "domain": [64, 96, 128],
                "edit_op": "set_choice",
                "current_value": 128,
            }
        ],
        inactive_parameters=[
            {
                "name": "WARMDOWN_RATIO",
                "type": "float",
                "domain": [0.0, 0.95],
                "current_value": 0.5,
            }
        ],
        action={
            "type": "expand_design_space",
            "reasoning": "Active edits have plateaued.",
            "payload": {"activate": "WARMDOWN_RATIO", "initial_value": 0.5},
            "summary": "Activate warmdown schedule control.",
        },
        request_description="Choose one valid action.",
        max_edits_per_candidate=2,
    )

    validate_ir_record(ir)
    assert "expand_design_space" in ir["request"]["allowed_actions"]
    rendered = render_record(ir)
    assert "Inactive expansion parameters" in rendered["instruction"]
    assert "WARMDOWN_RATIO" in rendered["output"]


def test_nanogpt_operation_engine_collects_validated_action(tmp_path, monkeypatch):
    from tasks.nanogpt.core.expansion_schema import load_operation_schema
    from tasks.nanogpt.core.search_core import SearchConfig
    from tasks.nanogpt.core.workflow import OperationSearchEngine, parse_args

    project_root = Path(__file__).resolve().parents[1] / "tasks" / "nanogpt"
    args = parse_args(
        [
            "--generator",
            "operation_mock",
            "--operation-schema",
            "resources/schemas/mock_operations.json",
            "--train-file",
            "resources/train/mock_train.py",
            "--mock-expand-every",
            "1",
        ]
    )
    schema = load_operation_schema(
        Path("resources/schemas/mock_operations.json"),
        project_root,
    )
    monkeypatch.setenv("LDM_DATA_COLLECTION_ENABLED", "1")
    monkeypatch.delenv("LDM_DATA_COLLECTION_DIR", raising=False)
    engine = OperationSearchEngine(
        SearchConfig(
            project_root=project_root,
            seed_train_path=project_root / "resources/train/mock_train.py",
            out_dir=tmp_path / "nanogpt_run",
            generator="operation_mock",
            show_progress=False,
            task_context="Optimize mock train.py under a fixed runtime budget.",
        ),
        schema,
        args,
    )
    engine.current_iteration = 3
    root = engine.create_seed_state()

    child = asyncio.run(engine.expand_state(root, 1))[0]
    expansion_child = asyncio.run(engine.expand_state(root, 1))[0]

    rows = [
        json.loads(line)
        for line in (tmp_path / "nanogpt_run/ldm_data/ldm_ir.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert child.status == "generated"
    assert len(rows) == 2
    assert rows[0]["task"]["id"] == "nanogpt"
    assert rows[0]["action"]["type"] == "propose"
    assert rows[0]["action"]["payload"]["candidates"][0]["parent"] == root.state_id
    assert rows[0]["search_state"]["round"] == 3
    assert rows[0]["collection"]["provenance"]["state_id"] == child.state_id
    assert rows[0]["raw_context"]["parent_train_py"]
    assert expansion_child.status == "generated"
    assert rows[1]["action"]["type"] == "expand_design_space"
    activated = rows[1]["action"]["payload"]["activate"]
    assert activated not in {
        parameter["name"]
        for parameter in rows[1]["search_state"]["design_space"]["active_parameters"]
    }
    assert activated in {
        parameter["name"]
        for parameter in rows[1]["search_state"]["design_space"]["inactive_parameters"]
    }


def test_antibody_run_collects_direct_action_and_skips_evaluator_context(
    tmp_path,
    monkeypatch,
):
    import numpy as np

    from tasks.antibody.core.ldm_light import ldm_acq

    sequence = "ADGHTKQNPRA"

    class DirectLLM:
        def call(self, _prompt, **_kwargs):
            return json.dumps([sequence])

    class Evaluator:
        def energy(self, _encoded):
            return np.array([-7.5]), [sequence]

    args = SimpleNamespace(
        out_root=str(tmp_path / "antibody_runs"),
        seed=17,
        method="llm_gen",
        parallel_budget=8,
        n_evals=1,
        batch_size=1,
        gen_m=1,
        n_strategies=1,
        planner_mode="choices",
        softmax_eta=1.0,
        per_strategy_budget=0,
        pool_score="acq",
        selection_score="acq",
        bias_weight=0.05,
        acq="ei",
        acq_beta=1.0,
        acq_xi=0.001,
        n_init=1,
        include_antigen_context=False,
        max_retries=1,
        history_top_k=10,
        fallback_random=False,
        temperature=0.0,
        timeout_s=10,
        device="cpu",
    )
    monkeypatch.setenv("LDM_DATA_COLLECTION_ENABLED", "1")
    monkeypatch.delenv("LDM_DATA_COLLECTION_DIR", raising=False)
    monkeypatch.setattr(ldm_acq, "make_llm_client", lambda: DirectLLM())
    monkeypatch.setattr(
        ldm_acq,
        "make_evaluator",
        lambda _config, _antigen, _run_id: (
            Evaluator(),
            {"tool": "random"},
        ),
    )

    run_dir = ldm_acq.run_one(
        {"seq_len": len(sequence), "bbox": {"tool": "random"}},
        "TEST_ANTIGEN",
        17,
        args,
    )

    ir_rows = [
        json.loads(line)
        for line in (run_dir / "ldm_data/ldm_ir.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    sft_row = json.loads(
        (run_dir / "ldm_data/ldm_sft.jsonl").read_text(encoding="utf-8")
    )
    assert len(ir_rows) == 1
    assert ir_rows[0]["task"]["id"] == "protein"
    assert ir_rows[0]["task"]["reasoning_available"] is False
    assert ir_rows[0]["action"]["payload"]["candidates"] == [
        {"design": sequence, "rationale": None}
    ]
    assert ir_rows[0]["collection"]["provenance"]["antigen"] == "TEST_ANTIGEN"
    assert "-7.5" not in sft_row["instruction"]


def test_antibody_collection_rejects_random_fallback(tmp_path):
    from tasks.antibody.core.ldm_light.ldm_acq import (
        collect_direct_sequence_action,
    )

    sink = DataCollectionSink(tmp_path)

    collected = collect_direct_sequence_action(
        sink,
        decision={"source": "llm_direct_fallback_random"},
        selected_candidates=[{"sequence": "ADGHTKQNPRA"}],
        rows=[],
        observed=set(),
        antigen="TEST_ANTIGEN",
        antigen_context=None,
        seq_len=11,
        history_top_k=10,
        seed=17,
        eval_start=0,
        method="llm_gen",
        run_dir=tmp_path,
    )

    assert collected is False
    assert not (tmp_path / "ldm_ir.jsonl").exists()


def test_antibody_direct_acquisition_collects_preselection_candidates(tmp_path):
    from tasks.antibody.core.ldm_light.ldm_acq import (
        collect_direct_sequence_action,
    )

    sink = DataCollectionSink(tmp_path)
    candidates = [
        {"sequence": "ADGHTKQNPRA", "acquisition_score": 0.9},
        {"sequence": "SDGHTKQNPRG", "acquisition_score": 0.1},
    ]

    collected = collect_direct_sequence_action(
        sink,
        decision={
            "source": "direct_max",
            "generation": {"source": "llm_direct"},
            "candidates": candidates,
            "selected_indices": [0],
        },
        selected_candidates=[candidates[0]],
        rows=[],
        observed=set(),
        antigen="TEST_ANTIGEN",
        antigen_context=None,
        seq_len=11,
        history_top_k=10,
        seed=17,
        eval_start=0,
        method="direct_max",
        run_dir=tmp_path,
    )

    ir = json.loads((tmp_path / "ldm_ir.jsonl").read_text(encoding="utf-8"))
    sft = json.loads((tmp_path / "ldm_sft.jsonl").read_text(encoding="utf-8"))
    assert collected is True
    assert [
        candidate["design"] for candidate in ir["action"]["payload"]["candidates"]
    ] == ["ADGHTKQNPRA", "SDGHTKQNPRG"]
    assert "acquisition_score" not in sft["output"]
