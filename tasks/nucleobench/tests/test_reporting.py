from __future__ import annotations

import csv
import json
from pathlib import Path

from ldm_tts.data import DataCollectionSink
from ldm_tts.engine import LDMEngine
from ldm_tts.engine.run_store import CampaignRuntime
from tasks.nucleobench.core.candidate import NucleoBenchCandidateDomain
from tasks.nucleobench.core.designer import (
    NucleoBenchDesigner,
    initialize_designer_state,
)
from tasks.nucleobench.core.evaluator import NucleoBenchEvaluator
from tasks.nucleobench.core.mock import (
    MOCK_CONTEXT,
    build_mock_expander,
    build_mock_task_spec,
)
from tasks.nucleobench.core.reporting import write_campaign_reports


def test_campaign_reporting_separates_pilot_and_official_semantics(
    tmp_path: Path,
) -> None:
    task_spec = build_mock_task_spec()
    runtime = CampaignRuntime.open(
        tmp_path / "campaign",
        task="nucleobench",
        task_spec=task_spec,
    )
    evaluator = NucleoBenchEvaluator(
        MOCK_CONTEXT,
        lambda sequences: [-float(sequence.count("C")) for sequence in sequences],
    )
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
        evaluations_per_step=1,
    )
    designer.run(1)
    runtime.finish(designer.summary())

    result = write_campaign_reports(
        runtime,
        execution={
            "execution_profile": "pilot_evaluation",
            "termination_kind": "rounds",
            "total_rounds": 2,
            "active_optimization_rounds": 1,
            "initialization_evaluations": 1,
            "benchmark_comparable": False,
            "case_id": MOCK_CONTEXT.case.case_id,
            "start_index": MOCK_CONTEXT.start_index,
            "start_set_digest": MOCK_CONTEXT.start_set_digest,
            "optimization_seed": 3,
            "hardware_profile": "test-cpu",
            "search_method": "bo",
            "method_preset_sha256": "1" * 64,
            "max_seconds": None,
        },
        oracle_manifest={"schema_version": 1, "oracle": "fixture"},
        official_outputs=[],
    )

    assert result["execution_profile"] == "pilot_evaluation"
    assert result["termination_kind"] == "rounds"
    assert result["benchmark_comparable"] is False
    assert result["total_rounds"] == 2
    assert result["active_optimization_rounds"] == 1
    assert result["best_found_utility"] == result["best_candidate"]["utility"]
    with (runtime.run_dir / "trajectory.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["round"]) for row in rows] == [0, 1]
    assert all(float(row["elapsed_seconds"]) >= 0 for row in rows)
    assert json.loads((runtime.run_dir / "oracle_manifest.json").read_text()) == {
        "oracle": "fixture",
        "schema_version": 1,
    }
    assert (runtime.run_dir / "proposal_trace.jsonl").is_file()

    official = write_campaign_reports(
        runtime,
        execution={
            "execution_profile": "official_benchmark",
            "termination_kind": "wall_time",
            "total_rounds": 2,
            "active_optimization_rounds": 1,
            "initialization_evaluations": 1,
            "benchmark_comparable": True,
            "case_id": MOCK_CONTEXT.case.case_id,
            "start_index": MOCK_CONTEXT.start_index,
            "start_set_digest": MOCK_CONTEXT.start_set_digest,
            "optimization_seed": 3,
            "hardware_profile": "test-cpu",
            "search_method": "bo",
            "method_preset_sha256": "1" * 64,
            "max_seconds": 28_800,
        },
        oracle_manifest={"schema_version": 1, "oracle": "fixture"},
        official_outputs=[],
    )

    assert official["benchmark_comparable"] is True
    assert official["trajectory_primary_axis"] == "elapsed_seconds"
    assert official["max_seconds"] == 28_800
