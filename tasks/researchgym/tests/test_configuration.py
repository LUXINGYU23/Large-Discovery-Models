"""Configs, contract profiles, pilot matrix, and pinned-source contract consistency."""

import json
from pathlib import Path

import pytest

from ldm_tts.cli.runner import build_plan, load_config
from ldm_tts.pilot_evaluation.config import load_pilot_evaluation_spec
from ldm_tts.pilot_evaluation.execution import _child_plan, _select_runs
from tasks.researchgym.core.cases import CASE_IDS, catalog
from tasks.researchgym.core.source import upstream_contract
from tasks.researchgym.core.workflow import parse_args

REPO = Path(__file__).resolve().parents[3]
CONFIGS = sorted((REPO / "config/researchgym").glob("*.yaml"))
ENV = {"RESEARCHGYM_ROOT": "/tmp/rg", "RESEARCHGYM_CASE_DATA_ROOT": "/tmp/rgd", "RESEARCHGYM_CASE_PYTHON": "python",
       "RESEARCHGYM_DEVICES": "0,1", "RESEARCHGYM_RUNS_ROOT": "/tmp/rgruns"}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_every_config_resolves_and_parses(path, monkeypatch):
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)
    plan = build_plan(load_config(path), path)
    assert plan["module"] == "tasks.researchgym.ldm_task.procedure"
    args = parse_args(plan["argv"])
    raw = load_config(path)
    if raw.get("contract_profile"):
        assert plan["contract_profile"] == raw["contract_profile"]
        assert args.search_method in raw["contract_profile"]


def test_pilot_matrix_children_parse_for_every_case_and_method(monkeypatch, tmp_path):
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("RESEARCHGYM_RUNS_ROOT", str(tmp_path))
    spec = load_pilot_evaluation_spec(REPO / "config/pilot_evaluation/researchgym.yaml")
    assert set(spec.methods) == {"llm", "ldm", "harness", "ldm_harness", "ldm_harness_compiled"}
    assert {case.case_id for case in spec.cases} == set(CASE_IDS)
    base = load_config(spec.base_config)
    runs = _select_runs(spec, cases=None, methods=None, seeds=None)
    for run in runs:
        args = parse_args(_child_plan(spec, base, run, resume=False)["argv"])
        assert (args.search_method, args.case, args.seed) == (run.method, run.case_id, run.seed)
        assert args.initialization_mode == "shared_start" and args.iterations == spec.iterations
        assert args.proposal_mode == ("openai" if run.method in ("llm", "ldm") else "harness")


def test_catalog_cases_are_pinned_and_consistent():
    contract = upstream_contract()
    assert contract["commit"] == "0adc08e93754b606d8a76055ed2f1c9574504312"
    experiment = json.loads((REPO / "tasks/researchgym/experiment.json").read_text())
    assert experiment["benchmark"]["source_commit"] == contract["commit"]
    reported = {m["name"] for m in experiment["metrics"]["reported"]}
    for case in catalog()["cases"]:
        pinned = contract["cases"][case["case_id"]]
        assert pinned["task_path"] == case["upstream_task"]
        for injection in case["injections"]:
            assert injection["path"] in pinned["files"]
        for reference in case["public_context"]["references"]:
            assert reference in pinned["files"]
        assert case["metric"]["name"] in reported
        assert case["protocol"]["official_grader"] and not case["protocol"]["official_protocol"]


@pytest.mark.parametrize("case_id,jobs", [("time_series_explanation", 25), ("continual_learning", 3),
                                          ("cross_modal_retrieval", 16), ("improving_replay_buffers", 2)])
def test_benchmark_job_budget_counts_the_full_job_matrix(case_id, jobs):
    from tasks.researchgym.core.cases import load_case
    from tasks.researchgym.core.workflow import jobs_per_evaluation
    args = parse_args(["--case", case_id, "--case-python", "python"])
    assert jobs_per_evaluation(args, load_case(case_id)) == jobs
