"""Tests for the Iron Mind proposal prompt policies."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from ldm_tts.engine.expansion import ExpansionRequest

from tasks.iron_mind.core.prompting import (
    BASELINE_PROMPT_POLICY,
    DIRECT_PROMPT_POLICY,
    PORTFOLIO_PROMPT_POLICY,
    build_reaction_prompt_messages,
    build_slot_plan,
    prompt_sha256,
    validate_prompt_policy,
    validate_slot_focus,
)
from tasks.iron_mind.core.schema import load_reaction_schemas


TASK_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = TASK_ROOT / "resources" / "reaction_schemas.json"


def _schema():
    return load_reaction_schemas(SCHEMA_PATH)["buchwald_hartwig"]


def _request(*, observations=(), reservoir_size: int = 64) -> ExpansionRequest:
    return ExpansionRequest(round_idx=2, reservoir_size=reservoir_size, observations=observations)


def _observation(score: float = 42.0):
    schema = _schema()
    conditions = {factor.name: factor.options[0] for factor in schema.factors}
    candidate = SimpleNamespace(
        payload={"dataset_id": schema.dataset_id, "conditions": conditions},
        canonical_key="known-condition",
    )
    return SimpleNamespace(candidate=candidate, metrics={"reaction_score": score})


def test_portfolio_allocates_unique_focuses_for_a_64_candidate_reservoir() -> None:
    schema = _schema()
    request = _request()

    plans = tuple(
        build_slot_plan(request, schema, proposal_index=index) for index in range(request.reservoir_size)
    )

    assert {plan.policy for plan in plans} == {PORTFOLIO_PROMPT_POLICY}
    assert len({plan.focus for plan in plans}) == request.reservoir_size
    assert all(plan.focus_capacity >= request.reservoir_size for plan in plans)
    assert len({plan.focus_payload()["additive"] for plan in plans}) == 24
    assert len({plan.focus_payload()["aryl_halide"] for plan in plans}) == 16
    assert {plan.role for plan in plans} == {
        "chemical_prior",
        "coverage_prior",
        "interaction_prior",
        "operational_contrast",
    }


def test_portfolio_rotates_to_a_disjoint_focus_batch_across_rounds_and_seeds() -> None:
    schema = _schema()
    first_request = ExpansionRequest(round_idx=0, reservoir_size=64, observations=())
    next_request = ExpansionRequest(round_idx=1, reservoir_size=64, observations=())
    first = {
        build_slot_plan(first_request, schema, proposal_index=index).focus
        for index in range(first_request.reservoir_size)
    }
    next_round = {
        build_slot_plan(next_request, schema, proposal_index=index).focus
        for index in range(next_request.reservoir_size)
    }
    next_seed = {
        build_slot_plan(first_request, schema, proposal_index=index, slot_seed=1).focus
        for index in range(first_request.reservoir_size)
    }

    assert first.isdisjoint(next_round)
    assert next_seed == next_round


def test_portfolio_prompt_includes_objective_history_role_and_focus() -> None:
    schema = _schema()
    request = _request(observations=(_observation(),), reservoir_size=4)
    plan = build_slot_plan(request, schema, proposal_index=0)

    messages = build_reaction_prompt_messages(request, schema, plan, proposal_index=0)
    content = "\n".join(message["content"] for message in messages)

    assert messages[0]["content"].endswith("Return JSON only.")
    assert '"direction":"maximize"' in content
    assert "higher is better" in content
    assert "Completed external evaluations: 1" in content
    assert "Observed option coverage:" in content
    assert "evidence_exploitation" in content
    assert '"slot_focus"' not in content
    assert "Required slot focus (hard allocation):" in content
    assert "known-condition" in content


def test_portfolio_focus_is_enforced_before_candidate_admission() -> None:
    schema = _schema()
    request = _request(reservoir_size=4)
    plan = build_slot_plan(request, schema, proposal_index=1)
    conditions = {factor.name: factor.options[0] for factor in schema.factors}
    payload = {"dataset_id": schema.dataset_id, "conditions": conditions}

    with pytest.raises(ValueError, match="assigned slot focus"):
        validate_slot_focus(payload, plan)

    conditions.update(plan.focus_payload())
    validate_slot_focus(payload, plan)


def test_baseline_prompt_has_no_focus_constraint() -> None:
    schema = _schema()
    request = _request(reservoir_size=4)
    plan = build_slot_plan(
        request,
        schema,
        proposal_index=0,
        policy=BASELINE_PROMPT_POLICY,
    )
    messages = build_reaction_prompt_messages(request, schema, plan, proposal_index=0)
    payload = {"dataset_id": schema.dataset_id, "conditions": {}}

    assert plan.focus == ()
    assert messages[0]["content"] == "You propose reaction conditions under an exact source-pinned schema."
    assert "Allowed factors and exact options:" in messages[1]["content"]
    assert "Objective:" not in messages[1]["content"]
    assert "Required slot focus" not in messages[1]["content"]
    validate_slot_focus(payload, plan)


def test_direct_prompt_uses_round_rotated_focus_without_gp_language() -> None:
    schema = _schema()
    first = build_slot_plan(_request(reservoir_size=1), schema, proposal_index=0, policy=DIRECT_PROMPT_POLICY)
    later_request = ExpansionRequest(round_idx=3, reservoir_size=1, observations=())
    later = build_slot_plan(later_request, schema, proposal_index=0, policy=DIRECT_PROMPT_POLICY)
    messages = build_reaction_prompt_messages(later_request, schema, later, proposal_index=0)

    assert first.focus != later.focus
    assert later.role == "direct_search"
    assert "without a GP selector" in messages[0]["content"]
    assert "this candidate is evaluated immediately" in messages[1]["content"]


def test_prompt_digests_are_slot_specific_and_policy_names_are_validated() -> None:
    schema = _schema()
    request = _request(reservoir_size=4)
    first = build_slot_plan(request, schema, proposal_index=0)
    second = build_slot_plan(request, schema, proposal_index=1)
    first_messages = build_reaction_prompt_messages(request, schema, first, proposal_index=0)
    second_messages = build_reaction_prompt_messages(request, schema, second, proposal_index=1)

    assert prompt_sha256(first_messages) != prompt_sha256(second_messages)
    assert validate_prompt_policy(PORTFOLIO_PROMPT_POLICY) == PORTFOLIO_PROMPT_POLICY
    with pytest.raises(ValueError, match="Unknown Iron Mind prompt policy"):
        validate_prompt_policy("unknown")
