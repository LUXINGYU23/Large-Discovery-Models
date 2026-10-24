import json
import pytest
from ldm_tts.engine.expansion import ExpansionRequest
from tasks.nucleobench.core.harness import _turn_message, direct_harness_profile, harness_tool_extensions
from tasks.nucleobench.core.harness_runtime import describe, proposal_profiles
from tasks.nucleobench.core.mock import MOCK_CONTEXT
from tasks.nucleobench.core.workflow import describe_ldm_task
from tasks.nucleobench.core.workflow_args import parse_args, method_preset_sha256


def test_blank_slate_keeps_tools_and_skills_but_removes_research_guidance():
    args = parse_args(['--search-method', 'harness', '--harness-blank-slate', '--dry-run'])
    profile = proposal_profiles(args)[0]
    assert profile.profile_id == 'blank_slate'
    assert profile.skill_dirs == direct_harness_profile()[0].skill_dirs
    assert profile.skill_dir_sha256 == direct_harness_profile()[0].skill_dir_sha256
    assert describe(args)['research_mode'] == 'blank_slate'
    assert not describe(args)['surrogate_query']
    assert 'query_surrogate' not in str(harness_tool_extensions())
    message = _turn_message(ExpansionRequest(round_idx=1, reservoir_size=16),
        candidate_count=16, observations=[], session_count=1,
        allow_repeated_occurrences=False, attach_empirical_q0=False,
        initial=True, history_from_seq=0, history_to_seq=0, history_digest='test',
        context=MOCK_CONTEXT, blank_slate=True)
    payload = json.loads(message.split('\n\n', 1)[1])
    assert payload['submission_contract']['candidate_count'] == 16
    assert payload['novelty_contract']['evaluated_candidates_are_forbidden']
    assert not payload['novelty_contract']['same_session_repeated_occurrences_are_allowed']
    assert 'first_submission_by_minute' not in payload['time_budget']
    assert 'hypothesis-driven' not in message
    assert 'contrasting hypothesis' not in message
    assert 'paired start' in message


def test_blank_slate_changes_resume_identity_and_rejects_parallel_method():
    blank = parse_args(['--search-method', 'harness', '--harness-blank-slate', '--dry-run'])
    normal = parse_args(['--search-method', 'harness', '--dry-run'])
    assert method_preset_sha256(blank, describe_ldm_task(blank)) != method_preset_sha256(normal, describe_ldm_task(normal))
    with pytest.raises(SystemExit):
        parse_args(['--search-method', 'ldm_harness', '--harness-blank-slate', '--dry-run'])


def test_blank_slate_factory_uses_direct_selection(tmp_path):
    from tasks.nucleobench.core.factory import build_proposal_expander
    expander = build_proposal_expander(
        'harness', MOCK_CONTEXT, seed=42, evaluations_per_round=16,
        harness_client=object(), harness_session_profiles=direct_harness_profile(blank_slate=True),
        harness_artifact_root=tmp_path, harness_candidates_per_session=16, campaign_id='blank')
    assert not expander.attach_empirical_q0
    assert not expander.allow_repeated_occurrences
    turns = expander._turns(ExpansionRequest(round_idx=1, reservoir_size=16))
    assert len(turns)==1
    assert turns[0].message.startswith('Submit the requested candidate batch')
