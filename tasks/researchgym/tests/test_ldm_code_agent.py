import json
import io

import pytest

from tasks.researchgym.scripts.ldm_code_agent import (
    AgentError,
    ChatCompletionsClient,
    MAX_ACTIONS_PER_TURN,
    Pricing,
    WorkspaceTools,
    parse_actions,
    resolve_api_url,
    run_agent,
    load_prior_receipt,
)
from tasks.researchgym.table3_protocol import protocol_manifest


def test_parse_actions_accepts_json_and_limits_count():
    analysis, actions = parse_actions('```json\n{"analysis":"x", "actions":[{"tool":"list","path":"."}]}\n```')
    assert analysis == "x" and actions == [{"tool": "list", "path": "."}]
    with pytest.raises(AgentError, match="non-empty"):
        parse_actions('{"actions": []}')
    accepted = '{"actions":[' + ','.join('{"tool":"list"}' for _ in range(MAX_ACTIONS_PER_TURN)) + ']}'
    assert len(parse_actions(accepted)[1]) == MAX_ACTIONS_PER_TURN
    rejected = '{"actions":[' + ','.join('{"tool":"list"}' for _ in range(MAX_ACTIONS_PER_TURN + 1)) + ']}'
    with pytest.raises(AgentError, match="more than"):
        parse_actions(rejected)


def test_parse_actions_repairs_only_string_literal_newlines_and_escapes():
    _, actions = parse_actions('{"actions":[{"tool":"bash","command":"printf one\ntwo \\d"}]}')
    assert actions[0]["command"] == "printf one\ntwo \\d"
    with pytest.raises(AgentError, match="malformed"):
        parse_actions('{"actions":[{"tool":"bash" "command":"missing comma"}]}')


def test_api_url_accepts_base_or_full_openai_compatible_endpoint():
    assert resolve_api_url("https://provider.example/v1") == "https://provider.example/v1/chat/completions"
    assert resolve_api_url("https://provider.example/v1/chat/completions/") == "https://provider.example/v1/chat/completions"


def test_workspace_tools_keep_file_paths_inside_workspace(tmp_path):
    tools = WorkspaceTools(tmp_path)
    ok = tools.execute({"tool": "write", "path": "nested/method.py", "content": "x = 1\n"})
    assert ok["ok"] and (tmp_path / "nested/method.py").read_text() == "x = 1\n"
    escaped = tools.execute({"tool": "read", "path": "../outside"})
    assert not escaped["ok"] and "escapes" in escaped["error"]
    external = tools.execute({"tool": "bash", "command": "python -c 'import urllib.request; urllib.request.urlopen(\"https://example.com\")'"})
    assert not external["ok"] and "network-capable" in external["error"]
    pretrained = tools.execute({"tool": "bash", "command": "python -c 'import timm; timm.create_model(\"vit\", pretrained=True)'"})
    assert not pretrained["ok"] and "network-capable" in pretrained["error"]
    edited = tools.execute({"tool": "replace", "path": "nested/method.py", "old": "42", "new": "7"})
    assert not edited["ok"] and (tmp_path / "nested/method.py").read_text() == "x = 1\n"
    edited = tools.execute({"tool": "replace", "path": "nested/method.py", "old": "x = 1", "new": "x = 7"})
    assert edited["ok"] and (tmp_path / "nested/method.py").read_text() == "x = 7\n"
    found = tools.execute({"tool": "search", "path": "nested/method.py", "query": "x ="})
    assert found["matches"] == [{"line": 1, "offset": 0, "text": "x = 7"}]


def test_pricing_handles_cached_input_and_reasoning_without_double_counting():
    pricing = Pricing(2.0, 0.5, 4.0, 0.1)
    usage = {"prompt_tokens": 1000, "completion_tokens": 300,
             "prompt_tokens_details": {"cached_tokens": 400},
             "completion_tokens_details": {"reasoning_tokens": 200}}
    # 600*2 + 400*.5 + max(300, 200)*4 = 2600 token-dollars / 1M
    assert pricing.cost(usage) == pytest.approx(0.0026)


@pytest.mark.parametrize("response,expected_cost,unknown", [
    ({"choices": [{"message": {"content": ""}}],
      "usage": {"prompt_tokens": 1000, "completion_tokens": 100}}, 0.0011, False),
    ({"choices": [{"message": {"content": ""}}]}, 0.1, True),
])
def test_empty_provider_reply_still_charges_call(monkeypatch, response, expected_cost, unknown):
    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    monkeypatch.setattr(
        "tasks.researchgym.scripts.ldm_code_agent.urllib.request.urlopen",
        lambda *_args, **_kwargs: FakeResponse(json.dumps(response).encode()),
    )
    client = ChatCompletionsClient(
        api_url="https://provider.example/v1/chat/completions", api_key="test",
        model="test", pricing=Pricing(1.0, 1.0, 1.0, 0.1),
        budget_usd=10.0, request_timeout=1,
    )
    with pytest.raises(AgentError, match="no assistant content|omitted token usage"):
        client.complete([{"role": "user", "content": "hello"}])
    assert client.total_cost == pytest.approx(expected_cost)
    assert len(client.usage_events) == 1
    assert client.usage_events[0].get("unknown_usage", False) is unknown


class _FakeClient:
    def __init__(self):
        self.total_cost = 0.01
        self.usage_events = [{"cost_usd": 0.01}]

    def complete(self, messages):
        return json.dumps({"analysis": "write, test, submit", "actions": [
            {"tool": "write", "path": "method.py", "content": "answer = 42\n"},
            {"tool": "bash", "command": "python method.py", "timeout_seconds": 10},
            {"tool": "submit", "summary": "wrote and executed method.py"},
        ]})


def test_agent_owns_workspace_and_leaves_auditable_transcript(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "task_description.md").write_text("Do real research.")
    manifest = protocol_manifest("table3", model="provider/model", harness="LDM-CodeAgent",
                                 cmr_primary_row="qgs_blip_base")
    receipt = run_agent(task_id="continual_learning", workspace=workspace, seed=0,
                        profile_name="table3", manifest=manifest, output_dir=tmp_path / "run",
                        client=_FakeClient(), max_turns=2)
    assert receipt["status"] == "submitted"
    assert (workspace / "method.py").read_text() == "answer = 42\n"
    transcript = json.loads((tmp_path / "run" / "transcript.json").read_text())
    assert transcript[0]["actions"][0]["tool"] == "write"
    assert transcript[0]["observations"][1]["tool"] == "bash"
    progress = json.loads((tmp_path / "run" / "progress.json").read_text())
    assert progress["status"] == "submitted" and progress["turns_completed"] == 1


def test_turn_limit_is_distinct_from_wall_clock_limit(tmp_path):
    class _InspectOnlyClient:
        total_cost = 0.01
        usage_events = [{"cost_usd": 0.01}]

        def complete(self, messages):
            return '{"actions":[{"tool":"list","path":"."}]}'

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manifest = protocol_manifest("table3", model="provider/model", harness="LDM-CodeAgent",
                                 cmr_primary_row="qgs_blip_base")
    receipt = run_agent(task_id="continual_learning", workspace=workspace, seed=0,
                        profile_name="table3", manifest=manifest, output_dir=tmp_path / "run",
                        client=_InspectOnlyClient(), max_turns=2)
    assert receipt["status"] == "turn_exhausted"
    assert json.loads((tmp_path / "run" / "progress.json").read_text())["turns_completed"] == 2


def test_inspection_only_trajectory_stops_before_spending_all_240_turns(tmp_path):
    class _InspectOnlyClient:
        total_cost = 0.01
        usage_events = [{"cost_usd": 0.01}]

        def complete(self, messages):
            return '{"actions":[{"tool":"list","path":"."}]}'

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manifest = protocol_manifest("table3", model="provider/model", harness="LDM-CodeAgent",
                                 cmr_primary_row="qgs_blip_base")
    receipt = run_agent(task_id="continual_learning", workspace=workspace, seed=0,
                        profile_name="table3", manifest=manifest, output_dir=tmp_path / "run",
                        client=_InspectOnlyClient(), max_turns=240)
    assert receipt["status"] == "no_candidate"
    assert json.loads((tmp_path / "run" / "progress.json").read_text())["turns_completed"] == 24
    transcript = json.loads((tmp_path / "run" / "transcript.json").read_text())
    assert transcript[16]["observations"][0]["tool"] == "source_change_guard"


class _FormattingThenSubmitClient:
    def __init__(self):
        self.total_cost = 0.02
        self.usage_events = [{"cost_usd": 0.01}, {"cost_usd": 0.01}]
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        if self.calls == 1:
            return '{"actions":[{"tool":"bash" "command":"missing comma"}]}'
        return json.dumps({"actions": [{"tool": "submit", "summary": "grader ran"}]})


def test_agent_recovers_from_one_invalid_model_action_without_executing_it(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manifest = protocol_manifest("table3", model="provider/model", harness="LDM-CodeAgent",
                                 cmr_primary_row="qgs_blip_base")
    receipt = run_agent(task_id="continual_learning", workspace=workspace, seed=0,
                        profile_name="table3", manifest=manifest, output_dir=tmp_path / "run",
                        client=_FormattingThenSubmitClient(), max_turns=3)
    assert receipt["status"] == "submitted"
    transcript = json.loads((tmp_path / "run" / "transcript.json").read_text())
    assert "malformed action JSON" in transcript[0]["error"]
    assert transcript[1]["observations"][0]["tool"] == "submit"


def test_prior_receipt_must_match_the_seed_and_preserves_accounting(tmp_path):
    manifest = protocol_manifest("table3", model="provider/model", harness="LDM-CodeAgent",
                                 cmr_primary_row="qgs_blip_base")
    prior = {
        "task_id": "continual_learning", "seed": 0, "profile": "table3",
        "protocol": manifest["protocol"], "source_commit": manifest["source_commit"],
        "model": manifest["model"], "harness": manifest["harness"], "status": "agent_error",
        "api_cost_usd": 1.25, "wall_clock_hours": 0.5, "api_calls": 4,
    }
    receipt_path = tmp_path / "prior.json"
    receipt_path.write_text(json.dumps(prior))
    loaded = load_prior_receipt(receipt_path, task_id="continual_learning", seed=0,
                                profile_name="table3", manifest=manifest)
    assert loaded["api_cost_usd"] == 1.25
    with pytest.raises(AgentError, match="seed"):
        load_prior_receipt(receipt_path, task_id="continual_learning", seed=1,
                           profile_name="table3", manifest=manifest)
