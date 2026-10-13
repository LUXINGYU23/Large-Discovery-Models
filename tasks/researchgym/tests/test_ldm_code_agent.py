import json

import pytest

from tasks.researchgym.scripts.ldm_code_agent import (
    AgentError,
    Pricing,
    WorkspaceTools,
    parse_actions,
    resolve_api_url,
    run_agent,
)
from tasks.researchgym.table3_protocol import protocol_manifest


def test_parse_actions_accepts_json_and_limits_count():
    analysis, actions = parse_actions('```json\n{"analysis":"x", "actions":[{"tool":"list","path":"."}]}\n```')
    assert analysis == "x" and actions == [{"tool": "list", "path": "."}]
    with pytest.raises(AgentError, match="non-empty"):
        parse_actions('{"actions": []}')


def test_api_url_accepts_base_or_full_openai_compatible_endpoint():
    assert resolve_api_url("https://provider.example/v1") == "https://provider.example/v1/chat/completions"
    assert resolve_api_url("https://provider.example/v1/chat/completions/") == "https://provider.example/v1/chat/completions"


def test_workspace_tools_keep_file_paths_inside_workspace(tmp_path):
    tools = WorkspaceTools(tmp_path)
    ok = tools.execute({"tool": "write", "path": "nested/method.py", "content": "x = 1\n"})
    assert ok["ok"] and (tmp_path / "nested/method.py").read_text() == "x = 1\n"
    escaped = tools.execute({"tool": "read", "path": "../outside"})
    assert not escaped["ok"] and "escapes" in escaped["error"]


def test_pricing_handles_cached_input_and_reasoning_without_double_counting():
    pricing = Pricing(2.0, 0.5, 4.0, 0.1)
    usage = {"prompt_tokens": 1000, "completion_tokens": 300,
             "prompt_tokens_details": {"cached_tokens": 400},
             "completion_tokens_details": {"reasoning_tokens": 200}}
    # 600*2 + 400*.5 + max(300, 200)*4 = 2600 token-dollars / 1M
    assert pricing.cost(usage) == pytest.approx(0.0026)


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
    manifest = protocol_manifest("table3", model="provider/model", harness="LDM-CodeAgent")
    receipt = run_agent(task_id="continual_learning", workspace=workspace, seed=0,
                        profile_name="table3", manifest=manifest, output_dir=tmp_path / "run",
                        client=_FakeClient(), max_turns=2)
    assert receipt["status"] == "submitted"
    assert (workspace / "method.py").read_text() == "answer = 42\n"
    transcript = json.loads((tmp_path / "run" / "transcript.json").read_text())
    assert transcript[0]["actions"][0]["tool"] == "write"
    assert transcript[0]["observations"][1]["tool"] == "bash"
