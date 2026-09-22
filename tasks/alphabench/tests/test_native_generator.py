import ast
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import random
from types import SimpleNamespace
from threading import Thread
import typing

import pytest

from ldm_tts.contracts.evaluation import EvaluationPaused
from ldm_tts.engine.run_store import BudgetExceededError, CampaignRuntime
from ldm_tts.transport import CallableProposalClient
from tasks.alphabench.core.collection import AcceptedActions
from tasks.alphabench.core.gateway import OracleGateway
from tasks.alphabench.core.native_generator import NativeGenerator
from tasks.alphabench.core.native_runtime import NativeRuntime
from tasks.alphabench.core.native_source import load_algorithms
from tasks.alphabench.core.protocol import T3Protocol, digest
from tasks.alphabench.core.quality import audit_quality
from tasks.alphabench.core.reporting import generation_costs
from tasks.alphabench.core.workflow import make_client
from tasks.alphabench.tests.test_native import SEEDS, SOURCE, source


def build(root, outputs, monkeypatch, *, protocol=None):
    monkeypatch.setattr("tasks.alphabench.core.native_generator.time.sleep", lambda _: None)
    protocol = protocol or T3Protocol(method="alphabench_ea")
    runtime = CampaignRuntime.open(root, task="alphabench", budget_limits=protocol.budgets,
                                   resume=(root / "campaign.json").exists())
    gateway = OracleGateway(protocol, runtime, mock=True)
    calls = []
    def propose(request):
        calls.append(request)
        value = outputs(request) if callable(outputs) else outputs[len(calls) - 1]
        return value if isinstance(value, str) else json.dumps(value)
    native = NativeGenerator(protocol, runtime, CallableProposalClient(propose), gateway, AcceptedActions(root), SOURCE)
    return native, NativeRuntime(runtime, gateway.host), calls


def invoke(native, scheduler, *, count=2, max_try=5, avoid_repeat=True):
    return scheduler.run(lambda: scheduler.callback("generation", native)(
        "Generate financial factors as JSON.", model=native.protocol.model, N=count,
        max_try=max_try, temperature=native.protocol.temperature, avoid_repeat=avoid_repeat))


def generation(root):
    return json.loads(next((root / "generation").glob("*.json")).read_text())


def test_full_raw_output_tail_sampling_and_idempotent_replay(tmp_path, source, monkeypatch):
    output = {"generated": [{"name": name, "expression": expr} for name, expr in
                           (("A", "$close"), ("B", "$open"), ("Tail", "$volume"))] + [17]}
    native, scheduler, calls = build(tmp_path, [output], monkeypatch)
    answer = invoke(native, scheduler)
    saved = generation(tmp_path)
    assert saved["counts"] == {"raw_items": 4, "normalized_items": 3, "visited_items": 2,
                               "checked_items": 2, "unique_accepted": 2, "unprocessed_items": 1}
    assert answer["quality"]["generated_num"] == 2
    assert answer["quality"]["first_accept_rate"] == [.75]
    assert [row["status"] for row in saved["occurrences"]] == ["accepted", "accepted", "unprocessed", "normalization_rejected"]
    assert native.runtime.budget.counters["model_requests"] == 1
    assert native.runtime.budget.counters["dynamic_checks"] == 2
    assert len(list((tmp_path / "accepted_actions").glob("*.json"))) == 1
    audit = audit_quality(native.protocol, native.runtime, native.gateway)
    assert audit["raw_occurrences"] == 4 and audit["accepted_occurrences"] == 2
    assert audit["qlib_dynamic_success_rate"] == .75 and audit["complete"]
    assert native.runtime.budget.counters["quality_checks"] == 1
    assert any(event["payload"]["kind"] == "sample" for event in native.runtime.events()
               if event["event_type"] == "native_generation_choice")
    before = (tmp_path / "budget.json").read_bytes()
    new_native, replay, new_calls = build(tmp_path, [], monkeypatch)
    assert invoke(new_native, replay) == answer
    assert not new_calls and len(calls) == 1 and (tmp_path / "budget.json").read_bytes() == before


def test_native_cold_start_retains_partial_factors_even_when_source_success_is_false(tmp_path, source, monkeypatch):
    output = {"generated": [{"name": "a", "expression": "$close"}, {"name": "b", "expression": "$open"}]}
    native, scheduler, calls = build(tmp_path, [output] + [{"generated": []}] * 4, monkeypatch,
        protocol=T3Protocol(method="alphabench_ea", cold_seed_count=6))
    seeds = scheduler.run(lambda: scheduler.callback("cold_start", native.cold_start)())
    assert len(seeds) == 2 and len(calls) == 5
    assert all(set(seed) == {"name", "expression"} for seed in seeds)
    assert generation(tmp_path)["native_result"]["success"] is False
    assert calls[0].messages[1]["content"].startswith("Generate diverse alpha factors for stock ranking.")


def test_abandoned_generation_audits_all_returned_items_without_fabricating_a_native_return(tmp_path, source, monkeypatch):
    output = {"generated": [{"name": str(index), "expression": expression}
                            for index, expression in enumerate(("$close", "$open", "$volume"))]}
    native, scheduler, calls = build(tmp_path, [output], monkeypatch)
    send = native.gateway._send
    def stop(request):
        scheduler.stop(BudgetExceededError("peer stopped"), notify=False)
        return send(request)
    native.gateway.before_dispatch = scheduler._check
    monkeypatch.setattr(native.gateway, "_send", stop)
    with pytest.raises(BudgetExceededError):
        invoke(native, scheduler, count=2)
    assert native.settle() == {"interrupted_generations": 1}
    saved = generation(tmp_path)
    assert saved["native_result"] is None and len(saved["occurrences"]) == 3
    assert len(calls) == 1 and not list((tmp_path / "accepted_actions").glob("*.json"))
    assert generation_costs([saved])["interrupted_steps"] == 1
    assert generation_costs([saved])["failed_steps"] == 0
    monkeypatch.setattr(native.gateway, "_send", send)
    audit = audit_quality(native.protocol, native.runtime, native.gateway)
    assert audit["raw_occurrences"] == 3 and audit["complete"]
    assert native.runtime.budget.counters["quality_checks"] == 2  # The completed in-generation check is reused.


@pytest.mark.parametrize("phase", ["model", "check"])
def test_native_stop_cannot_hide_an_unknown_generation_request(tmp_path, source, monkeypatch, phase):
    def output(request):
        if phase == "model":
            scheduler.stop(BudgetExceededError("peer stopped"), notify=False)
            raise OSError("model response lost")
        return {"generated": [{"name": "a", "expression": "$close"}]}
    native, scheduler, calls = build(tmp_path, output, monkeypatch)
    if phase == "check":
        def disconnect(request):
            scheduler.stop(BudgetExceededError("peer stopped"), notify=False)
            raise OSError("check response lost")
        monkeypatch.setattr(native.gateway, "_send", disconnect)
    native.gateway.before_dispatch = scheduler._check
    with pytest.raises(BudgetExceededError):
        invoke(native, scheduler, count=1)
    with pytest.raises(EvaluationPaused, match="reconciliation"):
        native.settle()
    assert len(calls) == 1 and not list((tmp_path / "generation").glob("*.json"))


@pytest.mark.parametrize("count,expected", [(5, True), (6, False)])
def test_five_attempt_repairs_preserve_native_partial_success_and_original_quality(tmp_path, source, monkeypatch, count, expected):
    outputs = ["invalid JSON", {"generated": []}, {"name": 17, "expression": "$close"},
               {"generated": [{"name": "A", "expression": "$close"}, {"name": "B", "expression": "$close"},
                               {"name": "Bad", "expression": "Ref($close,-1)"}]},
               {"name": "Open", "expression": "$open", "reason": 13}]
    native, scheduler, calls = build(tmp_path, outputs, monkeypatch)
    answer = invoke(native, scheduler, count=count)
    assert answer["success"] is expected and len(answer["factors"]) == 2
    assert len(calls) == answer["trynum"] == 5
    assert answer["quality"]["output_format_error"] == 3
    assert answer["quality"]["generated_num"] == 4 and answer["quality"]["accepted_num"] == 2
    saved = generation(tmp_path)
    assert saved["counts"]["raw_items"] == 5 and saved["counts"]["checked_items"] == 2
    assert "duplicate" in [row["status"] for row in saved["occurrences"]]
    assert "static_rejected" in [row["status"] for row in saved["occurrences"]]
    assert len(list((tmp_path / "accepted_actions").glob("*.json"))) == int(expected)
    assert "Do NOT repeat" in calls[-1].messages[-1]["content"]
    assert native.runtime.budget.counters["proposal_attempts"] == 5


@pytest.mark.parametrize("failure", ["timeout", "malformed"])
def test_check_infrastructure_failure_pauses_without_repairs_or_redispatch(tmp_path, source, monkeypatch, failure):
    output = {"generated": [{"name": "A", "expression": "$close"}]}
    sent = []
    def disconnect(request):
        sent.append(request)
        if failure == "malformed":
            return {"request_id": request["request_id"], "jobs": [{}]}
        raise OSError("response lost")
    for _ in range(2):
        native, scheduler, calls = build(tmp_path, [output], monkeypatch)
        monkeypatch.setattr(native.gateway, "_send", disconnect)
        with pytest.raises(EvaluationPaused):
            invoke(native, scheduler, count=1)
    assert len(sent) == 1 and not calls
    assert native.runtime.budget.counters["model_requests"] == 1
    assert native.runtime.budget.counters["dynamic_checks"] == 1


def test_nonempty_valid_generator_matches_original_source(tmp_path, source, monkeypatch):
    monkeypatch.setattr("tasks.alphabench.core.native_generator.uuid.uuid4", lambda: "12345678-1234-5678-1234-567812345678")
    outputs = [{"generated": [{"name": "A", "expression": "$close"}]},
               {"generated": [{"name": "B", "expression": "$open"}, {"name": "Tail", "expression": "$volume"}]}]
    native, scheduler, calls = build(tmp_path, outputs, monkeypatch)
    actual = invoke(native, scheduler, count=2)
    original = (source / "agent/generator_qlib_search.py").read_text()
    names = {"get_system_searcher_prompt", "_normalize_llm_output", "_unique_name", "call_qlib_search"}
    nodes = [node for node in ast.parse(original).body if isinstance(node, ast.FunctionDef) and node.name in names or
             isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "SYSTEM_SEARCHER_PROMPT" for target in node.targets)]
    reference_calls = []
    def llm(prompt, **kwargs):
        reference_calls.append(prompt)
        return json.dumps(outputs[len(reference_calls)-1])
    env = {**native.prompts, "json": json, "Any": typing.Any, "Dict": typing.Dict, "Set": typing.Set, "Optional": typing.Optional,
           "call_llm": llm, "is_assay": lambda: False, "check_factor_via_api": lambda _: {"success": True},
           "time": SimpleNamespace(sleep=lambda _: None), "random": random.Random(42),
           "uuid": SimpleNamespace(uuid4=lambda: "12345678-1234-5678-1234-567812345678")}
    exec("\n\n".join(ast.get_source_segment(original, node) for node in nodes), env)
    expected = env["call_qlib_search"]("Generate financial factors as JSON.", model="deepseek-flash", N=2,
                                     temperature=.7, avoid_repeat=True)
    assert actual == expected
    assert [request.messages[-1]["content"] for request in calls] == reference_calls


def test_assay_native_prompt_and_separate_dialects_reach_the_same_check_gateway(tmp_path, source, monkeypatch):
    protocol = T3Protocol(method="alphabench_tot", backend="assay", market="nasdaq100",
                          filter_profile="assay_code_filter_v1")
    output = {"generated": [{"name": "Qlib", "expression": "Mean($close,5)"},
                            {"name": "Assay", "expression": "ts_mean(close,5)"}]}
    native, scheduler, calls = build(tmp_path, [output], monkeypatch, protocol=protocol)
    assert invoke(native, scheduler)["success"]
    assert native.prompts["ASSAY_GENERATE_INSTRUCTION"] in calls[0].messages[0]["content"]
    receipts = [json.loads(path.read_text()) for path in native.gateway.receipts.root.glob("*.json")]
    assert {row["request"]["dialect"] for row in receipts} == {"qlib", "assay"}
    assert all(row["request"]["protocol"]["market"] == "nasdaq100" for row in receipts)
    assert native.runtime.budget.counters["lint_checks"] == 2
    assert native.runtime.budget.counters.get("dynamic_checks", 0) == 0


@pytest.mark.parametrize("method", ["cot", "tot", "ea"])
def test_real_native_algorithm_uses_real_source_generator_and_metered_requests(tmp_path, source, monkeypatch, method):
    protocol = T3Protocol(method="alphabench_" + method)
    def outputs(request):
        number = int(digest(request.messages)[:8], 16) % 30 + 1
        return {"generated": [{"name": f"F{i}", "expression": f"Mean($close,{number+i})"} for i in range(3)]}
    native, scheduler, calls = build(tmp_path, outputs, monkeypatch, protocol=protocol)
    def evaluate(factors):
        return [{"success": True, "metrics": {"ic": .02, "rank_ic": .02}} for _ in factors]
    def stage(module):
        algo = module.create_algo(method, {"rounds": 2, "workers": 2, "N": 2, "top_k": 2,
                                  "model": protocol.model, "temperature": protocol.temperature},
            evaluate_fn=lambda expression: evaluate([expression])[0], batch_evaluate_fn=evaluate,
            batch_evaluate_fn_dict=lambda factors: {f["name"]: value for f, value in zip(factors, evaluate(factors))},
            search_fn=scheduler.callback("generate", native))
        return algo.run(SEEDS, str(tmp_path / "native"))
    with load_algorithms(source, tmp_path / "private", scheduler) as module:
        answer = scheduler.run(lambda: stage(module))
    assert answer["history"] and calls
    assert native.runtime.budget.counters["model_requests"] == len(calls)
    assert native.runtime.budget.counters["dynamic_checks"] > 0
    new_native, replay, new_calls = build(tmp_path, [], monkeypatch, protocol=protocol)
    native, scheduler = new_native, replay
    with load_algorithms(source, tmp_path / "private", scheduler) as module:
        assert scheduler.run(lambda: stage(module)) == answer
    assert not new_calls


def test_resume_after_generation_publication_preserves_rng_and_paired_collection(tmp_path, source, monkeypatch):
    monkeypatch.setenv("LDM_DATA_COLLECTION_ENABLED", "1")
    monkeypatch.setenv("LDM_DATA_COLLECTION_DIR", str(tmp_path / "collection"))
    monkeypatch.setattr("tasks.alphabench.core.native_generator.uuid.uuid4", lambda: "12345678-1234-5678-1234-567812345678")
    valid = ("$open", "$close", "$volume")
    output = {"generated": [{"name": str(i), "expression": expr} for i, expr in
                           enumerate(("$open", "Ref($close,-1)", "$close", "$volume"))]}
    native, scheduler, calls = build(tmp_path, [output, output], monkeypatch)
    accept = native.generator.collection.accept
    def interrupted(*args, **kwargs):
        accept(*args, **kwargs)
        raise EvaluationPaused("process stopped after action publication")
    native.generator.collection.accept = interrupted
    def stage():
        generate = scheduler.callback("generation", native)
        return [generate(f"Generate JSON factors, stage {i}.", model=native.protocol.model, N=3,
                         temperature=native.protocol.temperature) for i in range(2)]
    with pytest.raises(EvaluationPaused, match="after action publication"):
        scheduler.run(stage)
    assert len(calls) == 1
    native, scheduler, resumed_calls = build(tmp_path, [output], monkeypatch)
    answers = scheduler.run(stage)
    reference = random.Random(42)
    expected = [[valid[i] for i in reference.sample(range(3), 3)] for _ in range(2)]
    assert [[item["expression"] for item in answer["factors"]] for answer in answers] == expected
    assert len(resumed_calls) == 1 and native.runtime.budget.counters["model_requests"] == 2
    assert len(list((tmp_path / "accepted_actions").glob("*.json"))) == 2
    assert native.runtime.budget.counters["dynamic_checks"] == 6
    pointer = next((tmp_path / "collection").rglob("current.json"))
    published = json.loads(pointer.read_text())
    assert published["count"] == 2
    for name in ("ldm_ir.jsonl", "ldm_sft.jsonl"):
        assert len((pointer.parent / published["generation"] / name).read_text().splitlines()) == 2


@pytest.mark.parametrize("status", [200, 503])
def test_native_responses_wire_uses_json_max_thinking_and_no_implicit_retry(tmp_path, source, monkeypatch, status):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-token")
    protocol = T3Protocol(method="alphabench_cot", temperature=.5)
    native, scheduler, _ = build(tmp_path, [], monkeypatch, protocol=protocol)
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_POST(self):
            requests.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            body = {"model": protocol.model, "status": "completed", "usage": {"total_tokens": 3},
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(
                        {"generated": [{"name": "A", "expression": "$close"}]})}]}]}
            self.wfile.write(json.dumps(body if status == 200 else {"error": "fixture outage"}).encode())
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = make_client(protocol, False)
        client.url = f"http://127.0.0.1:{server.server_port}/responses"
        native.generator.client = client
        if status == 200:
            assert invoke(native, scheduler, count=1)["success"]
        else:
            with pytest.raises(EvaluationPaused, match="model request failed"):
                invoke(native, scheduler, count=1)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    assert len(requests) == 1
    path, body = requests[0]
    assert path == "/responses" and body["model"] == "deepseek-flash"
    assert body["reasoning"] == {"effort": "max"} and body["text"] == {"format": {"type": "json_object"}}
    assert body["temperature"] == .5 and body["max_output_tokens"] == protocol.max_model_tokens
    assert native.runtime.budget.counters["model_requests"] == 1
