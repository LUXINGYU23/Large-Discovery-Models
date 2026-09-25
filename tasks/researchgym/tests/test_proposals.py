"""Strict submissions, repair, measured-only exclusion, q0, and request accounting."""

import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from ldm_tts.engine.run_store import BudgetLedger
from ldm_tts.transport import ProposalResponse
from ldm_tts.transport.openai import EndpointRequestError
from tasks.researchgym.core.candidate import ProgramDomain
from tasks.researchgym.core.cases import load_case
from tasks.researchgym.core.proposals import (
    DirectProgramSource, MockProposalClient, ProposalExhausted, validate_rows,
)
from tasks.researchgym.core.sampling import Q0_KEY, attach_empirical_base_measure, empirical_base_masses
from tasks.researchgym.core.usage import ApiBudgetExhausted, ApiPricing

CASE = load_case("improving_replay_buffers")


def program(tag):
    return CASE.seed_program() + f"\n\ndef _variant_{tag}():\n    return {tag}\n"


def row(tag):
    return {"program": program(tag), "change_summary": f"variant {tag}", "rationale": "test"}


def test_validator_enforces_count_distinctness_and_measured_exclusion_only():
    measured = {CASE.canonical_key(program(1).strip() + "\n")}
    rows, errors = validate_rows([row(1), row(2)], case=CASE, count=2, measured_keys=measured, measured_ids=set())
    assert not rows and errors[0]["code"] == "already_measured" and errors[0]["path"] == "/candidates/0/program"
    rows, errors = validate_rows([row(2), row(2)], case=CASE, count=2, measured_keys=set(), measured_ids=set())
    assert errors[0]["code"] == "duplicate_in_submission" and errors[0]["path"] == "/candidates/1/program"
    rows, errors = validate_rows([row(2)], case=CASE, count=2, measured_keys=set(), measured_ids=set())
    assert errors[0]["code"] == "wrong_candidate_count"
    rows, errors = validate_rows([{**row(3), "comparison_candidate_ids": ["rg-unknown"]}], case=CASE, count=1,
                                 measured_keys=set(), measured_ids={"rg-known"})
    assert errors[0]["code"] == "unknown_comparison_candidate"
    # A program proposed earlier but never evaluated is not excluded.
    rows, errors = validate_rows([row(2), row(3)], case=CASE, count=2, measured_keys=measured, measured_ids=set())
    assert not errors and len(rows) == 2


class Runtime:
    def __init__(self, tmp_path, limits=None):
        self.budget = BudgetLedger(limits=dict(limits or {}), path=tmp_path / "budget.json")

    def consume_many(self, amounts, *, usage_key=None):
        return self.budget.consume_many(amounts, usage_key=usage_key)


def args(**overrides):
    base = dict(max_repair_requests=2, llm_transport_retries=1, mock=True)
    return Namespace(**{**base, **overrides})


class ScriptedClient:
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []

    def propose(self, request):
        self.requests.append(request)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return ProposalResponse(text=json.dumps({"candidates": item}), usage={"input_tokens": 10, "output_tokens": 5})


def source(tmp_path, client, runtime=None, pricing=ApiPricing(), **overrides):
    result = DirectProgramSource(args=args(**overrides), case=CASE, client=client, run_dir=tmp_path, pricing=pricing)
    result.runtime = runtime
    return result


def test_direct_repair_replaces_only_invalid_entries_and_counts_every_request(tmp_path):
    bad = {**row(1), "program": "def nope():\n    pass\n"}
    client = ScriptedClient([[bad, row(2)], [row(1), row(2)], [row(3), row(4)]])
    runtime = Runtime(tmp_path)
    batch = source(tmp_path, client, runtime).collect(round_idx=0, count=4, batch_size=2, records=[],
                                                      measured_keys=set(), recovery_pass=0)
    assert len(batch.rows) == 4 and len(batch.attempts) == 3
    repair_message = client.requests[1].messages[-1]["content"]
    assert "/candidates/0/program" in repair_message and "missing_entry" in repair_message
    counters = runtime.budget.counters
    assert counters["llm_requests"] == 3 and counters["llm_input_tokens"] == 30 and counters["llm_output_tokens"] == 15


def test_exhausted_repair_pauses_with_all_attempts_instead_of_shrinking(tmp_path):
    bad = {**row(1), "program": "def nope():\n    pass\n"}
    client = ScriptedClient([[bad, row(2)]] * 3)
    with pytest.raises(ProposalExhausted) as raised:
        source(tmp_path, client, Runtime(tmp_path)).collect(round_idx=0, count=2, batch_size=2, records=[],
                                                             measured_keys=set(), recovery_pass=0)
    assert len(raised.value.attempts) == 3 and raised.value.metadata["errors"][0]["code"] == "missing_entry"


def test_failed_transport_attempts_are_counted_with_unknown_usage(tmp_path, monkeypatch):
    monkeypatch.setattr("tasks.researchgym.core.proposals.time.sleep", lambda seconds: None)
    runtime = Runtime(tmp_path)
    client = ScriptedClient([EndpointRequestError("503"), [row(1)]])
    batch = source(tmp_path, client, runtime).collect(round_idx=0, count=1, batch_size=1, records=[],
                                                      measured_keys=set(), recovery_pass=0)
    assert len(batch.rows) == 1
    counters = runtime.budget.counters
    assert counters["llm_requests"] == 2 and counters["llm_failed_requests"] == 1 and counters["llm_usage_unknown"] == 1
    client = ScriptedClient([EndpointRequestError("503")] * 2)
    with pytest.raises(EndpointRequestError):
        source(tmp_path / "second", client, Runtime(tmp_path / "second")).collect(
            round_idx=0, count=1, batch_size=1, records=[], measured_keys=set(), recovery_pass=0)


def test_resume_replays_cached_responses_and_never_reuses_an_in_flight_attempt(tmp_path):
    runtime = Runtime(tmp_path)
    first = source(tmp_path, ScriptedClient([[row(1)]]), runtime)
    first.collect(round_idx=0, count=1, batch_size=1, records=[], measured_keys=set(), recovery_pass=0)
    replay = source(tmp_path, ScriptedClient([]), runtime)
    assert replay.collect(round_idx=0, count=1, batch_size=1, records=[], measured_keys=set(), recovery_pass=0).rows
    assert runtime.budget.counters["llm_requests"] == 1
    # Simulate a process killed while its second-round request was in flight.
    marker = tmp_path / "proposal_requests" / "p000-r000001-b0000-a00.t0.pending"
    marker.write_text("{}")
    runtime.consume_many({"llm_requests": 1}, usage_key="direct:p000-r000001-b0000-a00:t0")
    resumed = source(tmp_path, ScriptedClient([[row(2)]]), runtime)
    resumed.collect(round_idx=1, count=1, batch_size=1, records=[], measured_keys=set(), recovery_pass=0)
    counters = runtime.budget.counters
    assert counters["llm_requests"] == 3 and counters["llm_usage_unknown"] == 1


def test_api_budget_stops_before_a_new_request(tmp_path):
    runtime = Runtime(tmp_path)
    pricing = ApiPricing(input_usd_per_mtok=1_000_000.0, output_usd_per_mtok=0.0, budget_usd=5.0)
    client = ScriptedClient([[row(1)], [row(2)]])
    direct = source(tmp_path, client, runtime, pricing=pricing)
    direct.collect(round_idx=0, count=1, batch_size=1, records=[], measured_keys=set(), recovery_pass=0)
    assert runtime.budget.counters["api_cost_usd"] == 10.0  # measured overshoot of the in-flight request
    with pytest.raises(ApiBudgetExhausted):
        direct.collect(round_idx=1, count=1, batch_size=1, records=[], measured_keys=set(), recovery_pass=0)
    assert len(client.requests) == 1


def test_q0_counts_occurrences_before_canonical_deduplication():
    domain = ProgramDomain(CASE)
    from ldm_tts.contracts import RawProposal
    proposals = [RawProposal({"program": program(t)}, "test", {"research_note": {"source": s}})
                 for t, s in ((1, "a"), (1, "b"), (2, "a"))]
    annotated = attach_empirical_base_measure(proposals, domain, measured_keys=set())
    records = [p.metadata[Q0_KEY] for p in annotated]
    assert [r["occurrence_count"] for r in records] == [2, 2, 1] and records[0]["valid_occurrence_count"] == 3
    assert [n["source"] for n in annotated[0].metadata["research_annotations"]] == ["a", "b"]
    candidates = {}
    for proposal in annotated:
        candidate = domain.admit(proposal)
        candidates.setdefault(candidate.canonical_key, candidate)
    assert sorted(empirical_base_masses(list(candidates.values())).tolist()) == pytest.approx([1 / 3, 2 / 3])


def test_mock_client_is_synthetic_and_passes_through_the_real_parser():
    client = MockProposalClient(CASE, seed=0)
    request = SimpleNamespace(metadata={"round_idx": 1, "minibatch_index": 0, "repair": 0, "count": 3},
                              messages=({"role": "user", "content": ""},))
    rows = json.loads(client.propose(request).text)["candidates"]
    parsed, errors = validate_rows(rows, case=CASE, count=3, measured_keys=set(), measured_ids=set())
    assert not errors and all("synthetic" in r["rationale"] for r in rows)
