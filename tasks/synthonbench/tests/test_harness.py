"""File admission, measured feedback, and persistent sampling contracts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from contextlib import nullcontext
from dataclasses import replace
from itertools import product
from pathlib import Path

import pytest

from ldm_tts.contracts import Candidate, EvaluationResult, Observation, RawProposal
from ldm_tts.engine.expansion import ExpansionRequest
from ldm_tts.engine.run_store import BudgetLedger
from ldm_tts.harness import (
    HarnessError,
    HarnessSubmissionRequest,
    HarnessSubmittedArtifact,
    HarnessTurnResult,
    canonical_sha256,
)
from synthonbench.space import Synthon, SynthonSpace
from tasks.synthonbench.core.candidate import SynthonCandidateDomain
from tasks.synthonbench.core.constants import Q0_METADATA_KEY
from tasks.synthonbench.core.space_order import (
    ordered_positions,
    ordered_reactions,
    ordered_synthon_ids,
)
from tasks.synthonbench.core.harness import (
    HARNESS_PROFILE_IDS,
    HARNESS_SKILL_IDS,
    SynthonHarnessExpander,
    _validate_submission,
    direct_harness_profile,
    harness_profiles,
    write_harness_space_catalog,
)
from tasks.synthonbench.core.research import (
    MEASURED_HISTORY_FILE,
    write_measured_history,
)
from tasks.synthonbench.core.workflow import describe_ldm_task, main, parse_args
from tasks.synthonbench.core.workflow_args import validate_args


class FakeHarnessClient:
    def __init__(self, artifact_root, candidates_by_profile, *, replacements=None):
        self.artifact_root = artifact_root
        self.candidates_by_profile = candidates_by_profile
        self.replacements = replacements or {}
        self.batches = []
        self.rejections = {}

    def run_turn(self, turns, *, submission_validator, recovery_timeout_seconds):
        assert recovery_timeout_seconds == 3600
        self.batches.append(turns)
        results = []
        for turn in turns:
            submitted = _submission(
                self.artifact_root,
                {
                    "candidates": [
                        _annotated(item)
                        for item in self.candidates_by_profile[turn.profile_id]
                    ]
                },
                profile_id=turn.profile_id,
                turn_id=turn.turn_id,
            )
            validation = submission_validator(submitted)
            if validation.decision != "accept":
                self.rejections[turn.profile_id] = validation.errors
                submitted = _submission(
                    self.artifact_root,
                    {
                        "candidates": [
                            _annotated(item)
                            for item in self.replacements[turn.profile_id]
                        ]
                    },
                    profile_id=turn.profile_id,
                    turn_id=turn.turn_id,
                    attempt=2,
                )
                validation = submission_validator(submitted)
            assert validation.decision == "accept", validation.errors
            results.append(
                HarnessTurnResult(
                    profile_id=turn.profile_id,
                    session_id=f"session-{turn.profile_id}",
                    turn_id=turn.turn_id,
                    round_index=turn.round_index,
                    history_from_seq=turn.history_from_seq,
                    history_to_seq=turn.history_to_seq,
                    history_digest=turn.history_digest,
                    input_digest=turn.input_digest,
                    replayed=False,
                    submission_status="accepted",
                    submission_id=f"submission-{turn.profile_id}",
                    submission_digest=canonical_sha256(submitted.submission),
                    submission=submitted.submission,
                    submitted_artifacts=submitted.artifacts,
                    validation_errors=(),
                    usage={
                        "providerCalls": 2,
                        "toolCalls": {"web_search": 1, "submit_candidates": 1},
                        "artifactBytes": 50,
                    },
                    tool_budget={"web_search": {"limit": 4, "used": 1, "remaining": 3}},
                    artifacts={"turn": f"turns/{turn.turn_id}"},
                )
            )
        return tuple(results)


@pytest.fixture
def domain_and_payloads():
    space = SynthonSpace(
        [
            *(
                Synthon(index, 1, "r1", smiles)
                for index, smiles in enumerate(
                    ("CC", "CO", "CN", "CF", "CCl", "CBr"), start=1
                )
            ),
            *(
                Synthon(index, 2, "r1", smiles)
                for index, smiles in enumerate(("N", "O", "S"), start=11)
            ),
        ]
    )
    return SynthonCandidateDomain(space, ("r1",), "kif11"), [
        {"reaction_id": "r1", "synthon_ids": list(ids)}
        for ids in product(range(1, 7), range(11, 14))
    ]


def _annotated(payload):
    return {
        **payload,
        "change_summary": "Change one component relative to the measured control.",
        "rationale": "Test this interaction against the alternative chemical hypothesis.",
    }


def _submission(
    root, payload, *, profile_id=HARNESS_PROFILE_IDS[0], turn_id="turn-1", attempt=1
):
    body = json.dumps(payload).encode("utf-8")
    snapshot = root / profile_id / turn_id / str(attempt) / "snapshot.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(body)
    return HarnessSubmissionRequest(
        profile_id,
        turn_id,
        attempt,
        {"artifact_path": "candidates.json"},
        artifacts=(
            HarnessSubmittedArtifact(
                "/artifact_path",
                "candidates.json",
                snapshot.relative_to(root).as_posix(),
                hashlib.sha256(body).hexdigest(),
                len(body),
            ),
        ),
    )


def _expander(client, domain, *, per_profile=1, account=None):
    return SynthonHarnessExpander(
        client,
        domain,
        target="kif11",
        profiles=harness_profiles(),
        candidates_per_profile=per_profile,
        campaign_id="test-campaign",
        first_active_round=0,
        attach_empirical_q0=True,
        artifact_root=client.artifact_root,
        account=account,
    )


def _observation(domain, payload, *, round_idx, failed=False):
    candidate = domain.admit(
        RawProposal(
            payload,
            "test",
            {
                "research_annotations": [
                    {
                        "change_summary": "Measured control.",
                        "rationale": "Compare the alternative interaction.",
                    },
                ]
            },
        )
    )
    assert isinstance(candidate, Candidate)
    return Observation(
        candidate,
        EvaluationResult(
            candidate.candidate_id,
            "failed" if failed else "succeeded",
            {} if failed else {"synthon_utility": float(round_idx)},
        ),
        round_idx=round_idx,
    )


def test_independent_agreement_preserves_q0_and_all_research_notes(
    tmp_path, domain_and_payloads
):
    domain, payloads = domain_and_payloads
    indices = ((0, 1), (0, 2), (1, 2), (0, 2))
    client = FakeHarnessClient(
        tmp_path,
        {
            profile: [payloads[index] for index in pair]
            for profile, pair in zip(HARNESS_PROFILE_IDS, indices, strict=True)
        },
    )
    budget = BudgetLedger(limits={})
    result = _expander(client, domain, per_profile=2, account=budget.consume_many).expand(
        ExpansionRequest(round_idx=0, reservoir_size=8)
    )
    assert len(result.proposals) == 8
    assert budget.counters["proposal_attempts"] == budget.counters["harness_turns"] == 4
    assert budget.counters["llm_requests"] == 8
    assert budget.counters["harness_tool_calls"] == 8
    for proposal in result.proposals:
        candidate = domain.admit(proposal)
        assert isinstance(candidate, Candidate)
        occurrence_count = sum(
            item.payload == proposal.payload for item in result.proposals
        )
        assert candidate.metadata[Q0_METADATA_KEY]["probability"] == pytest.approx(
            occurrence_count / 8
        )
        notes = candidate.metadata["research_annotations"]
        assert len(notes) == occurrence_count
        assert len({note["profile_id"] for note in notes}) == occurrence_count
        assert (
            candidate.metadata["harness_lineage"]
            == proposal.metadata["harness_lineage"]
        )


def test_failed_usage_and_replayed_turns_are_accounted_once(
    tmp_path, domain_and_payloads, monkeypatch
):
    domain, payloads = domain_and_payloads
    client = FakeHarnessClient(
        tmp_path, {profile: [payloads[0]] for profile in HARNESS_PROFILE_IDS}
    )
    run_turn = client.run_turn
    budget_path = tmp_path / "budget.json"
    budget = BudgetLedger(
        limits={"proposal_attempts": 4, "harness_turns": 4, "llm_requests": 8},
        path=budget_path,
    )
    request = ExpansionRequest(round_idx=0, reservoir_size=4)

    def fail(*args, **kwargs):
        raise HarnessError("provider disconnected", turn_usage={
            HARNESS_PROFILE_IDS[0]: {
                "providerCalls": 1, "toolCalls": {"web_search": 1}, "artifactBytes": 20,
            },
            HARNESS_PROFILE_IDS[1]: {"validationSubmissions": 1},
        })

    monkeypatch.setattr(client, "run_turn", fail)
    with pytest.raises(HarnessError, match="provider disconnected"):
        _expander(client, domain, account=budget.consume_many).expand(request)
    assert budget.counters["llm_requests"] == 1
    assert budget.counters["harness_tool_calls"] == 1
    assert budget.counters["harness_artifact_bytes"] == 20
    assert budget.counters["harness_wall_time_seconds"] >= 0
    assert sum("llm_requests" in row for row in budget.metadata["cumulative_usage"].values()) == 1

    def replay(*args, **kwargs):
        return tuple(replace(result, replayed=True) for result in run_turn(*args, **kwargs))

    monkeypatch.setattr(client, "run_turn", replay)
    for _ in range(2):
        budget = BudgetLedger.load(budget_path)
        result = _expander(client, domain, account=budget.consume_many).expand(request)
        assert len(result.proposals) == 4
        assert budget.counters["proposal_attempts"] == budget.counters["harness_turns"] == 4
        assert budget.counters["llm_requests"] == 8
        assert budget.counters["harness_tool_calls"] == 8
        assert budget.counters["harness_artifact_bytes"] == 200


def test_compact_delta_and_measured_only_history(tmp_path, domain_and_payloads):
    domain, payloads = domain_and_payloads
    client = FakeHarnessClient(
        tmp_path, {profile: [payloads[3]] for profile in HARNESS_PROFILE_IDS}
    )
    expander = _expander(client, domain)
    history = (
        _observation(domain, payloads[0], round_idx=0),
        _observation(domain, payloads[1], round_idx=1),
        _observation(domain, payloads[2], round_idx=1, failed=True),
    )
    request = ExpansionRequest(round_idx=2, reservoir_size=4, observations=history)
    expander.expand(request)
    expander.expand(replace(request, round_idx=3))
    rows = json.loads((tmp_path / MEASURED_HISTORY_FILE).read_text())["observations"]
    assert len(rows) == 3
    assert rows[1]["research_annotations"]
    assert rows[2]["evaluation_status"] == "failed"
    assert rows[2]["synthon_utility"] is None
    for turn in client.batches[0]:
        message = json.loads(turn.message.split("\n\n", 1)[1])
        assert turn.history_from_seq == 1 and turn.history_to_seq == 3
        assert message["new_measured_observations"] == [
            {
                key: row[key]
                for key in ("candidate_id", "round_index", "synthon_utility")
            }
            for row in rows[1:]
        ]
        assert message["evaluated_candidates"]["count"] == 3
        assert message["evaluated_candidates"]["tool"] == "get_measured_history"
        assert message["novelty_contract"][
            "prior_unmeasured_submissions_may_be_reproposed"
        ]
        assert not message["novelty_contract"][
            "same_session_repeated_occurrences_are_allowed"
        ]


def test_historical_rejection_is_repaired_before_q0(tmp_path, domain_and_payloads):
    domain, payloads = domain_and_payloads
    first = HARNESS_PROFILE_IDS[0]
    client = FakeHarnessClient(
        tmp_path,
        {
            profile: [payloads[index]]
            for index, profile in enumerate(HARNESS_PROFILE_IDS)
        },
        replacements={first: [payloads[4]]},
    )
    result = _expander(client, domain).expand(
        ExpansionRequest(
            round_idx=1,
            reservoir_size=4,
            observations=(_observation(domain, payloads[0], round_idx=0),),
        )
    )
    assert client.rejections[first][0].code == "historical_duplicate"
    assert "already evaluated" in client.rejections[first][0].message
    assert len(result.proposals) == 4
    assert all(
        item.metadata[Q0_METADATA_KEY]["valid_occurrence_count"] == 4
        for item in result.proposals
    )


def test_rejection_reasons_identify_invalid_values_notes_and_duplicates(
    tmp_path, domain_and_payloads
):
    domain, payloads = domain_and_payloads
    invalid = json.loads(json.dumps(payloads[1]))
    invalid["synthon_ids"][0] = 999
    history = _observation(domain, payloads[0], round_idx=0, failed=True)
    rows = [
        _annotated(item)
        for item in (payloads[0], invalid, payloads[1], payloads[1], payloads[2])
    ]
    rows[-1]["rationale"] = " "
    validation = _validate_submission(
        _submission(tmp_path, {"candidates": rows}),
        domain,
        {history.canonical_key},
        measured_candidate_ids={history.candidate_id},
        artifact_root=tmp_path,
        candidate_count=5,
    )
    assert validation.decision == "retry"
    assert [error.code for error in validation.errors] == [
        "historical_duplicate",
        "invalid_candidate",
        "same_session_duplicate",
        "invalid_annotation",
    ]
    assert [error.path for error in validation.errors] == [
        "/candidates/0",
        "/candidates/1",
        "/candidates/3",
        "/candidates/4/rationale",
    ]
    assert all(error.message and error.hint for error in validation.errors)


@pytest.mark.parametrize(
    "payload", [[], {"candidates": []}, {"candidates": [], "notes": "extra"}]
)
def test_invalid_file_shape_and_count(tmp_path, domain_and_payloads, payload):
    domain, _ = domain_and_payloads
    validation = _validate_submission(
        _submission(tmp_path, payload),
        domain,
        set(),
        artifact_root=tmp_path,
        measured_candidate_ids=set(),
        candidate_count=1,
    )
    assert validation.decision == "retry"
    assert validation.errors[0].code == "invalid_candidate_file"


def test_comparison_reference_repair_and_annotation_preservation(tmp_path, domain_and_payloads):
    domain, payloads = domain_and_payloads
    measured = _observation(domain, payloads[0], round_idx=0)
    candidate = _annotated(payloads[1])
    candidate["comparison_candidate_ids"] = ["mistyped-id"]
    def validate():
        return _validate_submission(
            _submission(tmp_path, {"candidates": [candidate]}), domain, {measured.canonical_key},
            measured_candidate_ids={measured.candidate_id}, artifact_root=tmp_path, candidate_count=1,
        )
    rejected = validate()
    assert rejected.errors[0].code == "unknown_comparison_candidate"
    assert rejected.errors[0].path == "/candidates/0/comparison_candidate_ids/0"
    assert "mistyped-id" in rejected.errors[0].message
    candidate["comparison_candidate_ids"] = [measured.candidate_id]
    assert validate().decision == "accept"
    from tasks.synthonbench.core.harness import _submission_candidate
    prepared, annotation = _submission_candidate(candidate, domain)
    assert annotation["comparison_candidate_ids"] == [measured.candidate_id]
    assert "comparison_candidate_ids" not in prepared.payload
    candidate["comparison_candidate_ids"] = measured.candidate_id
    assert validate().errors[0].code == "invalid_comparison_reference"


def test_snapshot_admission_rejects_tampering(tmp_path, domain_and_payloads):
    domain, payloads = domain_and_payloads
    submitted = _submission(tmp_path, {"candidates": [_annotated(payloads[0])]})
    (tmp_path / "candidates.json").write_text("uncommitted workspace edit")

    def validate(value):
        return _validate_submission(
            value, domain, set(), measured_candidate_ids=set(), artifact_root=tmp_path, candidate_count=1
        )

    assert validate(submitted).decision == "accept"
    (tmp_path / "escaped").symlink_to(tmp_path.parent, target_is_directory=True)
    outside = replace(
        submitted,
        artifacts=(
            replace(submitted.artifacts[0], snapshot_path="escaped/outside.json"),
        ),
    )
    assert validate(outside).decision == "retry"
    (tmp_path / submitted.artifacts[0].snapshot_path).write_text("{}")
    assert "digest or size" in validate(submitted).errors[0].message


def test_defaults_load_independent_template_sessions_and_skills():
    profiles = harness_profiles()
    assert len({profile.profile_id for profile in profiles}) == 4
    assert len({profile.agents_path for profile in profiles}) == 1
    assert all(
        len(profile.skill_dirs) == len(HARNESS_SKILL_IDS) for profile in profiles
    )
    assert all(
        len(digest) == 64 for profile in profiles for digest in profile.skill_dir_sha256
    )
    args = parse_args(
        ["--mock", "--search-method", "ldm_harness", "--proposal-mode", "none"]
    )
    spec = describe_ldm_task(args)
    assert spec.proposal_search.parameters["skills_loaded"] is True
    assert spec.proposal_search.parameters["profile_count"] == 4
    assert spec.proposal_search.parameters["candidates_per_session"] == 16
    assert "web_search=8" in args.harness_tool_budget
    without_context7 = parse_args(["--mock", "--no-harness-context7"])
    assert all(
        "query-docs" not in value for value in without_context7.harness_tool_budget
    )
    invalid = parse_args(["--mock", "--harness-tool-budget", "submit_candidates=1"])
    with pytest.raises(SystemExit, match="submit_candidates"):
        validate_args(invalid)


def test_history_tool_reads_fresh_pages_and_checks_exact_membership(
    tmp_path, domain_and_payloads
):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to execute the Pi task extension")
    domain, payloads = domain_and_payloads
    catalog_path = tmp_path / "catalog.json"
    write_harness_space_catalog(domain.space, domain.allowed_reactions, catalog_path)
    assert "synthon_utility" not in catalog_path.read_text()
    write_measured_history(tmp_path, (), domain.space)
    history_path = tmp_path / MEASURED_HISTORY_FILE
    rows = []
    for index in range(3):
        observation = _observation(
            domain, payloads[index], round_idx=index, failed=index == 2
        )
        rows.append(
            {
                "candidate_id": observation.candidate_id,
                "round_index": index,
                **payloads[index],
                "synthon_utility": None if index == 2 else float(index),
                "evaluation_status": "failed" if index == 2 else "succeeded",
                "research_annotations": observation.candidate.metadata[
                    "research_annotations"
                ],
            }
        )
    extension = Path(__file__).parents[1] / "resources/harness/tools/synthon_space.mjs"
    script = """
import assert from 'node:assert/strict';
import {readFileSync, writeFileSync} from 'node:fs';
import {dirname} from 'node:path';
import {createHash} from 'node:crypto';
const {default: load} = await import(process.argv[1]);
const tools = new Map();
load({registerTool: tool => tools.set(tool.name, tool)});
const ctx = {cwd: dirname(process.env.LDM_SYNTHONBENCH_HISTORY)};
const read = async args => (await tools.get('get_measured_history').execute('test', args, undefined, undefined, ctx)).details;
const exported = result => {
  const body = readFileSync(result.guest_file.path.replace('/workspace', ctx.cwd));
  assert.equal(createHash('sha256').update(body).digest('hex'), result.guest_file.sha256);
  return JSON.parse(body);
};
const validate = async args => (await tools.get('validate_synthon_candidate').execute('test', args)).details;
const rows = JSON.parse(process.argv[2]);
const candidates = JSON.parse(process.argv[3]);
const search = async args => (await tools.get('search_synthon_space').execute('test', args, undefined, undefined, ctx)).details;
const reaction = JSON.parse(readFileSync(process.env.LDM_SYNTHON_SPACE_CATALOG, 'utf8')).reactions[0];
const space = await search({reaction_id: reaction.reaction_id, limit: 1});
assert.deepEqual(exported(space).slots, reaction.positions);
assert.equal(space.slots[0].synthons.length, 1);
const position = reaction.positions[0].position;
const query = String(reaction.positions[0].synthons[0].synthon_id);
const filtered = exported(await search({reaction_id: reaction.reaction_id, position, query}));
assert.deepEqual(filtered.slots, [{position, synthons: reaction.positions[0].synthons.filter(s => s.smiles.toLowerCase().includes(query) || String(s.synthon_id).includes(query))}]);
assert.equal((await read({})).total, 0);
assert.equal((await validate(candidates[0])).already_evaluated, false);
writeFileSync(process.env.LDM_SYNTHONBENCH_HISTORY, JSON.stringify({observations: rows}));
assert.equal((await read({limit: 1})).next_offset, 1);
assert.equal(exported(await read({limit: 1})).observations.length, rows.length);
assert.equal(exported(await read({})).complete_evaluated_history, true);
assert.equal(exported(await read({round_index: 0})).complete_evaluated_history, false);
assert.deepEqual(exported(await read({candidate_ids: [rows[0].candidate_id]})).observations, [rows[0]]);
assert.deepEqual((await read({offset: 1, limit: 1, response_format: 'detailed'})).observations, [rows[1]]);
assert.deepEqual((await read({round_index: 0, response_format: 'detailed'})).observations, [rows[0]]);
assert.equal((await read({candidate_ids: [rows[0].candidate_id]})).observations[0].research_annotations, undefined);
assert.deepEqual((await read({candidate_ids: ['unmeasured']})).unmeasured_or_unknown_ids, ['unmeasured']);
assert.deepEqual((await read({sort_by: 'utility_desc'})).observations.map(r => r.synthon_utility), [1, 0, null]);
assert.deepEqual((await read({sort_by: 'utility_asc'})).observations.map(r => r.synthon_utility), [0, 1, null]);
assert.equal((await validate(candidates[2])).already_evaluated, true);
assert.equal((await validate(candidates[2])).candidate_id, rows[2].candidate_id);
assert.equal((await validate(candidates[3])).already_evaluated, false);
"""
    subprocess.run(
        [
            node,
            "--input-type=module",
            "-e",
            script,
            extension.resolve().as_uri(),
            json.dumps(rows),
            json.dumps(payloads),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "LDM_SYNTHON_SPACE_CATALOG": str(catalog_path),
            "LDM_SYNTHONBENCH_HISTORY": str(history_path),
        },
    )


def test_mock_campaign_routes_harness_candidates_through_the_existing_engine(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    def fake_client(_args, _runtime, _provider, benchmark, _mcp):
        reaction_id = ordered_reactions(benchmark.task.allowed_reactions)[0]
        candidate = {
            "reaction_id": reaction_id,
            "synthon_ids": [
                ordered_synthon_ids(benchmark.task.space, reaction_id, position)[0]
                for position in ordered_positions(benchmark.task.space, reaction_id)
            ],
        }
        return nullcontext(
            FakeHarnessClient(
                _runtime.run_dir / "harness",
                {profile: [candidate] for profile in HARNESS_PROFILE_IDS},
            )
        )

    monkeypatch.setattr(
        "tasks.synthonbench.core.workflow._harness_client",
        fake_client,
    )

    assert (
        main(
            [
                "--mock",
                "--search-method",
                "ldm_harness",
                "--proposal-mode",
                "none",
                "--proposal-samples",
                "4",
                "--harness-candidates-per-session",
                "1",
                "--bo-pool-size",
                "2",
                "--iterations",
                "1",
                "--llm-url",
                "http://provider.invalid/v1",
                "--llm-model-name",
                "fake-model",
                "--api-key",
                "fake-key",
                "--out-dir",
                str(tmp_path),
                "--run-name",
                "harness_mock",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    run_dir = Path(payload["run_dir"])
    budget = json.loads((run_dir / "budget.json").read_text(encoding="utf-8"))
    assert budget["counters"]["proposal_attempts"] == 4
    assert budget["counters"]["harness_turns"] == 4
    assert budget["counters"]["llm_requests"] == 8
    assert budget["counters"]["harness_tool_calls"] == 8
    assert budget["counters"]["benchmark_jobs"] == 1
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    metadata = checkpoint["state"]["observations"][0]["candidate"]["metadata"]
    assert len(metadata["research_annotations"]) == 4
    measured = json.loads((run_dir / "harness" / MEASURED_HISTORY_FILE).read_text())[
        "observations"
    ]
    assert measured[0]["research_annotations"] == metadata["research_annotations"]
    lineage = metadata["harness_lineage"]
    assert set(lineage) == {
        "campaign_id",
        "round_index",
        "profile_id",
        "session_id",
        "turn_id",
        "submission_id",
        "item_index",
    }


def test_mock_direct_harness_evaluates_all_sixteen_submissions(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    submitted_payloads = []

    def fake_client(_args, _runtime, _provider, benchmark, _mcp):
        candidates = []
        for reaction_id in ordered_reactions(benchmark.task.allowed_reactions):
            choices = [
                ordered_synthon_ids(benchmark.task.space, reaction_id, position)
                for position in ordered_positions(benchmark.task.space, reaction_id)
            ]
            candidates.extend(
                {
                    "reaction_id": reaction_id,
                    "synthon_ids": list(synthon_ids),
                }
                for synthon_ids in product(*choices)
            )
            if len(candidates) >= 16:
                break
        submitted_payloads.extend(candidates[:16])
        return nullcontext(
            FakeHarnessClient(
                _runtime.run_dir / "harness",
                {direct_harness_profile()[0].profile_id: candidates[:16]},
            )
        )

    monkeypatch.setattr(
        "tasks.synthonbench.core.workflow._harness_client",
        fake_client,
    )

    assert (
        main(
            [
                "--mock",
                "--search-method",
                "harness",
                "--proposal-mode",
                "none",
                "--proposal-samples",
                "16",
                "--evaluations-per-round",
                "16",
                "--iterations",
                "1",
                "--llm-url",
                "http://provider.invalid/v1",
                "--llm-model-name",
                "fake-model",
                "--api-key",
                "fake-key",
                "--out-dir",
                str(tmp_path),
                "--run-name",
                "direct_harness_mock",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    run_dir = Path(payload["run_dir"])
    budget = json.loads((run_dir / "budget.json").read_text(encoding="utf-8"))
    counters = budget["counters"]
    assert counters["proposal_attempts"] == 1
    assert counters["harness_turns"] == 1
    assert counters["benchmark_jobs"] == 16
    assert counters["successful_evaluations"] == 16
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text())
    observations = checkpoint["state"]["observations"]
    assert [row["candidate"]["payload"] for row in observations] == submitted_payloads
    assert all(
        Q0_METADATA_KEY not in row["candidate"]["metadata"] for row in observations
    )
    assert all(row["surrogate"] is None for row in observations)
