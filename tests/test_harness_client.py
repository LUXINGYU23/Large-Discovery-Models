from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from ldm_tts.harness.pi import PiGuestRuntime, PiHarnessConfig
from ldm_tts.harness import (
    HarnessClient,
    HarnessPoolConfig,
    HarnessError,
    HarnessProfile,
    HarnessSubmissionContract,
    HarnessSubmissionError,
    HarnessSubmissionValidation,
    HarnessSubmittedArtifact,
    HarnessTurn,
    canonical_sha256,
    directory_sha256,
    file_sha256,
    parse_tool_call_budgets,
)


TEST_GUEST_RUNTIME = PiGuestRuntime(
    image_ref="ldm/fixture-research:aaaaaaaaaaaa",
    recipe_sha256="a" * 64,
    rootfs_size="4G",
    install_policy="session_overlay",
)


def test_generic_pool_configuration_does_not_require_pi_provider_or_guest(tmp_path):
    config = HarnessPoolConfig(
        artifact_root=tmp_path,
        profiles=(HarnessProfile("research", Path("/resources/AGENTS.md"), agents_sha256="a" * 64),),
        campaign_id="campaign", task_id="fixture", case_id="case", seed=0,
        submission_contract=_candidate_contract(),
    )
    payload = config.initialize_payload()
    assert payload["taskId"] == "fixture"
    assert not {"model", "baseUrl", "wireApi", "guestRuntime", "mcpServers"} & payload.keys()


def _candidate_contract(
    *, max_validation_attempts: int | None = None
) -> HarnessSubmissionContract:
    candidate_schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    return HarnessSubmissionContract(
        contract_id="candidate_batch",
        tool_name="submit_candidates",
        payload_schema={
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": candidate_schema,
                    "minItems": 1,
                    "maxItems": 1,
                },
            },
            "required": ["candidates"],
            "additionalProperties": False,
        },
        max_validation_attempts=max_validation_attempts,
    )


def test_submission_contract_digest_covers_transmitted_json(tmp_path: Path) -> None:
    contract = _candidate_contract()
    config = PiHarnessConfig(
        artifact_root=tmp_path,
        base_url="https://provider.example/v1",
        model="test-model",
        profiles=(HarnessProfile(
            "chemist",
            Path("/resources/AGENTS.md"),
            agents_sha256="a" * 64,
        ),),
        campaign_id="campaign-1",
        task_id="fixture",
        case_id="case-1",
        seed=1,
        submission_contract=contract,
        guest_runtime=TEST_GUEST_RUNTIME,
    )

    frame = config.initialize_payload()

    contract_json = frame["submissionContractJson"]
    assert isinstance(contract_json, str)
    assert json.loads(contract_json) == contract.to_dict()
    assert frame["submissionContractSha256"] == hashlib.sha256(
        contract_json.encode("utf-8")
    ).hexdigest()
    assert "candidateSchemaJson" not in frame
    assert "protocolVersion" not in frame
    assert "requestId" not in frame
    assert frame["guestRuntime"] == TEST_GUEST_RUNTIME.to_dict()
    assert frame["webSearch"] == {
        "providers": ["parallel-mcp", "exa", "duckduckgo"],
        "fallbackOn": [
            "transient",
            "quota",
            "network",
            "invalid-response",
            "unsupported",
        ],
    }
    assert "webProvider" not in frame
    assert frame["mcpServers"] == []
    assert frame["limits"] == {"wallTimeSeconds": 1800, "toolCallBudgets": {}}


def test_directory_digest_uses_stable_utf8_filename_order(tmp_path: Path) -> None:
    skill_root = tmp_path / "skill"
    (skill_root / "agents").mkdir(parents=True)
    (skill_root / "references").mkdir()
    (skill_root / "SKILL.md").write_text("skill", encoding="utf-8")
    (skill_root / "agents" / "openai.yaml").write_text("agent", encoding="utf-8")
    (skill_root / "references" / "notes.md").write_text("notes", encoding="utf-8")

    expected = canonical_sha256(
        [
            {
                "path": "SKILL.md",
                "sha256": file_sha256(skill_root / "SKILL.md"),
            },
            {
                "path": "agents/openai.yaml",
                "sha256": file_sha256(skill_root / "agents" / "openai.yaml"),
            },
            {
                "path": "references/notes.md",
                "sha256": file_sha256(skill_root / "references" / "notes.md"),
            },
        ]
    )

    assert directory_sha256(skill_root) == expected


def test_tool_call_budget_parser_rejects_duplicates_and_terminal_limits() -> None:
    assert parse_tool_call_budgets(("web_search=4", "mcp__literature__search=0")) == {
        "mcp__literature__search": 0,
        "web_search": 4,
    }
    with pytest.raises(ValueError, match="duplicate"):
        parse_tool_call_budgets(("web_search=4", "web_search=2"))
    with pytest.raises(ValueError, match="submit_candidates"):
        parse_tool_call_budgets(
            ("submit_candidates=1",),
            excluded_tools=("submit_candidates",),
        )


def test_submitted_artifact_requires_safe_digest_bound_paths() -> None:
    artifact = HarnessSubmittedArtifact(
        path_pointer="/artifact_path",
        relative_path="optimization_policy.py",
        snapshot_path="turns/turn-1/attempts/0001/artifacts/artifact-00.py",
        sha256="a" * 64,
        size_bytes=12,
    )
    assert artifact.to_dict()["snapshotPath"].endswith("artifact-00.py")
    with pytest.raises(ValueError, match="safe POSIX relative path"):
        HarnessSubmittedArtifact(
            path_pointer="/artifact_path",
            relative_path="../escape.py",
            snapshot_path="turns/turn-1/artifact.py",
            sha256="a" * 64,
            size_bytes=12,
        )


def test_persistent_harness_client_runs_one_profile_batch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "fake_harness_sidecar.py"
    config = HarnessPoolConfig(
        artifact_root=tmp_path,
        profiles=(HarnessProfile(
            "chemist",
            Path("/resources/AGENTS.md"),
            agents_sha256="a" * 64,
        ),),
        campaign_id="campaign-1",
        task_id="fixture",
        case_id="case-1",
        seed=1,
        submission_contract=_candidate_contract(),
    )
    monkeypatch.setenv("HARNESS_TEST_SECRET", "test-secret")
    monkeypatch.setenv("HARNESS_MCP_SECRET", "mcp-secret")
    client = HarnessClient(
        (sys.executable, "-u", str(fixture)),
        api_key="test-secret",
        config=config,
        named_secrets={"mcp.fixture.env.token": "mcp-secret"},
        response_timeout_seconds=5,
    )
    turn = HarnessTurn(
        profile_id="chemist",
        turn_id="round_0_chemist",
        round_index=0,
        history_from_seq=0,
        history_to_seq=0,
        history_digest="c" * 64,
        message="research",
    )

    with client:
        result = client.run_turn(
            (turn,),
            submission_validator=lambda _request: HarnessSubmissionValidation(),
        )

    assert result[0].session_id == "session-chemist"
    assert result[0].input_digest == turn.input_digest
    assert result[0].submission == {"candidates": [{"value": "chemist"}]}
    assert result[0].submission_status == "accepted"
    assert result[0].usage["providerCalls"] == 1
    assert result[0].usage["toolCalls"] == {}
    assert result[0].usage["validationSubmissions"] == 1
    assert result[0].tool_budget == {}


@pytest.mark.parametrize("failure", ["measured", "unknown", "wrong-turn"])
def test_failed_turn_preserves_known_usage_and_checks_identity(tmp_path, monkeypatch, failure):
    monkeypatch.setenv("HARNESS_TEST_TURN_FAILURE", failure)
    config = HarnessPoolConfig(
        artifact_root=tmp_path,
        profiles=(HarnessProfile("research", Path("/resources/AGENTS.md"), agents_sha256="a" * 64),),
        campaign_id="campaign", task_id="fixture", case_id="case", seed=0,
        submission_contract=_candidate_contract(),
    )
    turn = HarnessTurn(
        profile_id="research", turn_id="turn-1", round_index=1,
        history_from_seq=0, history_to_seq=1, history_digest="c" * 64, message="research",
    )
    with HarnessClient(
        (sys.executable, "-u", str(Path(__file__).parent / "fixtures/fake_harness_sidecar.py")),
        api_key="fixture", config=config, response_timeout_seconds=5,
    ) as client:
        with pytest.raises(HarnessError) as caught:
            client.run_turn((turn,), submission_validator=lambda _: HarnessSubmissionValidation())
    usage = caught.value.turn_usage["research"]
    assert usage["validationSubmissions"] == 0
    if failure == "measured":
        assert usage == {"providerCalls": 3, "toolCalls": {"bash": 2}, "artifactBytes": 120, "validationSubmissions": 0}
    else:
        assert "providerCalls" not in usage
    assert str(caught.value) == (
        "harness error usage does not match the requested turn" if failure == "wrong-turn" else "provider 502"
    )


@pytest.mark.parametrize(
    "environment_variable",
    ("HARNESS_TEST_SKIP_VALIDATION", "HARNESS_TEST_CHANGE_AFTER_VALIDATION"),
)
def test_persistent_harness_client_rejects_unvalidated_submission(
    tmp_path: Path,
    monkeypatch,
    environment_variable: str,
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "fake_harness_sidecar.py"
    config = PiHarnessConfig(
        artifact_root=tmp_path,
        base_url="https://provider.example/v1",
        model="test-model",
        profiles=(HarnessProfile(
            "chemist",
            Path("/resources/AGENTS.md"),
            agents_sha256="a" * 64,
        ),),
        campaign_id="campaign-1",
        task_id="fixture",
        case_id="case-1",
        seed=1,
        submission_contract=_candidate_contract(),
        guest_runtime=TEST_GUEST_RUNTIME,
    )
    monkeypatch.setenv(environment_variable, "1")
    client = HarnessClient(
        (sys.executable, "-u", str(fixture)),
        api_key="test-secret",
        config=config,
        response_timeout_seconds=5,
    )
    turn = HarnessTurn(
        profile_id="chemist",
        turn_id="round_0_chemist",
        round_index=0,
        history_from_seq=0,
        history_to_seq=0,
        history_digest="c" * 64,
        message="research",
    )

    with client, pytest.raises(
        HarnessError,
        match="matching task validation|submissionJson does not match",
    ):
        client.run_turn(
            (turn,),
            submission_validator=lambda _request: HarnessSubmissionValidation(),
        )


def test_submission_retry_preserves_the_turn_until_acceptance(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "fake_harness_sidecar.py"
    config = PiHarnessConfig(
        artifact_root=tmp_path,
        base_url="https://provider.example/v1",
        model="test-model",
        profiles=(HarnessProfile(
            "chemist",
            Path("/resources/AGENTS.md"),
            agents_sha256="a" * 64,
        ),),
        campaign_id="campaign-1",
        task_id="fixture",
        case_id="case-1",
        seed=1,
        submission_contract=_candidate_contract(),
        guest_runtime=TEST_GUEST_RUNTIME,
    )
    attempts: list[int] = []

    def validate(request):
        attempts.append(request.attempt_index)
        if request.attempt_index == 1:
            return HarnessSubmissionValidation("retry", (
                HarnessSubmissionError(
                    "/candidates/0",
                    "historical_duplicate",
                    "The candidate was already evaluated.",
                    "Replace this entry.",
                ),
            ))
        return HarnessSubmissionValidation()

    turn = HarnessTurn("chemist", "round_0_chemist", 0, 0, 0, "c" * 64, "research")
    with HarnessClient(
        (sys.executable, "-u", str(fixture)),
        api_key="test-secret",
        config=config,
        response_timeout_seconds=5,
    ) as client:
        result = client.run_turn((turn,), submission_validator=validate)

    assert attempts == [1, 2]
    assert result[0].submission == {"candidates": [{"value": "chemist-2"}]}


def test_maximum_validation_attempts_rejects_the_turn(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "fake_harness_sidecar.py"
    error = HarnessSubmissionError(
        "/artifact_path",
        "invalid_artifact",
        "The artifact is invalid.",
        "Repair the file.",
    )
    config = PiHarnessConfig(
        artifact_root=tmp_path,
        base_url="https://provider.example/v1",
        model="test-model",
        profiles=(HarnessProfile(
            "architect",
            Path("/resources/AGENTS.md"),
            agents_sha256="a" * 64,
        ),),
        campaign_id="campaign-1",
        task_id="fixture",
        case_id="case-1",
        seed=1,
        submission_contract=_candidate_contract(max_validation_attempts=3),
        guest_runtime=TEST_GUEST_RUNTIME,
    )
    turn = HarnessTurn("architect", "round_0_architect", 0, 0, 0, "c" * 64, "research")
    attempts: list[int] = []

    def validate(request):
        attempts.append(request.attempt_index)
        return HarnessSubmissionValidation("retry", (error,))

    with HarnessClient(
        (sys.executable, "-u", str(fixture)),
        api_key="test-secret",
        config=config,
        response_timeout_seconds=5,
    ) as client:
        result = client.run_turn((turn,), submission_validator=validate)

    assert attempts == [1, 2, 3]
    assert result[0].submission_status == "rejected"
    assert result[0].validation_errors == (error,)
