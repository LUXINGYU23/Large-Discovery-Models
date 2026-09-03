"""Provenance requirements for archived pilot-evaluation releases."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ldm_tts.pilot_evaluation import execution


def test_repository_state_uses_explicit_archive_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_git(*_args: object, **_kwargs: object) -> str:
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(execution, "_git", no_git)
    monkeypatch.setenv("LDM_PILOT_EVALUATION_COMMIT", "archive-commit")

    assert execution._repository_state() == {
        "commit": "archive-commit",
        "dirty": False,
        "source": "environment",
    }


def test_repository_state_rejects_unversioned_archive(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_git(*_args: object, **_kwargs: object) -> str:
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(execution, "_git", no_git)
    monkeypatch.delenv("LDM_PILOT_EVALUATION_COMMIT", raising=False)

    with pytest.raises(RuntimeError, match="LDM_PILOT_EVALUATION_COMMIT"):
        execution._repository_state()


def test_harness_provenance_keeps_only_redacted_release_fields(tmp_path: Path) -> None:
    run_dir = tmp_path / "campaigns" / "case" / "harness" / "seed_0"
    manifest_path = run_dir / "harness" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "backend": "pi",
                "baseUrl": "https://provider.example/v1",
                "model": "research-model",
                "thinking": "max",
                "profiles": [{"profileId": "direct_research"}],
                "mcpServers": [
                    {"serverId": "literature", "secretSources": []}
                ],
                "limits": {"toolCallBudgets": {"web_search": 4}},
                "internal_secret": "must-not-be-copied",
            }
        ),
        encoding="utf-8",
    )

    provenance = execution._harness_manifest_provenance(manifest_path, tmp_path)

    assert provenance["model"] == "research-model"
    assert provenance["profiles"] == [{"profileId": "direct_research"}]
    assert provenance["mcp_servers"] == [
        {"serverId": "literature", "secretSources": []}
    ]
    assert provenance["tool_call_budgets"] == {"web_search": 4}
    assert "must-not-be-copied" not in repr(provenance)


def test_compiled_harness_provenance_validates_the_dual_pool_identity(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "campaigns" / "case" / "ldm_harness_compiled" / "seed_0"
    _write_harness_manifest(
        run_dir / "harness" / "manifest.json",
        case_id="case-1",
        profiles=4,
        contract_id="candidates",
    )
    _write_harness_manifest(
        run_dir / "policy_harness" / "manifest.json",
        case_id="case-1:optimization_policy",
        profiles=1,
        contract_id="optimization_policy",
    )

    provenance = execution._compiled_harness_provenance(run_dir, tmp_path)

    assert provenance["topology"] == "dual_pool"
    assert provenance["proposal"]["case_id"] == "case-1"
    assert provenance["policy"]["case_id"] == "case-1:optimization_policy"
    assert provenance["proposal"]["wire_api"] == "responses"
    assert provenance["policy"]["submission_contract"]["contractId"] == (
        "optimization_policy"
    )


def test_compiled_harness_provenance_rejects_mismatched_pool_identity(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "campaigns" / "case" / "ldm_harness_compiled" / "seed_0"
    _write_harness_manifest(
        run_dir / "harness" / "manifest.json",
        case_id="case-1",
        profiles=4,
        contract_id="candidates",
    )
    _write_harness_manifest(
        run_dir / "policy_harness" / "manifest.json",
        case_id="wrong:optimization_policy",
        profiles=1,
        contract_id="optimization_policy",
    )

    with pytest.raises(ValueError, match="policy case identity"):
        execution._compiled_harness_provenance(run_dir, tmp_path)


def _write_harness_manifest(
    path: Path,
    *,
    case_id: str,
    profiles: int,
    contract_id: str,
) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "protocolVersion": "1.2.3",
                "campaignId": "campaign-1",
                "taskId": "fixture",
                "caseId": case_id,
                "seed": 0,
                "backend": "pi",
                "baseUrl": "https://provider.example/v1",
                "wireApi": "responses",
                "model": "research-model",
                "thinking": "max",
                "profileSetSha256": "a" * 64,
                "profiles": [
                    {"profileId": f"profile-{index}"} for index in range(profiles)
                ],
                "guestRuntime": {
                    "imageRef": "fixture:latest",
                    "manifestSha256": "b" * 64,
                },
                "submissionContract": {"contractId": contract_id},
                "mcpServers": [],
                "limits": {"toolCallBudgets": {"web_search": 8}},
            }
        ),
        encoding="utf-8",
    )
