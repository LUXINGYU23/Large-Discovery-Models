"""Provenance requirements for archived pilot-evaluation releases."""

from __future__ import annotations

import subprocess

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
