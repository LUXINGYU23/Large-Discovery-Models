"""Admission checks run in trusted code without importing submitted programs."""

import hashlib
from types import SimpleNamespace

import pytest

from tasks.copper_inversion.core.harness import parse_snapshot
from tasks.copper_inversion.core.candidate import Domain, seal
from ldm_tts.contracts import RawProposal, CandidateRejection


def test_snapshot_contract(tmp_path):
    source = (
        b'def solve(problem, budget, seed):\n    return problem["initial_level_set"]\n'
    )
    (tmp_path / "solver.py").write_bytes(source)
    sha = hashlib.sha256(source).hexdigest()
    artifact = SimpleNamespace(
        snapshot_path="solver.py",
        relative_path="candidates/0/solver.py",
        path_pointer="/artifact_path",
        sha256=sha,
        size_bytes=len(source),
    )
    submission = dict(
        artifact_path="candidates/0/solver.py",
        parent_sha256=None,
        change_summary="test",
        rationale="test",
    )
    assert parse_snapshot(submission, (artifact,), tmp_path, set()) == source
    with pytest.raises(ValueError, match="already evaluated"):
        parse_snapshot(submission, (artifact,), tmp_path, {sha})
    with pytest.raises(ValueError, match="unknown parent"):
        parse_snapshot(
            {**submission, "parent_sha256": "unknown"}, (artifact,), tmp_path, set()
        )
    (tmp_path / "solver.py").write_bytes(source + b"# tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        parse_snapshot(submission, (artifact,), tmp_path, set())


def test_domain_rejects_link_and_tampering(tmp_path):
    payload = seal(tmp_path, b"def solve(problem,budget,seed): return 0\n")
    link = tmp_path / "link.py"
    link.symlink_to(tmp_path / payload["source_path"])
    result = Domain(tmp_path).admit(
        RawProposal({**payload, "source_path": "link.py"}, "test")
    )
    assert isinstance(result, CandidateRejection)
    result = Domain(tmp_path).admit(RawProposal({**payload, "sha256": "bad"}, "test"))
    assert isinstance(result, CandidateRejection)


def test_docker_cli_timeout_pauses_unknown_outcome(monkeypatch):
    import subprocess
    from tasks.copper_inversion.core import execution
    from ldm_tts.contracts.evaluation import EvaluationPaused

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('docker', 40)

    monkeypatch.setattr(execution.subprocess, 'run', timeout)
    with pytest.raises(EvaluationPaused, match='reconcile'):
        execution.docker('start', 'test-job')


def test_preflight_receipt_replays_usage_without_provider(tmp_path):
    import json
    from tasks.copper_inversion.core.harness_runtime import _preflight

    (tmp_path / 'preflight.json').write_text(json.dumps({'usage': {'providerCalls': 3}}))
    calls = []
    _preflight(None, tmp_path, None, '', lambda counts, **kw: calls.append((counts, kw)))
    assert calls[0][0]['llm_requests'] == 3
    assert calls[0][1]['usage_key'] == 'copper:preflight:usage'


def test_default_provider_is_official_deepseek_low():
    from tasks.copper_inversion.core.workflow import parse_args

    args = parse_args([])
    assert args.llm_url == "https://api.deepseek.com"
    assert args.llm_model == "deepseek-flash"
    assert args.thinking == "low"


def test_research_uses_shared_bash_budget(tmp_path, monkeypatch):
    from contextlib import ExitStack
    from tasks.copper_inversion.core import harness_runtime as h
    from tasks.copper_inversion.core.workflow import parse_args
    from tasks.copper_inversion.core.data import load_problem

    captured = []
    class Client:
        def __init__(self, *args, **kwargs):
            captured.append(kwargs["config"])
        def start(self):
            pass
        def close(self):
            pass
    monkeypatch.setenv("LLM_API_KEY", "test-only-key")
    monkeypatch.setattr(h, "HarnessClient", Client)
    monkeypatch.setattr(h, "_preflight", lambda *args: None)
    with ExitStack() as stack:
        h.open_client(parse_args([]), tmp_path, load_problem("drillholes_sketch_2d", fixture=True), stack, lambda *a, **k: None)
    assert captured[0].limits.tool_call_budgets["bash"] == 8
    assert captured[0].thinking == "low"
    assert captured[0].force_first_tool_call is False
