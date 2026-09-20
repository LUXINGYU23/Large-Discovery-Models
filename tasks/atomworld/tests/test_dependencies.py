from tasks.atomworld.ldm_task import dependencies


def by_name(checks):
    return {check.name: check for check in checks}


def test_mock_direct_dependency_check_does_not_require_provider():
    checks = by_name(
        dependencies.check_dependencies(
            {"task": "atomworld", "mode": "mock", "argv": ["--mock"]}
        )
    )

    assert checks["mock"].status == "ok"
    assert checks["proposal_provider"].status == "ok"
    assert "LLM URL" not in checks


def test_mock_harness_dependency_check_requires_provider_settings(tmp_path, monkeypatch):
    for name in (
        "LLM_BASE_URL",
        "LDM_LLM_URL",
        "OPENAI_BASE_URL",
        "LLM_MODEL_NAME",
        "LDM_LLM_MODEL",
        "OPENAI_MODEL",
        "LLM_API_KEY",
        "LDM_LLM_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(dependencies.shutil, "which", lambda name: "/usr/bin/docker")

    checks = by_name(
        dependencies.check_dependencies(
            {
                "task": "atomworld",
                "mode": "mock",
                "argv": ["--mock", "--search-method", "harness"],
                "cwd": str(tmp_path),
            }
        )
    )

    assert checks["mock"].status == "ok"
    assert checks["Harness Docker client"].status == "ok"
    assert checks["LLM URL"].status == "fail"
    assert checks["LLM model"].status == "fail"
    assert checks["LLM API key"].status == "fail"
    assert "proposal_provider" not in checks


def test_mock_harness_dependency_check_accepts_masked_provider_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(dependencies.shutil, "which", lambda name: "/usr/bin/docker")
    checks = by_name(
        dependencies.check_dependencies(
            {
                "task": "atomworld",
                "mode": "mock",
                "argv": [
                    "--mock",
                    "--search-method",
                    "harness_public_audit",
                    "--llm-url",
                    "https://provider.example/v1",
                    "--llm-model-name",
                    "model",
                ],
                "env_overrides": {"LLM_API_KEY": "secret-value"},
                "cwd": str(tmp_path),
            }
        )
    )

    assert checks["LLM URL"].status == "ok"
    assert checks["LLM model"].status == "ok"
    assert checks["LLM API key"].status == "ok"
    assert checks["LLM API key"].detail == "***"
