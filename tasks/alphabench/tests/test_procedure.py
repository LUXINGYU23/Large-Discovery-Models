from tasks.alphabench.ldm_task.procedure import main


def test_mock_procedure(tmp_path, capsys) -> None:
    assert main([
        "--mock",
        "--iterations",
        "1",
        "--out-dir",
        str(tmp_path / "mock_run"),
    ]) == 0
    assert '"task": "alphabench"' in capsys.readouterr().out
    assert (tmp_path / "mock_run" / "events.jsonl").is_file()
    assert (tmp_path / "mock_run" / "summary.json").is_file()
