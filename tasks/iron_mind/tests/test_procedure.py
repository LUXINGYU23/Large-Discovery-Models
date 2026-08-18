from tasks.iron_mind.ldm_task.procedure import main, parse_args


def test_mock_procedure(tmp_path, capsys) -> None:
    assert parse_args(["--mock", "--iterations", "0"]).iterations == 0
    assert main([
        "--mock",
        "--iterations",
        "1",
        "--out-dir",
        str(tmp_path / "mock_run"),
    ]) == 0
    assert '"task": "iron_mind"' in capsys.readouterr().out
    assert (tmp_path / "mock_run" / "events.jsonl").is_file()
    assert (tmp_path / "mock_run" / "summary.json").is_file()
