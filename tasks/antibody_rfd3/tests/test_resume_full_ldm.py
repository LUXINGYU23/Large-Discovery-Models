import json
import sqlite3

from tasks.antibody_rfd3.repro.h20.resume_full_ldm import backup_database, resume_case


def test_resume_keeps_initial_command_and_does_not_mutate_original(tmp_path):
    original = {"case": {"command": ["python", "scripts/run_ldm_tts.py", "original.yaml"],
                         "effective_args": {"iterations": 640}, "exit_code": 1, "finished_at": "before"}}
    frozen = json.dumps(original)
    first = resume_case(original, tmp_path / "campaign", tmp_path / "attempt-001")
    assert json.dumps(original) == frozen
    assert first["effective_args"]["iterations"] == 640
    assert first["effective_args"]["resume_run"] == str(tmp_path / "campaign")
    assert "exit_code" not in first and "finished_at" not in first
    next_manifest = {"initial_command": original["case"]["command"], "case": first}
    second = resume_case(next_manifest, tmp_path / "campaign", tmp_path / "attempt-002")
    assert second["command"] == first["command"]
    assert second["log"] != first["log"]


def test_backup_includes_committed_wal_without_modifying_source(tmp_path):
    source, target = tmp_path / "live.sqlite3", tmp_path / "snapshot.sqlite3"
    connection = sqlite3.connect(source)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE evaluations(id INTEGER PRIMARY KEY, energy REAL)")
        connection.execute("INSERT INTO evaluations VALUES(1, -100)")
        connection.commit()
        backup_database(source.resolve(), target)
        connection.execute("INSERT INTO evaluations VALUES(2, -101)")
        connection.commit()
        backup = sqlite3.connect(target)
        try:
            assert backup.execute("SELECT * FROM evaluations").fetchall() == [(1, -100.0)]
            assert connection.execute("SELECT count(*) FROM evaluations").fetchone()[0] == 2
        finally:
            backup.close()
    finally:
        connection.close()
