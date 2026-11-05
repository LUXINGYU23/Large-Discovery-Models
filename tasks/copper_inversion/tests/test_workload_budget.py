import pytest
from tasks.copper_inversion.core import runner


def test_remaining_budget_deducts_boot_and_import_time(monkeypatch):
    monkeypatch.setattr(runner.time, 'monotonic', lambda: 102.25)
    assert runner.remaining_seconds(5, 105) == 2.75
    assert runner.remaining_seconds(120, 105) == 2.75
    assert runner.remaining_seconds(1, 105) == 1


def test_exhausted_budget_does_not_start_candidate(monkeypatch):
    monkeypatch.setattr(runner.time, 'monotonic', lambda: 106)
    with pytest.raises(TimeoutError):
        runner.remaining_seconds(5, 105)
