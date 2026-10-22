import pytest

from tasks.antibody.core import workflow
from tasks.antibody.core.ldm_light import ldm_acq


@pytest.mark.parametrize("interrupt", [False, True])
def test_mock_client_is_scoped_to_each_run(tmp_path, monkeypatch, interrupt):
    factory = ldm_acq.make_llm_client
    clients = []

    def run_one(config, antigen, seed, args, *, llm=None):
        assert ldm_acq.make_llm_client is factory
        clients.append(llm)
        if interrupt and len(clients) == 1:
            raise RuntimeError("interrupted mock run")
        return tmp_path / f"run-{len(clients)}"

    monkeypatch.setattr(ldm_acq, "run_one", run_one)
    shared = ["--antigen", "FIXTURE", "--out-dir", str(tmp_path), "--n-trials", "1"]
    if interrupt:
        with pytest.raises(RuntimeError, match="interrupted mock run"):
            workflow.run(workflow.parse_args(shared + ["--mock"]))
    else:
        workflow.run(workflow.parse_args(shared + ["--mock"]))
    for flags in ([], ["--mock"], []):
        workflow.run(workflow.parse_args(shared + flags))
    assert isinstance(clients[0], workflow.DeterministicAntBOTTSLLM)
    assert isinstance(clients[2], workflow.DeterministicAntBOTTSLLM)
    assert clients[0] is not clients[2]
    assert clients[1] is clients[3] is None
    assert ldm_acq.make_llm_client is factory
