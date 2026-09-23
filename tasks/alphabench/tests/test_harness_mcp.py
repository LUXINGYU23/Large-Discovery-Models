import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from ldm_tts.engine.expansion import ExpansionRequest
from ldm_tts.engine.run_store import CampaignRuntime
from tasks.alphabench.core.gateway import OracleGateway
from tasks.alphabench.core.harness_tools import HarnessToolService
from tasks.alphabench.core.protocol import T3Protocol


@pytest.mark.skipif(not os.environ.get("ALPHABENCH_T3_SIDECAR_IMAGE"),
    reason="requires the built Pi sidecar image")
def test_mcp_sidecar_calls_host_tools_without_guest_data_access(tmp_path):
    image = os.environ["ALPHABENCH_T3_SIDECAR_IMAGE"]
    protocol = T3Protocol(method="ldm_harness", harness_surrogate_query=True,
        budgets={**T3Protocol().budgets, "surrogate_queries": 400})
    runtime = CampaignRuntime.open(tmp_path / "run", task="alphabench", budget_limits={
        "dynamic_checks": 1, "oracle_job_slots": 2, "surrogate_queries": 1})
    gateway = OracleGateway(protocol, runtime, mock=True)
    tool_script = Path(__file__).resolve().parents[1] / "resources/harness/tools/t3-mcp.mjs"
    smoke_script = Path(__file__).with_name("t3_mcp_smoke.mjs")
    with HarnessToolService(protocol, gateway, Path(tempfile.gettempdir()), surrogate_query=True) as tools:
        def run():
            snapshot = tools.prepare(ExpansionRequest(0, 4))
            command = ["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
                "--mount", f"type=bind,src={tools.socket_dir},dst=/t3-host",
                "--mount", f"type=bind,src={tool_script},dst=/app/t3-mcp.mjs,readonly",
                "--mount", f"type=bind,src={smoke_script},dst=/app/t3-mcp-smoke.mjs,readonly",
                "--env", "LDM_T3_HOST_SOCKET=/t3-host/host.sock", "--entrypoint", "node", image,
                "/app/t3-mcp-smoke.mjs", snapshot["snapshot_id"]]
            result = subprocess.run(command, capture_output=True, text=True, timeout=90, check=True)
            tools.clear()
            return json.loads(result.stdout)
        result = gateway.host.run(run)
    assert result["contract"]["backend"] == "qlib"
    assert result["validated"]["canonical_expression"] == "Mean($close,5)"
    assert result["history"]["total"] == 0
    assert result["checked"]["success"] is True
    assert result["query"]["fit_status"] == "prior"
    assert runtime.budget.counters["dynamic_checks"] == 1
    assert runtime.budget.counters["surrogate_queries"] == 1
