import json
import os
import subprocess
from pathlib import Path

import pytest

from tasks.alphabench.core.harness_runtime import runtime_root


@pytest.mark.skipif(not os.environ.get("ALPHABENCH_T3_SIDECAR_IMAGE"),
    reason="requires the built Pi sidecar and task guest")
def test_task_guest_cannot_reach_host_private_files_or_tools(tmp_path):
    artifact_root = tmp_path / "artifacts"
    private = artifact_root / "private"
    resources = artifact_root / "sessions/isolation/workspace/.ldm-resources"
    private.mkdir(parents=True)
    resources.mkdir(parents=True)
    (private / "secret.canary").write_text("host_only\n")
    task_root = Path(__file__).resolve().parents[1]
    (resources / "AGENTS.md").write_bytes((task_root / "resources/harness/profiles/research/AGENTS.md").read_bytes())
    smoke_script = Path(__file__).with_name("t3_guest_isolation.mjs")
    cache = runtime_root() / "cache"
    command = ["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
        "--group-add", str(os.stat("/dev/kvm").st_gid), "--device", "/dev/kvm",
        "--mount", f"type=bind,src={artifact_root},dst=/artifacts",
        "--mount", f"type=bind,src={task_root},dst=/tasks/alphabench,readonly",
        "--mount", f"type=bind,src={cache},dst=/runtime-home/.cache/gondolin",
        "--mount", f"type=bind,src={smoke_script},dst=/app/t3-guest-isolation.mjs,readonly",
        "--entrypoint", "node", os.environ["ALPHABENCH_T3_SIDECAR_IMAGE"],
        "/app/t3-guest-isolation.mjs"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=90, check=True)
    assert json.loads(result.stdout)["status"] == "ok"
    assert (resources.parent / "candidate.json").read_text() == "ok"
    assert (private / "secret.canary").read_text() == "host_only\n"
