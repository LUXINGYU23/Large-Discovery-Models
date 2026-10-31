from dataclasses import replace
import json
import subprocess
import sys

import pytest

from tasks.alphabench.core.protocol import T3Protocol
from tasks.alphabench.rebuild_report import rebuild


@pytest.mark.parametrize("method", ["llm", "ldm"])
@pytest.mark.parametrize("point", ["search", "validation", "test", "quality_set_frozen"])
def test_process_death_after_durable_receipt_reuses_paid_work(tmp_path, point, method):
    protocol = tmp_path / "protocol.json"
    frozen = replace(T3Protocol(), method=method, cold_seed_count=0, rounds=1,
        evaluations=2, batch_size=2, factor_select_n=2).to_dict()
    protocol.write_text(json.dumps(frozen))
    script = r'''
import json, os, sys
from pathlib import Path
from tasks.alphabench.core.gateway import OracleGateway
from tasks.alphabench.core.receipts import Receipts
from tasks.alphabench.core import workflow
from tasks.alphabench.ldm_task.procedure import main

run, protocol, point = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
marker = run / "crashed.marker"
original_send, original_evaluate, original_accept = OracleGateway._send, OracleGateway.evaluate, Receipts.accept
original_client = workflow.make_client


def make_client(*args, **kwargs):
    with (run / "clients.txt").open("a") as stream:
        stream.write("created\n")
    return original_client(*args, **kwargs)

def send(self, request):
    with (run / "physical.jsonl").open("a") as stream:
        stream.write(json.dumps({"phase": request["phase"], "id": request["request_id"]}) + "\n")
    return original_send(self, request)

def evaluate(self, candidate, *, phase, position, fast=True):
    response = original_evaluate(self, candidate, phase=phase, position=position, fast=fast)
    if phase == point and not marker.exists():
        marker.write_text(point)
        os._exit(92)
    return response

def accept(self, identity, payload):
    result = original_accept(self, identity, payload)
    if identity == point and not marker.exists():
        marker.write_text(point)
        os._exit(92)
    return result

OracleGateway._send, OracleGateway.evaluate, Receipts.accept = send, evaluate, accept
workflow.make_client = make_client
args = ["--mock", "--protocol-file", protocol, "--out-dir", str(run)]
if (run / "campaign.json").exists():
    args.extend(["--resume-run", str(run)])
raise SystemExit(main(args))
'''

    def child(run, boundary):
        return subprocess.run([sys.executable, "-c", script, str(run), str(protocol), boundary],
                              capture_output=True, text=True, timeout=30)

    interrupted, baseline = tmp_path / "interrupted", tmp_path / "baseline"
    killed = child(interrupted, point)
    assert killed.returncode == 92, killed.stderr
    if point == "search":
        calls = (interrupted / "physical.jsonl").read_bytes()
        clients = (interrupted / "clients.txt").read_bytes()
        changes = (frozen | {"data_digest": "different-data"},
                   frozen | {"environment_digest": "different-patch-or-environment"},
                   frozen | {"max_model_tokens": frozen["max_model_tokens"] + 1},
                   frozen | {"budgets": frozen["budgets"] | {"model_requests": frozen["budgets"]["model_requests"] + 1}})
        for altered in changes:
            protocol.write_text(json.dumps(altered))
            rejected = child(interrupted, point)
            assert rejected.returncode != 0 and "resume protocol identity mismatch" in rejected.stderr
            assert (interrupted / "physical.jsonl").read_bytes() == calls
            assert (interrupted / "clients.txt").read_bytes() == clients
        protocol.write_text(json.dumps(frozen))
    resumed = child(interrupted, point)
    assert resumed.returncode == 0, resumed.stderr
    assert child(baseline, "none").returncode == 0

    def projection(run):
        report = json.loads((run / "result.json").read_text())
        checkpoint = json.loads((run / "checkpoint.json").read_text())
        calls = [json.loads(line) for line in (run / "physical.jsonl").read_text().splitlines()]
        assert len({row["id"] for row in calls}) == len(calls)
        return {
            "phases": [row["phase"] for row in calls],
            "budget": json.loads((run / "budget.json").read_text())["counters"],
            "observations": [(row["candidate"]["candidate_id"], row["evaluation"]["status"])
                             for row in checkpoint["state"]["observations"]],
            "search": report["search"],
            "test_metrics": [row["raw"]["metrics"] for row in report["test"]],
            "quality_coverage": report["quality_audit"]["coverage"],
            "completeness": report["completeness"],
        }

    assert projection(interrupted) == projection(baseline)
    assert len(rebuild(interrupted, tmp_path / "rebuilt")["matched"]) == 4


def test_process_death_after_reservation_does_not_repeat_authorization(tmp_path):
    script = r'''
import os, sys
from pathlib import Path
from tasks.alphabench.core import receipts as module

root = Path(sys.argv[1])
receipts = module.Receipts(root / "receipts")
marker = root / "reserved.marker"
original_write = module.atomic_json_write

def write(path, value):
    original_write(path, value)
    if value.get("state") == "reserved" and not marker.exists():
        marker.write_text("reserved")
        os._exit(93)

def count(name):
    with (root / "effects.txt").open("a") as stream:
        stream.write(name + "\n")

def operation():
    count("physical")
    return {"ok": True}

module.atomic_json_write = write
assert receipts.execute("one", {"value": 1}, authorize=lambda: count("authorize"),
    reserve=lambda: count("reserve"), operation=operation) == {"ok": True}
'''
    command = [sys.executable, "-c", script, str(tmp_path)]
    first = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert first.returncode == 93, first.stderr
    assert subprocess.run(command, capture_output=True, text=True, timeout=30).returncode == 0
    assert subprocess.run(command, capture_output=True, text=True, timeout=30).returncode == 0
    assert (tmp_path / "effects.txt").read_text().splitlines() == ["authorize", "reserve", "physical"]
    receipt_path = next((tmp_path / "receipts").glob("*.json"))
    receipt = json.loads(receipt_path.read_text())
    receipt["request"]["value"] = 2
    receipt_path.write_text(json.dumps(receipt))
    corrupted = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert corrupted.returncode != 0 and "different request" in corrupted.stderr
    assert (tmp_path / "effects.txt").read_text().splitlines() == ["authorize", "reserve", "physical"]


@pytest.mark.parametrize("phase", ["initialization", "search"])
def test_validation_budget_exhaustion_pauses_without_losing_paid_receipts(tmp_path, phase):
    from tasks.alphabench.ldm_task.procedure import main

    base = T3Protocol()
    protocol = replace(base, cold_seed_count=2 if phase == "initialization" else 0,
        rounds=1, evaluations=2, batch_size=2,
        budgets={**base.budgets, "validation_evaluations": 1})
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol.to_dict()))
    run = tmp_path / "run"
    assert main(["--mock", "--protocol-file", str(protocol_path), "--out-dir", str(run)]) == 2
    stage = run / "initialization" if phase == "initialization" else run
    assert json.loads((stage / "status.json").read_text())["status"] == "paused_budget"
    assert not (run / "result.json").exists()
    observations = json.loads((stage / "checkpoint.json").read_text())["state"]["observations"]
    assert len(observations) == 1 and observations[0]["evaluation"]["status"] == "succeeded"
    receipts = [json.loads(path.read_text()) for path in (stage / "private/oracle").glob("*.json")]
    completed = [row for row in receipts if row.get("state") == "completed"]
    assert sum(row["identity"].get("phase") == phase for row in completed) == 2
    assert sum(row["identity"].get("phase") == "validation" for row in completed) == 1
    budget = json.loads((stage / "budget.json").read_text())["counters"]
    assert budget["validation_evaluations"] == 1
    assert budget["external_evaluations"] == 2
