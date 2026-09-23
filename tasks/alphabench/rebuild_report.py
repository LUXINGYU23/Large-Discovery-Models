"""Rebuild T3 run reports from frozen local artifacts without model or oracle dispatch."""

import argparse
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from ldm_tts.engine import LDMEngineState
from ldm_tts.engine.run_store import BudgetExceededError

from .core.finalization import finalize
from .core.gateway import OracleGateway
from .core.protocol import T3Protocol, digest


class StoredOracleReceipts:
    def __init__(self, root, protocol):
        self.root, self.protocol = root, protocol

    def load(self, identity):
        path = self.root / (digest(identity) + ".json")
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        request, response = record["request"], record["response"]
        if (record["state"] != "completed" or digest(record["identity"]) != digest(identity) or
                record["request_digest"] != digest(request) or
                record["response_digest"] != digest(response) or
                request["request_id"] != digest(identity) or response["request_id"] != request["request_id"] or
                request["protocol"] != self.protocol.to_dict() or request["phase"] != identity["phase"]):
            raise ValueError("stored Oracle receipt failed offline identity verification")
        return record


class OfflineGateway:
    identity = OracleGateway.identity
    evaluation_request = OracleGateway.evaluation_request

    def __init__(self, protocol, source, run_id, mock, budget):
        self.protocol, self.mock = protocol, mock
        self.runtime = SimpleNamespace(run_id=run_id)
        self.receipts = StoredOracleReceipts(source / "private/oracle", protocol)
        self.budget = budget

    def evaluate(self, candidate, *, phase, position, fast=True):
        identity, request, _ = self.evaluation_request(candidate, phase=phase, position=position, fast=fast)
        record = self.receipts.load(identity)
        if record is None:
            if phase == "quality":
                counters, limits = self.budget["counters"], self.budget["limits"]
                if (counters.get("quality_checks", 0) + 1 > limits["quality_checks"] or
                        counters.get("oracle_job_slots", 0) + request["job_permits"] > limits["oracle_job_slots"]):
                    raise BudgetExceededError("quality audit budget exhausted")
            raise ValueError("required Oracle receipt is missing: " + phase)
        if record["request"] != request:
            raise ValueError("stored Oracle request differs from the frozen evaluation contract")
        return record["response"]

    def combine(self, candidates, *, selection_digest):
        identity = {"run": self.runtime.run_id, "protocol": self.protocol.identity, "phase": "analysis",
                    "selection": selection_digest, "candidates": [item.candidate_id for item in candidates]}
        record = self.receipts.load(identity)
        if record is None:
            raise ValueError("independent combination receipt is missing")
        request = record["request"]
        if (request["operation"] != "combine" or request["expressions"] !=
                [item.payload["expression"] for item in candidates] or
                (request["start"], request["end"]) != self.protocol.interval("test") or
                request["fast"] is not False or request["job_permits"] != 8):
            raise ValueError("independent combination request differs from the frozen selection")
        return record["response"]


def rebuild(run_dir, out_dir):
    source, target = Path(run_dir), Path(out_dir)
    if target.exists():
        raise FileExistsError("offline rebuild output directory already exists")
    if target.resolve().is_relative_to(source.resolve()):
        raise ValueError("offline rebuild output must be outside the source run")
    protocol = T3Protocol(**json.loads((source / "protocol.json").read_text(encoding="utf-8")))
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config["protocol"] != protocol.to_dict() or type(config["mock"]) is not bool:
        raise ValueError("run config differs from the frozen protocol")
    campaign = json.loads((source / "campaign.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((source / "checkpoint.json").read_text(encoding="utf-8"))
    if checkpoint["run_id"] != campaign["run_id"] or checkpoint["task"] != "alphabench":
        raise ValueError("search checkpoint differs from the campaign")
    observations = LDMEngineState.from_checkpoint(checkpoint["state"]).observations
    execution_path = source / "private/stages" / (digest("search_execution") + ".json")
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    initial_manifest = json.loads((source / "initialization/seed_manifest.json").read_text(encoding="utf-8"))
    if initial_manifest["protocol"] != protocol.identity:
        raise ValueError("initialization differs from the frozen protocol")
    paths = [source / name for name in
             ("protocol.json", "budget.json", "initialization/budget.json", "initialization/seed_manifest.json")]
    for pattern in ("generation/*.json", "generation_attempts/*.json", "initialization/generation/*.json",
                    "policy_harness/rounds/round_*/result.json"):
        paths.extend(sorted(source.glob(pattern)))
    if any(path.is_symlink() for path in paths):
        raise ValueError("offline rebuild input cannot be a symlink")
    target.mkdir(parents=True)
    for path in paths:
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    budget = json.loads((source / "budget.json").read_text(encoding="utf-8"))
    initial_budget = json.loads((source / "initialization/budget.json").read_text(encoding="utf-8"))
    runtime = SimpleNamespace(run_dir=target, run_id=campaign["run_id"],
        budget=SimpleNamespace(snapshot=lambda: budget),
        status=SimpleNamespace(update=lambda *args, **kwargs: None))
    gateway = OfflineGateway(protocol, source, campaign["run_id"], config["mock"], budget)
    finalize(protocol, runtime, gateway, observations, initial_manifest["pool"], initial_budget, execution=execution)
    names = ("result.json", "report.md", "trajectory.csv", "report_manifest.json")
    for name in names:
        if (source / name).exists() and (source / name).read_bytes() != (target / name).read_bytes():
            raise ValueError("offline report differs from the published artifact: " + name)
    return {"source": str(source.resolve()), "output": str(target.resolve()), "matched": [
        name for name in names if (source / name).exists()]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(rebuild(args.run_dir, args.out_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
