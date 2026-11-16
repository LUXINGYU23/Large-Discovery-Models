"""Exercise real isolated scoring and GP against synthetic fixture programs."""

import argparse
import json
from pathlib import Path

from tasks.copper_inversion.core.encoding import CodeEncoder
from tasks.copper_inversion.rsi.client import RiskClient
from tasks.copper_inversion.rsi.controller import Config, Controller
from tasks.copper_inversion.rsi.scripted import ScriptedAgent
from tasks.copper_inversion.rsi.selection import select
from tasks.copper_inversion.v2.public_catalog import catalog_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, required=True)
    args = parser.parse_args()
    deployment = json.loads(args.deployment.read_text())
    config = Config("ldm4geo", 7, rounds=1, proposals_per_branch=1, batch=4, reset_period=1)
    evaluator = RiskClient(args.root, deployment["socket"], deployment["contract_digest"],
                           service_uid=deployment["service_uid"])
    evaluator.verify_contract(catalog_digest(args.catalog), 10, config.slots)
    encoder = CodeEncoder(args.root, "fusion", args.embedding, "cuda:0")
    controller = Controller(args.root, config, ScriptedAgent(), evaluator,
                            lambda pool, rows, count: select(pool, rows, encoder, count),
                            "Synthetic integration fixture.", {"synthetic_test": True,
                                                               "service_contract": deployment["contract_digest"]})
    result = controller.run()
    seal = evaluator.seal_training()
    if result["charged_records"] != 5 or seal["used_slots"] != 5:
        raise ValueError("synthetic smoke budget did not reconcile")
    print(json.dumps({"scope": "synthetic_fixture_only", "result": result,
                      "private_slots": seal["used_slots"]}, sort_keys=True))


if __name__ == "__main__":
    main()
