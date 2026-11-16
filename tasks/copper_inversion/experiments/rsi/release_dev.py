"""Make the private, one-layout `rsi-dev` development release from planned GemPy data.

This command does not qualify a formal release or publish any held-out data. It
uses the shared RSI release tool with a development generation plan.
"""

import argparse
import json
import os
from pathlib import Path

from .plan import load_generation_plan
from .release import build


def release(dataset, output):
    plan, _ = load_generation_plan(Path(dataset).resolve())
    if plan["stage"] != "development":
        raise ValueError("a development generation plan is required")
    return build(dataset, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.geteuid() != 991:
        raise PermissionError("development release requires the isolated evaluator account")
    print(json.dumps(release(args.dataset, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
