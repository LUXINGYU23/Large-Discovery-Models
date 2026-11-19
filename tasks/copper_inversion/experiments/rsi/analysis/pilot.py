"""Internal pilot (spec §7, AT-36): pooled within-arm SD, pooled mean, MDE table.

    python -m tasks.copper_inversion.experiments.rsi.analysis.pilot --archive RUNS \
        (--selection-root PRIVATE/<protocol_id>/selection | --outcomes TABLES)

Reads the validation IoU of the whole-run frozen solvers of the first seeds of the
two primary arms. Before the unseal there is no M1 export (T7 refuses it), so the
evaluator account runs this with --selection-root, which reads the private
selection reports (<exp>/<run_id>/full/selection.json) without writing anything.
The output never contains a per-arm or per-run value.
"""

import argparse
import json
from pathlib import Path
from statistics import mean

from . import stats
from .bundle import load_config, verify_protocol_runs
from .outcomes import Outcomes
from .rundir import discover


class SelectionReports:
    """Validation IoU straight from private v2/selection.py reports."""

    def __init__(self, root):
        self.root = Path(root)

    def value(self, run):
        path = self.root / run.experiment / run.run_id / "full" / "selection.json"
        if not path.is_file():
            return None
        report = json.loads(path.read_text())
        chosen = [c for c in report["candidates"] if c["source_sha256"] == report["selected_sha256"]]
        return chosen[0]["metrics"]["mean_iou"] if len(chosen) == 1 else None


class TableValues:
    def __init__(self, outcomes):
        self.outcomes = outcomes

    def value(self, run):
        final = self.outcomes.final(run.experiment, run.run_id)
        return None if final is None else self.outcomes.validation_value(run.experiment, run.run_id, final, "iou")


def pilot(runs, source, config):
    groups = []
    for arm in config["primary"]:
        chosen = sorted((r for r in runs if r.experiment == "M1" and r.arm == arm and not r.intervention),
                        key=lambda r: r.seed)[:config["pilot_seeds"]]
        values = []
        for run in chosen:
            value = source.value(run)
            if value is None:
                raise ValueError("pilot requires the validation IoU of every pilot run")
            values.append(value)
        if len(values) != config["pilot_seeds"]:
            raise ValueError("pilot requires the configured number of completed runs per arm")
        groups.append(values)
    s_v = stats.pooled_sd(groups)
    m_v = mean(v for g in groups for v in g)
    bound = config["mde_bound_relative"] * m_v
    z = config["mde_z"]
    table = {"bound": round(bound, 6), "bound_relative": config["mde_bound_relative"], "z": z,
             "rows": [{k: round(v, 6) if isinstance(v, float) else v for k, v in row.items()}
                      for row in stats.mde_table(s_v, config["mde_ns"], bound, z)],
             "decision": stats.replicate_decision(s_v, bound, z)}
    return {"s_v": round(s_v, 6), "m_v": round(m_v, 6), "mde_table": table}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--selection-root")
    group.add_argument("--outcomes")
    parser.add_argument("--config")
    parser.add_argument("--protocol")
    args = parser.parse_args(argv)
    source = SelectionReports(args.selection_root) if args.selection_root else TableValues(Outcomes.load(args.outcomes))
    runs = discover(args.archive)
    pid = verify_protocol_runs(runs, args.protocol)
    if pid is not None and args.outcomes and Outcomes.load(args.outcomes).protocol_id != pid:
        raise ValueError("outcome protocol differs from supplied protocol")
    result = pilot(runs, source, load_config(args.config, args.protocol))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
