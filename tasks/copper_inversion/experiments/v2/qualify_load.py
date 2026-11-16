"""Load qualification (parallelism.md §5 rule 2, AT-44) on public fixtures only.

A fixed, compute-only reference solver loops the same numpy operation until its
deadline and returns how often it got through. It runs in real solver containers
(2 CPUs, the solver tier's seconds) through the scheduling queue at three loads:
alone, half of the slots (G/2 at once) and every slot (G at once). The slot count
G qualifies when, under full load, the median count is at least 95% of the idle
median and the worst 5% (5th percentile) at least 85% of it. Otherwise lower G and
run again. The record's SHA-256 goes into the scheduling policy as
load_qualification_sha256, and the launcher refuses formal batches without it.

    python -m tasks.copper_inversion.experiments.v2.qualify_load --root DIR --image sha256:... \
        --values scheduling-values.json --seconds 30 [--waves 2] [--idle-jobs 8] [--threads 2]
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import time

import numpy as np

from ldm_tts.engine.run_store import atomic_json_write
from tasks.copper_inversion.core import execution
from tasks.copper_inversion.core.execution import Execution, digest, source_identity
from tasks.copper_inversion.v2.data import save_problem
from tasks.copper_inversion.v2.fixtures import fixtures

MEDIAN_FLOOR, TAIL_FLOOR, TAIL = 0.95, 0.85, 0.05
# The r3 evaluator image of the previous qualification; qualify_failure_case.py imports it.
IMAGE = 'sha256:f6856385a0da9444b1e5edb5048b2f6081593fe568c3473c2fda524d0c4489f5'
REFERENCE = '''import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
THREADS = {threads}
def solve(problem, budget, seed):
    # Fixed work per iteration; numpy releases the GIL, so THREADS cores are used.
    deadline = time.monotonic() + max(0., budget["wall_time_seconds"] - 3.)
    def spin(_):
        x, n = np.linspace(.1, 1., 20000), 0
        while time.monotonic() < deadline:
            x = np.sin(x) + .1
            n += 1
        return n
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        count = sum(pool.map(spin, range(THREADS)))
    return np.full(problem["shape"], float(count))
'''


def percentile(values, q):
    """Nearest-rank percentile: the value below which a fraction q of the runs lie."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def verdict(idle, full, half=None):
    """Rule 2 thresholds; counts are the reference solver's iterations per run."""
    if not idle or not full:
        raise ValueError("qualification needs idle and full-load runs")
    base = statistics.median(idle)
    row = {"idle_median": base, "full_median": statistics.median(full), "full_p5": percentile(full, TAIL),
           "median_ratio": statistics.median(full) / base, "p5_ratio": percentile(full, TAIL) / base,
           "thresholds": {"median": MEDIAN_FLOOR, "p5": TAIL_FLOOR}}
    if half:
        row.update(half_median=statistics.median(half), half_p5=percentile(half, TAIL))
    row["passed"] = row["median_ratio"] >= MEDIAN_FLOOR and row["p5_ratio"] >= TAIL_FLOOR
    return row


def prepare_queue(root, values):
    from tasks.copper_inversion.experiments.v2 import parallel_runtime
    queue = Path(root) / "queue"
    (queue / "startup").mkdir(parents=True)
    for folder, count in ((queue, values["global_jobs"]), (queue / "startup", values["startup_slots"])):
        for index in range(count):
            (folder / f"{index:03}.lock").touch()
    policy = dict(values, queue_root=str(queue), interface_slots=0,
                  jobs_per_process=values["global_jobs"],
                  runtime_sha256=hashlib.sha256(Path(parallel_runtime.__file__).read_bytes()).hexdigest())
    policy["policy_sha256"] = digest(policy)
    parallel_runtime.install(policy)
    return policy


def run_level(root, image, name, concurrency, jobs, seconds, threads):
    work = Path(root) / name
    work.mkdir()
    inputs = work / "input"
    save_problem(next(iter(fixtures())), inputs)
    atomic_json_write(inputs / "budget.json", dict(seed=47, wall_time_seconds=seconds, cpu_count=2, memory_mb=4096))
    source = work / "reference"
    source.mkdir()
    (source / "solver.py").write_text(REFERENCE.format(threads=threads))
    ex = Execution(work / "jobs", image)

    def one(index):
        started = time.time()
        folder, receipt = ex.run(dict(level=name, index=index, reference=digest(REFERENCE), threads=threads),
                                 phase="solver", inputs=inputs, source=source, timeout=seconds, cpus=2,
                                 memory_mb=4096, science_module="v2")
        if receipt["exit_code"] or receipt["reason"] or receipt["oom"]:
            raise RuntimeError(f"reference run {name}/{index} failed: {receipt['reason'] or receipt['exit_code']}")
        count = float(np.load(folder / "output/model.npy", allow_pickle=False).flat[0])
        scheduling = json.loads((Path(ex.root) / "scheduling" / f"{folder.name}.json").read_text())
        return {"index": index, "count": count, "wall_seconds": receipt["wall_seconds"], "started": started,
                "slots_in_use_at_start": scheduling["slots_in_use_at_start"],
                "start_retries": len(scheduling["start_retries"])}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(one, range(jobs)))
    return {"concurrency": concurrency, "jobs": jobs, "counts": [r["count"] for r in rows], "runs": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--values", type=Path, required=True,
                   help="scheduling values: global_jobs, startup_slots, start_retry_seconds, ...")
    p.add_argument("--seconds", type=int, required=True, help="solver seconds of the tier being qualified")
    p.add_argument("--waves", type=int, default=2, help="full-load and half-load jobs = waves x concurrency")
    p.add_argument("--idle-jobs", type=int, default=8)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    if not a.image.startswith("sha256:"):
        raise ValueError("pin the evaluator image digest")
    values = json.loads(a.values.read_text())
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    policy = prepare_queue(root, values)
    g = values["global_jobs"]
    levels = {}
    for name, concurrency, jobs in (("idle", 1, a.idle_jobs), ("half", max(1, g // 2), max(1, g // 2) * a.waves),
                                    ("full", g, g * a.waves)):
        levels[name] = run_level(root, a.image, name, concurrency, jobs, a.seconds, a.threads)
        print(json.dumps({"level": name, "jobs": jobs, "median": statistics.median(levels[name]["counts"])}), flush=True)
    result = verdict(levels["idle"]["counts"], levels["full"]["counts"], levels["half"]["counts"])
    record = {"schema": "rsi-load-qualification-v1", "status": "passed" if result["passed"] else "failed",
              "global_jobs": g, "startup_slots": values["startup_slots"], "solver_seconds": a.seconds,
              "threads": a.threads, "image": a.image, "implementation": source_identity(),
              "reference_sha256": hashlib.sha256(REFERENCE.encode()).hexdigest(), "policy": policy,
              "verdict": result, "levels": levels, "host": socket.gethostname(),
              "cpu_count": os.cpu_count(), "finished_at": time.time()}
    out = a.out or root / "qualification.json"
    atomic_json_write(out, record)
    print(json.dumps({"status": record["status"], "sha256": hashlib.sha256(Path(out).read_bytes()).hexdigest(),
                      **{k: result[k] for k in ("median_ratio", "p5_ratio")}}), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
