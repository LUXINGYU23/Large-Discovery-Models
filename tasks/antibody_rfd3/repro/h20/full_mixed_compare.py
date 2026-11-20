"""Run 20 rounds of 64 online scores for the matched HER2 comparison."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
import json
import os
from pathlib import Path
import subprocess

from .launch_utils import ACTIVE_ROOT, REPO, ROOT, now, private_environment_values, run_case, save_summary
from .paired_compare import ANTIGEN, METHODS, SEED_COUNT, audit, make_case, observations, query
from tasks.antibody_rfd3.core.local_proposals import LOCAL_SEARCH_TIMEOUT_SECONDS


SEED_BANK = ROOT / "runs/her2-tk-ard-feedback-pilot-20261120-r2/shared-initialization.json"
SCORES_PER_GENERATION = 64
RESEARCH_ROUNDS = 20
ONLINE_SCORES = RESEARCH_ROUNDS * SCORES_PER_GENERATION
TOTAL_SCORES = SEED_COUNT + ONLINE_SCORES
LDM_OVERRIDES = {
    "pool-size": 256, "research-candidates-per-role": 64, "local-proposal-count": 128,
    "research-rounds": RESEARCH_ROUNDS, "scores-per-generation": SCORES_PER_GENERATION,
    "evaluation-window": 32, "concurrency": 8, "backbones-per-round": 32,
    "sequences-per-round": 1024,
}


def enable_native_recovery():
    """Use the previously verified native restart if the original optimizer stalls."""
    import numpy as np
    from tasks.antibody_rfd3.core.baselines import NativeAntBO

    original = NativeAntBO.propose

    @wraps(original)
    def propose(self, history, pending=()):
        try:
            try:
                return original(self, history, pending)
            except RuntimeError as error:
                if str(error) != "AntBO could not suggest an admissible new CDRH3 in 32 attempts":
                    raise
            self.optimizer.guided_restart = False
            self.optimizer.restart()
            candidate = original(self, history, pending)
            candidate["antbo_proposal"]["recovery"] = "native_unguided_restart"
            print(f"Native AntBO trust-region recovery at {len(history)} observations", flush=True)
            return candidate
        finally:
            self.optimizer.x_init = np.asarray(self.optimizer.x_init).reshape(-1, 11)

    NativeAntBO.propose = propose


def wrap_runner(command, setup):
    index = command.index("scripts/run_ldm_tts.py")
    command[index:index + 1] = ["-c", setup + '; import runpy; runpy.run_path("scripts/run_ldm_tts.py", run_name="__main__")']


def make_cases(output):
    cases = {method: make_case(method, "full", output, SEED_BANK) for method in METHODS}
    for case in cases.values():
        command = case["command"]
        index = next(i for i, flag in enumerate(command) if flag.startswith("args.iterations="))
        command[index] = f"args.iterations={TOTAL_SCORES}"
        case["effective_args"]["iterations"] = TOTAL_SCORES
    ldm = cases["ldm_harness"]
    for key, value in LDM_OVERRIDES.items():
        ldm["command"].extend(("--set", f"args.{key}={json.dumps(value)}"))
    ldm["effective_args"].update({k.replace("-", "_"): v for k, v in LDM_OVERRIDES.items()})
    ldm["policy_execution_timeout_seconds"] = 30
    ldm["local_proposal_timeout_seconds"] = LOCAL_SEARCH_TIMEOUT_SECONDS
    wrap_runner(ldm["command"], "from functools import partial; from tasks.antibody_rfd3.core import policy; "
                "policy.DockerPolicyExecutor = partial(policy.DockerPolicyExecutor, timeout_seconds=30)")
    antbo = cases["antbo"]
    antbo["recovery_on_repeated_suggestions"] = "native_unguided_restart"
    wrap_runner(antbo["command"], "from tasks.antibody_rfd3.repro.h20.full_mixed_compare import enable_native_recovery; "
                "enable_native_recovery()")
    return cases


def audit_absolut_receipts(directory, method):
    rows = observations(directory, method)
    if method == "ldm_harness":
        service = {eid: (state, spec, result) for eid, state, spec, result in query(
            directory / "experiment-service.sqlite3", "SELECT experiment_id,state,spec,result FROM experiments")}
        if len(service) != len(rows):
            raise RuntimeError("Experiment service and observation counts differ")
        for row in rows:
            state, spec, result = service[row["experiment_id"]]
            expected = {"antigen": ANTIGEN, "oracle": "absolut", "sequence": row["candidate"]["canonical_key"]}
            if (state != "SUCCEEDED" or json.loads(spec) != expected
                    or json.loads(result)["absolut_energy"] != row["evaluation"]["metrics"]["absolut_energy"]):
                raise RuntimeError("An LDM score does not match its successful Absolut experiment receipt")
    else:
        for row in rows:
            evaluation = row["evaluation"]
            artifacts = evaluation["artifacts"]
            receipt = json.loads(Path(artifacts["result"]).read_text())
            expected = {"sequence": row["candidate"]["canonical_key"],
                        "absolut_energy": evaluation["metrics"]["absolut_energy"]}
            if (receipt["antigen"] != ANTIGEN or receipt["oracle"] != "Absolut"
                    or receipt["results"] != [expected] or not Path(artifacts["raw"]).is_file()):
                raise RuntimeError(f"{method} has a score without its matching Absolut receipt")
    return len(rows)


def audit_ldm(directory):
    path = directory / "controller.sqlite3"
    generations = query(path, "SELECT generation,base_ingest_seq,state,outcome_json FROM generations ORDER BY generation")
    if len(generations) != RESEARCH_ROUNDS:
        raise RuntimeError(f"LDM did not complete {RESEARCH_ROUNDS} research generations")
    expected = {"agent": 128, "antbo_local": 128}
    pools = {}
    for index, head, state, outcome in generations:
        data = json.loads(outcome)
        has_feedback = (head == SEED_COUNT if index == 1
                        else head > SEED_COUNT + (index - 2) * SCORES_PER_GENERATION)
        if (state != "ready" or not has_feedback or data["eligible"] != 256
                or data["metadata"]["pool_sources"] != expected
                or data["metadata"]["candidate_counts"] != {"explorer": 64, "refiner": 64}):
            raise RuntimeError(f"LDM generation {index} did not meet the pool or feedback requirement")
        pools[index] = {p["candidate"]["canonical_key"] for p in data["proposals"]}
        if len(data["proposals"]) != 256 or len(pools[index]) != 256:
            raise RuntimeError(f"LDM generation {index} did not submit 256 unique proposals")
    measured = query(path, f"""SELECT d.snapshot_json, o.status, o.canonical_key
        FROM decisions d LEFT JOIN observations o ON o.decision_id=d.decision_id AND o.experiment_id=d.experiment_id
        WHERE d.decision_id>{SEED_COUNT} ORDER BY d.decision_id""")
    quotas = Counter()
    selected = set()
    for snapshot, status, key in measured:
        index = json.loads(snapshot)["pacing_state"]["current_generation"]
        if status != "succeeded" or key in selected or key not in pools.get(index, ()):
            raise RuntimeError(f"LDM generation {index} has a missing, repeated or out-of-pool measurement")
        quotas[index] += 1
        selected.add(key)
    if quotas != dict.fromkeys(range(1, RESEARCH_ROUNDS + 1), SCORES_PER_GENERATION):
        raise RuntimeError("LDM did not complete the exact successful measurement count in each generation")
    return {"pool_sources": expected, "successful_online_measurements_per_generation": dict(sorted(quotas.items()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if ACTIVE_ROOT not in root.parents:
        parser.error("Use a run root beneath the H20 active data directory")
    bank = json.loads(SEED_BANK.read_text())
    if len(bank["candidates"]) != SEED_COUNT or len(bank["backbones"]) != 4:
        parser.error("Expected 16 common starts with the four preserved seed backbones")
    output = root / "full"
    cases = make_cases(output)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    budgets = {"shared_scores": SEED_COUNT, "online_scores": ONLINE_SCORES, "total_scores": TOTAL_SCORES,
               "comparison_rounds": RESEARCH_ROUNDS, "online_scores_per_round": SCORES_PER_GENERATION}
    if args.plan:
        print(json.dumps({"commit": commit, **budgets, "cases": cases}, indent=2))
        return 0
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip():
        parser.error("Use the clean scheduled checkout")
    env = dict(os.environ, ANTBO_HOME=str(ROOT / "env/sources/HEBO-77636146/AntBO"),
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
               HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false", TMPDIR=str(ROOT / "tmp"),
               LLM_REASONING_EFFORT="low", NO_PROXY="localhost,127.0.0.1,172.17.0.1",
               no_proxy="localhost,127.0.0.1,172.17.0.1")
    private = private_environment_values(env, env["LLM_API_KEY"])
    output.mkdir()
    report = {"status": "running", "commit": commit, "started_at": now(), **budgets, "methods": {}, "errors": {}}
    report_path = output / "report.json"
    save_summary(report_path, report)

    def run(method, case):
        run_case(case, env, 172800, private, lambda: save_summary(output / (method + ".json"), case))
        result = audit({method: case}, bank)[method]
        directory = Path(case["run_dir"])
        result["absolut_receipts_verified"] = audit_absolut_receipts(directory, method)
        result["best_by_online_score_count"] = {
            count: result["best_so_far"][SEED_COUNT + count - 1]
            for count in range(SCORES_PER_GENERATION, ONLINE_SCORES + 1, SCORES_PER_GENERATION)
        }
        if method == "ldm_harness":
            result.update(audit_ldm(directory))
        return result

    with ThreadPoolExecutor(max_workers=4) as workers:
        futures = {workers.submit(run, method, case): method for method, case in cases.items()}
        for future in as_completed(futures):
            method = futures[future]
            try:
                report["methods"][method] = future.result()
            except Exception as error:
                report["errors"][method] = str(error)
            save_summary(report_path, report)
    report.update(status="failed" if report["errors"] else "passed", finished_at=now())
    save_summary(report_path, report)
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
