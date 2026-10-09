"""Remote-only qualification, four-seed experiment and historical comparison.

Usage: python run.py /mnt/data1/.../research/main-research-forks-20261009
Requires a source snapshot, built image, and the two named historical experiments.
No API secret is copied into artifacts. Existing campaigns are never overwritten.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import yaml

SEEDS = (42, 43, 3407, 10086)
METHOD = "ldm_harness_compiled"
CASE = "enformer_muscle_not_liver"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def prepare(work, project, image):
    source = work / "source"
    reference = project / "research/sol-pi-validation-20260921"
    historical = json.loads((reference / "experiment.json").read_text())
    # Reuse the already-qualified guest byte-for-byte. Its pinned base is unchanged;
    # the historical deployment recipe uses reachable OCI and Debian mirrors.
    recipe_relative = Path("tasks/nucleobench/resources/harness/image/Dockerfile")
    recipe_source = project / "research/pure-harness-pi-ablation-20261007/source" / recipe_relative
    recipe_sha256 = hashlib.sha256(recipe_source.read_bytes()).hexdigest()
    shutil.copyfile(recipe_source, source / recipe_relative)
    config = historical["config"]
    config.update(name="enformer_main_research_forks", algorithm="main_research_round_forks",
                  description="Native Pi main-session forks; historical comparison, not isolated causal attribution.")
    config["args"].update({
        "harness-main-research": True,
        "harness-profile": ["target_biology", "regulatory_grammar", "sequence_landscape", "evidence_critic"],
        "harness-sidecar-image": image,
        "harness-cache-dir": str(project / "research/pure-harness-pi-ablation-20261007/guest-cache"),
        "api-key-file": str(project / "research/pure-harness-pi-ablation-20261007/secrets/deepseek.key"),
        "out-dir": str(work / "runs"),
    })
    assert not config["args"].get("harness-sol-pi-config")
    files = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(source.rglob("*")) if p.is_file()
             and not {"node_modules", "dist", "__pycache__", ".pytest_cache", ".git"}.intersection(p.parts)
             and p.suffix != ".pyc"}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    save(work / "source-manifest.json", {"sha256": digest, "files": files})
    for directory in ("config", "logs", "status", "tmp"):
        (work / directory).mkdir(exist_ok=True)
    base_path = work / "config/base.yaml"
    base_path.write_text(yaml.safe_dump(config, sort_keys=False))
    for stage, seeds, rounds in (("smoke", (42,), 2), ("runs", SEEDS, 20)):
        for seed in seeds:
            spec = {
                "schema_version": 1, "name": f"enformer_{stage}_forks_{seed}",
                "task": "nucleobench", "base_config": str(base_path),
                "cases": [{"id": CASE, "overrides": []}], "methods": [METHOD],
                "method_overrides": {METHOD: []},
                "seeds": [seed], "optimization_rounds": rounds, "initialization_mode": "shared_start",
                "output_root": str(work / stage / f"seed_{seed}"),
                "trajectory": {"step_column": "round", "step_kind": "round",
                               "objective_column": "utility", "direction": "maximize"},
                "result_fields": {"best_found_utility": "best_found_utility", "best_found_energy": "best_found_energy"},
            }
            (work / "config" / f"{stage}_{seed}.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    save(work / "experiment.json", {
        "source_sha256": digest, "image": image, "config": config, "seeds": SEEDS,
        "guest_recipe_deployment": {"source": str(recipe_source), "sha256": recipe_sha256,
                                    "reason": "Reuse previously qualified guest; mirror-only recipe change"},
        "reference": str(reference), "active_rounds": 20, "evaluations_per_round": 16,
        "proposal_sessions_per_round": 4, "persistent_main_sessions": 1,
        "persistent_policy_sessions": 1, "candidates_per_session": 32, "bo_pool_size": 64,
        "confounders": ["historical endpoint/time", "different role prompts", "source revision", "rebuilt sidecar"],
        "cost_scope": "all main, child and policy provider responses, including failed calls and compaction",
        "success_rule": "Report paired per-seed utility and total/cached/uncached/output tokens; do not claim stable improvement from a mean alone.",
    })
    return digest


def environment(work, project, digest):
    return {**os.environ, "PYTHONPATH": f"{work / 'source'}:{project / 'nucleobench/source'}",
            "PATH": f"{project}/runtime/node-v24.7.0/bin:{os.environ['PATH']}",
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "TMPDIR": str(work / "tmp"),
            "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "16", "MKL_NUM_THREADS": "16",
            "OPENBLAS_NUM_THREADS": "16", "HF_HOME": str(project / "cache/huggingface"),
            "HF_ENDPOINT": "https://hf-mirror.com", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "XDG_CACHE_HOME": str(project / "cache/xdg"), "MPLCONFIGDIR": str(work / "matplotlib"),
            "LDM_DATA_COLLECTION_ENABLED": "1", "LDM_PILOT_EVALUATION_COMMIT": "snapshot:" + digest}


def campaign(work, stage, seed):
    return work / stage / f"seed_{seed}" / "campaigns" / CASE / METHOD / f"seed_{seed}"


def run(work, stage, seed, env):
    name = f"{stage}_{seed}"
    state_path = work / "status" / f"{name}.json"
    if state_path.exists() or campaign(work, stage, seed).exists():
        raise FileExistsError(f"Refusing to overwrite {name}")
    state = {"status": "running", "started_at": time.time(), "seed": seed, "stage": stage}
    save(state_path, state)
    with (work / "logs" / f"{name}.log").open("x") as log:
        process = subprocess.Popen([sys.executable, "scripts/run_pilot_evaluation.py",
                                    str(work / "config" / f"{name}.yaml")], cwd=work / "source", env=env,
                                   stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        state["pid"] = process.pid
        save(state_path, state)
        code = process.wait()
    state.update(status="completed" if code == 0 else "failed", returncode=code, finished_at=time.time())
    save(state_path, state)
    if code:
        raise RuntimeError(f"Campaign failed: {name}; see its log")


def verify_forks(root, rounds):
    observations = json.loads((root / "checkpoint.json").read_text())["state"]["observations"]
    assert len(observations) == 1 + 16 * rounds
    assert observations[0]["evaluation"]["metrics"]["utility"] == -3009.28515625
    main = root / "harness/main"
    main_commits = [json.loads(path.read_text()) for path in main.glob("turns/*/turn_committed.json")]
    assert len(main_commits) == rounds
    assert len({item["sessionId"] for item in main_commits}) == 1
    assert len(list((root / "policy_harness").glob("turns/*/turn_committed.json"))) == rounds
    children = []
    for index in range(1, rounds + 1):
        branch = root / "harness/branches" / f"round_{index:04d}"
        panel = json.loads((branch / "panel.json").read_text())
        assert len(panel["proposals"]) == 128
        commits = list(branch.glob("turns/*/turn_committed.json"))
        assert len(commits) == 4
        manifest = json.loads((branch / "manifest.json").read_text())
        assert manifest["forkSource"]["historyCursor"] == 1 + 16 * (index - 1)
        assert manifest["forkSource"]["sessionSha256"] == panel["main_session_sha256"]
        for path in branch.glob("sessions/*/pi-session/*.jsonl"):
            with path.open() as stream:
                header = json.loads(next(stream))
            assert header["parentSession"] == "/artifacts/fork-source/session.jsonl"
            children.append(header["id"])
    assert len(children) == 4 * rounds and len(set(children)) == len(children)
    return {"evaluations": len(observations), "verified_main_turns": rounds, "verified_native_forks": len(children)}


def usage(root):
    total, groups = Counter(), {}
    for name, directory in (("main", root / "harness/main"), ("branches", root / "harness/branches"),
                            ("policy", root / "policy_harness")):
        counts = Counter()
        for path in directory.rglob("*.request.bin"):
            if path.parent.name == "provider": counts["provider_calls"] += 1
        for path in directory.rglob("*.response.bin"):
            if path.parent.name != "provider": continue
            value = None
            for line in path.read_text(errors="replace").splitlines():
                if not line.startswith("data:"): continue
                try: event = json.loads(line[5:].strip())
                except json.JSONDecodeError: continue
                if isinstance(event.get("response", {}).get("usage"), dict): value = event["response"]["usage"]
            if value is None:
                counts["responses_missing_usage"] += 1
                continue
            counts["responses_with_usage"] += 1
            counts["input_tokens"] += value.get("input_tokens", 0)
            counts["output_tokens"] += value.get("output_tokens", 0)
            counts["cached_input_tokens"] += (value.get("input_tokens_details") or {}).get("cached_tokens", 0)
            counts["reasoning_tokens"] += (value.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
        counts["uncached_input_tokens"] = counts["input_tokens"] - counts["cached_input_tokens"]
        counts["total_tokens"] = counts["input_tokens"] + counts["output_tokens"]
        counts["requests_without_usage"] = counts["provider_calls"] - counts["responses_with_usage"]
        groups[name] = dict(counts)
        total.update(counts)
    return dict(total), groups


def compare(work, project):
    baseline = json.loads((project / "research/sol-pi-validation-20260921/comparison/summary.json").read_text())
    old = {row["seed"]: row for row in baseline["campaigns"] if row["arm"] == "without_sol_pi"}
    rows = []
    for seed in SEEDS:
        root = campaign(work, "runs", seed)
        verification = verify_forks(root, 20)
        points = json.loads((root / "checkpoint.json").read_text())["state"]["observations"]
        score = max(p["evaluation"]["metrics"]["utility"] for p in points)
        total, groups = usage(root)
        state = json.loads((work / "status" / f"runs_{seed}.json").read_text())
        rows.append({"seed": seed, "best_utility": score, "historical_best_utility": old[seed]["best_utility"],
                     "utility_difference": score - old[seed]["best_utility"], "usage": total, "usage_by_role": groups,
                     "historical_usage": old[seed], "verification": verification,
                     "wall_seconds": state["finished_at"] - state["started_at"]})
    result = {"campaigns": rows, "mean_utility": statistics.mean(r["best_utility"] for r in rows),
              "historical_mean_utility": statistics.mean(r["historical_best_utility"] for r in rows),
              "mean_total_tokens": statistics.mean(r["usage"]["total_tokens"] for r in rows),
              "historical_mean_total_tokens": statistics.mean(r["historical_usage"]["total_tokens"] for r in rows),
              "paired_utility_wins": sum(r["utility_difference"] > 0 for r in rows),
              "limitations": "Four-seed historical comparison; roles, time/endpoint, source and image differ. No causal or stable-improvement claim."}
    result["qualification_usage"] = usage(campaign(work, "smoke", 42))[0]
    aborted = work / "qualification-open-ended-main/qualification-aborted.json"
    if aborted.exists():
        result["aborted_qualification"] = json.loads(aborted.read_text())
    save(work / "comparison/summary.json", result)
    trajectories = []
    for seed in SEEDS:
        historical = (project / "research/sol-pi-validation-20260921/runs/without_sol_pi"
                      / f"seed_{seed}" / "campaigns" / CASE / METHOD / f"seed_{seed}")
        for arm, root in (("historical_ldm", historical), ("main_research_forks", campaign(work, "runs", seed))):
            observations = json.loads((root / "checkpoint.json").read_text())["state"]["observations"]
            for index in range(21):
                scores = [p["evaluation"]["metrics"]["utility"] for p in observations if p["round_idx"] <= index]
                trajectories.append({"arm": arm, "seed": seed, "round": index,
                                     "evaluations": len(scores), "best_utility": max(scores)})
    with (work / "comparison/trajectories.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trajectories[0]))
        writer.writeheader()
        writer.writerows(trajectories)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for arm, color, label in (("historical_ldm", "#777777", "Historical LDM"),
                              ("main_research_forks", "#007f86", "Main research forks")):
        for seed in SEEDS:
            selected = [row for row in trajectories if row["arm"] == arm and row["seed"] == seed]
            axes[0].plot([r["round"] for r in selected], [r["best_utility"] for r in selected], color=color, alpha=.2, lw=.8)
        means = [statistics.mean(r["best_utility"] for r in trajectories if r["arm"] == arm and r["round"] == i) for i in range(21)]
        axes[0].plot(range(21), means, color=color, lw=2.2, label=label)
    axes[0].set(xlabel="Active round (16 evaluations)", ylabel="Best measured utility", title="Four seeds: mean and individual runs")
    axes[0].legend(frameon=False)
    positions = range(len(SEEDS))
    axes[1].bar([i - .2 for i in positions], [r["historical_usage"]["total_tokens"] / 1e6 for r in rows], width=.4, color="#777777", label="Historical LDM")
    axes[1].bar([i + .2 for i in positions], [r["usage"]["total_tokens"] / 1e6 for r in rows], width=.4, color="#007f86", label="Main research forks")
    axes[1].set(xticks=list(positions), xticklabels=[str(s) for s in SEEDS], xlabel="Seed", ylabel="Input + output tokens (millions)", title="All roles, including cached input")
    axes[1].legend(frameon=False)
    figure.savefig(work / "comparison/comparison.png", dpi=180)
    figure.savefig(work / "comparison/comparison.svg")
    plt.close(figure)
    lines = ["# Main research forks: historical comparison", "", "| Seed | Fork utility | Historical LDM | Difference | Fork tokens (M) | Historical tokens (M) |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['seed']} | {row['best_utility']:.1f} | {row['historical_best_utility']:.1f} | {row['utility_difference']:+.1f} | {row['usage']['total_tokens']/1e6:.2f} | {row['historical_usage']['total_tokens']/1e6:.2f} |")
    lines += ["", result["limitations"], "", "All main, branch and policy usage is included. See summary.json for cache, reasoning, missing usage and per-role costs.", ""]
    (work / "comparison/report.md").write_text("\n".join(lines))
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--project", type=Path, default=Path("/mnt/data1/Large-Discovery-Models"))
    parser.add_argument("--image", default="ldm-pi-harness:main-research-forks-20261009")
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    work = args.work.resolve()
    if args.compare_only:
        compare(work, args.project)
        return
    if (work / "experiment.json").exists():
        raise FileExistsError("Experiment already prepared; do not silently restart or overwrite it")
    image = subprocess.check_output(["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], text=True).strip()
    # Keep the image in use between short-lived pools on a shared pruning host.
    lease = subprocess.check_output(["docker", "run", "-d", "--rm", "--label",
                                    "ldm.experiment=main-research-forks", "--entrypoint",
                                    "/bin/sleep", image, "infinity"], text=True).strip()
    save(work / "runtime-lease.json", {"container": lease, "image": image})
    try:
        digest = prepare(work, args.project, image)
        env = environment(work, args.project, digest)
        with (work / "logs/preflight.log").open("x") as log:
            for command in ([sys.executable, "scripts/check_task_dependencies.py", str(work / "config/base.yaml")],
                            [sys.executable, "scripts/run_pilot_evaluation.py", str(work / "config/runs_42.yaml"), "--dry-run"]):
                subprocess.run(command, cwd=work / "source", env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        run(work, "smoke", 42, env)
        save(work / "smoke-verification.json", verify_forks(campaign(work, "smoke", 42), 2))
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(run, work, "runs", seed, env) for seed in SEEDS]
            for future in futures: future.result()
        compare(work, args.project)
    finally:
        subprocess.run(["docker", "rm", "-f", lease], check=False, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
