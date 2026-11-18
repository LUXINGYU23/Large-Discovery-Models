"""Run the sequence campaign through a separate HTTP experiment service."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from ldm_tts.async_runtime import (CampaignConfig, DedupOnlySelector, ESClient, PacedController, PacingConfig,
                                   RunLock, RuntimeOptions, campaign_extra, check_invariants)
from ldm_tts.async_runtime.history_bridge import HistoryBridge
from ldm_tts.contracts import ObjectiveSet, ObjectiveSpec
from ldm_tts.engine.run_store import atomic_json_write, unique_run_dir
from ldm_tts.registration.experiment import load_active_experiment_contract
from .experiments import AntibodyCodec, KIND, history_projection
from .research import InitializationPacing, NativeResearch, PolicyRoundPacing
from .worker import DesignWorker

REPOSITORY = Path(__file__).resolve().parents[3]


class ServiceProcess:
    """Track an ES launched by this process or recovered from a crashed controller."""

    def __init__(self, pid, config_path, child=None):
        self.pid, self.config_path, self.child = pid, str(config_path), child

    def matches(self):
        if self.child is not None and self.child.poll() is not None:
            return False
        try:
            command = Path(f"/proc/{self.pid}/cmdline").read_bytes().split(b"\0")
            return b"experiment_service" in command and os.fsencode(self.config_path) in command
        except OSError:
            return False

    def poll(self):
        if self.child is not None:
            return self.child.poll()
        return None if self.matches() else 1

    def stop(self):
        if not self.matches():
            return
        os.kill(self.pid, signal.SIGTERM)
        deadline = time.monotonic() + 10
        while self.matches() and time.monotonic() < deadline:
            time.sleep(0.05)
        if self.matches():
            os.kill(self.pid, signal.SIGKILL)
        if self.child is not None:
            self.child.wait()


@contextmanager
def experiment_service(directory, args):
    options = {"workers": args.concurrency, "antigen": args.antigen, "oracle": args.oracle,
               "work_dir": str(directory / "absolut-attempts"),
               "absolut_path": str(args.absolut_path.resolve()) if args.absolut_path else None}
    config = {"database": str(directory / "experiment-service.sqlite3"), "host": "127.0.0.1", "port": 0,
              "pools": {"default": args.concurrency}, "tasks": [str(REPOSITORY / "tasks/antibody_rfd3/task.json")],
              "python_path": [str(REPOSITORY)], "executor_options": {KIND: options}, "poll_interval_seconds": 0.05}
    config_path, log_path = directory / "experiment-service.json", directory / "experiment-service.log"
    state_path = directory / "experiment-service-process.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Experiment service configuration differs from the original campaign")
    else:
        atomic_json_write(config_path, config)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        process = ServiceProcess(state["pid"], config_path)
        if process.matches():
            # A matching but unresponsive ES must not be replaced: it may still own the database.
            ESClient(state["url"], timeout=5).kind(KIND)
            try:
                yield process, state["url"]
            finally:
                process.stop()
            return
    with log_path.open("a", encoding="utf-8") as stream:
        start_offset = stream.tell()
        child = subprocess.Popen([sys.executable, "-m", "experiment_service", "--config", str(config_path)],
                                 cwd=REPOSITORY, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        process = ServiceProcess(child.pid, config_path, child)
        try:
            deadline = time.monotonic() + 30
            prefix = "experiment service listening on "
            while process.poll() is None and time.monotonic() < deadline:
                with log_path.open(encoding="utf-8") as reader:
                    reader.seek(start_offset)
                    lines = reader.read().splitlines()
                for line in lines:
                    if line.startswith(prefix):
                        url = line[len(prefix):].strip()
                        atomic_json_write(state_path, {"pid": process.pid, "url": url})
                        yield process, url
                        return
                time.sleep(0.05)
            raise RuntimeError(f"Experiment service did not start; see {log_path}")
        finally:
            process.stop()


def run(args, task_spec):
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:args.cpu_limit])
    if args.resume_run:
        directory = args.resume_run.resolve()
        if not directory.is_dir():
            raise ValueError("Resume campaign directory does not exist")
    else:
        directory = unique_run_dir(args.out_dir.resolve())
        directory.mkdir(parents=True)
    # The outer lock covers service discovery and sidecar startup as well as
    # the controller, so a second resume cannot take over and stop a live ES.
    with RunLock(directory / "campaign.lock"):
        return _run_locked(args, task_spec, directory)


def _run_locked(args, task_spec, directory):
    config_data = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    ignored = {"out_dir", "dry_run", "resume_run"}
    scientific = {key: value for key, value in config_data.items() if key not in ignored}
    if args.resume_run:
        saved = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if {key: value for key, value in saved.items() if key not in ignored} != scientific:
            raise ValueError("Resume arguments differ from the original campaign configuration")
        frozen_task = json.loads(json.dumps(task_spec.to_dict()))
        if json.loads((directory / "ldm_task_spec.json").read_text(encoding="utf-8")) != frozen_task:
            raise ValueError("Resume task specification differs from the original campaign")
    else:
        atomic_json_write(directory / "config.json", config_data)
        atomic_json_write(directory / "ldm_task_spec.json", task_spec.to_dict())
    contract, profile = load_active_experiment_contract()
    if contract:
        atomic_json_write(directory / "experiment_contract.json", contract.to_dict())
    metric = "mock_energy" if args.oracle == "mock" else "absolut_energy"
    worker = DesignWorker(directory, args)
    bridge = HistoryBridge(directory / "history", history_projection(metric))
    dynamic_policy = args.method == "ldm_harness"
    pacing = PacingConfig.balanced(concurrency=args.concurrency, reservoir_size=args.pool_size,
                                   dispatch_during_research="continue", research_budget=args.research_rounds,
                                   research_timeout_seconds=3900 if dynamic_policy else 1200)
    config = CampaignConfig("antibody_rfd3", budget=args.iterations, concurrency=args.concurrency,
                            seed=args.seed, extra=campaign_extra(pacing, task=scientific))
    objectives = ObjectiveSet.from_specs([ObjectiveSpec(metric, "minimize")])
    with ExitStack() as stack:
        process, url = stack.enter_context(experiment_service(directory, args))
        client = ESClient(url, timeout=10)
        if args.method in ("harness", "ldm_harness"):
            from .harness import open_agent
            research = stack.enter_context(open_agent(args, directory, worker, bridge, config.campaign_id))
        else:
            research = NativeResearch(args, worker)
        policy = None
        if dynamic_policy:
            from .policy import make_selector, open_policy
            selector = make_selector()
            policy = stack.enter_context(open_policy(args, directory, config.campaign_id, selector))
        else:
            selector = DedupOnlySelector()
        ctrl = stack.enter_context(PacedController(directory / "controller.sqlite3", config,
            codec=AntibodyCodec(args.antigen, args.oracle, utility=dynamic_policy), client=client,
            research=research, history=bridge, selector=selector, policy=policy,
            pacing=PolicyRoundPacing() if dynamic_policy else InitializationPacing(), objectives=objectives,
            options=RuntimeOptions(poll_interval=0.1, long_poll_seconds=0, anti_entropy_interval=30)))
        error = None
        try:
            while not ctrl.finished:
                exit_code = process.poll()
                if exit_code is not None:
                    raise RuntimeError(f"Experiment service exited with code {exit_code}")
                progressed = ctrl.step()
                view = ctrl.view()
                if view.lane.state == "blocked":
                    generation = ctrl.generations()[-1]
                    raise RuntimeError(f"Research generation blocked: {generation['error']}")
                if view.paused:
                    raise RuntimeError(f"Campaign paused: {', '.join(view.paused)}")
                if not progressed:
                    time.sleep(0.05)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            observations = ctrl.observations()
            successful = [o for o in observations if o.evaluation.succeeded]
            incumbent = min(successful, key=lambda o: o.evaluation.metrics[metric], default=None)
            items, cursor = [], ""
            try:
                while True:
                    page = client.changes(config.campaign_id, cursor, limit=1000)
                    items.extend(page["items"])
                    next_cursor = page["cursor"]
                    if not page["items"] or next_cursor == cursor:
                        break
                    cursor = next_cursor
                problems = check_invariants(ctrl.store, config, items, synced=error is None)
            except Exception as exc:
                detail = f"Experiment service audit failed: {type(exc).__name__}: {exc}"
                error = f"{error}; {detail}" if error else detail
                problems = check_invariants(ctrl.store, config)
            jobs = list(worker.records())
            generation_usage = {"jobs": len(jobs),
                                "backbones_requested": sum(job["action"] == "design_h3" for job in jobs),
                                "sequences_requested": sum(job["count"] for job in jobs),
                                "failed_jobs": sum(job["status"] != "succeeded" for job in jobs)}
            harness_usage = {"turns": 0, "provider_calls": 0, "tool_calls": 0}
            generations = ctrl.generations()
            for generation in generations:
                outcome = json.loads(generation.get("outcome_json") or "{}")
                for session in (outcome.get("metadata", {}).get("sessions") or {}).values():
                    usage = session.get("usage") or {}
                    harness_usage["turns"] += 1
                    harness_usage["provider_calls"] += usage.get("providerCalls", 0)
                    harness_usage["tool_calls"] += sum(usage.get("toolCalls", {}).values())
            policy_summary = None
            if policy is not None:
                rounds = []
                for generation in generations:
                    outcome = json.loads(generation.get("outcome_json") or "{}")
                    record = {"generation": generation["generation"], "state": generation["state"],
                              **(outcome.get("metadata", {}).get("policy_round") or {})}
                    path = policy.research_controller.root / "rounds" / f"round_{generation['generation']:03d}" / "result.json"
                    if path.is_file():
                        result = json.loads(path.read_text(encoding="utf-8"))
                        record["harness_turn"] = result.get("harness_turn")
                        record["validation_errors"] = result.get("validation_errors", [])
                        record["failed_harness_usage"] = result.get("failed_harness_usage")
                    rounds.append(record)
                applications = policy.applications()
                decisions = ctrl.decisions()
                by_application = {a["application_id"]: a for a in applications}
                used_generations = set()
                decision_records = []
                for decision in decisions:
                    snapshot = decision["snapshot"]
                    app_id = snapshot.get("inputs", {}).get("prior_application_id")
                    application = by_application.get(app_id)
                    if application is not None:
                        used_generations.add(application["generation"])
                    decision_records.append({"decision_id": decision["decision_id"], "application_id": app_id,
                                             "generation": application["generation"] if application else None,
                                             "probability": snapshot.get("probability"),
                                             "weights": (snapshot.get("record", {}).get("selection", {})
                                                         .get("parameters", {}))})
                    problems.extend(f"decision {decision['decision_id']}: {problem}"
                                    for problem in ctrl.audit(decision["decision_id"]))
                problems.extend(policy.check())
                expected = set(range(1, args.research_rounds + 1))
                if {r["generation"] for r in rounds if r["state"] == "ready" and r.get("status") == "accepted"} != expected:
                    problems.append("not every requested research generation has an accepted policy turn")
                if used_generations != expected:
                    problems.append("not every requested generation's policy application was used by a decision")
                if any(a["degraded"] for a in applications):
                    problems.append("one or more policy applications degraded to the default")
                policy_summary = {"research_budget": args.research_rounds, "evaluation_budget": args.iterations,
                                  "rounds": rounds, "applications": applications, "decisions": decision_records,
                                  "accepted_turns": sum(r.get("status") == "accepted" for r in rounds),
                                  "fallback_turns": sum(r.get("action") == "fallback" for r in rounds)}
            if ctrl.anomalies():
                problems.append("controller has open anomalies")
            summary = {"task": "antibody_rfd3", "method": args.method, "oracle": args.oracle,
                       "metric": metric, "observation_count": len(observations),
                       "successful_evaluation_count": len(successful),
                       "failed_evaluation_count": len(observations) - len(successful),
                       "incumbent": incumbent.to_dict() if incumbent else None,
                       "stop_reason": ctrl.ledger.campaign_row()["stop_reason"],
                       "invariant_problems": problems, "error": error,
                       "generation_usage": generation_usage, "harness_usage": harness_usage,
                       "policy": policy_summary,
                       "experiment_contract_sha256": contract.digest if contract else "",
                       "experiment_contract_profile": profile}
            atomic_json_write(directory / "budget.json", ctrl.budget().to_dict())
            atomic_json_write(directory / "summary.json", summary)
            (directory / "events.jsonl").write_text("".join(json.dumps(e, sort_keys=True, allow_nan=False) + "\n"
                                                         for e in ctrl.events()), encoding="utf-8")
    print(json.dumps({"run_dir": str(directory), "oracle": args.oracle,
                      "stop_reason": summary["stop_reason"], "error": error}))
    return int(bool(error or problems) or len(successful) != args.iterations)
