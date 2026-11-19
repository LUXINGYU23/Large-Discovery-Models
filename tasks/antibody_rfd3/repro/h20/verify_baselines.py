"""Audit paired real baselines and summarize measured resource use."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from tasks.antibody.core.engine_adapters import _sequence_violations
from tasks.antibody.tools.structure_design.common import extract_h3
from tasks.antibody_rfd3.repro.h20.verify_real_sessions import read, verify


def resources(run):
    generated_seconds = replayed_seconds = 0.0
    max_memory = 0
    calls = {"design_h3": 0, "sequence_h3": 0}
    replayed = []
    for path in sorted(run.glob("design_jobs/*/receipt.json")):
        receipt = read(path)
        if receipt.get("initialization_source"):
            replayed.append(receipt["initialization_source"])
        else:
            calls[receipt["action"]] += 1
        for execution in (path.parent / "output").glob("*.execution.json"):
            if not (execution.name == "rfd3.execution.json" or execution.name.startswith("proteinmpnn-")):
                continue
            record = read(execution)
            if receipt.get("initialization_source"):
                replayed_seconds += record["elapsed_seconds"]
            else:
                generated_seconds += record["elapsed_seconds"]
                max_memory = max(max_memory, record.get("sampled_peak_process_gpu_memory_mib", 0) or 0)
    tokens = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0}
    for trace in run.glob("harness/sessions/*/pi-session/*.jsonl"):
        for line in trace.read_text().splitlines():
            message = json.loads(line).get("message", {})
            if message.get("role") == "assistant":
                usage = message.get("usage", {})
                for key in tokens:
                    tokens[key] += usage.get(key, 0)
    return {"executed_model_calls": calls, "generation_model_seconds": generated_seconds,
            "generation_process_gpu_hours": generated_seconds / 3600,
            "sampled_peak_generation_memory_mib": max_memory,
            "replayed_initialization_sources": replayed,
            "replayed_original_model_seconds": replayed_seconds,
            "llm_tokens_native_trace": tokens}


def verify_baselines(root, initialization):
    references = verify(initialization)
    report = {"run_root": str(root), "initialization_root": str(initialization),
              "oracle": "Absolut", "sessions": [], "ldm_reference": references["sessions"],
              "resource_notes": "Generation GPU-hours are model subprocess wall time / 3600. "
              "Replayed initialization model time is listed separately. LLM tokens come from Pi assistant traces."}
    for reference in report["ldm_reference"]:
        reference["resources"] = resources(Path(reference["run"]))
    for method in ("antbo", "fixed", "harness"):
        for target, antigen in (("her2", "1N8Z_C"), ("pd1", "5GGR_Y_FuC_5.25")):
            run = root / method / target
            config = read(run / "config.json")
            assert config["method"] == method and config["oracle"] == "absolut" and not config["mock"]
            assert config["seed"] == 42 and config["iterations"] == 3
            assert Path(config["initialization_run"]) == initialization / target
            assert read(run / "summary.json")["successful_evaluation_count"] == 3
            events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
            observations = [e["payload"] for e in events if e["event_type"] == "candidate_evaluated"]
            assert len(observations) == 3
            reference = next(r for r in references["sessions"] if r["target"] == target)
            assert observations[0]["candidate"]["payload"]["sequence"] == reference["observations"][0]["sequence"]
            assert observations[0]["evaluation"]["metrics"]["absolut_energy"] == reference["observations"][0]["energy"]
            jobs = [read(p) for p in sorted(run.glob("design_jobs/*/receipt.json"))]
            assert jobs[0]["initialization_source"] == str(initialization / target / "design_jobs/job-00000")
            rows, seen = [], set()
            session = run / "harness/sessions/cdrh3_research"
            for index, obs in enumerate(observations):
                candidate, evaluation = obs["candidate"]["payload"], obs["evaluation"]
                seq = candidate["sequence"]
                assert not _sequence_violations(seq, 11) and seq not in seen
                assert evaluation["status"] == "succeeded"
                energy = evaluation["metrics"]["absolut_energy"]
                oracle = read(Path(evaluation["artifacts"]["result"]))
                assert oracle["antigen"] == antigen and oracle["results"] == [
                    {"sequence": seq, "absolut_energy": energy}]
                with Path(evaluation["artifacts"]["raw"]).open() as handle:
                    next(handle)
                    raw = list(csv.DictReader(handle, delimiter="\t"))
                assert raw and all(r["CDR3"] == seq and r["Slide"] == seq for r in raw)
                assert min(float(r["Energy"]) for r in raw) == energy
                if method != "antbo" or index == 0:
                    design = read(run / candidate["artifact"])["candidates"][candidate["index"]]
                    assert extract_h3(design, 11) == seq
                    assert Path(design["backbone_path"]).is_relative_to(run)
                if method == "fixed" or index == 0:
                    eligible = [r for j in jobs if j["round"] == index
                                for r in j["candidates"] if not r["violations"] and r["sequence"] not in seen]
                    assert candidate["design_id"] == eligible[0]["design_id"]
                if method == "antbo" and index:
                    proposal = candidate["antbo_proposal"]
                    previous = observations[index - 1]
                    assert proposal["optimizer"] == "bo.optimizer.Optimizer"
                    assert proposal["history_count"] == index and proposal["suggest_calls"] >= 1
                    assert proposal["observed"] == [{
                        "sequence": previous["candidate"]["payload"]["sequence"],
                        "energy": previous["evaluation"]["metrics"]["absolut_energy"]}]
                row = {"round": index, "sequence": seq, "energy": energy,
                       "backbone_id": candidate.get("backbone_id"), "result": evaluation["artifacts"]["result"]}
                if method == "harness":
                    turn = session / "turns" / f"design-{index}"
                    history = json.loads(read(turn / "input.json")["message"])["measured_history"]
                    assert len(history) == index
                    for old, item in zip(observations[:index], history):
                        assert item["energy"] == old["evaluation"]["metrics"]["absolut_energy"]
                        assert item["backbone_id"] == old["candidate"]["payload"]["backbone_id"]
                        assert item["result_artifacts"] == old["evaluation"]["artifacts"]
                    submission = read(turn / "submission.json")["submission"]["submission"]["design_ids"]
                    assert submission == [candidate["design_id"]]
                    row["tools"] = read(turn / "tool-budget.json")["used"]
                seen.add(seq)
                rows.append(row)
            if method == "harness":
                traces = list((session / "pi-session").glob("*.jsonl"))
                assert len(traces) == 1
                calls = [b for line in traces[0].read_text().splitlines()
                         for b in json.loads(line).get("message", {}).get("content", [])
                         if isinstance(b, dict) and b.get("type") == "toolCall"]
                assert calls[0]["name"] == "read" and calls[0]["arguments"]["path"].endswith("/cdrh3-design/SKILL.md")
                assert any(j["action"] == "design_h3" and j["round"] > 0 and j["status"] == "succeeded" for j in jobs)
                request = session / "turns/design-0/provider/design-0-provider-2.request.bin"
                assert b"WGGDGFYAMDY" in request.read_bytes()
            budget = read(run / "budget.json")["counters"]
            assert budget["expensive_evaluation_attempts"] == budget["successful_evaluations"] == 3
            status = read(run / "status.json")
            report["sessions"].append({"method": method, "target": target, "run": str(run),
                "seconds": status["updated_at_unix"] - status["started_at_unix"],
                "failed_evaluations": read(run / "summary.json")["failed_evaluation_count"],
                "observations": rows, "budget": budget, "resources": resources(run),
                "events_sha256": hashlib.sha256((run / "events.jsonl").read_bytes()).hexdigest()})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--initialization-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_baselines(args.run_root.resolve(), args.initialization_root.resolve()), indent=2))
