"""Verify real research sessions against Oracle output and native Pi traces."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from tasks.antibody.tools.structure_design.common import extract_h3


def read(path):
    return json.loads(path.read_text())


def verify(root):
    report = {"run_root": str(root), "oracle": "Absolut", "sessions": []}
    for target, antigen in (("her2", "1N8Z_C"), ("pd1", "5GGR_Y_FuC_5.25")):
        run = root / target
        config = read(run / "config.json")
        assert config["oracle"] == "absolut" and not config["mock"]
        assert config["method"] == "ldm_harness" and config["seed"] == 42
        summary = read(run / "summary.json")
        assert summary["successful_evaluation_count"] == 3
        events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
        evaluations = [e["payload"] for e in events if e["event_type"] == "candidate_evaluated"]
        assert len(evaluations) == 3
        session = run / "harness/sessions/cdrh3_research"
        rows = []
        for index, obs in enumerate(evaluations):
            candidate = obs["candidate"]["payload"]
            evaluation = obs["evaluation"]
            sequence = candidate["sequence"]
            assert evaluation["status"] == "succeeded"
            energy = evaluation["metrics"]["absolut_energy"]
            oracle = read(Path(evaluation["artifacts"]["result"]))
            assert oracle["antigen"] == antigen and oracle["results"] == [
                {"sequence": sequence, "absolut_energy": energy}]
            assert oracle["binary_sha256"] == evaluation["metadata"]["binary_sha256"]
            with Path(evaluation["artifacts"]["raw"]).open() as handle:
                next(handle)
                raw = list(csv.DictReader(handle, delimiter="\t"))
            assert all(r["CDR3"] == sequence and r["Slide"] == sequence for r in raw)
            assert min(float(r["Energy"]) for r in raw) == energy
            designs = read(run / candidate["artifact"])["candidates"]
            assert extract_h3(designs[candidate["index"]], 11) == sequence
            turn = session / "turns" / f"design-{index}"
            history = json.loads(read(turn / "input.json")["message"])["measured_history"]
            assert len(history) == index
            for old, record in zip(evaluations[:index], history):
                assert record["energy"] == old["evaluation"]["metrics"]["absolut_energy"]
                assert record["backbone_id"] == old["candidate"]["payload"]["backbone_id"]
                assert record["design_id"] == old["candidate"]["payload"]["design_id"]
                assert record["result_artifacts"]["result"] == old["evaluation"]["artifacts"]["result"]
            row = {"round": index, "sequence": sequence, "energy": energy,
                   "backbone_id": candidate["backbone_id"], "design_id": candidate["design_id"],
                   "result": evaluation["artifacts"]["result"],
                   "tools": read(turn / "tool-budget.json")["used"]}
            if index:
                selection = next(e["payload"] for e in events
                                 if e["event_type"] == "candidates_selected" and e["iteration"] == index)
                scores = selection["metadata"]["candidates_scored"]
                assert len({s["sequence"] for s in scores}) >= 2
                assert max(s["acquisition_score"] for s in scores) == next(
                    s["acquisition_score"] for s in scores if s["sequence"] == sequence)
                row["gp_candidates"] = scores
            rows.append(row)
        traces = list((session / "pi-session").glob("*.jsonl"))
        assert len(traces) == 1
        calls = [block for line in traces[0].read_text().splitlines()
                 for block in json.loads(line).get("message", {}).get("content", [])
                 if isinstance(block, dict) and block.get("type") == "toolCall"]
        assert calls[0]["name"] == "read"
        assert calls[0]["arguments"]["path"].endswith("/cdrh3-design/SKILL.md")
        assert any(c["name"] == "design_h3" for c in calls)
        provider = session / "turns/design-0/provider"
        assert b"cdrh3-design" in (provider / "design-0-provider-1.request.bin").read_bytes()
        assert b"WGGDGFYAMDY" in (provider / "design-0-provider-2.request.bin").read_bytes()
        budget = read(run / "budget.json")["counters"]
        assert budget["expensive_evaluation_attempts"] == budget["successful_evaluations"] == 3
        status = read(run / "status.json")
        report["sessions"].append({"target": target, "antigen": antigen, "run": str(run),
            "model": read(run / "harness/manifest.json")["model"],
            "seconds": status["updated_at_unix"] - status["started_at_unix"],
            "observations": rows, "budget": budget,
            "events_sha256": hashlib.sha256((run / "events.jsonl").read_bytes()).hexdigest(),
            "trace": str(traces[0]), "trace_sha256": hashlib.sha256(traces[0].read_bytes()).hexdigest()})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.run_root.resolve()), indent=2))
