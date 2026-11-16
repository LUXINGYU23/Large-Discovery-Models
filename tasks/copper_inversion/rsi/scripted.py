"""Synthetic collaborators for controller verification, never scientific results."""

import hashlib


class ScriptedAgent:
    def propose(self, *, branch, round_index, count, **_):
        rows = []
        for index in range(count):
            source = ("from tasks.copper_inversion.v2.baselines import initialize\n"
                      "def solve(problem, budget, seed):\n"
                      f"    marker_{branch}_{round_index}_{index} = {branch + round_index + index}\n"
                      "    return initialize(problem)\n")
            rows.append({"source": source, "parent_sha256": None,
                         "change_summary": f"private_note_{branch}_{round_index}",
                         "rationale": f"private_rationale_{branch}_{round_index}",
                         "predicted_risk": .5, "interval80_low": .2, "interval80_high": .8})
        return rows, {"requests": 0, "input_tokens": 0, "cached_input_tokens": 0,
                      "output_tokens": 0, "responses_missing_usage": 0, "compactions": 0}


class ScriptedEvaluator:
    """Fake risk only for tests; its output must never be reported as measurement."""
    def evaluate_source(self, path):
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        risk = .1 + int(sha[:8], 16) / 0xffffffff * .8
        return {"source_sha256": sha, "status": "valid", "risk": risk,
                "risk_by_case": {"fixture-case": risk}, "legal_rate": 1., "failed_solves": 0,
                "seconds_per_solve": 0., "message": ""}
