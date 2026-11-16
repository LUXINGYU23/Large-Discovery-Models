"""Synthetic collaborators for controller verification, never scientific results.

Every proposal carries unique markers: ``ident_<tag>`` in an identifier (survives
stripping), ``comment_<tag>`` and ``docstring_<tag>`` (must be stripped),
``summary_<tag>``/``rationale_<tag>`` in the annotations, and ``interval_digits(...)``
as the trailing digits of the three interval numbers. ``tag`` is ``b<b>r<rr>p<pp>x``.
"""

import hashlib

FIXTURE_CASES = ("fixture-case-a", "fixture-case-b", "fixture-case-c")


def marker_tag(branch, round_index, index):
    return f"b{branch}r{round_index:02d}p{index:02d}x"


def interval_digits(branch, round_index, index):
    return f"7{branch:01d}{round_index:02d}{index:02d}3141"


class ScriptedAgent:
    """Pure function of its arguments, hence safe for concurrent branch calls.

    Sources also embed a digest of the received message, so any difference in a
    message changes every later source and record.
    """

    def __init__(self, input_tokens=1000, cached_fraction=.25, missing_usage=0):
        self.input_tokens, self.cached_fraction, self.missing_usage = input_tokens, cached_fraction, missing_usage

    def propose(self, *, branch, round_index, count, message="", view=(), **_):
        digest = hashlib.sha256(message.encode()).hexdigest()[:12]
        parent = view[-1]["program_id"] if view else None
        rows = []
        for index in range(count):
            tag, digits = marker_tag(branch, round_index, index), interval_digits(branch, round_index, index)
            source = ('"""docstring_' + tag + '"""\n'
                      "from tasks.copper_inversion.v2.baselines import initialize\n"
                      f"# comment_{tag}\n"
                      "def solve(problem, budget, seed):\n"
                      f'    """docstring_{tag} function"""\n'
                      f"    ident_{tag} = {branch + round_index + index}  # comment_{tag}\n"
                      f"    message_digest = '{digest}'\n"
                      "    return initialize(problem)\n")
            rows.append({"source": source, "parent_sha256": parent if index % 2 else None,
                         "change_summary": f"summary_{tag}", "rationale": f"rationale_{tag}",
                         "predicted_risk": float("0.5" + digits), "interval80_low": float("0.2" + digits),
                         "interval80_high": float("0.8" + digits)})
        tokens = self.input_tokens + 10 * branch + round_index
        return rows, {"requests": 1 + self.missing_usage, "input_tokens": tokens,
                      "cached_input_tokens": int(tokens * self.cached_fraction), "output_tokens": 100 * count,
                      "responses_missing_usage": self.missing_usage, "compactions": 0, "wall_seconds": 0.0}


class ScriptedEvaluator:
    """Fake risk only for tests; its output must never be reported as measurement."""

    cases = FIXTURE_CASES

    def __init__(self, cases=FIXTURE_CASES):
        self.cases = tuple(cases)

    def evaluate_source(self, path):
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        risk = .1 + int(sha[:8], 16) / 0xffffffff * .8
        spread = [(i - (len(self.cases) - 1) / 2) * .05 for i in range(len(self.cases))]
        cases = {case: risk + delta for case, delta in zip(self.cases, spread)}
        return {"source_sha256": sha, "status": "valid", "risk": sum(cases.values()) / len(cases),
                "risk_by_case": cases, "legal_rate": 1., "failed_solves": 0,
                "seconds_per_solve": int(sha[8:12], 16) / 0xffff * 30, "message": ""}
