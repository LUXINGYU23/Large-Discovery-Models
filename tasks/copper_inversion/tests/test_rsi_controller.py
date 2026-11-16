"""Controller semantics with synthetic collaborators; no scientific claims."""

import ast
import dis
import hashlib
import io
import json
import os
import random
import subprocess
import sys
import threading
import time
import tokenize
import types
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from tasks.copper_inversion.core.candidate import validate_source
from tasks.copper_inversion.core.encoding import encoding_spec
from tasks.copper_inversion.rsi.arms import ARMS, Arm, policy
from tasks.copper_inversion.rsi.controller import (Config, Controller, Start, make_message, profile_name,
                                                   session_round)
from tasks.copper_inversion.rsi.audit import scan_blind, verify_briefs
from tasks.copper_inversion.rsi.exposure import risk_by_case, validate_feedback
from tasks.copper_inversion.rsi.ledger import read
from tasks.copper_inversion.rsi.render import _case_labels, render, stripped_source, write_sources
from tasks.copper_inversion.rsi.scripted import ScriptedAgent, ScriptedEvaluator, interval_digits, marker_tag
from tasks.copper_inversion.rsi.selection import reserve_and_rank, select
from tasks.copper_inversion.rsi.service import RiskOnlyService
from ldm_tts.optimization.records import SurrogateVector


def scripted_select(pool, history, count):
    predictions = [{"program_id": p["program_id"], "branch": p["branch"],
                    "mean": .5, "std": .1,
                    "acquisition": int(p["program_id"][:8], 16) / 0xffffffff} for p in pool]
    return {"predictions": predictions, "selected": reserve_and_rank(predictions, count)}


def make_controller(root, arm="ldm4geo", reset=2, evaluator=None, agent=None, *, rounds=3, n=2, batch=4,
                    cap=10**9, arm_policy=None, start=None, **settings):
    return Controller(root, Config(arm, 7, rounds=rounds, proposals_per_branch=n, batch=batch,
                                  reset_period=reset, token_cap_input=cap, policy=arm_policy, **settings),
                      agent or ScriptedAgent(), evaluator or ScriptedEvaluator(), scripted_select,
                      "Fixed task brief.", {"synthetic_test": True}, start=start)


def briefs(root):
    """(branch, round) -> message bytes; the epoch directory is the clearing mechanism, not content."""
    found = {}
    for path in sorted(Path(root, "harness").glob("epoch_*/branch_*/briefs/round_*.json")):
        key = (int(path.parent.parent.name[7:]), int(path.stem[6:]))
        assert key not in found
        found[key] = path.read_bytes()
    return found


def tree(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(Path(root).rglob("*"))
            if p.is_file() and not p.name.endswith(".tmp")}


def test_projection_keeps_parent_macro_and_rejects_extra_fields():
    rows = [{"parent_id": "private-a", "case_id": "public-a", "status": "succeeded",
             "metrics": {"confirmation_loss_total": 3., "iou": .999}},
            {"parent_id": "private-a", "case_id": "public-a", "status": "failed", "metrics": {}},
            {"parent_id": "private-b", "case_id": "public-b", "status": "succeeded",
             "metrics": {"confirmation_loss_total": 0., "iou": .123}}]
    assert risk_by_case(rows, ["public-a", "public-b"]) == {"public-a": .875, "public-b": 0.}
    value = {"source_sha256": "a" * 64, "status": "valid", "risk": .4375,
             "risk_by_case": {"public-a": .875, "public-b": 0.}, "legal_rate": 2/3,
             "failed_solves": 1, "seconds_per_solve": 1., "message": ""}
    assert validate_feedback(value) == value
    with pytest.raises(ValueError, match="unexpected"):
        validate_feedback({**value, "mean_iou": .999})


@pytest.mark.parametrize("outcome", ["succeeded", "invalid_interface", "all_failed"])
def test_server_only_returns_allowlisted_feedback(tmp_path, outcome):
    service = RiskOnlyService.__new__(RiskOnlyService)
    service.root = tmp_path
    service.identity = {"test": True}
    service.public_contract = "c" * 64
    from tasks.copper_inversion.rsi.integrity import shared_identity
    service.shared_implementation = shared_identity()
    service.audit = {"sentinel": "SECRET-SENTINEL", "private_parent_ids": ["private-parent"], "case_ids": ["case"]}
    service.evaluator = SimpleNamespace(evaluation_attempt_usage_key=lambda _: "key",
                                        manifest={"cases": [{"case_id": "case", "parent_id": "private-parent"}],
                                                  "seeds": [47, 48]})
    sha = "d" * 64
    failed = outcome in {"invalid_interface", "all_failed"}
    metrics = {"mean_iou": .987, "sentinel": "SECRET-SENTINEL",
               "loss_total": 1. if failed else .5, "legal_rate": 0. if failed else 1.,
               "failed_solves": 2.0 if failed else 0.0, "mean_solver_wall_seconds": 0. if failed else 3.}
    status = "invalid_interface" if outcome == "invalid_interface" else "succeeded"
    result = {"status": status, "source_sha256": sha, "metrics": metrics}
    if status == "succeeded":
        path = tmp_path / "evaluations/key/private-case-results.json"
        path.parent.mkdir(parents=True)
        rows = [{"parent_id": "private-parent", "case_id": "case", "seed": seed,
                 "status": "failed" if failed else "succeeded",
                 "metrics": {} if failed else {"confirmation_loss_total": 1., "iou": .987}}
                for seed in (47, 48)]
        path.write_text(json.dumps(rows))
    with patch("tasks.copper_inversion.v2.service.EvaluatorService.dispatch", return_value=result):
        reply = service.dispatch({"operation": "evaluate", "source": "source", "contract_digest": "c" * 64})
    validate_feedback(reply)
    assert set(reply["risk_by_case"]) == {"case"}
    assert reply["status"] == ("rejected" if outcome == "invalid_interface" else "failed" if failed else "valid")
    assert "SECRET-SENTINEL" not in json.dumps(reply)
    assert "private-parent" not in json.dumps(reply)
    assert "iou" not in json.dumps(reply).lower()
    assert "SECRET-SENTINEL" in (tmp_path / "rsi-feedback" / (sha + ".json")).read_text()


def test_renderer_removes_comments_and_docstrings_only():
    source = ('"""module note"""\n# comment\nclass C:\n    """class note"""\n'
              '    def f(self):\n        """function note"""\n        return "data"\n')
    shown = stripped_source(source)
    assert "note" not in shown and "comment" not in shown
    assert "data" in shown
    namespace = {}
    exec(compile(shown, "display.py", "exec"), namespace)
    assert namespace["C"]().f() == "data"


@pytest.mark.parametrize("reset", [1, 2])
def test_three_epoch_visibility_and_deterministic_replay(tmp_path, reset):
    root = tmp_path / "run"
    controller = make_controller(root, reset=reset)
    result = controller.run()
    assert result["charged_records"] == 13
    ledger = read(root / "ledger.jsonl")
    assert [r["seq"] for r in ledger] == list(range(13))
    prefix = [r for r in ledger if r["round"] < 3]
    _, view = make_message(controller.config, root, prefix, 0, 3, controller.task_brief, [])
    start = ((3 - 1) // reset) * reset + 1
    assert all(r["round"] < start or r["branch"] == 0 for r in view)
    rendered = render(view)
    assert "summary_" not in rendered and "rationale_" not in rendered
    source_dir = tmp_path / "view"
    write_sources(source_dir, view, root)
    assert not any("summary_" in p.read_text() for p in source_dir.glob("*.py"))
    snapshot = (root / "ledger.jsonl").read_bytes()
    assert make_controller(root, reset=reset).run(resume=True) == result
    assert (root / "ledger.jsonl").read_bytes() == snapshot
    assert verify_briefs(root)["briefs_checked"] == 12


def test_blind_scanner_detects_positive_and_negative_controls(tmp_path):
    root = tmp_path / "run"
    make_controller(root).run()
    request = root / "harness/epoch_00/branch_0/turn.request.bin"
    request.write_bytes(b"allowed public task and risk")
    private = tmp_path / "private"
    private.mkdir()
    (private / "rsi-private-audit.json").write_text(json.dumps({
        "sentinel": "secret-canary-123", "private_parent_ids": ["private-body-123"]}))
    assert scan_blind(root, private)["provider_requests_checked"] == 1
    request.write_bytes(b"request contains secret-canary-123")
    with pytest.raises(ValueError, match="contaminated"):
        scan_blind(root, private)


def test_matching_history_yields_identical_reset_boundary_messages(tmp_path):
    source = tmp_path / "run"
    ctrl = make_controller(source)
    ctrl.run()
    rows = [r for r in read(source / "ledger.jsonl") if r["round"] < 3]
    a = make_message(ctrl.config, source, rows, 0, 3, ctrl.task_brief, [])[0]
    b = make_message(Config("accumulated", 7, rounds=3, proposals_per_branch=2,
                            batch=4, reset_period=2, token_cap_input=10**9),
                     source, rows, 0, 3, ctrl.task_brief, [])[0]
    assert a == b


def test_planted_note_is_only_in_the_target_round_messages(tmp_path):
    root = tmp_path / "run"
    note = {"round": 2, "target_tag": "fixture_family", "note": "PLANTED-CLAIM-ONLY"}
    ctrl = Controller(root, Config("ldm4geo", 7, rounds=3, proposals_per_branch=1,
                                  batch=4, reset_period=2), ScriptedAgent(), ScriptedEvaluator(),
                      scripted_select, "Fixed task brief.", {"synthetic_test": True}, intervention=note)
    ctrl.run()
    messages = sorted((root / "harness").glob("epoch_*/branch_*/briefs/round_*.json"))
    assert sum("PLANTED-CLAIM-ONLY" in p.read_text() for p in messages) == 4
    assert all(("PLANTED-CLAIM-ONLY" in p.read_text()) == (p.stem == "round_02") for p in messages)
    assert "PLANTED-CLAIM-ONLY" not in (root / "ledger.jsonl").read_text()
    assert verify_briefs(root)["briefs_checked"] == 12


def test_interrupted_evaluation_resumes_without_recharging(tmp_path):
    baseline = tmp_path / "baseline"
    make_controller(baseline).run()
    class Interrupted(ScriptedEvaluator):
        def __init__(self): self.count = 0
        def evaluate_source(self, path):
            self.count += 1
            if self.count == 4:
                raise RuntimeError("synthetic interruption")
            return super().evaluate_source(path)
    interrupted = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        make_controller(interrupted, evaluator=Interrupted()).run()
    make_controller(interrupted).run(resume=True)
    assert (baseline / "ledger.jsonl").read_bytes() == (interrupted / "ledger.jsonl").read_bytes()


def test_reserved_selection_then_highest_remaining():
    """AT-13: every branch with proposals gets its reservation; the rest are the best remaining."""
    rng = np.random.default_rng(5)
    for trial in range(200):
        branches = int(rng.integers(1, 5))
        reserve = int(rng.integers(0, 3))
        sizes = {b: int(rng.integers(0, 9)) for b in range(branches)}
        if trial % 3 == 0:
            sizes = {b: 8 for b in range(4)}
        values = [{"program_id": f"{b * 100 + i:064x}", "branch": b, "acquisition": float(rng.normal())}
                  for b, size in sizes.items() for i in range(size)]
        if not values:
            continue
        held = sum(min(reserve, size) for size in sizes.values())
        if held > len(values):
            continue
        count = 16 if trial % 3 == 0 else int(rng.integers(max(held, 1), len(values) + 1))
        chosen = reserve_and_rank(values, count, reserve)
        assert len(chosen) == count == len({p["program_id"] for p in chosen})
        got = {b: sum(1 for p in chosen if p["branch"] == b and p["selection"] == "reserved") for b in sizes}
        assert got == {b: min(reserve, size) for b, size in sizes.items()}
        for b in sizes:
            mine = sorted((p for p in values if p["branch"] == b), key=lambda p: (-p["acquisition"], p["program_id"]))
            assert [p["program_id"] for p in chosen if p["branch"] == b and p["selection"] == "reserved"] == \
                [p["program_id"] for p in mine[:reserve]]
        reserved = {p["program_id"] for p in chosen if p["selection"] == "reserved"}
        rest = sorted((p for p in values if p["program_id"] not in reserved),
                      key=lambda p: (-p["acquisition"], p["program_id"]))[:count - len(reserved)]
        assert [p["program_id"] for p in chosen[len(reserved):]] == [p["program_id"] for p in rest]
    with pytest.raises(ValueError, match="reservations"):
        reserve_and_rank([{"program_id": f"{i:064x}", "branch": i, "acquisition": 0.} for i in range(4)], 3)


def test_existing_fusion_gp_receives_only_code_features():
    version = encoding_spec("fusion").version
    class Encoder:
        def encode(self, candidate):
            vector = np.zeros(1280)
            vector[int(candidate.candidate_id[:4], 16) % len(vector)] = 1.
            return SurrogateVector(tuple(vector), version, source_id=candidate.candidate_id)
    history = [{"program_id": "a" * 64, "risk": .4}]
    pool = [{"program_id": f"{i+1:064x}", "branch": i} for i in range(4)]
    chosen = select(pool, history, Encoder(), 4)
    assert len(chosen["selected"]) == 4
    assert {p["branch"] for p in chosen["selected"]} == set(range(4))


DATA = Path(__file__).parent / "data"
TASK = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
EXTENSION = ("single_conversation", "separate_runs", "program_evolution")


class Crash(Exception):
    pass


class RecordingAgent(ScriptedAgent):
    """Snapshots what each branch could read at the moment of its turn."""

    def __init__(self, **settings):
        super().__init__(**settings)
        self.seen, self.lock = {}, threading.Lock()

    def propose(self, *, branch, round_index, turn_root, message, **settings):
        evidence = Path(turn_root) / "public" / "evidence"
        text = message + "".join(p.read_text() for p in sorted(evidence.rglob("*")) if p.is_file())
        with self.lock:
            self.seen[(branch, round_index)] = {"text": text, "turn_root": Path(turn_root),
                                                **{k: settings[k] for k in ("session_round", "epoch", "profile")}}
        return super().propose(branch=branch, round_index=round_index, message=message, turn_root=turn_root,
                               **settings)


class FailingAgent(ScriptedAgent):
    def __init__(self, at):
        super().__init__()
        self.at, self.calls = at, []

    def propose(self, *, branch, round_index, **settings):
        self.calls.append((round_index, branch))
        if (branch, round_index) == self.at:
            raise Crash("synthetic kill")
        return super().propose(branch=branch, round_index=round_index, **settings)


class FailingEvaluator(ScriptedEvaluator):
    def __init__(self, at):
        super().__init__()
        self.at, self.count = at, 0

    def evaluate_source(self, path):
        self.count += 1
        if self.count == self.at:
            raise Crash("synthetic kill")
        return super().evaluate_source(path)


def tags_by_program(root):
    tags = {}
    for path in Path(root, "proposals").glob("round_*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            tags[row["program_id"]] = row["change_summary"].removeprefix("summary_")
    return tags


def test_render_matches_golden_and_ignores_key_order():
    """AT-06."""
    rows = json.loads((DATA / "rsi_render_input.json").read_text())
    golden = (DATA / "rsi_render_golden.txt").read_bytes()
    assert render(rows).encode() == golden
    rng = random.Random(11)
    for _ in range(25):
        shuffled = []
        for row in rng.sample(rows, len(rows)):
            items = list(row.items())
            rng.shuffle(items)
            row = dict(items)
            cases = list(row["risk_by_case"].items())
            rng.shuffle(cases)
            shuffled.append({**row, "risk_by_case": dict(cases)})
        assert render(shuffled).encode() == golden
    text = golden.decode()
    assert text.splitlines()[1] == "# id            ep rd br  status    risk      legal  s/solve  sel"
    assert sum(line.endswith("  interface rejected") for line in text.splitlines()) == 1
    assert sum(line.endswith("  all planned solves failed") for line in text.splitlines()) == 1
    with pytest.raises(ValueError, match="unexpected"):
        render([{**rows[0], "rationale": "agent text"}])


def test_case_labels_are_first_eight_characters_unless_they_collide():
    assert _case_labels(["3c1f9a07be52", "8e20c4d7f1a9"]) == ["3c1f9a07", "8e20c4d7"]
    assert _case_labels(["abcdefgh1x", "abcdefgh2x", "zz"]) == ["abcdefgh1", "abcdefgh2", "zz"]


JUMPS = set(dis.hasjabs) | set(dis.hasjrel)


def _program(code):
    """Instructions without NOPs (a removed docstring leaves one), jumps as instruction indices."""
    ins = [i for i in dis.get_instructions(code) if i.opname != "NOP"]
    offsets = [i.offset for i in ins]
    out = []
    for i in ins:
        if i.opcode in JUMPS:
            arg = next(k for k, offset in enumerate(offsets) if offset >= i.argval)
        elif isinstance(i.argval, types.CodeType):
            arg = _program(i.argval)
        else:
            arg = i.argrepr
        out.append((i.opname, arg))
    return (code.co_name, tuple(out), tuple(c for c in code.co_consts if not isinstance(c, types.CodeType)),
            code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars)


STRIP_SAMPLE = '''"""module docstring_marker"""
# comment_marker at the top
import math  # comment_marker trailing


class Model:
    """class docstring_marker"""
    scale = 2.5  # comment_marker

    def fit(self, values):
        """method docstring_marker"""
        # comment_marker inside
        kept = "a string constant that stays"
        return [math.sqrt(v) * self.scale for v in values if v > 0] or kept

    async def later(self):
        """async docstring_marker"""
        return f"{self.scale:.2f}"


def only_docstring():
    """docstring_marker only"""


def solve(problem, budget, seed):
    """solve docstring_marker"""
    try:
        model = Model()
        assert budget >= 0, "budget"
        return {"x": model.fit([1, 2, 3]), "f": lambda z: z + 1}
    except ValueError:  # comment_marker
        return None
'''


def test_stripping_removes_comments_docstrings_and_keeps_bytecode():
    """AT-07: no comments or docstrings; constants and bytecode equal except docstrings."""
    agent = ScriptedAgent()
    corpus = [STRIP_SAMPLE] + [r["source"] for r in agent.propose(branch=1, round_index=2, count=3)[0]]
    corpus += [(TASK / "v2" / name).read_text() for name in
               ("baselines.py", "geometry.py", "objective.py", "geological_policy.py", "contract.py")]
    for source in corpus:
        shown = stripped_source(source)
        assert "comment_" not in shown and "docstring_" not in shown
        assert not any(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(shown).readline))
        tree_ = ast.parse(shown)
        assert all(ast.get_docstring(node, clean=False) is None for node in ast.walk(tree_)
                   if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)))
        for optimize in (2,):
            assert _program(compile(source, "solver.py", "exec", optimize=optimize)) == \
                _program(compile(shown, "solver.py", "exec", optimize=optimize))
    assert "a string constant that stays" in stripped_source(STRIP_SAMPLE)
    assert "ident_b1r02p00x" in stripped_source(corpus[1])
    # Across the whole package: the displayed tree is the original tree minus docstrings.
    from tasks.copper_inversion.rsi.render import _RemoveDocstrings
    for path in sorted(TASK.rglob("*.py")):
        text = path.read_text()
        assert ast.dump(ast.parse(stripped_source(text))) == ast.dump(_RemoveDocstrings().visit(ast.parse(text)))


def test_agent_fields_never_reach_records_views_or_messages(tmp_path):
    """AT-08: annotation, interval, comment and docstring markers stay in proposals and raw sources."""
    root = tmp_path / "run"
    agent = RecordingAgent()
    make_controller(root, agent=agent, rounds=3).run()
    markers = []
    for b in range(4):
        for r in range(1, 4):
            for i in range(2):
                tag = marker_tag(b, r, i)
                markers += [f"summary_{tag}", f"rationale_{tag}", interval_digits(b, r, i),
                            f"comment_{tag}", f"docstring_{tag}"]
    public = [p for p in root.rglob("*") if p.is_file() and "proposals" not in p.parts
              and "candidates" not in p.parts]
    assert any("evidence" in p.parts for p in public) and any("briefs" in p.parts for p in public)
    for path in public:
        text = path.read_text()
        assert not [m for m in markers if m in text], path
    for seen in agent.seen.values():
        assert not [m for m in markers if m in seen["text"]]
    # The search is able to find them where they belong.
    proposals = "".join(p.read_text() for p in (root / "proposals").glob("round_*.jsonl"))
    assert all(f"summary_{marker_tag(b, 1, 0)}" in proposals and interval_digits(b, 1, 0) in proposals
               for b in range(4))
    raw = "".join(p.read_text() for p in (root / "candidates").rglob("solver.py"))
    assert f"comment_{marker_tag(0, 1, 0)}" in raw


def test_ldm4geo_and_accumulated_runs_send_identical_briefs(tmp_path):
    """AT-11: two full runs; only the clearing (directories and session cursor) differs."""
    agents = {arm: RecordingAgent() for arm in ("ldm4geo", "accumulated")}
    for arm, agent in agents.items():
        make_controller(tmp_path / arm, arm=arm, agent=agent, rounds=5, reset=2).run()
    a, b = briefs(tmp_path / "ldm4geo"), briefs(tmp_path / "accumulated")
    assert len(a) == 20 and a == b
    assert (tmp_path / "ldm4geo/ledger.jsonl").read_bytes() == (tmp_path / "accumulated/ledger.jsonl").read_bytes()
    assert sorted(p.name for p in (tmp_path / "ldm4geo/harness").iterdir()) == ["epoch_00", "epoch_01", "epoch_02"]
    assert sorted(p.name for p in (tmp_path / "accumulated/harness").iterdir()) == ["epoch_00"]
    assert agents["ldm4geo"].seen[(0, 4)]["session_round"] == 1
    assert agents["accumulated"].seen[(0, 4)]["session_round"] == 3
    assert {json.loads(v)["kind"] for (_, r), v in a.items() if r in (1, 3, 5)} == {"boundary"}
    for arm in agents:
        assert verify_briefs(tmp_path / arm)["briefs_checked"] == 20


@pytest.mark.parametrize("arm", ["ldm4geo", "accumulated"])
def test_branch_markers_hidden_within_epoch_and_visible_after_boundary(tmp_path, arm):
    """AT-12: identifier markers survive stripping; siblings' markers appear only after a boundary."""
    root, reset = tmp_path / "run", 2
    agent = RecordingAgent()
    make_controller(root, arm=arm, agent=agent, rounds=5, reset=reset).run()
    ledger = read(root / "ledger.jsonl")
    tags = tags_by_program(root)
    charged = {r["program_id"] for r in ledger}
    checked = {"hidden": 0, "visible": 0}
    for (branch, round_index), seen in agent.seen.items():
        start = (round_index - 1) // reset * reset + 1
        for record in ledger[1:]:
            marker = "ident_" + tags[record["program_id"]]
            if record["round"] >= round_index:
                assert marker not in seen["text"]
            elif record["branch"] != branch and record["round"] >= start:
                assert marker not in seen["text"]
                checked["hidden"] += 1
            else:
                assert marker in seen["text"]
                checked["visible"] += record["branch"] != branch
        for program, tag in tags.items():
            if program not in charged:
                assert "ident_" + tag not in seen["text"]
    assert checked["hidden"] > 0 and checked["visible"] > 0


@pytest.mark.parametrize("kind,at", [("agent", (1, 2)), ("agent", (2, 3)), ("evaluator", 7), ("evaluator", 10)])
def test_interruption_mid_epoch_and_at_boundary_resumes_identically(tmp_path, kind, at):
    """AT-14 (exception at a step): mid-epoch round 2 and boundary round 3, proposals and evaluation."""
    baseline = tmp_path / "baseline"
    make_controller(baseline, rounds=4, reset=2).run()
    interrupted = tmp_path / "interrupted"
    failing = {"agent": FailingAgent(at)} if kind == "agent" else {"evaluator": FailingEvaluator(at)}
    with pytest.raises(Crash):
        make_controller(interrupted, rounds=4, reset=2, **failing).run()
    make_controller(interrupted, rounds=4, reset=2).run(resume=True)
    assert tree(interrupted) == tree(baseline)


KILL_CHILD = r"""
import os, signal, sys, time
from tasks.copper_inversion.tests import test_rsi_controller as t
root, kind, a, b = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
class Agent(t.ScriptedAgent):
    def propose(self, *, branch, round_index, **kw):
        if kind == "agent" and (branch, round_index) == (a, b):
            time.sleep(0.05)
            os.kill(os.getpid(), signal.SIGKILL)
        return super().propose(branch=branch, round_index=round_index, **kw)
class Evaluator(t.ScriptedEvaluator):
    count = 0
    def evaluate_source(self, path):
        Evaluator.count += 1
        if kind == "evaluator" and Evaluator.count == a:
            os.kill(os.getpid(), signal.SIGKILL)
        return super().evaluate_source(path)
t.make_controller(root, rounds=4, reset=2, agent=Agent(), evaluator=Evaluator()).run()
"""


@pytest.mark.parametrize("kind,a,b", [("agent", 1, 2), ("agent", 0, 3), ("evaluator", 7, 0), ("evaluator", 10, 0)])
def test_killed_process_resumes_identically(tmp_path, kind, a, b):
    """AT-14 (SIGKILL): the process dies mid-epoch or at the boundary; resume equals an uninterrupted run."""
    baseline = tmp_path / "baseline"
    make_controller(baseline, rounds=4, reset=2).run()
    killed = tmp_path / "killed"
    child = subprocess.run([sys.executable, "-c", KILL_CHILD, str(killed), kind, str(a), str(b)], cwd=REPO,
                           env={**os.environ, "PYTHONPATH": str(REPO)}, capture_output=True, timeout=300)
    assert child.returncode == -9, child.stderr.decode()[-2000:]
    assert not (killed / "result.json").exists()
    make_controller(killed, rounds=4, reset=2).run(resume=True)
    assert tree(killed) == tree(baseline)


def _round_tokens(r, cached):
    totals = [1000 + 10 * b + r for b in range(4)]
    return sum(t if cached else t - int(t * .25) for t in totals)


def test_token_cap_stops_after_that_rounds_evaluation(tmp_path):
    """AT-18: stop only after the capped round's evaluation; records stay complete and freezable."""
    root = tmp_path / "run"
    cap = _round_tokens(1, True) + 1
    result = make_controller(root, rounds=4, cap=cap).run()
    assert (result["status"], result["stop_reason"], result["rounds_completed"]) == ("stopped", "token_cap", 2)
    ledger = read(root / "ledger.jsonl")
    assert result["charged_records"] == len(ledger) == 1 + 2 * 4
    assert [sum(r["round"] == k for r in ledger) for k in range(4)] == [1, 4, 4, 0]
    assert result["input_tokens"] == _round_tokens(1, True) + _round_tokens(2, True)
    assert sorted(p.name for p in (root / "usage").iterdir()) == ["round_01.json", "round_02.json"]
    usage = json.loads((root / "usage/round_02.json").read_text())
    assert set(usage["by_profile"]) == {profile_name(b) for b in range(4)}
    assert usage["run_total_input_tokens"] == result["input_tokens"]
    assert not any(k[1] == 3 for k in briefs(root)) and not (root / "proposals/round_03").exists()
    assert verify_briefs(root)["briefs_checked"] == 8
    best = sorted((r for r in ledger if r["status"] == "valid"), key=lambda r: (r["risk"], -r["legal_rate"], r["seconds_per_solve"], r["program_id"]))
    assert result["training_finalists"] == [r["program_id"] for r in best[:3]]
    assert all((root / "candidates" / p / "solver.py").is_file() for p in result["training_finalists"])
    assert make_controller(root, rounds=4, cap=cap).run(resume=True) == result
    # Open author decision: excluding cached input lets the same cap run one more round.
    cap = _round_tokens(1, False) + _round_tokens(2, False) + 1
    counted = make_controller(tmp_path / "cached", rounds=4, cap=cap).run()
    uncached = make_controller(tmp_path / "uncached", rounds=4, cap=cap, token_cap_counts_cached=False).run()
    assert (counted["rounds_completed"], uncached["rounds_completed"]) == (2, 3)
    assert uncached["token_cap_basis"] == "uncached_input_tokens"
    assert make_controller(tmp_path / "nocap", rounds=4, cap=0).run()["stop_reason"] == "completed"
    last = make_controller(tmp_path / "last", rounds=2, cap=_round_tokens(1, True) + _round_tokens(2, True)).run()
    assert (last["stop_reason"], last["rounds_completed"]) == ("completed", 2)


def test_missing_usage_is_counted_never_a_reason_to_stop(tmp_path):
    root = tmp_path / "run"
    result = make_controller(root, agent=ScriptedAgent(missing_usage=2), rounds=3).run()
    assert result["stop_reason"] == "completed" and result["responses_missing_usage"] == 2 * 4 * 3
    usage = json.loads((root / "usage/round_03.json").read_text())
    assert all(v["responses_missing_usage"] == 2 for v in usage["by_profile"].values())
    assert usage["run_total_responses_missing_usage"] == 24


def test_branches_propose_concurrently_with_order_independent_results(tmp_path):
    class Concurrent(ScriptedAgent):
        def __init__(self):
            super().__init__()
            self.barrier, self.finished, self.lock = threading.Barrier(4, timeout=30), [], threading.Lock()
            self.roots = set()

        def propose(self, *, branch, round_index, profile, turn_root, **settings):
            self.barrier.wait()  # every branch is inside propose at the same time
            time.sleep(.05 * (3 - branch))
            with self.lock:
                self.finished.append((round_index, branch))
                self.roots.add((round_index, profile, str(turn_root)))
            return super().propose(branch=branch, round_index=round_index, **settings)

    baseline, parallel = tmp_path / "baseline", tmp_path / "parallel"
    make_controller(baseline, rounds=3).run()
    agent = Concurrent()
    make_controller(parallel, agent=agent, rounds=3).run()
    assert tree(parallel) == tree(baseline)
    assert agent.finished[:4] == [(1, 3), (1, 2), (1, 1), (1, 0)]
    assert len({root for r, _, root in agent.roots if r == 1}) == 4
    assert {p for r, p, _ in agent.roots} == {profile_name(b) for b in range(4)}


def test_failed_branch_keeps_sibling_receipts_and_resume_asks_only_it(tmp_path):
    root = tmp_path / "run"
    with pytest.raises(Crash):
        make_controller(root, agent=FailingAgent((2, 1))).run()
    assert sorted(p.name for p in (root / "proposals/round_01").iterdir()) == \
        ["branch_0.json", "branch_1.json", "branch_3.json"]
    assert not (root / "selection").exists()
    agent = FailingAgent(None)
    make_controller(root, agent=agent).run(resume=True)
    assert [c for c in agent.calls if c[0] == 1] == [(1, 2)]
    baseline = tmp_path / "baseline"
    make_controller(baseline).run()
    assert tree(root) == tree(baseline)


PROTOCOL = {
    "protocol_version": 1,
    "tiers": {"S": {"solver_seconds": 30, "slots": 321, "train_manifest": "train_s/manifest.json"},
              "F": {"solver_seconds": 120, "slots": 321}},
    "loop": {"rounds": 20, "branches": 4, "proposals_per_branch": 8, "batch": 16, "reserved_per_branch": 1,
             "reset_period": 5, "finalists": 3, "harness_seconds": 1800, "bash_calls": 8},
    "model": {"base_url": "https://model.invalid", "name": "fixture-model", "thinking": "low"},
    "token_cap_input": 0,
    "arms": {name: asdict(arm) for name, arm in ARMS.items()},
}


def test_config_from_protocol_reads_loop_tier_cap_model_and_arm():
    config = Config.from_protocol(PROTOCOL, "accumulated", "S", seed=101)
    assert (config.arm, config.seed, config.rounds, config.batch, config.reset_period, config.finalists,
            config.solver_seconds, config.tier, config.token_cap_input, config.slots) == \
        ("accumulated", 101, 20, 16, 5, 3, 30, "S", 0, 321)
    assert config.policy == ARMS["accumulated"] and config.model["name"] == "fixture-model"
    assert config.token_cap_counts_cached is True
    assert Config(**json.loads(json.dumps(asdict(config)))) == config
    assert replace(Config.from_protocol(PROTOCOL, "ldm4geo", "F"), seed=5).seed == 5
    custom = {**PROTOCOL, "arms": {"k3": {**asdict(ARMS["ldm4geo"])}}, "token_cap_counts_cached": False}
    assert Config.from_protocol(custom, "k3", "S").token_cap_counts_cached is False
    broken = [{**PROTOCOL, "loop": {k: v for k, v in PROTOCOL["loop"].items() if k != "batch"}},
              {**PROTOCOL, "loop": {**PROTOCOL["loop"], "reset_peroid": 5}},
              {**PROTOCOL, "tiers": {"S": {"solver_seconds": 30, "slots": 320}}},
              {**PROTOCOL, "arms": {"accumulated": {**asdict(ARMS["accumulated"]), "extra": 1}}},
              {**PROTOCOL, "arms": {}}, {k: v for k, v in PROTOCOL.items() if k != "model"}]
    for protocol in broken:
        with pytest.raises(ValueError):
            Config.from_protocol(protocol, "accumulated", "S")
    with pytest.raises(ValueError):
        Config.from_protocol(PROTOCOL, "accumulated", "M")


def test_arm_table_and_strict_definitions():
    assert set(ARMS) == {"ldm4geo", "accumulated", "restart_best", "independent_sampling", *EXTENSION}
    assert ARMS["single_conversation"] == Arm(1, 4, "never", "history", "all", "shared")
    assert Arm.from_dict(asdict(ARMS["separate_runs"])).budget_split == "separate"
    for bad in ({"branches": 4}, {**asdict(ARMS["ldm4geo"]), "clear": "sometimes"},
                {**asdict(ARMS["ldm4geo"]), "branches": 0}):
        with pytest.raises(ValueError):
            Arm.from_dict(bad)
    with pytest.raises(ValueError):
        policy("tree_search")


@pytest.mark.parametrize("arm", EXTENSION)
def test_extension_arms_are_refused_clearly(tmp_path, arm):
    config = Config.from_protocol(PROTOCOL, arm, "S")
    with pytest.raises(ValueError, match="extension arm"):
        Controller(tmp_path / "run", config, ScriptedAgent(), ScriptedEvaluator(), scripted_select, "brief", {})


def test_controller_reads_fields_not_arm_names(tmp_path):
    make_controller(tmp_path / "named", rounds=3).run()
    make_controller(tmp_path / "custom", arm="renamed-arm", arm_policy=ARMS["ldm4geo"], rounds=3).run()
    assert briefs(tmp_path / "named") == briefs(tmp_path / "custom")
    two = Arm(2, 1, "every_k", "history", "own", "shared")
    result = make_controller(tmp_path / "two", arm="two-branches", arm_policy=two, rounds=3, n=3, batch=4).run()
    ledger = read(tmp_path / "two/ledger.jsonl")
    assert result["charged_records"] == 13 and {r["branch"] for r in ledger[1:]} == {0, 1}
    assert {k[0] for k in briefs(tmp_path / "two")} == {0, 1}
    pooled = Arm(2, 1, "every_k", "history", "all", "shared")
    agent = RecordingAgent()
    make_controller(tmp_path / "all", arm="pooled", arm_policy=pooled, agent=agent, rounds=2, reset=2).run()
    ledger = read(tmp_path / "all/ledger.jsonl")
    tags = tags_by_program(tmp_path / "all")
    sibling = next(r for r in ledger if r["round"] == 1 and r["branch"] == 1)
    assert "ident_" + tags[sibling["program_id"]] in agent.seen[(0, 2)]["text"]


@pytest.mark.parametrize("first", [3, 4])
def test_start_api_continues_a_ledger_prefix_with_identical_messages(tmp_path, first):
    """Resume API for continuations: prefix + previous pool reproduce the source run's later messages."""
    source = tmp_path / "source"
    make_controller(source, arm="accumulated", rounds=4, reset=2).run()
    prefix = [r for r in read(source / "ledger.jsonl") if r["round"] < first]
    pool = [json.loads(line) for line in (source / f"proposals/round_{first - 1:02d}.jsonl").read_text().splitlines()]
    start = Start(first, prefix, str(source), pool)
    target = tmp_path / "continuation"
    controller = make_controller(target, arm="accumulated", rounds=4, reset=2, start=start,
                                 evaluator=FailingEvaluator(3))
    controller.prepare()
    assert controller.turn_root(first, 2) == target / "harness/epoch_00/branch_2"  # forked sessions go here
    with pytest.raises(Crash):
        controller.run(resume=True)
    result = make_controller(target, arm="accumulated", rounds=4, reset=2, start=start).run(resume=True)
    assert briefs(target) == {k: v for k, v in briefs(source).items() if k[1] >= first}
    assert (target / "ledger.jsonl").read_bytes() == (source / "ledger.jsonl").read_bytes()
    assert result["charged_records"] == len(read(source / "ledger.jsonl"))
    assert verify_briefs(target)["briefs_checked"] == 4 * (5 - first)
    assert json.loads((target / "manifest.json").read_text())["start"]["round"] == first


def test_start_with_fresh_sessions_opens_with_a_boundary(tmp_path):
    source = tmp_path / "source"
    make_controller(source, arm="accumulated", rounds=4, reset=2).run()
    prefix = [r for r in read(source / "ledger.jsonl") if r["round"] < 4]
    start = Start(4, prefix, str(source), (), fresh_sessions=True)
    agent = RecordingAgent()
    make_controller(tmp_path / "fresh", arm="accumulated", rounds=4, reset=2, start=start, agent=agent).run()
    assert {seen["session_round"] for seen in agent.seen.values()} == {0}
    message = json.loads(briefs(tmp_path / "fresh")[(0, 4)])
    assert message["kind"] == "boundary" and message["brief"].startswith("Fixed task brief.")
    config = Config("accumulated", 7, reset_period=2)
    assert (session_round(config, 4), session_round(config, 4, start)) == (3, 0)
    with pytest.raises(ValueError, match="continuation"):
        Start(2, prefix, str(source))


@pytest.mark.parametrize("arm", ["ldm4geo", "accumulated", "restart_best", "independent_sampling"])
def test_core_arms_run_and_their_briefs_reconstruct(tmp_path, arm):
    root = tmp_path / arm
    agent = RecordingAgent()
    result = make_controller(root, arm=arm, agent=agent, rounds=3, reset=2).run()
    assert result["charged_records"] == 13 and verify_briefs(root)["briefs_checked"] == 12
    epochs = sorted(p.name for p in (root / "harness").iterdir())
    assert epochs == {"ldm4geo": ["epoch_00", "epoch_01"], "accumulated": ["epoch_00"],
                      "restart_best": ["epoch_00", "epoch_01"],
                      "independent_sampling": ["epoch_00", "epoch_01", "epoch_02"]}[arm]
    message = json.loads(briefs(root)[(1, 3)])
    assert message["kind"] == "boundary"
    ledger = read(root / "ledger.jsonl")
    if arm == "restart_best":
        best = min((r for r in ledger if r["round"] < 3), key=lambda r: (r["risk"], r["program_id"]))
        marker = tags_by_program(root).get(best["program_id"])
        assert ("ident_" + marker if marker else "return initialize(problem)") in message["brief"]
        assert "# history v1  records=1  rounds=0..%d\n" % best["round"] in message["brief"]
        assert best["program_id"][:12] in message["brief"]
    if arm == "independent_sampling":
        assert "solve(problem, budget, seed)" in message["brief"] and "# history" not in message["brief"]
        assert {seen["session_round"] for seen in agent.seen.values()} == {0}
        assert all(json.loads(v)["kind"] == "boundary" for v in briefs(root).values())
