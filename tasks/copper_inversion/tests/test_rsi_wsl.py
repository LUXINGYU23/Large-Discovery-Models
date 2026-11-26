"""LDM4Geo with published baselines library (WSL) protocol, controller, agent, and solver tests."""

import ast
import copy
import hashlib
import importlib.util
import json
import stat
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tasks.copper_inversion.core.candidate import validate_source
from tasks.copper_inversion.core.encoding import MAX_NORMALIZED_BYTES, normalized_code
from tasks.copper_inversion.experiments.rsi import launch
from tasks.copper_inversion.rsi import agent as agent_mod, audit, protocol as protocols, workflow
from tasks.copper_inversion.rsi.controller import (
    Config,
    Controller,
    INITIAL_SOURCE,
    profile_name,
    reference_library_paragraph,
)
from tasks.copper_inversion.rsi.ledger import read
from tasks.copper_inversion.rsi.scripted import ScriptedAgent, ScriptedEvaluator
from tasks.copper_inversion.rsi.selection import reserve_and_rank
from tasks.copper_inversion.v2 import fixtures
from tasks.copper_inversion.v2.baselines import REFERENCE_SOLVERS
from tasks.copper_inversion.v2.contract import validate as validate_problem
from tasks.copper_inversion.v2.geometry import canonicalize


EXAMPLE = Path(protocols.RESOURCES) / "rsi/protocol.example.json"
REF_LIB_DIR = Path(protocols.RESOURCES) / "reference_solvers"
REF_LIB_REL = "tasks/copper_inversion/resources/reference_solvers"
EXPECTED_SOLVERS = (
    "ellipsoid_fit",
    "fdi_smooth",
    "gp_classifier",
    "indicator_kriging",
    "knn_idw",
    "layered_horizon",
    "rbf_sdf",
    "stretched_nearest_class",
    "svm_rbf",
)
NOW = datetime(2026, 11, 26, 2, 0, tzinfo=timezone.utc)
SIDECAR = "sha256:" + "a" * 64
EVALUATOR = "sha256:" + "b" * 64


def _example():
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def _ref_library_spec():
    return {
        "dir": REF_LIB_REL,
        "files": {
            name: hashlib.sha256((REF_LIB_DIR / f"{name}.py").read_bytes()).hexdigest()
            for name in EXPECTED_SOLVERS
        },
    }


def _scripted_select(pool, history, count):
    predictions = [
        {
            "program_id": p["program_id"],
            "branch": p["branch"],
            "mean": 0.5,
            "std": 0.1,
            "acquisition": int(p["program_id"][:8], 16) / 0xFFFFFFFF,
        }
        for p in pool
    ]
    return {"predictions": predictions, "selected": reserve_and_rank(predictions, count)}


def _load_standalone_solver(name):
    path = REF_LIB_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"wsl_ref_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def freeze_sources(tmp_path, monkeypatch):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    (analysis / "metrics.py").write_text("X = 1\n", encoding="utf-8")
    (analysis / "tagger.py").write_text("RULES = ()\n", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    (model / "provenance.json").write_text('{"model": "fixture"}\n', encoding="utf-8")
    monkeypatch.setattr(protocols, "ANALYSIS", analysis)
    monkeypatch.setattr(protocols, "TAGGER", analysis / "tagger.py")
    return {"embedding_model_path": model, "sidecar_image": SIDECAR, "evaluator_image": EVALUATOR}


def test_reference_library_protocol_hashing_and_default_unchanged():
    base = _example()
    assert "reference_library" not in base
    assert protocols.resolve_reference_library(base) == {}
    before_sha = protocols.protocol_sha256(base)
    before_id = protocols.protocol_id(base)

    wsl = copy.deepcopy(base)
    wsl["reference_library"] = _ref_library_spec()
    protocols.validate(wsl)
    assert protocols.protocol_sha256(wsl) != before_sha
    assert protocols.protocol_id(wsl) != before_id
    assert protocols.content_sha256(wsl) != protocols.content_sha256(base)
    assert protocols.resolve_initial_program(wsl) == INITIAL_SOURCE
    resolved = protocols.resolve_reference_library(wsl)
    assert tuple(sorted(resolved)) == EXPECTED_SOLVERS
    for name in EXPECTED_SOLVERS:
        assert resolved[name] == (REF_LIB_DIR / f"{name}.py").read_bytes()


@pytest.mark.parametrize(
    "bad_spec",
    [
        None,
        "reference_solvers",
        {},
        {"dir": REF_LIB_REL},
        {"files": {"rbf_sdf": "a" * 64}},
        {"dir": REF_LIB_REL, "files": {}, "extra": 1},
        {"dir": "", "files": {"rbf_sdf": "a" * 64}},
        {"dir": "../outside", "files": {"rbf_sdf": "a" * 64}},
        {"dir": REF_LIB_REL, "files": {}},
        {"dir": REF_LIB_REL, "files": {"bad-name": "a" * 64}},
        {"dir": REF_LIB_REL, "files": {"rbf_sdf.py": "a" * 64}},
        {"dir": REF_LIB_REL, "files": {"rbf_sdf": "not-hex"}},
    ],
)
def test_protocol_rejects_malformed_reference_library(bad_spec):
    proto = _example()
    proto["reference_library"] = bad_spec
    with pytest.raises(ValueError, match="invalid protocol"):
        protocols.validate(proto)


def test_freeze_and_verify_bind_reference_library_files(tmp_path, freeze_sources):
    lib_dir = tmp_path / "ref_lib"
    lib_dir.mkdir()
    body_a = (REF_LIB_DIR / "ellipsoid_fit.py").read_bytes()
    body_b = (REF_LIB_DIR / "knn_idw.py").read_bytes()
    (lib_dir / "ellipsoid_fit.py").write_bytes(body_a)
    (lib_dir / "knn_idw.py").write_bytes(body_b)

    proto = _example()
    proto["reference_library"] = {
        "dir": str(lib_dir),
        "files": {
            "ellipsoid_fit": hashlib.sha256(body_a).hexdigest(),
            "knn_idw": hashlib.sha256(body_b).hexdigest(),
        },
    }
    proto_path = tmp_path / "protocol.json"
    proto_path.write_text(json.dumps(proto, indent=2, sort_keys=True), encoding="utf-8")

    record = protocols.freeze(proto_path, now=NOW, **freeze_sources)
    assert record["complete"] is True
    frozen = protocols.load(proto_path)
    current = protocols.local_hashes(**freeze_sources)
    assert protocols.verify(frozen, current) == []

    # Modifying or deleting any file in the library is refused by verify and resolve_reference_library.
    (lib_dir / "knn_idw.py").write_bytes(body_b + b"\n# modified\n")
    with pytest.raises(ValueError, match="reference_library"):
        protocols.resolve_reference_library(frozen)
    with pytest.raises(ValueError, match="frozen protocol hashes no longer match: reference_library"):
        protocols.verify(frozen, current)

    (lib_dir / "knn_idw.py").write_bytes(body_b)
    assert protocols.verify(frozen, current) == []

    # An 'iou' substring inside a reference solver is also refused.
    iou_solver = b"def solve(problem, budget=None, seed=0):\n    iou_val = 0.0\n    return problem['domain_mask'].astype(float)\n"
    (lib_dir / "bad_iou.py").write_bytes(iou_solver)
    bad_proto = copy.deepcopy(proto)
    bad_proto["reference_library"]["files"]["bad_iou"] = hashlib.sha256(iou_solver).hexdigest()
    with pytest.raises(ValueError, match="iou"):
        protocols.resolve_reference_library(bad_proto)


@pytest.mark.parametrize("arm", ["ldm4geo", "accumulated", "independent_sampling", "restart_best"])
def test_controller_delivers_readonly_reference_library_every_round_branch_and_clear_and_passes_blind_audit(
    tmp_path, arm
):
    root = tmp_path / f"run-{arm}"
    subset = {
        name: (REF_LIB_DIR / f"{name}.py").read_bytes()
        for name in ("ellipsoid_fit", "knn_idw", "rbf_sdf")
    }
    seen_in_turns = []

    class InspectingAgent(ScriptedAgent):
        def propose(self, *, branch, round_index, count, message="", view=(), turn_root=None, profile=None, **kw):
            prof = profile_name(branch) if profile is None else profile
            ref_dir = Path(turn_root) / "sessions" / prof / "workspace" / "reference_solvers"
            assert ref_dir.is_dir()
            assert stat.S_IMODE(ref_dir.stat().st_mode) == 0o555
            found = {p.stem: (p.read_bytes(), stat.S_IMODE(p.stat().st_mode)) for p in sorted(ref_dir.glob("*.py"))}
            assert set(found) == set(subset)
            for name, (body, mode) in found.items():
                assert body == subset[name]
                assert mode == 0o444
            # Write a fake preserved provider request so scan_blind(require_requests=True) scans the run.
            req_dir = Path(turn_root) / "sessions" / prof / "turns" / f"round_{round_index:03d}"
            req_dir.mkdir(parents=True, exist_ok=True)
            (req_dir / "000.request.bin").write_bytes(message.encode("utf-8"))
            seen_in_turns.append((round_index, branch))
            return super().propose(branch=branch, round_index=round_index, count=count, message=message, view=view, **kw)

    config = Config(arm, 801, rounds=3, proposals_per_branch=2, batch=4, reset_period=2, token_cap_input=10**9)
    base_brief = (Path(protocols.RESOURCES) / "rsi/TASK.md").read_text(encoding="utf-8")
    controller = Controller(
        root,
        config,
        InspectingAgent(),
        ScriptedEvaluator(),
        _scripted_select,
        base_brief,
        {"synthetic_test": True},
        reference_library=subset,
    )
    report = controller.run()
    assert report["status"] == "completed"
    # 3 rounds x branches turns all saw read-only reference_solvers/<name>.py, including after epoch clear at round 3.
    assert len(seen_in_turns) == 3 * config.policy.branches

    # Starting program at round 0 / slot 0 remains built-in INITIAL_SOURCE (A0).
    rows = read(root / "ledger.jsonl")
    assert rows[0]["round"] == 0
    assert rows[0]["program_id"] == validate_source(INITIAL_SOURCE)

    # Task brief contains the neutral paragraph (no scores, no ranking, no 'iou') and verify_briefs passes.
    paragraph = reference_library_paragraph(subset)
    assert audit.IOU.search(paragraph.encode("utf-8")) is None
    saved_brief = (root / "task_brief.txt").read_text(encoding="utf-8")
    assert paragraph in saved_brief
    assert audit.verify_briefs(root)["status"] == "passed"

    # Blind-channel audit passes with zero leaks and zero unexplained 'iou' review hits.
    private = tmp_path / f"private-{arm}"
    (private / "evaluations" / "e0").mkdir(parents=True)
    (private / "rsi-private-audit.json").write_text(
        json.dumps({"sentinel": "PRIVATE-SENTINEL-TOKEN", "private_parent_ids": ["priv-parent-01"], "case_ids": []}),
        encoding="utf-8",
    )
    (private / "evaluations" / "e0" / "result.json").write_text(
        json.dumps({"metrics": {"mean_iou": 0.51234567891234}}),
        encoding="utf-8",
    )
    blind = audit.scan_blind(root, private, require_requests=True)
    assert blind["status"] == "passed"
    assert blind["leaks"] == 0
    assert blind["review_required"] == 0
    assert blind["iou_hits_by_kind"]["unexplained"] == 0


def test_pi_agent_populates_reference_library_after_sidecar_client_start(tmp_path, monkeypatch):
    from tasks.copper_inversion.v2.data import save_problem
    from tasks.copper_inversion.v2.public_catalog import SCHEMA as CATALOG_SCHEMA, file_hash

    prob = next(fixtures.fixtures())
    case_dir = tmp_path / "case0"
    save_problem(prob, case_dir)
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "schema": CATALOG_SCHEMA,
                "stage": "development",
                "split": "train",
                "cases": [
                    {
                        "case_id": prob["case_id"],
                        "directory": "case0",
                        "sha256": {
                            "problem.json": file_hash(case_dir / "problem.json"),
                            "problem.npz": file_hash(case_dir / "problem.npz"),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    subset = {"rbf_sdf": (REF_LIB_DIR / "rbf_sdf.py").read_bytes()}
    base_brief = (Path(protocols.RESOURCES) / "rsi/TASK.md").read_text(encoding="utf-8")
    full_brief = base_brief.rstrip() + "\n\n" + reference_library_paragraph(subset) + "\n"

    monkeypatch.setattr(agent_mod, "load_pi_guest_runtime", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(agent_mod.PiAgent, "config", lambda self, **kw: SimpleNamespace(limits=SimpleNamespace(wall_time_seconds=10)))

    events = []

    class FakeClient:
        def run_turn(self, turns, *, submission_validator, recovery_timeout_seconds):
            # Inside run_turn, reference_solvers/rbf_sdf.py and updated public/TASK.md must already be present.
            ws_ref = turn_root / "sessions/researcher_0/workspace/reference_solvers/rbf_sdf.py"
            assert ws_ref.is_file()
            assert stat.S_IMODE(ws_ref.stat().st_mode) == 0o444
            assert stat.S_IMODE(ws_ref.parent.stat().st_mode) == 0o555
            assert (turn_root / "public/TASK.md").read_text(encoding="utf-8") == full_brief
            events.append("run_turn")
            cand_dir = turn_root / "sessions/researcher_0/turns/round_001/submitted"
            cand_dir.mkdir(parents=True, exist_ok=True)
            body = b"from tasks.copper_inversion.v2.baselines import initialize\ndef solve(p, b, s):\n    return initialize(p)\n"
            snap = cand_dir / "solver.py"
            snap.write_bytes(body)
            art = SimpleNamespace(
                path_pointer="/candidates/0/artifact_path",
                relative_path="candidates/0/solver.py",
                snapshot_path=str(snap.relative_to(turn_root)),
                size_bytes=len(body),
                sha256=hashlib.sha256(body).hexdigest(),
            )
            sub = {
                "candidates": [
                    {
                        "artifact_path": "candidates/0/solver.py",
                        "parent_sha256": None,
                        "change_summary": "init",
                        "rationale": "test",
                        "predicted_risk": 0.5,
                        "interval80_low": 0.3,
                        "interval80_high": 0.7,
                    }
                ]
            }
            assert submission_validator(SimpleNamespace(submission=sub, artifacts=(art,))) == agent_mod.HarnessSubmissionValidation("accept")
            return (
                SimpleNamespace(
                    session_id="s0",
                    artifacts={"turn": "sessions/researcher_0/turns/round_001"},
                    usage={"providerCalls": 0},
                    submission_status="accepted",
                    submission=sub,
                    submitted_artifacts=(art,),
                ),
            )

        def close(self):
            pass

    def fake_client(self, profile, root, config, name):
        # Before sidecar pinInitialization runs, turn_root/sessions must NOT exist yet.
        assert not (Path(root) / "sessions").exists()
        (Path(root) / "sessions" / profile / "workspace").mkdir(parents=True, exist_ok=True)
        events.append("client_started")
        return FakeClient()

    monkeypatch.setattr(agent_mod.PiAgent, "_client", fake_client)
    monkeypatch.setattr(agent_mod, "collect", lambda turn_dir, calls: {
        "requests": 1, "input_tokens": 10, "cached_input_tokens": 0,
        "output_tokens": 5, "responses_missing_usage": 0, "compactions": 0,
    })

    pi = agent_mod.PiAgent(
        public_catalog=catalog,
        api_key="test-key",
        sidecar_image=SIDECAR,
        cache_dir=tmp_path / "cache",
        arm="ldm4geo",
        reset_period=5,
        base_url="http://127.0.0.1:1",
        model="deepseek-flash",
        thinking="high",
        bash_calls=8,
        reference_library=subset,
        task_brief=full_brief,
    )
    turn_root = tmp_path / "turn"
    cand_root = tmp_path / "run"
    cand_root.mkdir()
    rows, _ = pi.propose(
        branch=0,
        round_index=1,
        epoch=0,
        count=1,
        session_round=0,
        message='{"kind":"boundary"}',
        view=[],
        turn_root=turn_root,
        candidate_root=cand_root,
        known=frozenset(),
        seed=801,
        harness_seconds=30,
    )
    assert events == ["client_started", "run_turn"]
    assert len(rows) == 1


def test_workflow_prepare_and_single_arm_launch_check_support_wsl(tmp_path, freeze_sources):
    from tasks.copper_inversion.experiments.rsi.analysis.tagger import DRAFT_RULES

    proto = _example()
    proto["release"] = {"name": "rsi-r2-h12", "qualification_sha256": "c" * 64}
    proto["token_cap_input"] = 300_000_000
    proto["seeds"]["WS"] = [801, 802, 803, 804]
    proto["analysis"] = {
        "checkpoint_experiment": "M1",
        "checkpoint_round": 10,
        "freeze_prefixes": {
            "M1": ["full", 5, 10],
            "E1": ["full", 12, 14, 15, 16, 18],
            "E2": ["full"],
            "WS": ["full", 5, 10],
        },
        "tagger": {"mode": "tags", "rules": DRAFT_RULES, "cluster_reference": None},
    }
    proto["reference_library"] = _ref_library_spec()
    proto_path = tmp_path / "protocol.json"
    proto_path.write_text(json.dumps(proto, indent=2, sort_keys=True), encoding="utf-8")
    protocols.freeze(proto_path, now=NOW, **freeze_sources)
    frozen = protocols.load(proto_path)

    batch = {
        "schema": "rsi-batch-v1",
        "name": "wsl-h12",
        "protocol": str(proto_path),
        "experiment": "WS",
        "tier": "S",
        "prefixes": ["full", 5, 10],
        "runs": [{"arm": "ldm4geo", "judge": "standard", "seed": seed} for seed in (801, 802, 803, 804)],
    }
    summary = launch.check_batch(batch, frozen, {"max_concurrent_runs": {"S": 8}, "max_agent_sessions": 32})
    assert summary["experiment"] == "WS"
    assert summary["arms"] == ["ldm4geo"]
    assert summary["runs"] == ["ldm4geo-s801", "ldm4geo-s802", "ldm4geo-s803", "ldm4geo-s804"]
    assert summary["agent_sessions"] == 16

    plan = workflow.prepare(
        workflow.parse_args(
            [
                "--protocol", str(proto_path),
                "--experiment", "WS",
                "--arm", "ldm4geo",
                "--tier", "S",
                "--judge", "standard",
                "--seed", "801",
                "--out-dir", str(tmp_path / "ldm4geo-s801"),
                "--public-catalog", str(tmp_path / "catalog.json"),
                "--evaluation-socket", str(tmp_path / "e.sock"),
                "--evaluation-contract", "0" * 64,
                "--evaluation-service-uid", "1000",
                "--embedding-model-path", str(freeze_sources["embedding_model_path"]),
                "--sidecar-image", SIDECAR,
                "--harness-cache-dir", str(tmp_path / "cache"),
                "--api-key-file", str(tmp_path / "key"),
            ]
        )
    )
    assert plan["initial_source"] == INITIAL_SOURCE
    assert tuple(sorted(plan["reference_library"])) == EXPECTED_SOLVERS
    assert plan["manifest"]["reference_library"] == _ref_library_spec()


def test_all_nine_self_contained_reference_solvers_hygiene_and_exact_equivalence():
    present = tuple(sorted(p.stem for p in REF_LIB_DIR.glob("*.py")))
    assert present == EXPECTED_SOLVERS

    forbidden = {
        "tasks.copper_inversion.v2.lit_baselines",
        "tasks.copper_inversion.v2.baseline_harness",
        "tasks.copper_inversion.v2.baselines",
        "tasks.copper_inversion.v2.fdi_baseline",
        "tasks.copper_inversion.v2.offline_baselines",
    }
    allowed_task_imports = {
        "tasks.copper_inversion.v2.contract",
        "tasks.copper_inversion.v2.geometry",
    }

    probs = list(fixtures.fixtures())
    base_prob = copy.deepcopy(probs[0])
    origin = np.asarray(base_prob["origin"], dtype=float)
    spacing = np.asarray(base_prob["spacing"], dtype=float)
    pts, labs, grps = [], [], []
    for h_idx, (ix, iy) in enumerate([(1, 1), (1, 6), (3, 2), (4, 5), (6, 1), (7, 6)]):
        for iz in range(6):
            pts.append(origin + (np.array([ix, iy, iz], dtype=float) + 0.5) * spacing)
            is_ore = (iz in (2, 3)) and (h_idx % 2 == 0 or iz == 2)
            labs.append(1.0 if is_ore else 0.0)
            grps.append(h_idx)
        pts.append(origin + (np.array([ix, iy, 1.5], dtype=float) + 0.5) * spacing)
        labs.append(0.5)
        grps.append(h_idx)
    cv_prob = validate_problem(
        {
            **base_prob,
            "observations": [
                {
                    **base_prob["observations"][0],
                    "points": np.asarray(pts, dtype=float),
                    "labels": np.asarray(labs, dtype=float),
                    "groups": np.asarray(grps, dtype=np.int64),
                    "label_error": 0.05,
                }
            ],
        }
    )
    barren_prob = copy.deepcopy(cv_prob)
    barren_prob["observations"][0] = {
        **barren_prob["observations"][0],
        "labels": np.zeros(len(labs), dtype=float),
    }
    barren_prob = validate_problem(barren_prob)
    budget = {"wall_time_seconds": 30.0}

    for name in EXPECTED_SOLVERS:
        raw = (REF_LIB_DIR / f"{name}.py").read_bytes()
        validate_source(raw)
        assert len(normalized_code(raw).encode("utf-8")) <= MAX_NORMALIZED_BYTES
        assert audit.IOU.search(raw) is None, f"{name}.py contains 'iou'"

        tree = ast.parse(raw.decode("utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, f"{name}.py uses relative import"
                imported.add(node.module or "")
        assert not (imported & forbidden), f"{name}.py imports {imported & forbidden}"
        for mod_name in imported:
            if mod_name.startswith("tasks."):
                assert mod_name in allowed_task_imports, f"{name}.py imports {mod_name}"

        mod = _load_standalone_solver(name)
        ref_fn = REFERENCE_SOLVERS[name]
        for prob in [*probs, cv_prob, barren_prob]:
            expected = ref_fn(prob, budget, 0)
            actual = mod.solve(prob, budget, 0)
            canonicalize(prob, actual)
            assert np.array_equal(actual, expected), f"mismatch for {name} on {prob['case_id']}"
