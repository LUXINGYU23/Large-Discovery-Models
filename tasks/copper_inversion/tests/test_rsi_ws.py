"""Warm-start (WS) protocol/controller support and self-contained rbf_sdf solver tests."""

import ast
import copy
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from tasks.copper_inversion.core.candidate import validate_source
from tasks.copper_inversion.core.encoding import MAX_NORMALIZED_BYTES, normalized_code
from tasks.copper_inversion.experiments.rsi import launch
from tasks.copper_inversion.rsi import audit, protocol as protocols, workflow
from tasks.copper_inversion.rsi.controller import Config, Controller, INITIAL_SOURCE
from tasks.copper_inversion.rsi.ledger import read
from tasks.copper_inversion.rsi.render import stripped_source
from tasks.copper_inversion.rsi.scripted import ScriptedAgent, ScriptedEvaluator
from tasks.copper_inversion.rsi.selection import reserve_and_rank
from tasks.copper_inversion.v2 import fixtures
from tasks.copper_inversion.v2.contract import validate as validate_problem
from tasks.copper_inversion.v2.geometry import canonicalize
from tasks.copper_inversion.v2.lit_baselines import rbf_sdf as reference_rbf_sdf


EXAMPLE = Path(protocols.RESOURCES) / "rsi/protocol.example.json"
RBF_SDF_PATH = Path(protocols.RESOURCES) / "rsi/initial_programs/rbf_sdf.py"
RBF_SDF_REL = "tasks/copper_inversion/resources/rsi/initial_programs/rbf_sdf.py"
NOW = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
SIDECAR = "sha256:" + "a" * 64
EVALUATOR = "sha256:" + "b" * 64

CUSTOM_SOLVER = (
    b'"""Warm-start test solver."""\n'
    b"import numpy as np\n"
    b"from tasks.copper_inversion.v2.baselines import initialize\n\n"
    b"WARM_MARKER = 42\n\n"
    b"def solve(problem, budget=None, seed=0):\n"
    b"    _ = WARM_MARKER\n"
    b"    return initialize(problem)\n"
)


def _example():
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


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


def _load_self_contained_rbf_sdf():
    spec = importlib.util.spec_from_file_location("ws_rbf_sdf_solver", RBF_SDF_PATH)
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


def test_ws_experiment_registered_and_default_protocol_unchanged():
    assert "WS" in protocols.EXPERIMENTS
    base = _example()
    assert "initial_program" not in base
    assert protocols.resolve_initial_program(base) == INITIAL_SOURCE
    before_sha = protocols.protocol_sha256(base)
    before_id = protocols.protocol_id(base)

    warm = copy.deepcopy(base)
    warm["initial_program"] = {
        "path": RBF_SDF_REL,
        "sha256": hashlib.sha256(RBF_SDF_PATH.read_bytes()).hexdigest(),
    }
    protocols.validate(warm)
    assert protocols.protocol_sha256(warm) != before_sha
    assert protocols.protocol_id(warm) != before_id
    assert protocols.content_sha256(warm) != protocols.content_sha256(base)
    assert protocols.resolve_initial_program(warm) == RBF_SDF_PATH.read_bytes()


@pytest.mark.parametrize(
    "bad_spec",
    [
        None,
        "rbf_sdf.py",
        {},
        {"path": RBF_SDF_REL},
        {"sha256": "a" * 64},
        {"path": RBF_SDF_REL, "sha256": "a" * 64, "extra": 1},
        {"path": "", "sha256": "a" * 64},
        {"path": "../outside.py", "sha256": "a" * 64},
        {"path": RBF_SDF_REL, "sha256": "not-hex"},
    ],
)
def test_protocol_rejects_malformed_initial_program_spec(bad_spec):
    proto = _example()
    proto["initial_program"] = bad_spec
    with pytest.raises(ValueError, match="invalid protocol"):
        protocols.validate(proto)


def test_freeze_and_verify_bind_initial_program_bytes(tmp_path, freeze_sources):
    solver_path = tmp_path / "start_solver.py"
    solver_path.write_bytes(CUSTOM_SOLVER)
    sha = hashlib.sha256(CUSTOM_SOLVER).hexdigest()

    proto = _example()
    proto["initial_program"] = {"path": str(solver_path), "sha256": sha}
    proto_path = tmp_path / "protocol.json"
    proto_path.write_text(json.dumps(proto, indent=2, sort_keys=True), encoding="utf-8")

    record = protocols.freeze(proto_path, now=NOW, **freeze_sources)
    assert record["complete"] is True
    frozen = protocols.load(proto_path)
    current = protocols.local_hashes(**freeze_sources)
    assert protocols.verify(frozen, current) == []

    # Tampering with the starting solver on disk must be refused by verify and resolve_initial_program.
    solver_path.write_bytes(CUSTOM_SOLVER + b"\n# tampered\n")
    with pytest.raises(ValueError, match="initial_program"):
        protocols.resolve_initial_program(frozen)
    with pytest.raises(ValueError, match="frozen protocol hashes no longer match: initial_program"):
        protocols.verify(frozen, current)

    # Restoring the exact bytes restores verification; deleting fails verification.
    solver_path.write_bytes(CUSTOM_SOLVER)
    assert protocols.verify(frozen, current) == []
    solver_path.unlink()
    with pytest.raises(ValueError, match="frozen protocol hashes no longer match: initial_program"):
        protocols.verify(frozen, current)


def test_resolve_initial_program_rejects_symlink_invalid_and_oversized_sources(tmp_path):
    real = tmp_path / "real.py"
    real.write_bytes(CUSTOM_SOLVER)
    link = tmp_path / "link.py"
    link.symlink_to(real)
    proto = _example()
    proto["initial_program"] = {
        "path": str(link),
        "sha256": hashlib.sha256(CUSTOM_SOLVER).hexdigest(),
    }
    with pytest.raises(ValueError, match="symlink"):
        protocols.resolve_initial_program(proto)

    no_solve = tmp_path / "no_solve.py"
    body = b"x = 1\n"
    no_solve.write_bytes(body)
    proto["initial_program"] = {"path": str(no_solve), "sha256": hashlib.sha256(body).hexdigest()}
    with pytest.raises(ValueError, match="solve"):
        protocols.resolve_initial_program(proto)

    huge = tmp_path / "huge.py"
    big_body = (
        "def solve(problem, budget=None, seed=0):\n"
        + "".join(f"    v_{i} = {i} * ({i} + 1)\n" for i in range(2500))
        + "    return v_0\n"
    ).encode("utf-8")
    assert len(normalized_code(big_body).encode("utf-8")) > MAX_NORMALIZED_BYTES
    huge.write_bytes(big_body)
    proto["initial_program"] = {"path": str(huge), "sha256": hashlib.sha256(big_body).hexdigest()}
    with pytest.raises(ValueError, match="normalized bytes"):
        protocols.resolve_initial_program(proto)


@pytest.mark.parametrize("arm", ["ldm4geo", "accumulated", "independent_sampling", "restart_best"])
def test_controller_uses_custom_initial_source_for_slot_zero_and_boundary_payloads(tmp_path, arm):
    root = tmp_path / f"run-{arm}"
    custom_sha = validate_source(CUSTOM_SOLVER)
    default_sha = validate_source(INITIAL_SOURCE)
    assert custom_sha != default_sha

    config = Config(arm, 801, rounds=3, proposals_per_branch=2, batch=4, reset_period=2, token_cap_input=10**9)
    controller = Controller(
        root,
        config,
        ScriptedAgent(),
        ScriptedEvaluator(),
        _scripted_select,
        "Fixed task brief.",
        {"synthetic_test": True},
        initial_source=CUSTOM_SOLVER,
    )
    report = controller.run()
    assert report["status"] == "completed"
    assert report["charged_records"] == 1 + 3 * 4

    rows = read(root / "ledger.jsonl")
    assert rows[0]["seq"] == 0
    assert rows[0]["round"] == 0
    assert rows[0]["branch"] == -1
    assert rows[0]["selection"] == "initial"
    assert rows[0]["program_id"] == custom_sha
    assert (root / "candidates" / custom_sha / "solver.py").read_bytes() == CUSTOM_SOLVER
    assert not (root / "candidates" / default_sha / "solver.py").exists()

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["initial_source_sha256"] == custom_sha
    assert audit.verify_briefs(root)["status"] == "passed"

    expected_display = stripped_source(CUSTOM_SOLVER).strip()
    if arm == "independent_sampling":
        # Every round is a boundary with boundary_payload == "initial_program".
        for r in (1, 2, 3):
            brief_obj = json.loads(
                (root / "harness" / f"epoch_{r - 1:02d}" / "branch_0" / "briefs" / f"round_{r:02d}.json").read_text(
                    encoding="utf-8"
                )
            )
            assert expected_display in brief_obj["brief"]
            assert "WARM_MARKER = 42" in brief_obj["brief"]
            assert (
                root / "harness" / f"epoch_{r - 1:02d}" / "branch_0" / "public" / "evidence" / "sources" / f"{custom_sha[:12]}.py"
            ).is_file()
    else:
        ev_source = (
            root / "harness" / "epoch_00" / "branch_0" / "public" / "evidence" / "sources" / f"{custom_sha[:12]}.py"
        )
        assert ev_source.is_file()
        assert "WARM_MARKER = 42" in ev_source.read_text(encoding="utf-8")

    # Resuming with a different initial_source must fail manifest check; same initial_source succeeds.
    wrong = Controller(
        root,
        config,
        ScriptedAgent(),
        ScriptedEvaluator(),
        _scripted_select,
        "Fixed task brief.",
        {"synthetic_test": True},
    )
    with pytest.raises(ValueError, match="resume identity differs"):
        wrong.run(resume=True)

    same = Controller(
        root,
        config,
        ScriptedAgent(),
        ScriptedEvaluator(),
        _scripted_select,
        "Fixed task brief.",
        {"synthetic_test": True},
        initial_source=CUSTOM_SOLVER,
    )
    assert same.run(resume=True)["charged_records"] == 13


def test_default_controller_manifest_excludes_initial_source_field(tmp_path):
    root = tmp_path / "default-run"
    config = Config("ldm4geo", 7, rounds=1, proposals_per_branch=2, batch=4, reset_period=2, token_cap_input=10**9)
    controller = Controller(
        root,
        config,
        ScriptedAgent(),
        ScriptedEvaluator(),
        _scripted_select,
        "Fixed task brief.",
        {"synthetic_test": True},
    )
    controller.run()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert "initial_source_sha256" not in manifest
    assert read(root / "ledger.jsonl")[0]["program_id"] == validate_source(INITIAL_SOURCE)


def test_workflow_prepare_and_launch_check_batch_support_ws(tmp_path, freeze_sources):
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
        "tagger": {"mode": "tags", "rules": protocols.validate_analysis({
            "checkpoint_experiment": "M1",
            "checkpoint_round": 10,
            "freeze_prefixes": {"M1": ["full"], "E1": ["full"], "E2": ["full"]},
            "tagger": {"mode": "tags", "rules": __import__(
                "tasks.copper_inversion.experiments.rsi.analysis.tagger", fromlist=["DRAFT_RULES"]
            ).DRAFT_RULES, "cluster_reference": None},
        })["tagger"]["rules"], "cluster_reference": None},
    }
    rbf_bytes = RBF_SDF_PATH.read_bytes()
    rbf_sha = hashlib.sha256(rbf_bytes).hexdigest()
    proto["initial_program"] = {"path": RBF_SDF_REL, "sha256": rbf_sha}
    proto_path = tmp_path / "protocol.json"
    proto_path.write_text(json.dumps(proto, indent=2, sort_keys=True), encoding="utf-8")
    protocols.freeze(proto_path, now=NOW, **freeze_sources)
    frozen = protocols.load(proto_path)

    batch = {
        "schema": "rsi-batch-v1",
        "name": "ws-h12",
        "protocol": str(proto_path),
        "experiment": "WS",
        "tier": "S",
        "prefixes": ["full", 5, 10],
        "runs": [
            {"arm": arm, "judge": "standard", "seed": seed}
            for seed in (801, 802, 803, 804)
            for arm in ("ldm4geo", "accumulated")
        ],
    }
    summary = launch.check_batch(batch, frozen, {"max_concurrent_runs": {"S": 8}, "max_agent_sessions": 32})
    assert summary["experiment"] == "WS"
    assert len(summary["runs"]) == 8

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
    assert plan["initial_source"] == rbf_bytes
    assert plan["manifest"]["initial_program"] == {"path": RBF_SDF_REL, "sha256": rbf_sha}


def test_self_contained_rbf_sdf_hygiene_and_exact_equivalence():
    raw = RBF_SDF_PATH.read_bytes()
    sha = validate_source(raw)
    assert len(sha) == 64

    norm = normalized_code(raw).encode("utf-8")
    assert len(norm) <= MAX_NORMALIZED_BYTES
    assert audit.IOU.search(raw) is None

    tree = ast.parse(raw.decode("utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative imports are forbidden in standalone solver.py"
            imported.add(node.module or "")
    forbidden = {
        "tasks.copper_inversion.v2.lit_baselines",
        "tasks.copper_inversion.v2.baseline_harness",
        "tasks.copper_inversion.v2.baselines",
        "tasks.copper_inversion.v2.fdi_baseline",
        "tasks.copper_inversion.v2.offline_baselines",
    }
    assert not (imported & forbidden)
    allowed_task_imports = {
        "tasks.copper_inversion.v2.contract",
        "tasks.copper_inversion.v2.geometry",
    }
    for mod in imported:
        if mod.startswith("tasks."):
            assert mod in allowed_task_imports

    mod = _load_self_contained_rbf_sdf()
    budget = {"wall_time_seconds": 30.0}
    probs = list(fixtures.fixtures())

    # Add multi-hole CV-active problem and all-barren problem to exercise CV + fallback paths.
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

    for prob in [*probs, cv_prob, barren_prob]:
        expected = reference_rbf_sdf(prob, budget, 0)
        actual = mod.solve(prob, budget, 0)
        canonicalize(prob, actual)
        assert np.array_equal(actual, expected)
