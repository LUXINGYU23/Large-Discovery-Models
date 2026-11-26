"""LDM4Geo with published baselines library (WSL) protocol, controller, agent, guest, and solver tests."""

import ast
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from ldm_tts.harness import HarnessArtifactRule, HarnessLimits, HarnessNetworkPolicy, HarnessProfile, HarnessSubmissionContract
from ldm_tts.harness.container import pi_sidecar_command
from ldm_tts.harness.pi import PiHarnessConfig, load_pi_guest_runtime
from ldm_tts.harness.pi_guest import PiGuestRuntime
from tasks.copper_inversion.core.candidate import validate_source
from tasks.copper_inversion.core.encoding import MAX_NORMALIZED_BYTES, normalized_code
from tasks.copper_inversion.experiments.baselines import fdi_baseline
from tasks.copper_inversion.experiments.rsi import launch
from tasks.copper_inversion.rsi import agent as agent_mod, audit, protocol as protocols, workflow
from tasks.copper_inversion.rsi.case_isolation import DEFAULT_CASE_ISOLATION, case_orientation, opaque_case_id
from tasks.copper_inversion.rsi.controller import (
    Config,
    Controller,
    INITIAL_SOURCE,
    profile_name,
    reference_library_paragraph,
    write_reference_library_dir,
)
from tasks.copper_inversion.rsi.ledger import read
from tasks.copper_inversion.rsi.scripted import ScriptedAgent, ScriptedEvaluator
from tasks.copper_inversion.rsi.selection import reserve_and_rank
from tasks.copper_inversion.v2 import fixtures
from tasks.copper_inversion.v2.baselines import REFERENCE_SOLVERS
from tasks.copper_inversion.v2.contract import validate as validate_problem
from tasks.copper_inversion.v2.data import save_problem
from tasks.copper_inversion.v2.geometry import canonicalize
from tasks.copper_inversion.v2.public_catalog import SCHEMA as CATALOG_SCHEMA, file_hash
from tasks.copper_inversion.v2.research import PublicTrainingExport
from tasks.copper_inversion.v2.suite import SuiteEvaluator


EXAMPLE = Path(protocols.RESOURCES) / "rsi/protocol.example.json"
REF_LIB_DIR = Path(protocols.RESOURCES) / "reference_solvers"
REF_LIB_REL = "tasks/copper_inversion/resources/reference_solvers"
V2_DIR = Path(protocols.TASK) / "v2"
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
# Exact file SHA-256 digests from pilot checkout /Aidata/shared/xingyulu/rsi4geo/code/r2 (4c0e8a4 on 1a65cfd).
PILOT_BASE_BRIEF_SHA256 = "26046a3c9b675359de22eeceb1133ce6cdb2265129bdbed70bef9569c78e9a26"
PILOT_EXPORT_FILE_SHA256 = {
    "TASK.md": PILOT_BASE_BRIEF_SHA256,
    "history.json": "37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570",
    "selfcheck.py": "0cb393b7736246582e94f298cea5d3d6a6bdabf24afbf8b08940b950b3adbbeb",
    "tasks/copper_inversion/v2/__init__.py": "d34141349dea989746e49b12afde42dd8b799c456561879728a986112cad7071",
    "tasks/copper_inversion/v2/baselines.py": "330637a602746a9de5f62cf238d5d36ea9c275b73dc009558461b348e85c225a",
    "tasks/copper_inversion/v2/contract.py": "ea47d979b8b61da42a03327831100c24e00617fbf8df9e4507ae7df51b0b8aa7",
    "tasks/copper_inversion/v2/data.py": "266f2e2dd70548149b3c32f7d993ee9c6af935acc089f16cca2a433850242988",
    "tasks/copper_inversion/v2/fixtures.py": "c7b443f27b21039d9beddd6364b9b8ab6089ec1e8c45b534a349492606d1382a",
    "tasks/copper_inversion/v2/geological_policy.py": "81ccd3166fa255ef80131b07afbd6482b892d623c67b83af0fc690fc9afd1400",
    "tasks/copper_inversion/v2/geometry.py": "3249afa9acd97c67574583e8120220c54a75da8cb5ddc8cccff2644f865ddc0b",
    "tasks/copper_inversion/v2/objective.py": "a8e8ccfcd75412059c422498270954ed38b1860f3ebadf6f95e4a435afccf991",
}
PILOT_V2_MODULES = frozenset({
    "__init__.py",
    "baseline_harness.py",
    "baselines.py",
    "budget.py",
    "contract.py",
    "data.py",
    "domain.py",
    "fixtures.py",
    "geological_policy.py",
    "geometry.py",
    "interface_checks.py",
    "lit_baselines.py",
    "metrics.py",
    "objective.py",
    "public_catalog.py",
    "qualification.py",
    "research.py",
    "runner.py",
    "scorer.py",
    "selection.py",
    "service.py",
    "service_client.py",
    "suite.py",
    "training_receipts.py",
})
PILOT_IMPLEMENTATION_IDENTITY = "77ef2d5f6d84409cc5507ff965faaef68bbcffd4369560f64abb99b876d94219"
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


def _make_catalog(tmp_path):
    prob = next(fixtures.fixtures())
    case_dir = tmp_path / "catalog_case0"
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
                        "directory": "catalog_case0",
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
    return catalog


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
    assert hashlib.sha256(protocols.effective_task_brief(base)).hexdigest() == PILOT_BASE_BRIEF_SHA256
    assert protocols.local_hashes(protocol=base)["task_brief"] == PILOT_BASE_BRIEF_SHA256
    before_sha = protocols.protocol_sha256(base)
    before_id = protocols.protocol_id(base)

    wsl = copy.deepcopy(base)
    wsl["reference_library"] = _ref_library_spec()
    protocols.validate(wsl)
    assert protocols.protocol_sha256(wsl) != before_sha
    assert protocols.protocol_id(wsl) != before_id
    assert protocols.content_sha256(wsl) != protocols.content_sha256(base)
    assert protocols.local_hashes(protocol=wsl)["task_brief"] != PILOT_BASE_BRIEF_SHA256
    assert protocols.resolve_initial_program(wsl) == INITIAL_SOURCE
    resolved = protocols.resolve_reference_library(wsl)
    assert tuple(sorted(resolved)) == EXPECTED_SOLVERS
    for name in EXPECTED_SOLVERS:
        assert resolved[name] == (REF_LIB_DIR / f"{name}.py").read_bytes()


def test_public_export_and_v2_runtime_match_pilot_bytes_and_wsl_adds_only_library_and_paragraph(tmp_path):
    # Candidate-visible v2 module set matches pilot exactly (no fdi_baseline.py or offline_baselines.py).
    actual_v2_modules = frozenset(p.name for p in V2_DIR.glob("*.py"))
    assert actual_v2_modules == PILOT_V2_MODULES
    for p in V2_DIR.glob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert "generator_matched" not in text, f"{p.name} leaks generator_matched"
        assert "potential_field" not in text, f"{p.name} mentions potential_field"

    catalog = _make_catalog(tmp_path)
    exporter = PublicTrainingExport(catalog, anonymize_cases=True, canonical_origin=True)

    base_proto = _example()
    wsl_proto = {**copy.deepcopy(base_proto), "reference_library": _ref_library_spec()}

    off_root = tmp_path / "turn_off"
    exporter(off_root, None)
    (off_root / "public/TASK.md").write_bytes(protocols.effective_task_brief(base_proto))

    on_root = tmp_path / "turn_on"
    exporter(on_root, None)
    (on_root / "public/TASK.md").write_bytes(protocols.effective_task_brief(wsl_proto))
    write_reference_library_dir(on_root / "reference_solvers", protocols.resolve_reference_library(wsl_proto))

    def tree_digests(base):
        return {
            p.relative_to(base).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(base.rglob("*"))
            if p.is_file()
        }

    off_public = tree_digests(off_root / "public")
    non_case_off = {
        k: v for k, v in off_public.items() if k != "cases.json" and not k.startswith("cases/")
    }
    assert non_case_off == PILOT_EXPORT_FILE_SHA256
    assert not (off_root / "reference_solvers").exists()

    on_all = tree_digests(on_root)
    off_all = tree_digests(off_root)
    added_paths = sorted(set(on_all) - set(off_all))
    assert added_paths == [f"reference_solvers/{name}.py" for name in EXPECTED_SOLVERS]
    changed_common = sorted(k for k in set(off_all) & set(on_all) if off_all[k] != on_all[k])
    assert changed_common == ["public/TASK.md"]


def test_real_public_export_blind_scan_matches_pilot_inherited_hits_with_zero_wsl_additions(tmp_path):
    catalog = _make_catalog(tmp_path)
    exporter = PublicTrainingExport(catalog, anonymize_cases=True, canonical_origin=True)
    private = tmp_path / "private"
    (private / "evaluations/e0").mkdir(parents=True)
    (private / "rsi-private-audit.json").write_text(
        json.dumps({"sentinel": "PRIVATE-SENTINEL-TOKEN", "private_parent_ids": ["priv-parent-01"], "case_ids": []}),
        encoding="utf-8",
    )
    (private / "evaluations/e0/result.json").write_text(
        json.dumps({"metrics": {"mean_iou": 0.51234567891234}}),
        encoding="utf-8",
    )

    reports = {}
    for mode, proto in (("off", _example()), ("on", {**_example(), "reference_library": _ref_library_spec()})):
        run = tmp_path / f"run-{mode}"
        turn = run / "harness/epoch_00/branch_0"
        exporter(turn, None)
        eff = protocols.effective_task_brief(proto)
        (turn / "public/TASK.md").write_bytes(eff)
        (run / "task_brief.txt").write_bytes(eff)
        if mode == "on":
            write_reference_library_dir(turn / "reference_solvers", protocols.resolve_reference_library(proto))
        req_dir = turn / "sessions/researcher_0/turns/round_001/provider"
        req_dir.mkdir(parents=True)
        (req_dir / "round_001-provider-1.request.bin").write_bytes(eff)
        reports[mode] = audit.scan_blind(run, private, require_requests=True)

    for mode in ("off", "on"):
        rep = reports[mode]
        assert rep["leaks"] == 0
        assert rep["status"] == "review_required"
        assert rep["review_required"] == 27
        unexplained_paths = {h["path"] for h in rep["iou_hits"] if h["kind"] == "unexplained"}
        assert unexplained_paths == {"harness/epoch_00/branch_0/public/tasks/copper_inversion/v2/baselines.py"}
    assert reports["on"]["review_required"] == reports["off"]["review_required"] == 27
    assert reports["on"]["review_sha256"] == reports["off"]["review_sha256"]


def test_case_transformation_key_pinned_to_pilot_while_cache_key_binds_source_identity(tmp_path, monkeypatch):
    from tasks.copper_inversion.tests.test_rsi_case_isolation import RiskOnlyService_with_checks, write_hex_suite
    from tasks.copper_inversion.tests.test_rsi_service import InProcessExecution, evaluate

    assert SuiteEvaluator.PILOT_CASE_TRANSFORMATION_IMPLEMENTATION == PILOT_IMPLEMENTATION_IDENTITY
    train = write_hex_suite(tmp_path / "release", "train", bodies=4)
    cases = json.loads(train.read_text(encoding="utf-8"))["cases"]

    # Orientation-sensitive solver: places ore in quadrant x < nx//2 (changes under horizontal rotation/flip).
    asym_solver = (
        "import numpy as np\n"
        "def solve(problem, budget, seed):\n"
        "    nx, ny, nz = problem['shape']\n"
        "    f = np.full((nx, ny, nz), -1.0)\n"
        "    f[:max(1, nx // 2), :, :] = 1.0\n"
        "    f[problem['forbidden_mask'] | ~problem['domain_mask']] = -1.0\n"
        "    return f\n"
    )

    records = {}
    for label, impl in (("pilot", PILOT_IMPLEMENTATION_IDENTITY), ("wsl_modified", "f" * 64)):
        monkeypatch.setattr("tasks.copper_inversion.v2.suite.source_identity", lambda val=impl: val)
        monkeypatch.setattr("tasks.copper_inversion.v2.service.source_identity", lambda val=impl: val)
        with patch("tasks.copper_inversion.v2.service.Execution", InProcessExecution):
            svc = RiskOnlyService_with_checks(tmp_path, train, f"svc-{label}", DEFAULT_CASE_ISOLATION, slots=10)
            fb_a0 = evaluate(svc, INITIAL_SOURCE.decode("utf-8"))
            fb_asym = evaluate(svc, asym_solver)
            cand_a0 = SimpleNamespace(canonical_key=fb_a0["source_sha256"])
            cand_asym = SimpleNamespace(canonical_key=fb_asym["source_sha256"])
            tkey_a0 = svc.evaluator.case_transformation_key(cand_a0)
            tkey_asym = svc.evaluator.case_transformation_key(cand_asym)
            records[label] = {
                "cache_key_a0": svc.evaluator.evaluation_attempt_usage_key(cand_a0),
                "tkey_a0": tkey_a0,
                "tkey_asym": tkey_asym,
                "orients_a0": [case_orientation(tkey_a0, c["case_id"]) for c in cases],
                "tokens_a0": [opaque_case_id(tkey_a0, c["case_id"]) for c in cases],
                "orients_asym": [case_orientation(tkey_asym, c["case_id"]) for c in cases],
                "tokens_asym": [opaque_case_id(tkey_asym, c["case_id"]) for c in cases],
                "risk_asym": fb_asym["risk"],
            }

    # Cache keys differ when source_identity changes, preserving cache/resume integrity...
    assert records["pilot"]["cache_key_a0"] != records["wsl_modified"]["cache_key_a0"]
    # ...while scientific transformation keys, orientations, opaque tokens, and orientation-sensitive risk match!
    for field in ("tkey_a0", "tkey_asym", "orients_a0", "tokens_a0", "orients_asym", "tokens_asym", "risk_asym"):
        assert records["pilot"][field] == records["wsl_modified"][field]


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


def test_freeze_and_verify_bind_reference_library_files_and_effective_brief_paragraph(tmp_path, freeze_sources, monkeypatch):
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
    expected_brief = protocols.effective_task_brief(frozen)
    assert frozen["hashes"]["task_brief"] == hashlib.sha256(expected_brief).hexdigest()
    assert frozen["hashes"]["task_brief"] != PILOT_BASE_BRIEF_SHA256

    current = protocols.local_hashes(protocol=frozen, **freeze_sources)
    assert protocols.verify(frozen, current) == []

    # Editing ONLY the appended paragraph causes verify to refuse task_brief, while no-field protocols stay unchanged.
    monkeypatch.setattr(
        "tasks.copper_inversion.rsi.controller.reference_library_paragraph",
        lambda names: "Altered paragraph wording for " + ", ".join(sorted(names)),
    )
    drifted = protocols.local_hashes(protocol=frozen, **freeze_sources)
    with pytest.raises(ValueError, match="frozen protocol hashes no longer match: task_brief"):
        protocols.verify(frozen, drifted)
    assert protocols.local_hashes(protocol=_example(), **freeze_sources)["task_brief"] == PILOT_BASE_BRIEF_SHA256
    monkeypatch.undo()

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
    assert len(seen_in_turns) == 3 * config.policy.branches

    rows = read(root / "ledger.jsonl")
    assert rows[0]["round"] == 0
    assert rows[0]["program_id"] == validate_source(INITIAL_SOURCE)

    paragraph = reference_library_paragraph(subset)
    assert audit.IOU.search(paragraph.encode("utf-8")) is None
    saved_brief = (root / "task_brief.txt").read_text(encoding="utf-8")
    assert paragraph in saved_brief
    assert audit.verify_briefs(root)["status"] == "passed"

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


def test_pi_harness_config_and_pi_agent_mount_controller_owned_reference_library(tmp_path, monkeypatch):
    contract = HarnessSubmissionContract(
        "rsi_solver",
        "submit_solver",
        {"type": "object", "additionalProperties": False, "properties": {}, "required": []},
        (HarnessArtifactRule("/a", (".py",), 1024),),
    )
    guest_rt = PiGuestRuntime("ldm/copper-inversion:0123456789ab", "c" * 64, "2G", "session_overlay")

    def make_cfg(mounts):
        return PiHarnessConfig(
            artifact_root=Path("/artifacts"),
            profiles=(HarnessProfile("researcher_0", Path("/resources/rsi/AGENTS.md"), agents_sha256="d" * 64),),
            campaign_id="c0",
            task_id="copper_inversion",
            case_id="public-training",
            seed=1,
            submission_contract=contract,
            base_url="https://api.deepseek.com",
            model="deepseek-flash",
            guest_runtime=guest_rt,
            context7_enabled=False,
            network_policy=HarnessNetworkPolicy(denied_hosts=("*",)),
            limits=HarnessLimits(wall_time_seconds=30, tool_call_budgets={"bash": 1}),
            read_only_workspace_mounts=mounts,
        )

    assert "readOnlyWorkspaceMounts" not in make_cfg({}).initialize_payload()
    assert make_cfg({"reference_solvers": "/artifacts/reference_solvers"}).initialize_payload()[
        "readOnlyWorkspaceMounts"
    ] == {"reference_solvers": "/artifacts/reference_solvers"}
    for bad in (
        {"../escape": "/artifacts/reference_solvers"},
        {"candidates": "/artifacts/reference_solvers"},
        {".web-evidence": "/artifacts/reference_solvers"},
        {"reference_solvers": "/artifacts/sessions/researcher_0/workspace"},
        {"reference_solvers": "/artifacts/turns/round_001"},
        {"reference_solvers": "/etc"},
    ):
        with pytest.raises(ValueError, match="read-only workspace mount"):
            make_cfg(bad)

    catalog = _make_catalog(tmp_path)
    subset = {"rbf_sdf": (REF_LIB_DIR / "rbf_sdf.py").read_bytes()}
    proto = {**_example(), "reference_library": {"dir": REF_LIB_REL, "files": {"rbf_sdf": hashlib.sha256(subset["rbf_sdf"]).hexdigest()}}}
    full_brief_bytes = protocols.effective_task_brief(proto)

    monkeypatch.setattr(agent_mod, "load_pi_guest_runtime", lambda *a, **k: guest_rt)
    events = []

    class FakeClient:
        def __init__(self, config):
            self.config = config

        def run_turn(self, turns, *, submission_validator, recovery_timeout_seconds):
            # Controller-owned source lives at turn_root/reference_solvers (outside sessions/*/workspace).
            ctrl_ref = turn_root / "reference_solvers/rbf_sdf.py"
            assert ctrl_ref.is_file()
            assert stat.S_IMODE(ctrl_ref.stat().st_mode) == 0o444
            assert stat.S_IMODE(ctrl_ref.parent.stat().st_mode) == 0o555
            assert not (turn_root / "sessions/researcher_0/workspace/reference_solvers").exists()
            assert self.config.read_only_workspace_mounts == {"reference_solvers": "/artifacts/reference_solvers"}
            assert (turn_root / "public/TASK.md").read_bytes() == full_brief_bytes
            events.append("run_turn")
            cand_dir = turn_root / "sessions/researcher_0/turns/round_001/submitted"
            cand_dir.mkdir(parents=True, exist_ok=True)
            body = INITIAL_SOURCE
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
        # Before sidecar pinInitialization runs, turn_root/sessions must NOT exist yet,
        # while turn_root/reference_solvers IS already materialized for the read-only mount.
        assert not (Path(root) / "sessions").exists()
        assert (Path(root) / "reference_solvers/rbf_sdf.py").is_file()
        (Path(root) / "sessions" / profile / "workspace").mkdir(parents=True, exist_ok=True)
        events.append("client_started")
        return FakeClient(config)

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
        base_url="https://api.deepseek.com",
        model="deepseek-flash",
        thinking="high",
        bash_calls=8,
        reference_library=subset,
        task_brief=full_brief_bytes,
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


def _resolve_guest_probe_env():
    image = os.environ.get("RSI_WSL_SIDECAR_IMAGE")
    cache = os.environ.get("RSI_WSL_GONDOLIN_CACHE")
    site_path = Path("/Aidata/shared/xingyulu/rsi4geo/launch-g0/site-r2-wsl.json")
    if (not image or not cache) and site_path.is_file():
        site = json.loads(site_path.read_text(encoding="utf-8"))
        image = image or site.get("sidecar_image")
        cache = cache or site.get("harness_cache_dir")
    if not image or not cache or not Path(cache).is_dir():
        return None, None
    try:
        res = subprocess.run(["docker", "image", "inspect", image], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, timeout=15)
    except OSError:
        return None, None
    if res.returncode != 0:
        return None, None
    return image, Path(cache)


def test_real_gondolin_guest_reference_library_readonly_and_candidate_writable(tmp_path):
    image, cache = _resolve_guest_probe_env()
    if image is None:
        pytest.skip("requires Docker sidecar image and Gondolin guest cache")

    base_work = Path(os.environ.get("RSI_DOCKER_TMPDIR", tmp_path)).resolve() / "wsl-guest-probe"
    if base_work.exists():
        shutil.rmtree(base_work)
    own = base_work / "harness/epoch_00/branch_0"
    subset = {
        name: (REF_LIB_DIR / f"{name}.py").read_bytes()
        for name in ("rbf_sdf", "knn_idw")
    }
    write_reference_library_dir(own / "reference_solvers", subset)
    (own / "sessions/researcher_0/workspace").mkdir(parents=True, exist_ok=True)

    catalog = _make_catalog(tmp_path)
    pi = agent_mod.PiAgent(
        public_catalog=catalog,
        api_key="probe-unused-key",
        sidecar_image=image,
        cache_dir=cache,
        arm="ldm4geo",
        reset_period=5,
        base_url="https://api.deepseek.com",
        model="deepseek-flash",
        thinking="high",
        bash_calls=8,
        reference_library=subset,
    )
    cfg = pi.config(profile="researcher_0", run="guestprobe", epoch=0, seed=801, count=1, harness_seconds=60)
    init_frame = {
        "type": "initialize",
        "requestId": "req-probe-1",
        **cfg.initialize_payload(),
    }
    runtime = load_pi_guest_runtime("copper_inversion", agent_mod.RESOURCES / "harness/image")
    probe_dir = base_work / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).with_name("rsi_wsl_guest_probe.mjs"), probe_dir / "rsi_wsl_guest_probe.mjs")
    (probe_dir / "spec.json").write_text(
        json.dumps(
            {
                "runtime": runtime.to_dict(),
                "initializeFrame": init_frame,
                "expectedSha256": {k: hashlib.sha256(v).hexdigest() for k, v in subset.items()},
            }
        ),
        encoding="utf-8",
    )

    cmd = pi_sidecar_command(
        own,
        image,
        cache_root=cache,
        mounts=(*agent_mod.sidecar_mounts(), (probe_dir, "/probe", True)),
    )
    at = cmd.index("run") + 1
    cname = agent_mod.container_name(own, "researcher_0", "wsl-guest-probe", 0)
    subprocess.run(["docker", "rm", "-f", cname], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cmd = cmd[:at] + ["--name", cname] + cmd[at:]
    argv = cmd[:-1] + ["--entrypoint", "node", cmd[-1], "/probe/rsi_wsl_guest_probe.mjs"]
    completed = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, f"stdout={completed.stdout}\nstderr={completed.stderr}"
    summary = json.loads((own / "wsl-guest-probe-result.json").read_text(encoding="utf-8"))
    assert summary["status"] == "passed"
    assert all(summary["write_tool_denied"].values())
    for key in (
        "overwrite_file",
        "create_file",
        "unlink_file",
        "chmod_file",
        "chmod_dir",
        "rename_inside",
        "rename_dir",
        "rm_rf_dir",
        "mv_replace_dir",
    ):
        assert key in summary["attempts_blocked"]


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
        "tasks.copper_inversion.experiments.baselines.fdi_baseline",
        "tasks.copper_inversion.experiments.baselines.offline_baselines",
    }
    allowed_task_imports = {
        "tasks.copper_inversion.v2.contract",
        "tasks.copper_inversion.v2.geometry",
    }
    ref_solvers = {**REFERENCE_SOLVERS, "fdi_smooth": fdi_baseline.fdi_smooth}

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
        ref_fn = ref_solvers[name]
        for prob in [*probs, cv_prob, barren_prob]:
            expected = ref_fn(prob, budget, 0)
            actual = mod.solve(prob, budget, 0)
            canonicalize(prob, actual)
            assert np.array_equal(actual, expected), f"mismatch for {name} on {prob['case_id']}"
