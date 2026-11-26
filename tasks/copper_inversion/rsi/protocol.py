"""Load, validate, hash and freeze the experiment protocol (contracts §2.1).

The protocol fixes every variable of a group of experiments. Its SHA-256 over
canonical JSON is the suffix of ``protocol_id``. Freezing (gate G3) fills the
component hashes and writes a timestamped record; a run refuses to start when
a frozen hash no longer matches the code and resources it would use.
"""

import argparse
import getpass
import hashlib
import json
import math
import os
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from ldm_tts.engine.run_store import atomic_json_write
from .case_isolation import LEGACY_CASE_ISOLATION, validate_case_isolation


TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[1]
RESOURCES = TASK / "resources"
ANALYSIS = TASK / "experiments/rsi/analysis"
TAGGER = ANALYSIS / "tagger.py"

EXPERIMENTS = ("smoke", "rehearsal", "Q1", "M1", "E1", "E2", "E3", "X1", "F1",
               "M0", "N25", "W25", "X4", "OR", "L40", "WS")
DEVELOPMENT_EXPERIMENTS = ("smoke", "rehearsal")
DEVELOPMENT_RELEASE = "rsi-dev"
HASH_FIELDS = ("task_brief", "renderer", "tool", "implementation", "evaluator_image",
               "sidecar_image", "embedding_model", "analysis", "tagger")
IMAGE_FIELDS = ("evaluator_image", "sidecar_image")
LOOP_FIELDS = ("rounds", "branches", "proposals_per_branch", "batch", "reserved_per_branch",
               "reset_period", "finalists", "harness_seconds", "bash_calls")
LOOP_PRESETS = {
    "r1": {"rounds": 20, "branches": 4, "proposals_per_branch": 8, "batch": 16,
           "reserved_per_branch": 1, "reset_period": 5, "finalists": 3,
           "harness_seconds": 1800, "bash_calls": 8},
    "r2": {"rounds": 20, "branches": 4, "proposals_per_branch": 4, "batch": 16,
           "reserved_per_branch": 4, "reset_period": 5, "finalists": 5,
           "harness_seconds": 1800, "bash_calls": 8},
}
ARM_FIELDS = {"branches": None, "turns_per_round": None,
              "clear": ("never", "every_k", "every_round"),
              "boundary_payload": ("history", "best_program", "initial_program", "parents"),
              "round_feed": ("own", "all", "none"),
              "budget_split": ("shared", "separate")}
OBJECTIVES = ("confirmation", "oracle_iou")
OBJECTIVE_VERSIONS = ("objective_v2", "objective_v3")
VALIDATION_JUDGES = ("standard", "match_training")
JUDGE_FIELDS = frozenset({"objective", "objective_version", "train_manifest", "validation_manifest",
                          "resample", "confirmation_packs"})
THINKING = ("off", "minimal", "low", "medium", "high", "xhigh", "max")
REQUIRED = {"protocol_version", "release", "tiers", "layout", "judges", "loop", "model",
            "token_cap_input", "arms", "seeds", "hashes", "decisions"}
_HEX = re.compile(r"[0-9a-f]{64}")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_MODULE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")
_PACK = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,31}")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sha256(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def protocol_sha256(protocol):
    return sha256(protocol)


def protocol_id(protocol):
    return f"{protocol['release']['name']}-{protocol_sha256(protocol)[:16]}"


def content_sha256(protocol):
    """Hash of everything except the component hashes, stable across freeze steps."""
    return sha256({k: v for k, v in protocol.items() if k != "hashes"})


def _fail(message):
    raise ValueError("invalid protocol: " + message)


def _int(value, name, low=1):
    if type(value) is not int or value < low:
        _fail(f"{name} must be an integer >= {low}")
    return value


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        _fail(name + " must be a finite number")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        _fail(name + " must be a non-empty string")
    return value


def _relative(value, name):
    path = Path(_text(value, name))
    if path.is_absolute() or ".." in path.parts:
        _fail(name + " must be a path inside the release")
    return value


def _dict(value, name, nonempty=True):
    if not isinstance(value, dict) or (nonempty and not value):
        _fail(name + " must be a non-empty object")
    for key in value:
        if not _NAME.fullmatch(key):
            _fail(f"{name} has an invalid key {key!r}")
    return value


def _strategy(config, name):
    # Research-tree forks are not part of any controller arm (AT-15).
    if config != {"name": "independent"}:
        _fail(name + " may only be {\"name\": \"independent\"}; fork strategies are refused")


def validate_strategy(config):
    if not isinstance(config, dict):
        _fail("research strategy configuration must be a JSON object")
    _strategy(config, "research strategy")
    return config


def validate_initial_program(spec):
    if not isinstance(spec, dict) or set(spec) != {"path", "sha256"}:
        _fail("initial_program needs path and sha256")
    path = Path(_text(spec["path"], "initial_program.path"))
    if ".." in path.parts:
        _fail("initial_program.path must not contain '..'")
    if not isinstance(spec["sha256"], str) or _HEX.fullmatch(spec["sha256"]) is None:
        _fail("initial_program.sha256 must be a SHA-256")
    return spec


def resolve_initial_program(protocol, *, repo=REPO):
    """Return the validated round-0 solver bytes (built-in A0 when omitted)."""
    from ..core.candidate import validate_source
    from ..core.encoding import MAX_NORMALIZED_BYTES, normalized_code
    from .controller import INITIAL_SOURCE

    if "initial_program" not in protocol:
        return INITIAL_SOURCE
    spec = validate_initial_program(protocol["initial_program"])
    raw = Path(spec["path"])
    base = Path(repo).resolve()
    target = raw if raw.is_absolute() else (base / raw)
    if target.is_symlink():
        raise ValueError("initial_program path cannot be a symlink")
    resolved = target.resolve()
    if not raw.is_absolute() and not resolved.is_relative_to(base):
        raise ValueError("initial_program path escapes repository")
    if not resolved.is_file():
        raise ValueError(f"initial_program file not found: {spec['path']}")
    body = resolved.read_bytes()
    digest = validate_source(body)
    if digest != spec["sha256"]:
        raise ValueError("initial_program sha256 does not match file on disk")
    size = len(normalized_code(body).encode("utf-8"))
    if size > MAX_NORMALIZED_BYTES:
        raise ValueError(f"initial_program is {size} normalized bytes; limit is {MAX_NORMALIZED_BYTES}")
    return body


def validate_reference_library(spec):
    if not isinstance(spec, dict) or set(spec) != {"dir", "files"}:
        _fail("reference_library needs dir and files")
    directory = Path(_text(spec["dir"], "reference_library.dir"))
    if ".." in directory.parts:
        _fail("reference_library.dir must not contain '..'")
    files = spec["files"]
    if not isinstance(files, dict) or not files:
        _fail("reference_library.files must be a non-empty object")
    for name, digest in files.items():
        if not isinstance(name, str) or _MODULE.fullmatch(name) is None:
            _fail(f"reference_library.files has invalid solver name {name!r}")
        if not isinstance(digest, str) or _HEX.fullmatch(digest) is None:
            _fail(f"reference_library.files.{name} must be a SHA-256")
    return spec


def resolve_reference_library(protocol, *, repo=REPO):
    """Return validated {name: bytes} for protocol.reference_library ({} when omitted)."""
    from ..core.candidate import validate_source
    from ..core.encoding import MAX_NORMALIZED_BYTES, normalized_code
    from .audit import IOU

    if "reference_library" not in protocol:
        return {}
    spec = validate_reference_library(protocol["reference_library"])
    raw_dir = Path(spec["dir"])
    base = Path(repo).resolve()
    target_dir = raw_dir if raw_dir.is_absolute() else (base / raw_dir)
    if target_dir.is_symlink():
        raise ValueError("reference_library.dir cannot be a symlink")
    resolved_dir = target_dir.resolve()
    if not raw_dir.is_absolute() and not resolved_dir.is_relative_to(base):
        raise ValueError("reference_library.dir escapes repository")
    if not resolved_dir.is_dir():
        raise ValueError(f"reference_library directory not found: {spec['dir']}")
    out = {}
    for name, expected_sha in sorted(spec["files"].items()):
        path = target_dir / f"{name}.py"
        if path.is_symlink():
            raise ValueError(f"reference_library file cannot be a symlink: {name}.py")
        resolved = path.resolve()
        if not resolved.is_relative_to(resolved_dir) or not resolved.is_file():
            raise ValueError(f"reference_library file not found: {name}.py")
        body = resolved.read_bytes()
        digest = validate_source(body)
        if digest != expected_sha:
            raise ValueError(f"reference_library sha256 does not match file on disk: {name}.py")
        size = len(normalized_code(body).encode("utf-8"))
        if size > MAX_NORMALIZED_BYTES:
            raise ValueError(f"reference_library {name}.py is {size} normalized bytes; limit is {MAX_NORMALIZED_BYTES}")
        if IOU.search(body) is not None:
            raise ValueError(f"reference_library {name}.py contains forbidden 'iou' substring")
        out[name] = body
    return out


def validate_analysis(value):
    """Validate the complete G3-bound analysis subset without reading external data."""
    if not isinstance(value, dict) or set(value) != {"freeze_prefixes", "checkpoint_experiment", "checkpoint_round", "tagger"}:
        _fail("analysis needs freeze_prefixes, checkpoint_experiment, checkpoint_round and tagger")
    prefixes = value["freeze_prefixes"]
    if not isinstance(prefixes, dict) or not {"M1", "E1", "E2"} <= set(prefixes) or set(prefixes) - set(EXPERIMENTS):
        _fail("analysis.freeze_prefixes must cover M1, E1 and E2")
    for name, entries in prefixes.items():
        if (not isinstance(entries, list) or not entries or "full" not in entries
                or any(x != "full" and (type(x) is not int or x < 1) for x in entries)
                or len(set(entries)) != len(entries)):
            _fail("invalid analysis.freeze_prefixes for " + name)
    if value["checkpoint_experiment"] not in EXPERIMENTS:
        _fail("invalid analysis.checkpoint_experiment")
    _int(value["checkpoint_round"], "analysis.checkpoint_round")
    tagger = value["tagger"]
    if not isinstance(tagger, dict) or set(tagger) != {"mode", "rules", "cluster_reference"} or tagger["mode"] not in ("tags", "clusters"):
        _fail("invalid analysis.tagger")
    from tasks.copper_inversion.experiments.rsi.analysis.tagger import validate_rules
    try:
        validate_rules(tagger["rules"])
    except (ValueError, TypeError, KeyError) as exc:
        _fail("invalid analysis.tagger.rules: " + str(exc))
    ref = tagger["cluster_reference"]
    if tagger["mode"] == "tags" and ref is not None:
        _fail("tags mode cannot have a cluster reference")
    if tagger["mode"] == "clusters" and (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
            or not isinstance(ref["path"], str) or not ref["path"] or not isinstance(ref["sha256"], str)
            or _HEX.fullmatch(ref["sha256"]) is None):
        _fail("clusters mode requires a pinned cluster reference")
    return value


def require_analysis(protocol):
    if "analysis" not in protocol:
        raise ValueError("formal experiment requires protocol analysis configuration")
    return validate_analysis(protocol["analysis"])


def verify_analysis_reference(protocol):
    analysis = require_analysis(protocol)
    spec = analysis["tagger"]
    if spec["mode"] == "clusters":
        from tasks.copper_inversion.experiments.rsi.analysis.tagger import load_cluster_reference
        load_cluster_reference(spec["cluster_reference"], spec["rules"])
    return analysis


def validate(protocol):
    """Return the protocol unchanged, or raise ValueError naming the first defect.

    Extra top-level fields are allowed (contracts: add, never rename or drop);
    ``tiers`` may be omitted only for a legacy release without a tier table.
    """
    if not isinstance(protocol, dict):
        _fail("top level must be an object")
    if "analysis" in protocol:
        validate_analysis(protocol["analysis"])
    missing = REQUIRED - set(protocol) - {"tiers"}
    if missing:
        _fail("missing " + ", ".join(sorted(missing)))
    if protocol["protocol_version"] != 1 or type(protocol["protocol_version"]) is not int:
        _fail("protocol_version must be 1")
    if "research_strategy" in protocol:
        _strategy(protocol["research_strategy"], "research_strategy")
    if "case_isolation" in protocol:
        try:
            validate_case_isolation(protocol["case_isolation"])
        except ValueError as exc:
            _fail(str(exc))
    if "initial_program" in protocol:
        validate_initial_program(protocol["initial_program"])
    if "reference_library" in protocol:
        validate_reference_library(protocol["reference_library"])
    release = protocol["release"]
    if not isinstance(release, dict) or not {"name", "qualification_sha256"} <= set(release):
        _fail("release needs name and qualification_sha256")
    if not isinstance(release["name"], str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,31}", release["name"]):
        _fail("release name must be lower-case letters, digits and dashes")
    if release["qualification_sha256"] is not None and not (
            isinstance(release["qualification_sha256"], str) and _HEX.fullmatch(release["qualification_sha256"])):
        _fail("release qualification_sha256 must be a SHA-256 or null")

    loop = protocol["loop"]
    if not isinstance(loop, dict) or not set(LOOP_FIELDS) <= set(loop):
        _fail("loop needs " + ", ".join(LOOP_FIELDS))
    for name in LOOP_FIELDS:
        _int(loop[name], "loop." + name, 0 if name == "bash_calls" else 1)
    if not loop["branches"] * loop["reserved_per_branch"] <= loop["batch"] <= loop["branches"] * loop["proposals_per_branch"]:
        _fail("loop.batch must cover the reserved slots and fit the proposal pool")
    if "loop_preset" in protocol:
        preset = protocol["loop_preset"]
        if preset not in LOOP_PRESETS:
            _fail(f"loop_preset must be one of {tuple(LOOP_PRESETS)}")
        if {k: loop[k] for k in LOOP_FIELDS} != LOOP_PRESETS[preset]:
            _fail(f"loop fields differ from loop_preset {preset!r}")
    if "objective_version" in protocol and protocol["objective_version"] not in OBJECTIVE_VERSIONS:
        _fail(f"objective_version must be one of {OBJECTIVE_VERSIONS}")
    if "validation_judge" in protocol and protocol["validation_judge"] not in VALIDATION_JUDGES:
        _fail(f"validation_judge must be one of {VALIDATION_JUDGES}")
    slots = 1 + loop["rounds"] * loop["batch"]

    if "tiers" in protocol:
        for name, tier in _dict(protocol["tiers"], "tiers").items():
            where = "tiers." + name
            needed = {"solver_seconds", "train_manifest", "train_bodies", "validation_manifest",
                      "sealed_manifests", "slots"}
            if not isinstance(tier, dict) or not needed <= set(tier):
                _fail(where + " needs " + ", ".join(sorted(needed)))
            _int(tier["solver_seconds"], where + ".solver_seconds")
            _int(tier["train_bodies"], where + ".train_bodies")
            _relative(tier["train_manifest"], where + ".train_manifest")
            _relative(tier["validation_manifest"], where + ".validation_manifest")
            if not isinstance(tier["sealed_manifests"], list) or not tier["sealed_manifests"]:
                _fail(where + ".sealed_manifests must be a non-empty list")
            for item in tier["sealed_manifests"]:
                _relative(item, where + ".sealed_manifests")
            if _int(tier["slots"], where + ".slots") != slots:
                _fail(f"{where}.slots must equal 1 + rounds * batch = {slots}")

    layout = protocol["layout"]
    if not isinstance(layout, dict) or not {"packs_per_body", "execution_seeds", "assignment_rule",
                                            "sealed_bodies"} <= set(layout):
        _fail("layout needs packs_per_body, execution_seeds, assignment_rule and sealed_bodies")
    _int(layout["packs_per_body"], "layout.packs_per_body")
    seeds = layout["execution_seeds"]
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        _fail("layout.execution_seeds must be a non-empty list of distinct integers")
    for value in seeds:
        _int(value, "layout.execution_seeds", 0)
    _text(layout["assignment_rule"], "layout.assignment_rule")
    for name, count in _dict(layout["sealed_bodies"], "layout.sealed_bodies").items():
        _int(count, "layout.sealed_bodies." + name)

    for name, judge in _dict(protocol["judges"], "judges").items():
        if not isinstance(judge, dict) or set(judge) - JUDGE_FIELDS or judge.get("objective") not in OBJECTIVES:
            _fail(f"judges.{name}.objective must be one of {OBJECTIVES}")
        if (judge["objective"] == "oracle_iou") != (name == "oracle"):
            _fail("only the judge named oracle may, and must, score oracle_iou")
        if "objective_version" in judge and judge["objective_version"] not in OBJECTIVE_VERSIONS:
            _fail(f"judges.{name}.objective_version must be one of {OBJECTIVE_VERSIONS}")
        for key in ("train_manifest", "validation_manifest"):
            if key in judge:
                _relative(judge[key], f"judges.{name}.{key}")
        if "resample" in judge:
            if judge["objective"] != "confirmation":
                _fail(f"judges.{name}.resample requires objective='confirmation'")
            spec = judge["resample"]
            if not isinstance(spec, dict) or not {"fraction"} <= set(spec) <= {"fraction", "outcrop_grid"}:
                _fail(f"judges.{name}.resample requires fraction (optional outcrop_grid)")
            frac = _number(spec["fraction"], f"judges.{name}.resample.fraction")
            if not 0 < frac < 1:
                _fail(f"judges.{name}.resample.fraction must lie in (0, 1)")
            if "outcrop_grid" in spec:
                _int(spec["outcrop_grid"], f"judges.{name}.resample.outcrop_grid")
        if "confirmation_packs" in judge:
            if judge["objective"] != "confirmation":
                _fail(f"judges.{name}.confirmation_packs requires objective='confirmation'")
            packs = judge["confirmation_packs"]
            if packs != "all" and (not isinstance(packs, list) or not packs or len(set(packs)) != len(packs)
                                   or any(not isinstance(p, str) or _PACK.fullmatch(p) is None for p in packs)):
                _fail(f"judges.{name}.confirmation_packs must be 'all' or a list of distinct pack names")
    if "standard" not in protocol["judges"]:
        _fail("judges must define standard")
    if validation_judge(protocol) == "standard":
        std_val = protocol["judges"]["standard"].get("validation_manifest")
        for name, judge in protocol["judges"].items():
            if name != "standard" and "validation_manifest" in judge and judge["validation_manifest"] != std_val:
                _fail(f"judges.{name}.validation_manifest is not allowed when validation_judge='standard'")

    model = protocol["model"]
    if not isinstance(model, dict) or not {"base_url", "name", "thinking"} <= set(model):
        _fail("model needs base_url, name and thinking")
    if not isinstance(model["base_url"], str) or not re.fullmatch(r"https?://\S+", model["base_url"]):
        _fail("model.base_url must be an http(s) URL")
    _text(model["name"], "model.name")
    if model["thinking"] not in THINKING:
        _fail(f"model.thinking must be one of {THINKING}")
    if {"api_key", "key", "token", "secret"} & {k.lower() for k in model}:
        _fail("model must not contain credentials")

    _int(protocol["token_cap_input"], "token_cap_input", 0)

    for name, arm in _dict(protocol["arms"], "arms").items():
        if not isinstance(arm, dict):
            _fail(f"arms.{name} must be an object")
        if {"fork_policy", "research_strategy"} & set(arm):
            _fail(f"arms.{name}: fork strategies are refused")
        if set(arm) != set(ARM_FIELDS):
            _fail(f"arms.{name} must have exactly the fields " + ", ".join(ARM_FIELDS))
        for field, choices in ARM_FIELDS.items():
            if choices is None:
                _int(arm[field], f"arms.{name}.{field}")
            elif arm[field] not in choices:
                _fail(f"arms.{name}.{field} must be one of {choices}")
        if arm["branches"] * loop["reserved_per_branch"] > loop["batch"]:
            _fail(f"arms.{name}: batch cannot reserve a slot per branch")

    used = {}
    for name, values in _dict(protocol["seeds"], "seeds", nonempty=False).items():
        if not isinstance(values, list) or len(set(values)) != len(values):
            _fail(f"seeds.{name} must be a list of distinct integers")
        for value in values:
            _int(value, "seeds." + name, 0)
            if value in used:
                _fail(f"seed {value} is in both seeds.{used[value]} and seeds.{name}")
            used[value] = name

    hashes = protocol["hashes"]
    if not isinstance(hashes, dict) or set(hashes) - set(HASH_FIELDS):
        _fail("hashes may only contain " + ", ".join(HASH_FIELDS))
    for name, value in hashes.items():
        if value is None:
            continue
        body = value[7:] if name in IMAGE_FIELDS and isinstance(value, str) and value.startswith("sha256:") else value
        if not isinstance(body, str) or not _HEX.fullmatch(body) or (name in IMAGE_FIELDS and body == value):
            _fail(f"hashes.{name} must be " + ("sha256:<hex>" if name in IMAGE_FIELDS else "a SHA-256") + " or null")

    decisions = protocol["decisions"]
    if not isinstance(decisions, dict):
        _fail("decisions must be an object")
    for name in ("mde_bound_relative", "q1_spearman_min"):
        if name in decisions:
            _number(decisions[name], "decisions." + name)
    if "mde_bound_relative" in decisions and not 0 < decisions["mde_bound_relative"] < 1:
        _fail("decisions.mde_bound_relative must lie in (0, 1)")
    return protocol


def _no_constants(name):
    raise ValueError("invalid protocol: JSON constant " + name + " is not allowed")


def load(path):
    try:
        protocol = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_no_constants)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid protocol: not JSON") from exc
    return validate(protocol)


def is_development(protocol):
    return protocol["release"]["name"] == DEVELOPMENT_RELEASE


def case_isolation(protocol):
    if "case_isolation" in protocol:
        return validate_case_isolation(protocol["case_isolation"])
    return dict(LEGACY_CASE_ISOLATION)


def validation_judge(protocol):
    """'standard' fixes validation to the standard judge in every cell; 'match_training' keeps legacy rsi-r1."""
    if "validation_judge" in protocol:
        return protocol["validation_judge"]
    judges = protocol.get("judges") or {}
    if (protocol.get("objective_version") == "objective_v3"
            or any(isinstance(j, dict) and ("resample" in j or "confirmation_packs" in j) for j in judges.values())
            or bool({"noisy25", "x4", "w25"} & set(judges))):
        return "standard"
    return "match_training"


def objective_version(protocol, judge_name=None):
    if judge_name is not None and judge_name in (protocol.get("judges") or {}):
        ver = protocol["judges"][judge_name].get("objective_version")
        if ver is not None:
            return ver
    return protocol.get("objective_version", "objective_v2")


def loop_preset(name):
    if name not in LOOP_PRESETS:
        raise ValueError(f"unknown loop preset {name!r}")
    return dict(LOOP_PRESETS[name])


# Component hashes -----------------------------------------------------------

def _file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tree(root, base):
    root = Path(root)
    return {str(p.relative_to(base)): _file(p) for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}


def implementation_identity(task=TASK, repo=REPO):
    """Task code and resources, shared ldm_tts code and sidecar sources.

    Unlike ``source_identity`` this leaves out ``resources/rsi/protocol*.json``,
    so a protocol kept in the tree never hashes itself.
    """
    task, repo = Path(task), Path(repo)
    files = {}
    for folder in ("core", "v2", "rsi", "resources", "experiments/rsi"):
        if (task / folder).is_dir():
            files.update(_tree(task / folder, task))
    files = {k: v for k, v in files.items() if k != "qualification_evidence.json"
             and "evidence" not in Path(k).parts
             and not (Path(k).parent == Path("resources/rsi") and Path(k).name.startswith("protocol"))}
    shared = {k: v for k, v in _tree(repo / "ldm_tts", repo).items() if k.endswith(".py")}
    pi = repo / "harnesses/pi"
    sidecar = {k: v for k, v in _tree(pi / "src", repo).items() if k.endswith(".ts")}
    for name in ("package-lock.json", "Dockerfile.rsi"):
        if (pi / name).is_file():
            sidecar[str((pi / name).relative_to(repo))] = _file(pi / name)
    return sha256({"task": files, "shared_python": shared, "sidecar_sources": sidecar})


def image_digest(reference):
    """Immutable image ID; a ``sha256:`` digest is accepted as given, else Docker resolves it."""
    if reference is None:
        return None
    if reference.startswith("sha256:"):
        if not _HEX.fullmatch(reference[7:]):
            raise ValueError("image digest must be sha256:<hex>")
        return reference
    try:
        result = subprocess.run(["docker", "image", "inspect", reference, "--format", "{{.Id}}"],
                                capture_output=True, text=True, timeout=40)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("cannot resolve image " + reference) from exc
    value = result.stdout.strip()
    if result.returncode or not value.startswith("sha256:") or not _HEX.fullmatch(value[7:]):
        raise ValueError("cannot resolve image " + reference)
    return value


def effective_task_brief(protocol=None, *, resources=None):
    """Exact UTF-8 task-brief bytes delivered to the agent, controller and brief audit."""
    res = Path(resources or RESOURCES)
    brief_path = res / "rsi/TASK.md"
    if protocol is None or "reference_library" not in protocol:
        return brief_path.read_bytes()
    from .controller import reference_library_paragraph
    spec = validate_reference_library(protocol["reference_library"])
    base = brief_path.read_text(encoding="utf-8")
    return (base.rstrip() + "\n\n" + reference_library_paragraph(spec["files"]) + "\n").encode("utf-8")


def local_hashes(*, protocol=None, resources=None, task=None, repo=None, embedding_model_path=None,
                 analysis_dir=None, tagger_path=None, sidecar_image=None, evaluator_image=None):
    """Every component hash computable here; ``None`` marks a pending component."""
    resources, task, repo = Path(resources or RESOURCES), Path(task or TASK), Path(repo or REPO)
    analysis_dir, tagger_path = Path(analysis_dir or ANALYSIS), Path(tagger_path or TAGGER)
    provenance = None if embedding_model_path is None else Path(embedding_model_path) / "provenance.json"
    analysis = _tree(analysis_dir, analysis_dir) if analysis_dir.is_dir() else {}
    return {
        "task_brief": hashlib.sha256(effective_task_brief(protocol, resources=resources)).hexdigest(),
        "renderer": sha256({name: _file(task / "rsi" / name) for name in ("render.py", "ledger.py")}),
        "tool": _file(resources / "rsi/rsi.mjs"),
        "implementation": implementation_identity(task, repo),
        "evaluator_image": image_digest(evaluator_image),
        "sidecar_image": image_digest(sidecar_image),
        "embedding_model": _file(provenance) if provenance is not None and provenance.is_file() else None,
        "analysis": sha256(analysis) if analysis else None,
        "tagger": _file(tagger_path) if tagger_path.is_file() else None,
    }


def mismatches(protocol, current, *, skip=()):
    """Frozen hash fields that differ from (or can no longer be computed as) ``current``."""
    return sorted(name for name, value in protocol["hashes"].items()
                  if value is not None and name not in skip and current.get(name) != value)


# Authors' decision 2026-11-22: code that does not change what the researcher sees or how it is
# scored may be fixed under the same protocol, so runs can be reused and resumed. Drift in these
# fields is returned for the record (every run and freeze stores its code commit), not refused.
# Task brief, renderer, tool, tagger, images, embedding model and every protocol field stay binding.
RECORDED_ONLY = ("implementation", "analysis")


def verify(protocol, current, *, skip=(), repo=REPO):
    """Refuse a binding hash that changed; return the recorded-only fields that drifted."""
    wrong = mismatches(protocol, current, skip=skip)
    binding = [name for name in wrong if name not in RECORDED_ONLY]
    if "initial_program" in protocol and "initial_program" not in skip:
        try:
            resolve_initial_program(protocol, repo=repo)
        except ValueError as exc:
            raise ValueError("frozen protocol hashes no longer match: initial_program") from exc
    if "reference_library" in protocol and "reference_library" not in skip:
        try:
            resolve_reference_library(protocol, repo=repo)
        except ValueError as exc:
            raise ValueError("frozen protocol hashes no longer match: reference_library") from exc
    if binding:
        raise ValueError("frozen protocol hashes no longer match: " + ", ".join(binding))
    return [name for name in wrong if name in RECORDED_ONLY]


# Freeze records (gate G3) ---------------------------------------------------

def _records(path):
    path = Path(path)
    return sorted(path.parent.glob(path.name + ".freeze-*.json"))


def latest_record(path):
    records = _records(path)
    return json.loads(records[-1].read_text()) if records else None


def code_commit(repo=REPO):
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=20)
        commit = head.stdout.strip()
        if head.returncode == 0 and re.fullmatch(r"[0-9a-f]{40,64}", commit):
            dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                                   capture_output=True, text=True, timeout=20)
            return commit + ("+dirty" if dirty.returncode or dirty.stdout.strip() else "")
    except (OSError, subprocess.TimeoutExpired):
        pass
    return os.environ.get("RSI_CODE_COMMIT") or None


def freeze(path, *, allow_pending=False, now=None, **sources):
    """Fill the component hashes, rewrite the protocol and write a timestamped record.

    A freeze is final once complete. An incomplete freeze (allowed only with
    ``allow_pending``) may later be completed, e.g. with image digests on the
    run host, provided no other field and no already frozen hash changed.
    """
    path = Path(path)
    protocol = load(path)
    previous = latest_record(path)
    if previous is not None:
        if previous["complete"]:
            raise ValueError("protocol is already frozen; start a new protocol version")
        if previous["protocol_sha256"] != protocol_sha256(protocol):
            raise ValueError("protocol changed since its partial freeze")
    if not is_development(protocol):
        verify_analysis_reference(protocol)
        if protocol["token_cap_input"] == 0:
            raise ValueError("a formal protocol must set token_cap_input before freezing")
        if protocol["release"]["qualification_sha256"] is None:
            raise ValueError("a formal protocol must name its release qualification record")
        if "tiers" not in protocol:
            raise ValueError("a formal protocol must define a tier table")
    if "initial_program" in protocol:
        resolve_initial_program(protocol, repo=sources.get("repo") or REPO)
    if "reference_library" in protocol:
        resolve_reference_library(protocol, repo=sources.get("repo") or REPO)
    current = local_hashes(protocol=protocol, **sources)
    hashes = {}
    for name in HASH_FIELDS:
        old = protocol["hashes"].get(name)
        new = current[name] if current[name] is not None else old
        if previous is not None and old is not None and new != old:
            raise ValueError(f"frozen hash {name} changed since the partial freeze")
        hashes[name] = new
    if previous is not None and hashes == protocol["hashes"]:
        raise ValueError("no pending hash could be filled")
    pending = [name for name in HASH_FIELDS if hashes[name] is None]
    if pending and not allow_pending:
        raise ValueError("cannot compute " + ", ".join(pending) + "; pass the sources or allow pending")
    frozen = validate({**protocol, "hashes": hashes})
    moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    record = {"schema": "rsi-protocol-freeze-v1", "frozen_at": moment.isoformat(timespec="seconds"),
              "protocol_file": path.name, "protocol_id": protocol_id(frozen),
              "protocol_sha256": protocol_sha256(frozen), "content_sha256": content_sha256(frozen),
              "hashes": hashes, "pending": pending, "complete": not pending,
              "previous_protocol_sha256": None if previous is None else previous["protocol_sha256"],
              "code_commit": code_commit(), "host": socket.gethostname(), "operator": getpass.getuser()}
    target = path.parent / f"{path.name}.freeze-{moment.strftime('%Y%m%dT%H%M%SZ')}.json"
    if target.exists():
        raise ValueError("a freeze record with this timestamp already exists")
    atomic_json_write(path, frozen)
    atomic_json_write(target, record)
    for written in (path, target):
        written.chmod(0o644)  # the evaluator account reads the frozen protocol
    return record


def require_frozen(path, protocol):
    """Gate G3: a complete freeze record exists and matches this protocol."""
    record = latest_record(path)
    if record is None or not record["complete"]:
        raise ValueError("protocol has no complete freeze record")
    if record["protocol_sha256"] != protocol_sha256(protocol) or any(
            protocol["hashes"].get(k) != v for k, v in record["hashes"].items()):
        raise ValueError("protocol differs from its freeze record")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "freeze", "check"))
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--embedding-model-path", type=Path)
    parser.add_argument("--sidecar-image", help="image reference or sha256: digest")
    parser.add_argument("--evaluator-image", help="image reference or sha256: digest")
    parser.add_argument("--analysis-dir", type=Path, help="default: experiments/rsi/analysis")
    parser.add_argument("--tagger-path", type=Path, help="default: experiments/rsi/analysis/tagger.py")
    parser.add_argument("--allow-pending", action="store_true",
                        help="write an incomplete freeze; complete it later on the run host")
    args = parser.parse_args(argv)
    sources = dict(embedding_model_path=args.embedding_model_path, sidecar_image=args.sidecar_image,
                   evaluator_image=args.evaluator_image, analysis_dir=args.analysis_dir,
                   tagger_path=args.tagger_path)
    if args.command == "validate":
        protocol = load(args.protocol)
        result = {"protocol_id": protocol_id(protocol), "protocol_sha256": protocol_sha256(protocol)}
    elif args.command == "freeze":
        result = freeze(args.protocol, allow_pending=args.allow_pending, **sources)
    else:
        protocol = load(args.protocol)
        current = local_hashes(protocol=protocol, **sources)
        skip = [name for name in IMAGE_FIELDS + ("embedding_model",) if current[name] is None]
        verify(protocol, current, skip=skip)
        record = latest_record(args.protocol)
        result = {"protocol_id": protocol_id(protocol), "unchecked": skip,
                  "freeze": None if record is None else {k: record[k] for k in ("frozen_at", "complete", "pending")}}
        if record is not None and record["complete"]:
            require_frozen(args.protocol, protocol)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
