"""Run one risk-only discovery campaign defined by a protocol file on the remote host."""

import argparse
import dataclasses
import hashlib
import inspect
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

from ldm_tts.engine.run_store import atomic_json_write
from ..core.encoding import CodeEncoder
from ..core.execution import digest, docker
from ..v2.public_catalog import catalog_digest, load_catalog
from . import continuation as continuations, intervention as interventions, protocol as protocols
from .agent import PiAgent, RESOURCES
from .arms import policy
from .client import RiskClient
from .controller import Config, Controller
from .integrity import shared_identity
from .selection import select


# The agent hard-coded these before it accepted model settings; a protocol may
# only use other values once the agent takes them as constructor arguments.
LEGACY_AGENT = {"base_url": "https://api.deepseek.com", "model": "deepseek-flash", "thinking": "low",
                "bash_calls": 8}
# Old-style limits, kept only for a release without a tier table.
LEGACY_TIER, LEGACY_MAX_SECONDS, LEGACY_FORMAL = "F", 120, (20, 8, 16, 120)
NO_TOKEN_CAP = 2 ** 63 - 1


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _shared_implementation():
    repo = Path(__file__).resolve().parents[3]
    files = ["harnesses/pi/src/history.ts",
             "harnesses/pi/src/session.ts", "harnesses/pi/package-lock.json"]
    return digest({"python": shared_identity(),
                   "sidecar_sources": {name: _file_hash(repo / name) for name in files}})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--experiment", required=True, choices=protocols.EXPERIMENTS)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--tier", help="a key of the protocol tier table (omit for a legacy release)")
    parser.add_argument("--judge", default="standard")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out-dir", required=True, type=Path, help="the run directory, named by its run_id")
    parser.add_argument("--public-catalog", required=True, type=Path)
    parser.add_argument("--evaluation-socket", required=True, type=Path)
    parser.add_argument("--evaluation-contract", required=True)
    parser.add_argument("--evaluation-service-uid", required=True, type=int)
    parser.add_argument("--embedding-model-path", required=True, type=Path)
    parser.add_argument("--embedding-device", default="cuda:0")
    parser.add_argument("--sidecar-image", required=True)
    parser.add_argument("--harness-cache-dir", required=True, type=Path)
    parser.add_argument("--api-key-file", required=True, type=Path)
    parser.add_argument("--harness-research-strategy-config", type=Path,
                        help="accepted only as {\"name\": \"independent\"}; fork strategies are refused")
    parser.add_argument("--solver-seconds", type=int,
                        help="legacy releases without a tier table only (default 120)")
    parser.add_argument("--checkpoint-round", type=int,
                        help="copy checkpoints/round_<t>/ after round t's evaluation (E1 sources)")
    parser.add_argument("--continuation", type=Path, metavar="CHECKPOINT",
                        help="E1/E2: start at round t+1 from <source run>/checkpoints/round_<t>")
    parser.add_argument("--continuation-mode", choices=sorted(continuations.MODES))
    parser.add_argument("--replicate", type=int)
    parser.add_argument("--planted", action="store_true",
                        help="E2: plant the templated note for the checkpoint's selected target tag")
    parser.add_argument("--fixture-only", action="store_true",
                        help="label a development integration fixture; never use for scientific results")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def run_id(experiment, arm, judge, seed):
    return f"{arm}-{judge}-s{seed}" if experiment == "E3" else f"{arm}-s{seed}"


def continuation_run_id(source_seed, mode, replicate, planted=False):
    return f"ck{source_seed}-{mode}{'-planted' if planted else ''}-r{replicate}"


def _continuation(args, protocol, tier):
    """Checkpoint, contract field and planted note of an E1/E2 run; None for an ordinary run."""
    continued = args.experiment in ("E1", "E2")
    if args.continuation is None:
        if continued or args.continuation_mode or args.replicate is not None or args.planted:
            raise ValueError("E1 and E2 runs, and only they, start from a checkpoint")
        return None
    if not continued and args.experiment not in protocols.DEVELOPMENT_EXPERIMENTS:
        raise ValueError("only E1, E2 and development runs continue from a checkpoint")
    mode = args.continuation_mode
    if mode is None or type(args.replicate) is not int or args.replicate < 1:
        raise ValueError("a continuation needs --continuation-mode and a positive --replicate")
    if args.checkpoint_round is not None:
        raise ValueError("a continuation does not take its own checkpoint")
    if args.arm != continuations.MODES[mode]:
        raise ValueError(f"continuation mode {mode} runs arm {continuations.MODES[mode]}")
    if args.planted != (args.experiment == "E2") and continued:
        raise ValueError("E2 continuations are planted and E1 continuations are not")
    if args.planted and mode == "best_program":
        raise ValueError("planted notes are tested in conversation and history modes only")
    checkpoint = continuations.open_checkpoint(args.continuation)
    source = checkpoint.source_manifest
    if (source.get("protocol_id") != protocols.protocol_id(protocol) or source.get("tier") != tier
            or source.get("judge") != args.judge):
        raise ValueError("checkpoint comes from another protocol, tier or judge")
    if continued and source.get("experiment") != "M1":
        raise ValueError("E1 and E2 continue checkpoints of M1 runs")
    if args.seed != checkpoint.source_seed:
        raise ValueError("a continuation keeps the source run's seed")
    if mode == "conversation" and source.get("arm") != args.arm:
        raise ValueError("a conversation continuation keeps the source arm")
    note = interventions.from_checkpoint(checkpoint, protocol if not protocols.is_development(protocol) else None) if args.planted else None
    return {"checkpoint": checkpoint, "mode": mode, "intervention": note,
            "manifest": checkpoint.identity(mode, args.replicate),
            "run_id": continuation_run_id(checkpoint.source_seed, mode, args.replicate, args.planted)}


def build_config(protocol, arm, tier, seed, solver_seconds=None):
    """Controller settings from the protocol.

    Uses ``Config.from_protocol`` when the controller provides it. Otherwise
    maps the protocol onto the current fixed-shape Config and refuses settings
    that the current controller would silently ignore.
    """
    if hasattr(Config, "from_protocol"):
        factory = Config.from_protocol
        if "tiers" not in protocol:
            # A legacy release has no tier table; give the controller the one tier it runs.
            loop = protocol["loop"]
            protocol = {**protocol, "tiers": {tier: {"solver_seconds": solver_seconds,
                                                     "slots": 1 + loop["rounds"] * loop["batch"]}}}
        if "seed" in inspect.signature(factory).parameters:
            return factory(protocol, arm, tier, seed=seed)
        return dataclasses.replace(factory(protocol, arm, tier), seed=seed)
    loop, definition = protocol["loop"], protocol["arms"][arm]
    try:
        legacy = policy(arm)
    except ValueError:
        legacy = None
    if (legacy is None or {k: definition[k] for k in ("clear", "boundary_payload", "round_feed")}
            != {k: getattr(legacy, k) for k in ("clear", "boundary_payload", "round_feed")}
            or (definition["branches"], definition["turns_per_round"], definition["budget_split"]) != (4, 1, "shared")
            or (loop["branches"], loop["reserved_per_branch"], loop["finalists"]) != (4, 1, 3)):
        raise ValueError("this controller cannot run the protocol's definition of arm " + arm)
    return Config(arm, seed, loop["rounds"], loop["proposals_per_branch"], loop["batch"],
                  loop["reset_period"], loop["harness_seconds"], protocol["token_cap_input"] or NO_TOKEN_CAP)


def agent_settings(protocol, agent_class=PiAgent):
    """Model and tool settings for the agent constructor, refusing any it would ignore."""
    wanted = {"base_url": protocol["model"]["base_url"], "model": protocol["model"]["name"],
              "thinking": protocol["model"]["thinking"], "bash_calls": protocol["loop"]["bash_calls"]}
    accepted = inspect.signature(agent_class).parameters
    ignored = sorted(k for k, v in wanted.items() if k not in accepted and v != LEGACY_AGENT[k])
    if ignored:
        raise ValueError("the agent cannot apply protocol settings: " + ", ".join(ignored))
    return {k: v for k, v in wanted.items() if k in accepted}


def _previous_manifest(args):
    path = args.out_dir / "manifest.json"
    return json.loads(path.read_text()) if args.resume and path.is_file() else {}


def prepare(args):
    """All checks that need neither Docker, the service nor the model; returns the run plan."""
    if args.harness_research_strategy_config is not None:
        protocols.validate_strategy(json.loads(args.harness_research_strategy_config.read_text()))
    protocol = protocols.load(args.protocol)
    if args.arm not in protocol["arms"]:
        raise ValueError("arm is not defined by the protocol")
    if args.judge not in protocol["judges"]:
        raise ValueError("judge is not defined by the protocol")
    if args.judge != "standard" and args.experiment not in ("E3",) + protocols.DEVELOPMENT_EXPERIMENTS:
        raise ValueError("judge strength varies only in E3")
    allowed = protocol["seeds"].get(args.experiment)
    if allowed is not None and args.seed not in allowed + protocol["seeds"].get(args.experiment + "_reserve", []):
        raise ValueError("seed is not in the protocol's seed table for this experiment")
    if args.experiment not in protocols.DEVELOPMENT_EXPERIMENTS:
        protocols.verify_analysis_reference(protocol)
        protocols.require_frozen(args.protocol, protocol)
    current = protocols.local_hashes(resources=RESOURCES, embedding_model_path=args.embedding_model_path)
    # Image digests are checked where they can be resolved: the sidecar below,
    # the evaluator image by its launcher.
    protocols.verify(protocol, current, skip=protocols.IMAGE_FIELDS)
    if "tiers" in protocol:
        if args.tier not in protocol["tiers"]:
            raise ValueError("tier is not defined by the protocol")
        if args.solver_seconds is not None:
            raise ValueError("solver seconds come from the protocol tier table")
        tier, solver_seconds = args.tier, protocol["tiers"][args.tier]["solver_seconds"]
    else:
        if args.tier not in (None, LEGACY_TIER):
            raise ValueError("a release without a tier table runs only the legacy full tier")
        tier = LEGACY_TIER
        solver_seconds = LEGACY_MAX_SECONDS if args.solver_seconds is None else args.solver_seconds
        if not 1 <= solver_seconds <= LEGACY_MAX_SECONDS:
            raise ValueError("solver limit must be between 1 and 120 seconds")
    continued = _continuation(args, protocol, tier)
    name = continued["run_id"] if continued else run_id(args.experiment, args.arm, args.judge, args.seed)
    if args.out_dir.name != name:
        raise ValueError("run directory must be named " + name)
    config = build_config(protocol, args.arm, tier, args.seed, solver_seconds)
    if getattr(config, "arm", args.arm) != args.arm or config.seed != args.seed:
        raise ValueError("controller settings differ from the requested arm or seed")
    if "tiers" in protocol and config.slots != protocol["tiers"][tier]["slots"]:
        raise ValueError("controller slots differ from the protocol tier")
    if args.checkpoint_round is not None and not 1 <= args.checkpoint_round < config.rounds:
        raise ValueError("checkpoint round must lie inside the campaign")
    if continued:
        continuations.check_config(continued["checkpoint"], config, continued["mode"])
    previous = _previous_manifest(args)
    manifest = {
        "run_id": name, "experiment": args.experiment, "protocol_id": protocols.protocol_id(protocol),
        "protocol_sha256": protocols.protocol_sha256(protocol),
        "arm": args.arm, "tier": tier, "judge": args.judge, "seed": args.seed,
        "objective": protocol["judges"][args.judge]["objective"],
        "blind": protocol["judges"][args.judge]["objective"] != "oracle_iou",
        "continuation": continued["manifest"] if continued else None,
        "checkpoint_round": args.checkpoint_round,
        "evaluation": {"socket": str(args.evaluation_socket), "contract_digest": args.evaluation_contract,
                       "service_uid": args.evaluation_service_uid},
        "code_commit": protocols.code_commit(),
        "started_at": previous.get("started_at") or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": previous.get("host") or socket.gethostname(),
        "solver_seconds": solver_seconds, "model": dict(protocol["model"]),
        "token_cap_input": protocol["token_cap_input"],
        "evaluator_image": protocol["hashes"].get("evaluator_image"),
        "case_isolation": protocols.case_isolation(protocol),
    }
    return {"protocol": protocol, "config": config, "solver_seconds": solver_seconds, "manifest": manifest,
            "agent": agent_settings(protocol), "continuation": continued}


def evaluation_concurrency(environ=os.environ):
    """Concurrent evaluations per round from the scheduling policy (parallel_runtime ``rsi`` mode).

    The runtime verifies the policy's hashes before exporting its path. Without a
    policy the controller evaluates serially, as before.
    """
    path = environ.get("LDM4GEO_SCHEDULING_POLICY")
    if not path:
        return 1
    value = json.loads(Path(path).read_text()).get("batch_workers", 1)
    if type(value) is not int or value < 1:
        raise ValueError("scheduling policy batch_workers must be a positive integer")
    return value


def main(argv=None):
    args = parse_args(argv)
    plan = prepare(args)
    protocol, config, solver_seconds = plan["protocol"], plan["config"], plan["solver_seconds"]
    config = dataclasses.replace(config, evaluation_concurrency=evaluation_concurrency())
    catalog = load_catalog(args.public_catalog)
    if args.fixture_only and catalog["stage"] != "development":
        raise ValueError("fixture-only runs require a development catalog")
    if (catalog["stage"] == "formal" and "tiers" in protocol and len(catalog["cases"])
            != protocol["tiers"][plan["manifest"]["tier"]]["train_bodies"] * protocol["layout"]["packs_per_body"]):
        raise ValueError("public catalog size differs from the protocol tier")
    if args.evaluation_service_uid == os.getuid():
        raise ValueError("the trusted evaluator must run under a different account")
    public_digest = catalog_digest(args.public_catalog)
    client = RiskClient(args.out_dir, args.evaluation_socket, args.evaluation_contract,
                        service_uid=args.evaluation_service_uid)
    continued = plan["continuation"]
    isolation = plan["manifest"]["case_isolation"]
    # A continuation's service imported exactly this checkpoint's prefix; any other run's was not started by import.
    prefix = continued["checkpoint"].snapshot["ledger_prefix_sha256"] if continued else None
    info = client.verify_contract(public_digest, solver_seconds, config.slots,
                                  import_ledger_prefix_sha256=prefix, case_isolation=isolation)
    if continued:
        continuations.check_service(info, continued["checkpoint"], resume=args.resume)
    elif not args.resume and info["used_slots"]:
        raise ValueError("a new run requires an unused private evaluator")
    if "tiers" not in protocol and catalog["stage"] == "formal" and (
            config.rounds, config.proposals_per_branch, config.batch, solver_seconds) != LEGACY_FORMAL:
        raise ValueError("formal full-tier discovery requires 20 rounds, 8 proposals, 16 selections and 120 s")
    sidecar_digest = docker("image", "inspect", args.sidecar_image, "--format", "{{.Id}}").stdout.strip()
    if not sidecar_digest.startswith("sha256:") or len(sidecar_digest) != 71:
        raise ValueError("sidecar image must resolve to an immutable digest")
    if protocol["hashes"].get("sidecar_image") not in (None, sidecar_digest):
        raise ValueError("frozen protocol hashes no longer match: sidecar_image")
    brief = (RESOURCES / "rsi/TASK.md").read_text(encoding="utf-8")
    identity = {**plan["manifest"], "public_digest": public_digest, "sidecar_image": sidecar_digest,
                "embedding_provenance_sha256": _file_hash(args.embedding_model_path / "provenance.json"),
                "shared_implementation_sha256": _shared_implementation(),
                "researcher_instructions_sha256": _file_hash(RESOURCES / "rsi/AGENTS.md"),
                "evidence_tool_sha256": _file_hash(RESOURCES / "rsi/rsi.mjs"),
                "stage": catalog["stage"], "synthetic_test": args.fixture_only}
    key = args.api_key_file.read_text().strip()
    if not key:
        raise ValueError("protected model API key file is empty")
    agent = pi_agent = PiAgent(public_catalog=args.public_catalog, api_key=key, sidecar_image=sidecar_digest,
                               cache_dir=args.harness_cache_dir, arm=args.arm, reset_period=config.reset_period,
                               case_isolation=isolation, **plan["agent"])
    encoder = CodeEncoder(args.out_dir, "fusion", args.embedding_model_path, args.embedding_device)
    if args.checkpoint_round is not None:
        agent = continuations.CheckpointingAgent(agent, args.out_dir, args.checkpoint_round)
    start = continuations.continuation_start(continued["checkpoint"], continued["mode"]) if continued else None
    controller = Controller(args.out_dir, config, agent, client,
                            lambda pool, rows, count: select(pool, rows, encoder, count), brief, identity,
                            intervention=continued["intervention"] if continued else None, start=start,
                            forbidden_case_tokens=pi_agent.forbidden_case_tokens)
    try:
        if continued:
            result = continuations.run_continuation(controller, continued["checkpoint"], continued["mode"],
                                                    resume=args.resume)
        else:
            result = controller.run(resume=args.resume)
        sealed = client.seal_training()
        if (sealed["used_slots"] != result["charged_records"]
                or sealed["source_sha256s"] != result["training_finalists"]):
            raise ValueError("public and private training records or finalists differ")
    except Exception as exc:
        if args.out_dir.exists():
            atomic_json_write(args.out_dir / "run-status.json", {"status": "paused", "error_type": type(exc).__name__})
        raise
    finally:
        # Sidecar clients stay open across rounds; stop them before the process exits.
        pi_agent.close()
    atomic_json_write(args.out_dir / "run-status.json", {"status": result["status"],
                                                         "stop_reason": result["stop_reason"]})
    print(json.dumps({"run_dir": str(args.out_dir), **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
