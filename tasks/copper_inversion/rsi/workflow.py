"""Run a new risk-only four-branch discovery campaign on the remote host."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from ldm_tts.engine.run_store import atomic_json_write
from ..core.encoding import CodeEncoder
from ..core.execution import digest, docker
from ..v2.public_catalog import catalog_digest, load_catalog
from .agent import PiAgent, RESOURCES
from .client import RiskClient
from .controller import Config, Controller
from .integrity import shared_identity
from .selection import select


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
    parser.add_argument("--public-catalog", required=True, type=Path)
    parser.add_argument("--evaluation-socket", required=True, type=Path)
    parser.add_argument("--evaluation-contract", required=True)
    parser.add_argument("--evaluation-service-uid", required=True, type=int)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--embedding-model-path", required=True, type=Path)
    parser.add_argument("--embedding-device", default="cuda:0")
    parser.add_argument("--sidecar-image", required=True)
    parser.add_argument("--harness-cache-dir", required=True, type=Path)
    parser.add_argument("--api-key-file", required=True, type=Path)
    parser.add_argument("--arm", required=True,
                        choices=("ldm4geo", "accumulated", "restart_best", "independent_sampling"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--proposals-per-branch", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--reset-period", type=int, default=5)
    parser.add_argument("--solver-seconds", type=int, default=120)
    parser.add_argument("--harness-seconds", type=int, default=1800)
    parser.add_argument("--token-cap-input", type=int, default=200_000_000)
    parser.add_argument("--intervention-file", type=Path)
    parser.add_argument("--fixture-only", action="store_true",
                        help="label a development integration fixture; never use for scientific results")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = Config(args.arm, args.seed, args.rounds, args.proposals_per_branch, args.batch,
                    args.reset_period, args.harness_seconds, args.token_cap_input)
    catalog = load_catalog(args.public_catalog)
    if args.fixture_only and catalog["stage"] != "development":
        raise ValueError("fixture-only runs require a development catalog")
    if args.solver_seconds < 1 or args.solver_seconds > 120:
        raise ValueError("solver limit must be between 1 and 120 seconds")
    if args.evaluation_service_uid == os.getuid():
        raise ValueError("the trusted evaluator must run under a different account")
    public_digest = catalog_digest(args.public_catalog)
    client = RiskClient(args.out_dir, args.evaluation_socket, args.evaluation_contract,
                        service_uid=args.evaluation_service_uid)
    info = client.verify_contract(public_digest, args.solver_seconds, config.slots)
    if not args.resume and info["used_slots"]:
        raise ValueError("a new run requires an unused private evaluator")
    if catalog["stage"] == "formal" and (config.rounds, config.proposals_per_branch,
                                          config.batch, args.solver_seconds) != (20, 8, 16, 120):
        raise ValueError("formal full-tier discovery requires 20 rounds, 8 proposals, 16 selections and 120 s")
    sidecar_digest = docker("image", "inspect", args.sidecar_image, "--format", "{{.Id}}").stdout.strip()
    if not sidecar_digest.startswith("sha256:") or len(sidecar_digest) != 71:
        raise ValueError("sidecar image must resolve to an immutable digest")
    brief = (RESOURCES / "rsi/TASK.md").read_text(encoding="utf-8")
    identity = {"public_digest": public_digest, "service_contract": args.evaluation_contract,
                "solver_seconds": args.solver_seconds, "sidecar_image": sidecar_digest,
                "model": "deepseek-flash", "base_url": "https://api.deepseek.com", "thinking": "low",
                "embedding_provenance_sha256": _file_hash(args.embedding_model_path / "provenance.json"),
                "shared_implementation_sha256": _shared_implementation(),
                "researcher_instructions_sha256": _file_hash(RESOURCES / "rsi/AGENTS.md"),
                "evidence_tool_sha256": _file_hash(RESOURCES / "rsi/rsi.mjs"),
                "stage": catalog["stage"], "synthetic_test": args.fixture_only}
    key = args.api_key_file.read_text().strip()
    if not key:
        raise ValueError("protected model API key file is empty")
    agent = PiAgent(public_catalog=args.public_catalog, api_key=key, sidecar_image=sidecar_digest,
                    cache_dir=args.harness_cache_dir, arm=args.arm, reset_period=args.reset_period)
    encoder = CodeEncoder(args.out_dir, "fusion", args.embedding_model_path, args.embedding_device)
    intervention = json.loads(args.intervention_file.read_text()) if args.intervention_file else None
    controller = Controller(args.out_dir, config, agent, client,
                            lambda pool, rows, count: select(pool, rows, encoder, count), brief, identity,
                            intervention=intervention)
    try:
        result = controller.run(resume=args.resume)
        sealed = client.seal_training()
        if (sealed["used_slots"] != result["charged_records"]
                or sealed["source_sha256s"] != result["training_finalists"]):
            raise ValueError("public and private training records or finalists differ")
    except Exception as exc:
        if args.out_dir.exists():
            atomic_json_write(args.out_dir / "run-status.json", {"status": "paused", "error_type": type(exc).__name__})
        raise
    atomic_json_write(args.out_dir / "run-status.json", {"status": result["status"],
                                                         "stop_reason": result["stop_reason"]})
    print(json.dumps({"run_dir": str(args.out_dir), **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
