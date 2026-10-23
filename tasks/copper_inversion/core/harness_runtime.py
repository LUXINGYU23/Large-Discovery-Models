"""Task resources and Pi client assembly, using the existing sidecar launcher."""

import os
import json
import time
from dataclasses import replace
from pathlib import Path
import shutil
from ldm_tts.harness import (
    HarnessClient,
    HarnessProfile,
    HarnessLimits,
    HarnessToolExtension,
    HarnessSubmissionContract,
    HarnessTurn,
    HarnessSubmissionValidation,
    HarnessError,
)
from ldm_tts.harness.pi import PiHarnessConfig, load_pi_guest_runtime
from ldm_tts.harness.container import pi_sidecar_command
from ldm_tts.harness.protocol import file_sha256
from ldm_tts.harness.protocol import canonical_sha256
from ldm_tts.harness.research_runtime import usage_counts
from ldm_tts.engine.run_store import atomic_json_write
from .harness import submission_contract
from .data import save_problem, RESOURCES


def open_client(args, root, problem, stack, account):
    root = Path(root) / "harness"
    public = root / "public"
    public.mkdir(parents=True, exist_ok=True)
    save_problem(problem, public)
    for name in ("selfcheck.py", "TASK.md"):
        shutil.copy2(RESOURCES / name, public / name)
    for name in ("objective.py", "geometry.py"):
        shutil.copy2(RESOURCES.parent / "core" / name, public / name)
    (public / "__init__.py").write_text("")
    resources = RESOURCES / "harness"
    profile = resources / "profiles/researcher/AGENTS.md"
    tool = resources / "tools/copper.mjs"
    key = os.environ.get("LLM_API_KEY", "")
    if args.api_key_file:
        key = Path(args.api_key_file).read_text().strip()
    if not key:
        raise ValueError("LLM_API_KEY or protected --api-key-file is required")
    config = PiHarnessConfig(
        artifact_root=Path("/artifacts"),
        profiles=(
            HarnessProfile(
                "researcher",
                Path("/resources/profiles/researcher/AGENTS.md"),
                agents_sha256=file_sha256(profile),
            ),
        ),
        campaign_id=Path(root).parent.name,
        task_id="copper_inversion",
        case_id=args.case_id,
        seed=47,
        submission_contract=submission_contract(),
        base_url=args.llm_url or os.environ.get("LLM_BASE_URL", ""),
        model=args.llm_model or os.environ.get("LLM_MODEL_NAME", ""),
        guest_runtime=load_pi_guest_runtime("copper_inversion", resources / "image"),
        thinking=args.thinking,
        tool_extensions=(
            HarnessToolExtension(
                Path("/resources/tools/copper.mjs"),
                file_sha256(tool),
                ("get_inversion_context",),
            ),
        ),
        context7_enabled=False,
        limits=HarnessLimits(
            wall_time_seconds=args.harness_seconds,
            tool_call_budgets={"web_search": 0, "web_fetch": 0},
        ),
    )
    _preflight(args, root, config, key, account)
    client = HarnessClient(
        pi_sidecar_command(
            root,
            args.sidecar_image,
            cache_root=args.harness_cache_dir,
            mounts=((resources, "/resources", True),),
        ),
        api_key=key,
        config=config,
        response_timeout_seconds=args.harness_seconds + 120,
    )
    stack.callback(client.close)
    client.start()
    return client


def _preflight(args, root, config, key, account):
    """Probe the actual Pi Responses wire path, guest tools and readonly resources."""
    receipt = root / "preflight.json"
    if receipt.exists():
        account(usage_counts(json.loads(receipt.read_text())["usage"]), usage_key="copper:preflight:usage")
        return
    probe_root = root / "preflight"
    probe_root.mkdir(exist_ok=True)
    resources = RESOURCES / "harness"
    profile = resources / "profiles/researcher/PROBE.md"
    probe = replace(
        config,
        profiles=(
            HarnessProfile(
                "preflight",
                Path("/resources/profiles/researcher/PROBE.md"),
                agents_sha256=file_sha256(profile),
            ),
        ),
        submission_contract=HarnessSubmissionContract(
            "readiness",
            "submit_readiness",
            {
                "type": "object",
                "properties": {"ready": {"type": "boolean", "const": True}},
                "required": ["ready"],
                "additionalProperties": False,
            },
        ),
        tool_extensions=(),
        thinking="off",
        limits=HarnessLimits(
            wall_time_seconds=120, tool_call_budgets={"web_search": 0, "web_fetch": 0}
        ),
    )
    turn = HarnessTurn(
        "preflight",
        "readiness",
        0,
        0,
        0,
        canonical_sha256([]),
        "Follow the runtime readiness instructions, then submit_readiness. Do not perform research.",
    )
    started = time.monotonic()
    client = HarnessClient(
        pi_sidecar_command(
            probe_root,
            args.sidecar_image,
            cache_root=args.harness_cache_dir,
            mounts=((resources, "/resources", True),),
        ),
        api_key=key,
        config=probe,
        response_timeout_seconds=180,
    )
    usage = {}
    try:
        with client:
            (result,) = client.run_turn(
                (turn,),
                submission_validator=lambda _: HarnessSubmissionValidation("accept"),
            )
            usage = result.usage
        proof = probe_root / "sessions/preflight/workspace/probe.txt"
        if (
            result.submission_status != "accepted"
            or proof.read_text().strip() != "ready"
        ):
            raise HarnessError("runtime readiness proof missing")
        models = set()
        for path in probe_root.rglob("*.response.bin"):
            for line in path.read_text().splitlines():
                if line.startswith("data: {"):
                    event = json.loads(line[6:])
                    if event.get("type") == "response.completed":
                        models.add(event["response"].get("model"))
        if models != {config.model}:
            raise HarnessError("Responses model identity differs from requested model")
        atomic_json_write(
            receipt,
            {
                "status": "ok",
                "request_model": config.model,
                "response_models": sorted(models),
                "latency_seconds": time.monotonic() - started,
                "wire_api": "Responses via Pi",
                "usage": usage,
                "readonly_guest_probe": True,
            },
        )
    except HarnessError as exc:
        usage = exc.turn_usage.get("preflight", usage)
        raise
    finally:
        account(usage_counts(usage), usage_key="copper:preflight:usage")
