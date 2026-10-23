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


def prepare_public(root, problem):
    root = Path(root)
    public = root / "public"
    public.mkdir(parents=True, exist_ok=True)
    save_problem(problem, public)
    for name in ("selfcheck.py", "TASK.md"):
        shutil.copy2(RESOURCES / name, public / name)
    for name in ("objective.py", "geometry.py"):
        shutil.copy2(RESOURCES.parent / "core" / name, public / name)
    (public / "__init__.py").write_text("")
    if not (public / "history.json").exists():
        atomic_json_write(public / "history.json", [])


def researcher_profiles(parallel=False):
    profile = RESOURCES / "harness/profiles/researcher/AGENTS.md"
    return tuple(HarnessProfile(name, Path("/resources/profiles/researcher/AGENTS.md"),
                                agents_sha256=file_sha256(profile))
                 for name in (("researcher_1", "researcher_2", "researcher_3", "researcher_4")
                              if parallel else ("researcher",)))


def api_key(args):
    key = os.environ.get("LLM_API_KEY", "")
    if args.api_key_file:
        key = Path(args.api_key_file).read_text().strip()
    if not key:
        raise ValueError("LLM_API_KEY or protected --api-key-file is required")
    return key


def create_client(args, campaign_root, problem, *, artifact_root=None, profiles=None,
                  contract=None, fork_sources=None):
    root = Path(artifact_root) if artifact_root is not None else Path(campaign_root) / "harness"
    prepare_public(root, problem)
    resources = RESOURCES / "harness"
    tool = resources / "tools/copper.mjs"
    config = PiHarnessConfig(
        artifact_root=Path("/artifacts"),
        profiles=profiles or researcher_profiles(args.search_method == "ldm_harness"),
        fork_sources=fork_sources or {},
        campaign_id=Path(campaign_root).name,
        task_id="copper_inversion",
        case_id=args.case_id,
        seed=47,
        submission_contract=contract or submission_contract(),
        base_url=args.llm_url or os.environ.get("LLM_BASE_URL", ""),
        model=args.llm_model or os.environ.get("LLM_MODEL_NAME", ""),
        guest_runtime=load_pi_guest_runtime("copper_inversion", resources / "image"),
        thinking=args.thinking,
        # DeepSeek thinking mode rejects tool_choice=required.
        force_first_tool_call=False,
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
            tool_call_budgets={"bash": 8, "web_search": 0, "web_fetch": 0},
        ),
    )
    return HarnessClient(
        pi_sidecar_command(
            root,
            args.sidecar_image,
            cache_root=args.harness_cache_dir,
            mounts=((resources, "/resources", True),),
        ),
        api_key=api_key(args),
        config=config,
        response_timeout_seconds=args.harness_seconds + 120,
    )


def open_client(args, root, problem, stack, account):
    client = create_client(args, root, problem)
    stack.callback(client.close)
    _preflight(args, Path(root) / "harness", client.config, api_key(args), account)
    client.start()
    return client


def open_search(args, root, problem, stack, checks, runtime):
    from .harness import Expander, serialize_history, write_history
    from .mock import MockClient
    from ldm_tts.harness.research_runtime import (
        ResearchCoordinator, MeasuredResearchExpander, main_research_profile, research_contract,
    )

    profiles = researcher_profiles(args.search_method == "ldm_harness")
    identities = tuple(p.profile_id for p in profiles)
    harness_root = Path(root) / "harness"

    def client_at(artifact_root, *, main=False, sources=None):
        if args.mock:
            prepare_public(artifact_root, problem)
            return MockClient(artifact_root, profiles=("main_research",) if main else identities,
                              fork_sources=sources)
        return create_client(args, root, problem, artifact_root=artifact_root,
                             profiles=main_research_profile(artifact_root, profiles[0]) if main else profiles,
                             contract=research_contract() if main else submission_contract(),
                             fork_sources=sources)

    def adapter(client, artifact_root, cursors=None):
        return Expander(root, client, checks, runtime.consume_many, artifact_root=artifact_root,
                        profiles=identities, inherited_history_counts=cursors,
                        selection_mode="acquisition" if args.search_method == "ldm_harness" else "reservoir_order")

    if not args._research_strategy.requires_main:
        if args.mock:
            client = client_at(harness_root)
        else:
            client = open_client(args, root, problem, stack, runtime.consume_many)
        return adapter(client, harness_root)
    if not args.mock:
        probe_client = client_at(harness_root)
        _preflight(args, harness_root, probe_client.config, api_key(args), runtime.consume_many)
    coordinator = ResearchCoordinator(
        harness_root, args._research_strategy, profiles,
        lambda: client_at(harness_root / "main", main=True), account=runtime.consume_many,
    )
    return MeasuredResearchExpander(
        coordinator, lambda artifact_root, sources: client_at(artifact_root, sources=sources), adapter,
        serialize_history=lambda observations: serialize_history(root, observations),
        write_history=lambda artifact_root, observations: write_history(root, artifact_root, observations),
        proposal_key=lambda proposal: proposal.payload["sha256"],
        objective_name="research_utility", first_active_round=1,
    )


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
                "base_url": config.base_url,
                "thinking": config.thinking,
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
