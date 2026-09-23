"""Bind the AlphaBench research tools and persistent Pi sessions to a campaign."""

from pathlib import Path

from ldm_tts.harness import (HarnessClient, HarnessLimits, HarnessMcpServer, HarnessMcpValue,
    HarnessNetworkPolicy, HarnessProfile, DockerPolicyExecutor, PolicyResearchController,
    canonical_sha256, directory_sha256, file_sha256, policy_submission_contract)
from ldm_tts.harness.container import docker_identity_args, resolve_container_user
from ldm_tts.harness.pi import PiHarnessConfig, load_pi_guest_runtime, policy_mcp_server
from .harness import HarnessExpander, submission_contract
from .harness_tools import HarnessToolService


RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources" / "harness"
SOCKET_ROOT = Path("/mnt/data1/Large-Discovery-Models/tmp/t3-sockets")
CACHE_ROOT = Path("/mnt/data1/Large-Discovery-Models/cache/alphabench-pi")
TOOL_NAMES = ("get_contract", "validate_expression", "check_expression", "get_history", "get_observation")


def _sidecar_command(artifact_root, sidecar_image, extra_mounts=()):
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    (CACHE_ROOT / "runtime-overlays").mkdir(exist_ok=True)
    user = resolve_container_user(None)
    command = ["docker", "run", "--rm", "-i", *docker_identity_args(user),
        "--device", "/dev/kvm"]
    environment = {"HOME": "/runtime-home", "XDG_CACHE_HOME": "/runtime-home/.cache",
        "GONDOLIN_IMAGE_STORE": "/runtime-home/.cache/gondolin/images",
        "GONDOLIN_SESSIONS_DIR": "/runtime-home/.cache/gondolin/sessions",
        "LDM_HARNESS_CACHE_ROOT": "/runtime-home/.cache/gondolin",
        "TMPDIR": "/runtime-home/.cache/gondolin/runtime-overlays"}
    for key, value in sorted(environment.items()):
        command.extend(("--env", f"{key}={value}"))
    for source_path, destination, readonly in (
        (artifact_root, "/artifacts", False), (CACHE_ROOT, "/runtime-home/.cache/gondolin", False),
        (RESOURCE_ROOT, "/resources", True), *extra_mounts):
        command.extend(("--mount", f"type=bind,src={source_path.resolve()},dst={destination}"
            + (",readonly" if readonly else "")))
    command.append(sidecar_image)
    return command


def build_harness(protocol, gateway, collection, *, api_key, sidecar_image):
    runtime = gateway.runtime
    artifact_root = runtime.run_dir / "harness"
    artifact_root.mkdir(parents=True, exist_ok=True)
    query = protocol.harness_surrogate_query
    tools = HarnessToolService(protocol, gateway, SOCKET_ROOT, surrogate_query=query)
    try:
        names = TOOL_NAMES + (("query_surrogate",) if query else ())
        source = RESOURCE_ROOT / "tools" / "t3-mcp.mjs"
        fields = {"server_id": "t3", "transport": "stdio", "tools": names,
            "command": "node", "args": ("/app/t3-mcp.mjs", "stdio"),
            "script_sha256": file_sha256(source), "socket": "/t3-host/host.sock"}
        mcp = HarnessMcpServer(server_id="t3", transport="stdio", tools=names,
            config_sha256=canonical_sha256(fields), command="node", args=fields["args"],
            env=(("LDM_T3_HOST_SOCKET", HarnessMcpValue(value=fields["socket"])),))
        profile_source = RESOURCE_ROOT / "profiles" / "research" / "AGENTS.md"
        count = 1 if protocol.method == "harness" else protocol.sessions
        variant = "direct" if protocol.method == "harness" else "gp_query" if query else "no_query"
        profiles = tuple(HarnessProfile(
            f"{variant}_{index:02d}", Path("/resources/profiles/research/AGENTS.md"),
            agents_sha256=file_sha256(profile_source)) for index in range(count))
        budgets = {f"mcp__t3__{name}": limit for name, limit in (
            ("get_contract", 5), ("validate_expression", 100), ("check_expression", 30),
            ("get_history", 40), ("get_observation", 40), ("query_surrogate", 50)) if name in names}
        budgets.update({"web_search": 0, "fetch_content": 0, "get_search_content": 0})
        config = PiHarnessConfig(artifact_root=Path("/artifacts"), profiles=profiles,
            campaign_id=runtime.run_id, task_id="alphabench",
            case_id=f"{protocol.backend}:{protocol.market}:{variant}", seed=protocol.random_seed,
            submission_contract=submission_contract(max(protocol.batch_size, protocol.candidates_per_session)),
            limits=HarnessLimits(wall_time_seconds=1800, tool_call_budgets=budgets),
            base_url=protocol.endpoint.removesuffix("/responses"), model=protocol.model,
            thinking=protocol.reasoning_effort,
            guest_runtime=load_pi_guest_runtime("alphabench", RESOURCE_ROOT / "image"),
            mcp_servers=(mcp,), context7_enabled=False,
            network_policy=HarnessNetworkPolicy(allowed_hosts=("offline.invalid",)),
            force_first_tool_call=False,
            provider_request_body={"reasoning": {"effort": protocol.reasoning_effort}, "store": False})
        command = _sidecar_command(artifact_root, sidecar_image,
            ((source, "/app/t3-mcp.mjs", True), (tools.socket_dir, "/t3-host", False)))
        client = HarnessClient(command, api_key=api_key, config=config,
            response_timeout_seconds=protocol.request_timeout + 1800)
        return client, tools, HarnessExpander(client, protocol, gateway, artifact_root,
            tools=tools, collection=collection)
    except BaseException:
        tools.close()
        raise


def build_policy_harness(protocol, gateway, adapter, meter, *, api_key, sidecar_image):
    artifact_root = gateway.runtime.run_dir / "policy_harness"
    artifact_root.mkdir(parents=True, exist_ok=True)
    profile_source = RESOURCE_ROOT / "profiles" / "policy_architect" / "AGENTS.md"
    skill_source = RESOURCE_ROOT / "skills" / "compile-ldm-policy"
    profile = HarnessProfile("policy_architect", Path("/resources/profiles/policy_architect/AGENTS.md"),
        skill_dirs=(Path("/resources/skills/compile-ldm-policy"),),
        agents_sha256=file_sha256(profile_source),
        skill_dir_sha256=(directory_sha256(skill_source),))
    budgets = {"mcp__ldm_policy__inspect_policy_contract": 10,
        "mcp__ldm_policy__validate_policy_draft": 20,
        "read": 40, "write": 20, "bash": 20,
        "web_search": 0, "fetch_content": 0, "get_search_content": 0}
    config = PiHarnessConfig(artifact_root=Path("/artifacts"), profiles=(profile,),
        campaign_id=gateway.runtime.run_id, task_id="alphabench",
        case_id=f"{protocol.backend}:{protocol.market}:compiled_policy", seed=protocol.random_seed,
        submission_contract=policy_submission_contract(),
        limits=HarnessLimits(wall_time_seconds=1800, tool_call_budgets=budgets),
        base_url=protocol.endpoint.removesuffix("/responses"), model=protocol.model,
        thinking=protocol.reasoning_effort,
        guest_runtime=load_pi_guest_runtime("alphabench", RESOURCE_ROOT / "image"),
        mcp_servers=(policy_mcp_server(draft_execution_enabled=False),), context7_enabled=False,
        network_policy=HarnessNetworkPolicy(allowed_hosts=("offline.invalid",)),
        force_first_tool_call=False,
        provider_request_body={"reasoning": {"effort": protocol.reasoning_effort}, "store": False})
    client = HarnessClient(_sidecar_command(artifact_root, sidecar_image), api_key=api_key,
        config=config, response_timeout_seconds=protocol.request_timeout + 1800)

    def authorize(request):
        from ldm_tts.contracts.evaluation import EvaluationPaused
        try:
            return meter.authorize(request)
        except ValueError as exc:
            raise EvaluationPaused("policy provider authorization identity mismatch", status="paused_provider") from exc

    controller = PolicyResearchController(client=client, adapter=adapter,
        executor=DockerPolicyExecutor(image=sidecar_image, container_user=resolve_container_user(None)),
        root=artifact_root, account=None, recovery_budget=lambda: 3600.0,
        provider_authorizer=authorize)
    return client, controller
