"""Persistent, sample-isolated research sessions with public-only feedback.

The only campaign request field read here is round_idx. Judge observations, parents,
and acquisition feedback are deliberately outside this module's data boundary.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from ldm_tts.contracts import RawProposal
from ldm_tts.engine.expansion import ExpansionResult
from ldm_tts.harness import (
    HarnessClient,
    HarnessError,
    HarnessLimits,
    HarnessNetworkPolicy,
    HarnessProfile,
    HarnessSubmissionContract,
    HarnessSubmissionError,
    HarnessSubmissionValidation,
    HarnessToolExtension,
    HarnessTurn,
    canonical_sha256,
    directory_sha256,
    file_sha256,
    parse_tool_call_budgets,
)
from ldm_tts.harness.container import docker_identity_args, resolve_container_user
from ldm_tts.harness.mcp import load_harness_mcp_config
from ldm_tts.harness.pi import PiHarnessConfig, load_pi_guest_runtime, policy_mcp_server
from tasks.atomworld.core.data import TASK_ROOT, write_json
from tasks.atomworld.core.proposals import (
    MAX_OUTPUT_CHARS,
    canonical_key,
    public_validation,
)

RESOURCE_ROOT = TASK_ROOT / "resources/harness"
PROFILE_IDS = ("geometry_research", "structure_audit")
TOOL_NAMES = ("get_public_task", "get_public_history", "get_geometry_contract")


def harness_profiles(ids=PROFILE_IDS):
    if (
        not ids
        or len(set(ids)) != len(ids)
        or set(ids) - {*PROFILE_IDS, "policy_architect"}
    ):
        raise ValueError("Select distinct known AtomWorld Harness profiles")
    return tuple(
        HarnessProfile(
            name,
            Path("/resources/profiles") / name / "AGENTS.md",
            skill_dirs=(
                Path("/resources/skills")
                / (
                    "public_audit_policy"
                    if name == "policy_architect"
                    else "crystal_geometry"
                ),
            ),
            agents_sha256=file_sha256(RESOURCE_ROOT / "profiles" / name / "AGENTS.md"),
            skill_dir_sha256=(
                directory_sha256(
                    RESOURCE_ROOT
                    / "skills"
                    / (
                        "public_audit_policy"
                        if name == "policy_architect"
                        else "crystal_geometry"
                    )
                ),
            ),
        )
        for name in ids
    )


def submission_contract(max_attempts=3):
    return HarnessSubmissionContract(
        contract_id="atomworld_public_answer",
        tool_name="submit_answer",
        payload_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "generated_output": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_OUTPUT_CHARS,
                },
                "rationale": {"type": "string", "minLength": 1, "maxLength": 8000},
            },
            "required": ["generated_output", "rationale"],
        },
        max_validation_attempts=max_attempts,
    )


def validate_answer(request, *, mock=False):
    value = request.submission
    errors = []
    if set(value) != {"generated_output", "rationale"} or request.artifacts:
        errors.append(
            HarnessSubmissionError(
                "",
                "invalid_fields",
                "Submit exactly generated_output and rationale, with no artifacts.",
            )
        )
    text = value.get("generated_output")
    if not isinstance(text, str) or not 0 < len(text) <= MAX_OUTPUT_CHARS:
        errors.append(
            HarnessSubmissionError(
                "/generated_output",
                "invalid_output",
                "Return a nonempty complete CIF answer within the output limit.",
            )
        )
    elif not public_validation(text, mock=mock)["parseable"]:
        errors.append(
            HarnessSubmissionError(
                "/generated_output",
                "invalid_cif",
                "The public CIF syntax check failed.",
                "Repair a complete <cif>...</cif> block and resubmit in this session.",
            )
        )
    rationale = value.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 8000:
        errors.append(
            HarnessSubmissionError(
                "/rationale",
                "invalid_rationale",
                "Provide a concise nonempty research rationale (at most 8000 characters).",
            )
        )
    return (
        HarnessSubmissionValidation("retry", tuple(errors))
        if errors
        else HarnessSubmissionValidation()
    )


def make_client(args, root, sample, *, policy=False):
    """Mount only task research resources and this sample's public artifact tree."""
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "public_task.json", sample)
    if not (root / "public_history.json").exists():
        write_json(root / "public_history.json", {"drafts": []})
    profiles = harness_profiles(
        ("policy_architect",) if policy else args.harness_profile
    )
    if policy:
        from ldm_tts.harness import policy_submission_contract

        contract = policy_submission_contract(args.harness_max_submission_attempts)
    else:
        contract = submission_contract(args.harness_max_submission_attempts)
    mcp = load_harness_mcp_config(args.harness_mcp_config)
    servers = (*mcp.servers, policy_mcp_server()) if policy else mcp.servers
    cache = args.harness_cache_dir.expanduser().resolve()
    (cache / "runtime-overlays").mkdir(parents=True, exist_ok=True)
    command = ["docker"]
    if args.harness_docker_host:
        command.extend(("--host", args.harness_docker_host))
    command.extend(("run", "--rm", "-i"))
    command.extend(
        docker_identity_args(
            resolve_container_user(
                args.harness_container_user, args.harness_docker_host
            ),
            args.harness_docker_host,
        )
    )
    command.extend(("--device", "/dev/kvm"))
    environment = {
        "HOME": "/runtime-home",
        "XDG_CACHE_HOME": "/runtime-home/.cache",
        "GONDOLIN_IMAGE_STORE": "/runtime-home/.cache/gondolin/images",
        "GONDOLIN_SESSIONS_DIR": "/runtime-home/.cache/gondolin/sessions",
        "LDM_HARNESS_CACHE_ROOT": "/runtime-home/.cache/gondolin",
        "TMPDIR": "/runtime-home/.cache/gondolin/runtime-overlays",
        "LDM_ATOMWORLD_PUBLIC_ROOT": "/artifacts",
    }
    for name, value in sorted(environment.items()):
        command.extend(("--env", f"{name}={value}"))
    for source, destination, readonly in (
        (root, "/artifacts", False),
        (cache, "/runtime-home/.cache/gondolin", False),
        (RESOURCE_ROOT.resolve(), "/resources", True),
    ):
        command.extend(
            (
                "--mount",
                f"type=bind,src={source},dst={destination}"
                + (",readonly" if readonly else ""),
            )
        )
    command.append(args.harness_sidecar_image)
    client = HarnessClient(
        command,
        api_key=os.environ.get(
            "LLM_API_KEY",
            os.environ.get("LDM_LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
        ),
        named_secrets=mcp.named_secrets,
        response_timeout_seconds=args.harness_response_timeout,
        config=PiHarnessConfig(
            artifact_root=Path("/artifacts"),
            profiles=profiles,
            campaign_id=canonical_sha256(
                {"run": str(root.parents[1]), "sample": sample}
            )[:32],
            task_id="atomworld",
            case_id=sample["sample_id"],
            seed=args.campaign_index,
            submission_contract=contract,
            base_url=args.llm_url,
            model=args.llm_model_name,
            guest_runtime=load_pi_guest_runtime("atomworld", RESOURCE_ROOT / "image"),
            tool_extensions=(
                HarnessToolExtension(
                    Path("/resources/tools/public_context.mjs"),
                    file_sha256(RESOURCE_ROOT / "tools/public_context.mjs"),
                    TOOL_NAMES,
                ),
            ),
            mcp_servers=servers,
            thinking=args.harness_thinking,
            limits=HarnessLimits(
                wall_time_seconds=args.harness_wall_time_seconds,
                tool_call_budgets=parse_tool_call_budgets(
                    args.harness_tool_budget, excluded_tools=(contract.tool_name,)
                ),
            ),
            network_policy=HarnessNetworkPolicy(
                forbidden_query_patterns=(
                    r"atomworld",
                    r"target_cif",
                    r"private\.jsonl",
                    r"benchmark.*answers?",
                )
            ),
        ),
    )
    client.start()
    return client


class AtomWorldHarnessExpander:
    def __init__(
        self,
        samples,
        args,
        run_dir,
        *,
        client_factory=make_client,
        policy_executor=None,
    ):
        self.samples = [dict(row) for row in samples]
        self.args, self.run_dir = args, Path(run_dir)
        self.client_factory, self.policy_executor = client_factory, policy_executor
        self.clients, self.controllers = {}, {}
        self.runtime = None

    def close(self):
        for client in self.clients.values():
            client.close()
        self.clients.clear()

    def _client(self, sample_index, *, policy=False):
        for old_key in list(self.clients):
            if old_key[0] != sample_index:
                self.clients.pop(old_key).close()
                self.controllers.pop(old_key[0], None)
        key = (sample_index, policy)
        root = (
            self.run_dir
            / ("policy_harness" if policy else "harness")
            / f"sample_{sample_index:06d}"
        )
        if key not in self.clients:
            self.clients[key] = self.client_factory(
                self.args, root, self.samples[sample_index], policy=policy
            )
        return self.clients[key], root

    def expand(self, request):
        index, attempt = divmod(request.round_idx, self.args.attempts_per_sample)
        sample = self.samples[index]
        path = self.run_dir / "attempts" / f"{request.round_idx:06d}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record["sample_id"] != sample["sample_id"]:
                raise ValueError("Resumed Harness answer differs from scheduled sample")
            return self._result(record)
        client, root = self._client(index)
        history = []
        for previous in range(attempt):
            ordinal = index * self.args.attempts_per_sample + previous
            draft = json.loads(
                (self.run_dir / "attempts" / f"{ordinal:06d}.json").read_text()
            )
            history.append(
                {
                    key: draft[key]
                    for key in ("round_idx", "generated_output", "public_validation")
                }
            )
        write_json(root / "public_history.json", {"drafts": history})
        delta = history[-1:]
        digest = canonical_sha256(delta)
        message = json.dumps(
            {
                "message_type": "public_history_delta"
                if attempt
                else "public_task_bootstrap",
                "sample": sample,
                "revision": attempt,
                "new_public_drafts": delta,
                "history_count": len(history),
                "history_access": "get_public_history",
                "required_answers_per_session": 1,
                "instructions": "Research the public geometry in your persistent sandbox. Read crystal_geometry skill, use structured tools, inspect CIF with ASE/pymatgen, and submit exactly one complete answer and rationale. Correct rejected syntax in the same session. Judges, private answers and accuracy feedback are unavailable. Prior drafts are hypotheses, not evidence of correctness. Each scheduled revision replaces the previous final answer.",
            },
            sort_keys=True,
        )
        turns = tuple(
            HarnessTurn(
                profile.profile_id,
                f"revision_{attempt:04d}_{canonical_sha256([profile.profile_id, message])[:20]}",
                request.round_idx,
                max(0, attempt - 1),
                attempt,
                digest,
                message,
                forbidden_query_terms=("atomworld", "target_cif", "private.jsonl"),
            )
            for profile in client.config.profiles
        )
        if self.runtime:
            self.runtime.consume("harness_turns", len(turns))
        try:
            results = client.run_turn(
                turns,
                submission_validator=lambda value: validate_answer(
                    value, mock=self.args.mock
                ),
                recovery_timeout_seconds=self.args.harness_recovery_seconds,
            )
        except HarnessError as exc:
            for turn in turns:
                self._account_usage(
                    index, turn.turn_id, exc.turn_usage.get(turn.profile_id, {})
                )
            raise
        for result in results:
            self._account_usage(index, result.turn_id, result.usage)
        # A strict barrier: neither missing nor rejected sessions reduce the requested count.
        by_id = {result.profile_id: result for result in results}
        if len(results) != len(turns) or set(by_id) != {
            turn.profile_id for turn in turns
        }:
            raise RuntimeError(
                "Harness must commit exactly one answer per independent session"
            )
        ordered = [by_id[turn.profile_id] for turn in turns]
        for result in ordered:
            from ldm_tts.harness import HarnessSubmissionRequest

            validation = validate_answer(
                HarnessSubmissionRequest(
                    result.profile_id,
                    result.turn_id,
                    1,
                    result.submission,
                    result.submitted_artifacts,
                ),
                mock=self.args.mock,
            )
            if (
                result.submission_status != "accepted"
                or validation.decision != "accept"
            ):
                raise RuntimeError(
                    f"Harness submission rejected for {result.profile_id}; repair allowance exhausted"
                )
        selected_index = (attempt + self.args.campaign_index) % len(ordered)
        selection = {"method": "predeclared_profile_rotation", "index": selected_index}
        if self.args.search_method == "blind_harness_compiled":
            from tasks.atomworld.core.optimization_policy import select_public_answer

            selected_index, selection = select_public_answer(
                self, index, request.round_idx, sample, ordered, history
            )
        selected = ordered[selected_index]
        record = {
            "sample_id": sample["sample_id"],
            "action_name": sample["action_name"],
            "attempt": attempt,
            "round_idx": request.round_idx,
            "generated_output": selected.submission["generated_output"],
            "canonical_key": canonical_key(
                sample["sample_id"], selected.submission["generated_output"]
            ),
            "public_validation": public_validation(
                selected.submission["generated_output"], mock=self.args.mock
            ),
            "sessions": [asdict(result) for result in ordered],
            "selection": selection,
        }
        write_json(path, record)
        return self._result(record)

    def _account_usage(self, sample_index, turn_id, usage):
        if self.runtime:
            self.runtime.consume_many(
                {
                    "llm_requests": int(usage.get("providerCalls", 0)),
                    "harness_tool_calls": sum(
                        int(value) for value in usage.get("toolCalls", {}).values()
                    ),
                    "harness_validation_submissions": int(
                        usage.get("validationSubmissions", 0)
                    ),
                    "harness_artifact_bytes": int(usage.get("artifactBytes", 0)),
                },
                usage_key=f"atomworld:{sample_index}:{turn_id}",
            )

    @staticmethod
    def _result(record):
        return ExpansionResult(
            proposals=(
                RawProposal(
                    {
                        key: record[key]
                        for key in ("sample_id", "action_name", "generated_output")
                    },
                    "atomworld_persistent_harness",
                    metadata={
                        "collectable": False,
                        "round_idx": record["round_idx"],
                        "selection": record["selection"],
                    },
                ),
            ),
            selection_mode="reservoir_order",
            metadata={
                "feedback": "public_syntax_only",
                "sessions": record["sessions"],
                "selection": record["selection"],
            },
        )
