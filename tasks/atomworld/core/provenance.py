"""Immutable schedule and task-resource identity for campaign recovery."""

from __future__ import annotations

import json

from ldm_tts.transport.openai import generation_body
from ldm_tts.harness.ucb_diagnostics import DIAGNOSTICS_PATH
from .data import TASK_ROOT, sha256_file, write_json


def bind_schedule(args, run_dir, samples, manifest, config, *, proposal_samples, encoder, ldm_mode, harness_mode):
    # These immutable values prevent resuming against a different task schedule or endpoint policy.
    immutable = {
        "samples": samples,
        "attempts_per_sample": args.attempts_per_sample,
        "dataset_manifest": manifest,
        "mock": args.mock,
        "proposal_format": args.proposal_format,
        "tool_source_sha256": {
            str(p.relative_to(TASK_ROOT)): sha256_file(p)
            for p in sorted(
                (TASK_ROOT / "resources/harness/image/atomworld_tools").glob("*.py")
            )
        }
        if args.proposal_format == "operations"
        else {},
        "llm_model_name": args.llm_model_name,
        "llm_max_tokens": args.llm_max_tokens,
        "llm_wire_api": args.llm_wire_api,
        "llm_reasoning": args.llm_reasoning,
        "llm_temperature": args.llm_temperature,
        "llm_extra_body_json": args.llm_extra_body_json,
        "resolved_generation_body": generation_body(wire_api=args.llm_wire_api, reasoning=args.llm_reasoning,
                                                    extra_body=json.loads(args.llm_extra_body_json)),
        "search_method": args.search_method,
        "campaign_index": args.campaign_index,
        "service_retry_allowance": args.service_retry_allowance,
        "ldm_configuration": {
            "feedback_protocol": "measured_correctness_optimization",
            "proposal_samples": proposal_samples,
            **{key: config[key] for key in ("bo_pool_size", "acquisition_alpha", "acquisition_eta", "ucb_beta",
                                            "z_clip", "gp_history_limit", "gp_lengthscale", "gp_noise")},
            "feature_version": encoder.version,
        } if ldm_mode else None,
    }
    if harness_mode:
        from ldm_tts.harness import canonical_sha256

        resource_root = TASK_ROOT / "resources/harness"
        immutable["harness_resources_sha256"] = canonical_sha256(
            {
                str(path.relative_to(resource_root)): sha256_file(path)
                for path in sorted(resource_root.rglob("*"))
                if path.is_file() and "__pycache__" not in path.parts
            }
        )
        immutable["harness_settings"] = {
            key: value
            for key, value in config.items()
            if key.startswith(("harness_", "policy_"))
        }
        if args.search_method == "ldm_harness_compiled":
            immutable["policy_diagnostics_sha256"] = sha256_file(DIAGNOSTICS_PATH)
    immutable_path = run_dir / "schedule.json"
    if args.resume and immutable_path.exists():
        if json.loads(immutable_path.read_text()) != immutable:
            raise ValueError(
                "Resume requires the same dataset, schedule, model and generation settings"
            )
    else:
        write_json(immutable_path, immutable)
