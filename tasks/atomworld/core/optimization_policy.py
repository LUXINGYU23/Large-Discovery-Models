"""Independent compiled public-audit policy; never consumes benchmark observations.

This optional extension ranks simultaneous public drafts using an untrained prior.
It is deliberately not advertised as objective-trained LDM/GP optimization.
"""

from __future__ import annotations

from io import StringIO
import hashlib

import numpy as np

from ldm_tts.harness import (
    DockerPolicyExecutor,
    PolicyCapabilityContract,
    PolicyResearchController,
    PolicyRoundInput,
)
from ldm_tts.harness.container import resolve_container_user
from tasks.atomworld.core.data import write_json
from tasks.atomworld.core.proposals import extract_cif, public_validation

FEATURE_NAMES = (
    "syntax_parseable",
    "atom_count",
    "element_count",
    "output_char_count",
    "geometry_features_available",
    "ase_input_atom_count",
    "ase_atom_count_delta",
    "cell_volume_ratio",
    "same_order_comparison_available",
    "same_order_mean_displacement_angstrom",
    "same_order_max_displacement_angstrom",
    "same_order_species_changes",
)


def _geometry_features(sample, text, *, mock=False):
    # Public input-to-draft invariants, never target-to-draft discrepancies.
    # Same-index comparison is explicitly qualified; CIF site ordering may differ.
    if mock:
        return [0.0] * 8
    from ase.io import read

    try:
        initial = read(StringIO(sample["input_cif"]), format="cif")
        draft = read(StringIO(extract_cif(text)), format="cif")
        same_count = len(initial) == len(draft)
        shifts = (
            np.linalg.norm(draft.positions - initial.positions, axis=1)
            if same_count
            else np.zeros(1)
        )
        values = [
            1.0,
            len(initial),
            len(draft) - len(initial),
            draft.get_volume() / initial.get_volume(),
            float(same_count),
            float(shifts.mean()),
            float(shifts.max()),
            int(np.sum(draft.numbers != initial.numbers)) if same_count else 0,
        ]
        return values if np.isfinite(values).all() else [0.0] * 8
    except (ValueError, TypeError, IndexError, KeyError, ZeroDivisionError):
        return [0.0] * 8


class PublicAuditPolicyAdapter:
    def capability_contract(self):
        return PolicyCapabilityContract(
            task_id="atomworld",
            api_version=1,
            enabled_capabilities=("prior_mean@1",),
            feature_names=FEATURE_NAMES,
            feature_groups={"public_structure": (0, len(FEATURE_NAMES))},
            mean_clip=4.0,
            default_alpha=0.0,
            default_eta=0.0,
        )

    def with_feedback(self, round_input, records):
        # No measured predictions, hidden scores, or selection feedback are legal.
        return round_input

    def validate_task_execution(self, execution, round_input):
        return ()


def public_policy_input(round_index, sample, answers, history, *, mock=False):
    features = []
    for answer in answers:
        text = answer.submission["generated_output"]
        check = public_validation(text, mock=mock)
        features.append(
            [
                float(check["parseable"]),
                check.get("atom_count", 0),
                len(check.get("composition", {})),
                len(text),
                *_geometry_features(sample, text, mock=mock),
            ]
        )
    # Empty histories are an explicit no-label protocol, not synthetic zero labels.
    return PolicyRoundInput(
        round_index=round_index,
        history_features=np.empty((0, len(FEATURE_NAMES))),
        history_utilities=np.empty(0),
        query_features=np.asarray(features, dtype=float),
        history_candidate_ids=(),
        history_rounds=(),
        measured_observations=(),
        research_snapshot={
            "public_task": {key: value for key, value in sample.items() if key != "input_cif"},
            "public_task_access": "get_public_task",
            "public_draft_history": [{k: v for k, v in row.items() if k != "generated_output"} for row in history],
            "current_drafts": [{"draft_id": f"current_{i}", "rationale": answer.submission["rationale"],
                "output_sha256": hashlib.sha256(answer.submission["generated_output"].encode()).hexdigest()}
                for i, answer in enumerate(answers)],
            "draft_access": "get_public_history with draft_ids and detail=detailed",
            "feedback_policy": "No correctness labels, hidden targets, judge errors or objective history. All priors are unverified public structural hypotheses. No GP is fitted. Only this revision's drafts are ranked; the last scheduled revision remains the final answer.",
        },
        execution_context={
            "mean_context": {
                "target_location": 0.0,
                "target_scale": 1.0,
                "feature_names": list(FEATURE_NAMES),
                "feature_semantics": "ASE geometry features compare only the public input to each draft. Availability flags mark unsupported parsing or unequal atom counts. Same-order displacements are Cartesian angstroms without periodic matching and may change under site reordering; species_changes counts substitutions at corresponding indices. They are public invariants, not correctness labels or hidden judge distances. Mock geometry features are explicitly unavailable.",
                "public_task": {key: value for key, value in sample.items() if key != "input_cif"},
                "scale_semantics": "Untrained standardized plausibility prior, never a measured correctness score",
            },
            "weight_context": {"default_alpha": 0.0, "default_eta": 0.0},
        },
    )


def select_public_answer(expander, sample_index, round_index, sample, answers, history):
    client, root = expander._client(sample_index, policy=True)
    write_json(root / "public_history.json", {"drafts": [*history, *[
        {"draft_id": f"current_{i}", "round_idx": round_index,
         "generated_output": answer.submission["generated_output"],
         "rationale": answer.submission["rationale"],
         "public_validation": public_validation(answer.submission["generated_output"], mock=expander.args.mock),
         "output_sha256": hashlib.sha256(answer.submission["generated_output"].encode()).hexdigest()}
        for i, answer in enumerate(answers)
    ]]})
    if sample_index not in expander.controllers:
        args = expander.args
        executor = expander.policy_executor or DockerPolicyExecutor(
            image=args.policy_runner_image,
            docker_host=args.harness_docker_host,
            container_user=resolve_container_user(
                args.harness_container_user, args.harness_docker_host
            ),
        )
        expander.controllers[sample_index] = PolicyResearchController(
            client=client,
            adapter=PublicAuditPolicyAdapter(),
            executor=executor,
            root=root,
            account=expander.runtime.consume_many
            if expander.runtime
            else None,
            recovery_budget=lambda: float(args.harness_recovery_seconds),
        )
    result = expander.controllers[sample_index].resolve(
        public_policy_input(
            round_index, sample, answers, history, mock=expander.args.mock
        )
    )
    selected = int(np.argmax(result.query_prior_mean))
    return selected, {
        "method": "compiled_public_audit_prior",
        "index": selected,
        "prior": result.query_prior_mean.tolist(),
        "source": result.source,
        "degraded": result.degraded,
        "epoch_id": result.epoch_id,
        "artifact_digest": result.artifact_digest,
        "objective_measurements_visible": False,
    }
