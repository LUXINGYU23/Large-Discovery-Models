"""LDM selection, zero-prior parity, and the compiled policy through the shared controller."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ldm_tts.contracts import Candidate
from ldm_tts.harness import (
    CompiledOptimizationPolicy, HarnessSubmissionError, PolicyResearchController, file_sha256,
)
from ldm_tts.harness.policy_execution import PolicyExecutionError, PolicyExecutionResult
from ldm_tts.optimization.records import BOObservation
from tasks.researchgym.core import workflow
from tasks.researchgym.core.candidate import FEATURE_NAMES, ProgramEncoder
from tasks.researchgym.core.cases import load_case
from tasks.researchgym.core.harness import RESOURCE_ROOT, create_client
from tasks.researchgym.core.optimization_policy import ResearchGymPolicyAdapter
from tasks.researchgym.core.sampling import Q0_KEY, tilted_distribution
from tasks.researchgym.core.selection import GPSettings, ProgramLDMSelector

CASE = load_case("cross_modal_retrieval")
ENCODER = ProgramEncoder(CASE)
FIXTURE = Path(__file__).with_name("fixtures") / "harness_sidecar.py"
RUNNER = Path(__file__).resolve().parents[3] / "harnesses/pi/policy_runner.py"
GP = GPSettings(lengthscale=0.5, noise=0.05, beta=1.0, history_limit=64, target_scale_floor=0.5)


def program(tag, calls=0):
    body = "".join("    x = x.softmax(1).topk(1)[0]\n" for _ in range(calls))
    return CASE.seed_program() + f"\n\ndef _variant_{tag}(x):\n{body}    return x * {tag}\n"


def candidate(tag, count=1, total=6, calls=0):
    source = program(tag, calls)
    key = CASE.canonical_key(source)
    return Candidate(f"rg-{key[:16]}", {"program": source}, key, metadata={
        Q0_KEY: {"occurrence_count": count, "valid_occurrence_count": total, "probability": count / total}})


def history():
    rows = []
    for i, (tag, calls, score) in enumerate(((10, 0, 60.0), (11, 2, 62.5), (12, 4, 63.0), (13, 1, 61.0))):
        c = candidate(tag, calls=calls)
        rows.append(BOObservation(c.candidate_id, (score,), ENCODER.encode(c), {"round_idx": i // 2}))
    return rows


def pool():
    candidates = [candidate(1, 2, calls=3), candidate(2, 1), candidate(3, 2, calls=1), candidate(4, 1, calls=5)]
    return candidates, {c.candidate_id: ENCODER.encode(c) for c in candidates}


def selector(**policy):
    return ProgramLDMSelector(GP, objective_name="cmr_coco_c_i2t_r1", alpha=1.0, eta=1.0, z_clip=5.0, seed=3,
                              pool_size=4, **policy)


def test_distribution_is_q0_power_times_exp_eta_robust_z():
    q0 = np.asarray([0.5, 0.25, 0.25])
    probability, logits, z = tilted_distribution(q0, [1.0, 2.0, 3.0], alpha=2.0, eta=0.5)
    expected = q0 ** 2 * np.exp(0.5 * z)
    assert probability == pytest.approx(expected / expected.sum())
    assert tilted_distribution(q0, [1.0, 2.0, 3.0], alpha=1.0, eta=0.0)[0] == pytest.approx(q0)


class ZeroPolicy:
    def __init__(self):
        self.records = []

    def resolve(self, round_input):
        return CompiledOptimizationPolicy(None, None, np.zeros(len(round_input.history_features)),
                                          np.zeros(len(round_input.query_features)), "default", 1.0, 1.0, "default", False)

    def record_predictions(self, round_index, records):
        self.records.append((round_index, records))


def adapter():
    return ResearchGymPolicyAdapter(case_id=CASE.case_id, metric="cmr_coco_c_i2t_r1", alpha=1.0, eta=1.0, seed=3,
                                    beta=1.0, z_clip=5.0, proposal_facts={"harness_sessions": 2})


def test_zero_prior_and_default_weights_reproduce_ldm_harness_selection():
    candidates, representations = pool()
    fixed = selector()
    fixed.fit(history())
    baseline = fixed.select(candidates, representations, count=2, round_idx=2)
    controller = ZeroPolicy()
    compiled = selector(policy_controller=controller, policy_adapter=adapter())
    compiled.fit(history())
    result = compiled.select(candidates, representations, count=2, round_idx=2)
    assert result.selected_candidate_ids == baseline.selected_candidate_ids
    for left, right in zip(baseline.metadata["distribution"], result.metadata["distribution"]):
        assert left["first_draw_probability"] == pytest.approx(right["first_draw_probability"])
        assert left["active_mean"] == pytest.approx(right["active_mean"])
    assert controller.records[0][0] == 2


def test_policy_inputs_keep_q0_and_acquisition_out_of_mean_features():
    candidates, representations = pool()
    fixed = selector()
    fixed.fit(history())
    baseline = fixed.select(candidates, representations, count=4, round_idx=2)
    pool_rows = [c for c in sorted(candidates, key=lambda c: c.candidate_id)]
    predictions = [next(p for p in baseline.predictions if p.candidate_id == c.candidate_id) for c in pool_rows]
    q0 = np.asarray([c.metadata[Q0_KEY]["occurrence_count"] for c in pool_rows], dtype=float)
    round_input = adapter().build_selection_round(
        round_index=2, history=tuple(history()), candidates=pool_rows, representations=representations,
        baseline_predictions=predictions, q0=q0 / q0.sum(), valid_proposal_occurrences=6,
        requested_evaluation_batch=1, gp=GP)
    assert round_input.query_features.shape == (4, len(FEATURE_NAMES))
    mean_context = round_input.execution_context["mean_context"]
    assert "q0" not in json.dumps(mean_context) and "acquisition" not in json.dumps(mean_context)
    assert "candidate_predictions" in round_input.execution_context["weight_context"]
    assert round_input.history_rounds == (0, 0, 1, 1)
    assert len(round_input.execution_context["validation_folds"]) == 1


class AuthoredFixtureExecutor:
    """Run only this test's static artifact through the actual restricted runner.

    Production uses DockerPolicyExecutor; this helper is not an isolation claim.
    """

    def execute(self, artifact_path, input_directory, output_directory):
        assert Path(artifact_path).read_text().startswith("# Authored static protocol-test fixture;")
        output = Path(output_directory)
        diagnostic = RESOURCE_ROOT / "policy_diagnostics.py"
        completed = subprocess.run([sys.executable, str(RUNNER), "execute", "--artifact", str(artifact_path),
                                    "--input", str(input_directory), "--output", str(output), "--diagnostics",
                                    str(diagnostic), "--diagnostics-sha256", file_sha256(diagnostic)],
                                   capture_output=True, text=True, timeout=30, check=False)
        response = json.loads(completed.stdout)
        if completed.returncode:
            raise PolicyExecutionError(tuple(HarnessSubmissionError(**row) for row in response["errors"]))
        metadata = json.loads((output / "result.json").read_text())
        with np.load(output / "arrays.npz", allow_pickle=False) as arrays:
            return PolicyExecutionResult(arrays["history_prior_mean"].copy(), arrays["query_prior_mean"].copy(),
                                         metadata["stage"], metadata["alpha"], metadata["eta"], metadata["prior_clip_count"])


def test_compiled_policy_repairs_in_session_and_changes_mean_and_weights(tmp_path):
    args = workflow.parse_args(["--mock", "--case", CASE.case_id, "--search-method", "ldm_harness_compiled",
                                "--harness-command-json", "[]"])
    args.provider_api_key = "synthetic-fixture-key"
    root = tmp_path / "policy_harness"
    client = create_client(args, root, "fixture-campaign", CASE, policy=True,
                           command=[sys.executable, "-u", str(FIXTURE), "repair"])
    client.start()
    try:
        controller = PolicyResearchController(client=client, adapter=adapter(), executor=AuthoredFixtureExecutor(),
                                              root=root)
        candidates, representations = pool()
        compiled = selector(policy_controller=controller, policy_adapter=controller.adapter)
        compiled.fit(history())
        result = compiled.select(candidates, representations, count=2, round_idx=2)
        policy = result.metadata["compiled_policy"]
        assert policy["source"] == "artifact" and policy["alpha"] == 0.5 and policy["eta"] == 1.5
        assert policy["prediction_mean_abs_change"] > 0
        again = compiled.select(candidates, representations, count=2, round_idx=2)
        assert again.selected_candidate_ids == result.selected_candidate_ids
    finally:
        client.close()
    decisions = [json.loads(line)["decision"] for line in (root / "sessions/policy_architect.jsonl").read_text().splitlines()]
    assert decisions == ["retry", "accept"]
    assert len((root / "fixture_requests.jsonl").read_text().splitlines()) == 1
