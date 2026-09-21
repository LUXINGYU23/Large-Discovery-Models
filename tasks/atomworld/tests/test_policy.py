"""Public-only compiled policy tested with the actual artifact validator/runner."""

import json
from types import SimpleNamespace

import numpy as np

from harnesses.pi.policy_runner import execute_artifact
from ldm_tts.harness import (
    HarnessSubmissionRequest,
    HarnessSubmittedArtifact,
    HarnessTurnResult,
    PolicyExecutionResult,
    file_sha256,
)
from tasks.atomworld.core.geometry import execute_operations
from tasks.atomworld.core.optimization_policy import FEATURE_NAMES, public_policy_input
from tasks.atomworld.core.workflow import parse_args, run
from tasks.atomworld.tests.test_harness import ResearchClient, fixture


class KnownArtifactExecutor:
    """Run only the test's fixed source locally; production uses Docker isolation."""

    def execute(self, artifact_path, input_directory, output_directory):
        result = execute_artifact(artifact_path, input_directory, output_directory)
        with np.load(output_directory / "arrays.npz", allow_pickle=False) as arrays:
            return PolicyExecutionResult(
                arrays["history_prior_mean"],
                arrays["query_prior_mean"],
                result["stage"],
                result["alpha"],
                result["eta"],
                result["prior_clip_count"],
            )


def test_public_geometric_prior_features_compare_input_only():
    sample = fixture()["public"][0]
    cif = execute_operations(
        sample["input_cif"], [{"op": "move", "index": 0, "d_pos": [1, 0, 0]}]
    )
    result = public_policy_input(
        0,
        sample,
        [
            SimpleNamespace(
                submission={
                    "generated_output": "<cif>" + cif + "</cif>",
                    "rationale": "public motion",
                }
            )
        ],
        [],
    )
    features = dict(zip(FEATURE_NAMES, result.query_features[0], strict=True))
    assert features["geometry_features_available"] == 1
    assert features["ase_atom_count_delta"] == 0
    assert features["cell_volume_ratio"] == 1
    assert np.isclose(features["same_order_max_displacement_angstrom"], 1)
    assert result.history_utilities.size == 0


class PolicyClient:
    expects_measured = False
    source = """import numpy as np
POLICY_API_VERSION = 1
CAPABILITIES = {"prior_mean": 1}
def compute_prior_mean(history_features, history_utilities, query_features, context):
    return query_features[:, 3] / 1000000.0
"""
    def __init__(self, args, root, sample, *, policy=False):
        self.root, self.turns, self.closed = root, [], False

    def close(self):
        self.closed = True

    def run_turn(self, turns, *, submission_validator, recovery_timeout_seconds=0):
        assert len(turns) == 1
        turn = turns[0]
        self.turns.append(turn)
        assert bool(json.loads(turn.message)["new_measured_observations"]) == self.expects_measured
        # Policy syntax/schema rejection returns to the same persistent session.
        bad = HarnessSubmissionRequest(
            turn.profile_id, turn.turn_id, 1, {"action": "guess"}
        )
        assert submission_validator(bad).decision == "retry"
        path = self.root / "snapshots" / f"{turn.round_index}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.source)
        artifact = HarnessSubmittedArtifact(
            "/artifact_path",
            "optimization_policy.py",
            str(path.relative_to(self.root)),
            file_sha256(path),
            path.stat().st_size,
        )
        payload = {"action": "replace", "artifact_path": "optimization_policy.py"}
        request = HarnessSubmissionRequest(
            turn.profile_id, turn.turn_id, 2, payload, (artifact,)
        )
        assert submission_validator(request).decision == "accept"
        return (
            HarnessTurnResult(
                turn.profile_id,
                "persistent_policy_session",
                turn.turn_id,
                turn.round_index,
                turn.history_from_seq,
                turn.history_to_seq,
                turn.history_digest,
                turn.input_digest,
                False,
                "accepted",
                turn.turn_id,
                request.digest,
                payload,
                (artifact,),
                (),
                {
                    "providerCalls": 0,
                    "toolCalls": {},
                    "validationSubmissions": 2,
                    "artifactBytes": artifact.size_bytes,
                },
                {},
                {},
            ),
        )


def test_independent_compiled_agent_executes_public_artifact_and_resumes(tmp_path):
    clients = []

    def factory(*args, **kwargs):
        client = (PolicyClient if kwargs.get("policy") else ResearchClient)(
            *args, **kwargs
        )
        clients.append(client)
        return client

    args = parse_args(
        [
            "--mock",
            "--search-method",
            "harness_public_audit",
            "--iterations",
            "2",
            "--out-dir",
            str(tmp_path / "run"),
        ]
    )
    result = run(
        args, harness_client_factory=factory, policy_executor=KnownArtifactExecutor()
    )
    assert len(clients) == 2 and all(client.closed for client in clients)
    policy = next(client for client in clients if isinstance(client, PolicyClient))
    assert len(policy.turns) == 2
    assert result.runtime.budget.counters["policy_harness_turns"] == 2
    for path in (policy.root / "rounds").glob("round_*/arrays.npz"):
        with np.load(path, allow_pickle=False) as arrays:
            assert len(arrays["history_utilities"]) == 0
    for path in (result.runtime.run_dir / "attempts").glob("*.json"):
        selection = json.loads(path.read_text())["selection"]
        assert selection["source"] == "artifact"
        assert selection["objective_measurements_visible"] is False
    args.resume, args.out_dir = True, result.runtime.run_dir
    resumed = run(
        args,
        harness_client_factory=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("unexpected call")
        ),
    )
    assert (
        resumed.projected["extended_final_accuracy"]
        == result.projected["extended_final_accuracy"]
    )
