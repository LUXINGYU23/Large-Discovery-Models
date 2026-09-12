"""Frozen research predictions must agree with selection without measuring drafts."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from ldm_tts.contracts import Candidate, EvaluationResult, Observation, RawProposal
from ldm_tts.engine.expansion import ExpansionRequest
from ldm_tts.optimization import BOObservation
from tasks.nucleobench.core.candidate import NucleoBenchCandidateDomain
from tasks.nucleobench.core.hamming_gp import HammingGPUCBConfig, HammingGPUCBSelector, NucleotideHammingEncoder
from tasks.nucleobench.core.harness import write_harness_sequence_context
from tasks.nucleobench.core.mock import MOCK_CONTEXT
from tasks.nucleobench.core.surrogate_query import write_surrogate_snapshot

TASK = Path(__file__).parents[1]


def observations():
    domain = NucleoBenchCandidateDomain(MOCK_CONTEXT)
    result = []
    for index, (position, base) in enumerate(((0, "C"), (0, "G"), (2, "C"), (2, "G"))):
        candidate = domain.admit(RawProposal({"mutations": [{"position": position, "base": base}]}, "test"))
        assert isinstance(candidate, Candidate)
        result.append(Observation(candidate, EvaluationResult(
            candidate.candidate_id, "succeeded", metrics={"utility": float(index * index - 4)},
        ), round_idx=index))
    return tuple(result)


@pytest.mark.parametrize("history_count", [1, 4])
def test_frozen_worker_matches_selection_and_is_batch_invariant(tmp_path, history_count):
    history = observations()[:history_count]
    request = ExpansionRequest(4, 128, history)
    config = HammingGPUCBConfig()
    info = write_surrogate_snapshot(request, MOCK_CONTEXT, config, tmp_path)
    assert write_surrogate_snapshot(request, MOCK_CONTEXT, config, tmp_path) == info
    changed = (replace(history[0], evaluation=replace(history[0].evaluation, metrics={"utility": 999.0})), *history[1:])
    with pytest.raises(ValueError, match="cannot change"):
        write_surrogate_snapshot(ExpansionRequest(4, 128, changed), MOCK_CONTEXT, config, tmp_path)
    encoder = NucleotideHammingEncoder(MOCK_CONTEXT)
    candidates = [item.candidate for item in observations()]
    features = {item.candidate_id: encoder.encode(item) for item in candidates}
    selector = HammingGPUCBSelector(objective_name="utility", feature_dimension=encoder.dimension,
                                    feature_version=encoder.version, config=config)
    selector.fit(tuple(BOObservation.from_observation(item, objective_names=("utility",),
                       feature=encoder.encode(item.candidate)) for item in history))
    expected = selector.select(candidates, features).predictions
    env = {**os.environ, "PYTHONPATH": str(TASK / "core")}

    def query(items):
        result = subprocess.run([sys.executable, str(TASK / "resources/harness/tools/query_posterior.py")],
            input=json.dumps({"snapshot_file": str(tmp_path / "surrogate/round_0004.json"),
                              "snapshot_id": info["snapshot_id"], "codes": items}),
            env=env, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)

    codes = [features[item.candidate_id].values for item in candidates]
    actual = query(codes)
    for row, prediction in zip(actual, expected, strict=True):
        np.testing.assert_allclose([row["mean"], row["std"], row["acquisition_score"]],
            [prediction.scalar_mean, prediction.scalar_std, prediction.acquisition_score], rtol=1e-10, atol=1e-10)
    single = query(codes[1:2])[0]
    np.testing.assert_allclose(list(single.values()), list(actual[1].values()), rtol=1e-10, atol=1e-10)
    assert len(request.observations) == history_count


def test_extension_batch_files_cache_and_snapshot_boundaries(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the Pi extension")
    version = subprocess.run([node, "--version"], check=True, capture_output=True, text=True).stdout
    if int(version.strip().lstrip("v").split(".")[0]) < 24:
        pytest.skip("The Pi task extension requires Node 24 or newer")
    context_path = tmp_path / "context.json"
    write_harness_sequence_context(MOCK_CONTEXT, context_path)
    history_path = tmp_path / "history.json"
    history_path.write_text('{"observations": []}')
    before_history = history_path.read_bytes()
    info = write_surrogate_snapshot(ExpansionRequest(4, 128, observations()), MOCK_CONTEXT, HammingGPUCBConfig(), tmp_path)
    script = """
import assert from 'node:assert/strict';
import {readFileSync, writeFileSync} from 'node:fs';
const {default: load} = await import(process.argv[1]);
const ctx = {cwd: process.env.TEST_WORKSPACE};
function session() {
  let tool; load({registerTool: value => { tool = value; }});
  return async args => (await tool.execute('test', args, undefined, undefined, ctx)).details;
}
const call = session();
const snapshot_id = process.env.TEST_SNAPSHOT;
const candidate = {mutations: [{position: 0, base: 'C'}], rationale: 'Annotated file input'};
const candidates = [candidate, candidate, {mutations: []}, {mutations: [{position: -1, base: 'G'}]}];
writeFileSync(ctx.cwd + '/candidates.json', JSON.stringify({candidates}));
const first = await call({snapshot_id, artifact_path: '/workspace/candidates.json'});
assert.equal(first.candidate_count, 4);
assert.equal(first.invalid_count, 1);
assert.equal(first.predicted_unique_candidates, 2);
assert.equal(first.cache_hits, 0);
assert.deepEqual(first.predictions[0].objectives, first.predictions[1].objectives);
assert.equal(first.predictions[3].error.code, 'invalid_mutations');
const exported = JSON.parse(readFileSync(first.guest_file.path.replace('/workspace', ctx.cwd)));
assert.deepEqual(exported.predictions, first.predictions);
const second = await call({snapshot_id, candidates: [candidate]});
assert.equal(second.predicted_unique_candidates, 0);
assert.equal(second.cache_hits, 1);
assert.deepEqual(second.predictions[0].objectives, first.predictions[0].objectives);
assert.equal((await session()({snapshot_id, candidates: [candidate]})).cache_hits, 0);
await assert.rejects(call({snapshot_id: 'stale', candidates}), /stale_snapshot/);
await assert.rejects(call({snapshot_id, candidates, artifact_path: 'candidates.json'}), /exactly one/);
await assert.rejects(call({snapshot_id, artifact_path: '../outside.json'}), /inside this session/);
"""
    (tmp_path.parent / "outside.json").write_text('{"candidates": []}')
    subprocess.run([node, "--input-type=module", "-e", script,
                    (TASK / "resources/harness/tools/query_surrogate.mjs").resolve().as_uri()],
        env={**os.environ, "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
             "PYTHONPATH": str(TASK / "core"), "TEST_WORKSPACE": str(tmp_path),
             "TEST_SNAPSHOT": info["snapshot_id"], "LDM_NUCLEOBENCH_CONTEXT": str(context_path),
             "LDM_NUCLEOBENCH_HISTORY": str(history_path), "LDM_NUCLEOBENCH_SURROGATE": str(tmp_path / "surrogate")},
        capture_output=True, text=True, check=True)
    assert history_path.read_bytes() == before_history
