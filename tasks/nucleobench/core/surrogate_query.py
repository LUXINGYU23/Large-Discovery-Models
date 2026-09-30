"""Publish a task-owned baseline posterior before candidate research begins."""

from dataclasses import asdict
import json
from pathlib import Path

from ldm_tts.engine.expansion import ExpansionRequest
from ldm_tts.engine.run_store import atomic_json_write
from ldm_tts.optimization import BOObservation
from tasks.nucleobench.core.candidate import MutationContext
from tasks.nucleobench.core.digests import canonical_json_sha256, file_digest
from tasks.nucleobench.core.hamming_gp import HammingGPUCBConfig, HammingGPUCBSelector, NucleotideHammingEncoder


def write_surrogate_snapshot(
    request: ExpansionRequest, context: MutationContext,
    config: HammingGPUCBConfig, artifact_root: Path,
) -> dict[str, object]:
    encoder = NucleotideHammingEncoder(context)
    history = tuple(
        BOObservation.from_observation(
            observation, objective_names=("utility",),
            feature=observation.surrogate or encoder.encode(observation.candidate),
        )
        for observation in request.observations if observation.evaluation.succeeded
    )
    selector = HammingGPUCBSelector(
        objective_name="utility", feature_dimension=encoder.dimension,
        feature_version=encoder.version, config=config,
    )
    selector.fit(history)
    snapshot = {
        "round_index": request.round_idx, "model_role": "baseline",
        "gp_config": asdict(config), "posterior": selector.posterior_snapshot(),
        "implementation_sha256": {
            name: file_digest(Path(__file__).with_name(name))
            for name in ("hamming_gp.py", "hamming_posterior.py", "surrogate_query.py")
        },
    }
    digest = canonical_json_sha256(snapshot)
    directory = artifact_root / "surrogate"
    path = directory / f"round_{request.round_idx:04d}.json"
    document = {"snapshot_id": digest, **snapshot}
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != json.loads(json.dumps(document)):
            raise ValueError("A frozen surrogate snapshot cannot change during round recovery")
    else:
        atomic_json_write(path, document)
    atomic_json_write(directory / "current.json", {
        "snapshot_id": digest, "file": path.name, "round_index": request.round_idx,
    })
    return {
        "tool": "query_surrogate", "snapshot_id": digest, "model_role": "baseline",
        "round_index": request.round_idx, "history_size": len(history),
        "working_set_size": snapshot["posterior"]["working_set_size"],
        "fit_status": snapshot["posterior"]["fit_status"],
        "acquisition": "ucb", "beta": config.beta,
        "std_kind": "latent", "objective_direction": "maximize",
    }
