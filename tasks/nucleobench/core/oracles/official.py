"""Digest-bound NucleoBench inputs and source-pinned oracle loading."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ldm_tts.engine.run_store import CampaignRuntime
from tasks.nucleobench.core.candidate import (
    MutationContext,
    normalize_editable_positions,
)
from tasks.nucleobench.core.cases import get_case
from tasks.nucleobench.core.constants import OFFICIAL_START_COUNT, UPSTREAM_COMMIT
from tasks.nucleobench.core.digests import canonical_json_sha256, file_digest
from tasks.nucleobench.core.source import require_clean_revision


@dataclass(frozen=True)
class PreparedCase:
    context: MutationContext
    model_artifact: Path
    model_init_args: dict[str, Any]


@dataclass(frozen=True)
class OfficialAPI:
    model: RecordingSequenceModel
    run_loop: Callable[..., Any]
    parsed_args_type: type


class RecordingSequenceModel:
    """Preserve official output while recording digest-only call traces."""

    def __init__(self, model: Callable[[list[str]], Any], runtime: CampaignRuntime):
        self.model = model
        self.runtime = runtime
        self.call_count = int(runtime.budget.counters.get("official_model_calls", 0))
        self.sequence_count = int(
            runtime.budget.counters.get("official_model_sequences", 0)
        )

    def __call__(self, sequences: Sequence[str]) -> Any:
        if isinstance(sequences, str):
            raise TypeError("official model input must be a sequence batch")
        batch = list(sequences)
        if not batch or any(not isinstance(sequence, str) for sequence in batch):
            raise ValueError("official model input must contain DNA strings")
        sequence_sha256 = [_sha256_text(item) for item in batch]
        self.runtime.consume_many(
            {
                "official_model_calls": 1,
                "official_model_sequences": len(batch),
            }
        )
        started = time.perf_counter()
        self.call_count += 1
        self.sequence_count += len(batch)
        try:
            raw = self.model(batch)
            energies = _finite_energies(raw, len(batch))
        except Exception as exc:
            self.runtime.record(
                "official_model_called",
                {
                    "status": "failed",
                    "call_index": self.call_count,
                    "batch_size": len(batch),
                    "cumulative_sequence_count": self.sequence_count,
                    "sequence_sha256": sequence_sha256,
                    "elapsed_seconds": time.perf_counter() - started,
                    "error": str(exc),
                },
            )
            raise
        self.runtime.record(
            "official_model_called",
            {
                "status": "succeeded",
                "call_index": self.call_count,
                "batch_size": len(batch),
                "cumulative_sequence_count": self.sequence_count,
                "sequence_sha256": sequence_sha256,
                "energies": list(energies),
                "utilities": [-energy for energy in energies],
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        return raw


def load_prepared_case(
    prepared_dir: Path,
    *,
    case_id: str,
    start_index: int,
) -> PreparedCase:
    case = get_case(case_id)
    if (
        isinstance(start_index, bool)
        or not isinstance(start_index, int)
        or not 0 <= start_index < OFFICIAL_START_COUNT
    ):
        raise ValueError(f"start_index must be in [0, {OFFICIAL_START_COUNT - 1}]")

    prepared_dir = Path(prepared_dir).resolve()
    manifest = _load_json(prepared_dir / "prepared_manifest.json")
    if manifest.get("case_id") != case_id:
        raise ValueError(f"prepared manifest is not for {case_id}")
    source = manifest.get("benchmark_source")
    if not isinstance(source, Mapping) or source.get("revision") != UPSTREAM_COMMIT:
        raise ValueError("prepared manifest benchmark revision is not pinned")

    starts_info = _mapping(manifest, "start_set")
    starts_path = prepared_dir / str(starts_info["path"])
    _require_file_digest(starts_path, starts_info)
    starts = _load_json(starts_path)
    _validate_starts(starts, case.sequence_length)
    start_digest = canonical_json_sha256(starts)
    if start_digest != starts_info.get("sha256"):
        raise ValueError("prepared start set digest mismatch")

    positions_info = _mapping(manifest, "editable_positions")
    positions_path = prepared_dir / str(positions_info["path"])
    _require_file_digest(positions_path, positions_info)
    positions_payload = _load_json(positions_path)
    if (
        "sha256" in positions_info
        and canonical_json_sha256(positions_payload) != positions_info["sha256"]
    ):
        raise ValueError("prepared editable-position digest mismatch")
    positions = _positions_for_start(
        positions_payload,
        start_index=start_index,
        sequence_length=case.sequence_length,
        expected_count=case.editable_position_count,
    )

    model_info = _mapping(manifest, "model_artifact")
    model_artifact = Path(str(model_info["path"]))
    if not model_artifact.is_absolute():
        model_artifact = prepared_dir / model_artifact
    model_artifact = model_artifact.resolve()
    _require_file_digest(model_artifact, model_info)

    return PreparedCase(
        context=MutationContext(
            case=case,
            start_set_digest=start_digest,
            start_index=start_index,
            start_sequence=starts[start_index],
            editable_positions=tuple(positions),
        ),
        model_artifact=model_artifact,
        model_init_args=dict(_mapping(manifest, "model_init_args")),
    )


def load_official_case(
    source_dir: Path,
    prepared: PreparedCase,
    runtime: CampaignRuntime,
) -> OfficialAPI:
    source_dir = Path(source_dir).resolve()
    require_clean_revision(source_dir, UPSTREAM_COMMIT)
    source_text = str(source_dir)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

    docker_entrypoint = _official_module(source_dir, "docker_entrypoint")
    argparse_lib = _official_module(source_dir, "nucleobench.common.argparse_lib")
    case = prepared.context.case
    if case.model_family == "malinois":
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            raise RuntimeError(
                "the pinned Malinois wrapper requires CPU execution; set "
                "CUDA_VISIBLE_DEVICES to an empty value"
            )
        module = _official_module(source_dir, "nucleobench.models.malinois.model_def")
        model = module.Malinois(
            model_artifact=str(prepared.model_artifact),
            **prepared.model_init_args,
        )
    elif case.model_family == "bpnet":
        model_module = _official_module(
            source_dir, "nucleobench.models.bpnet.model_def"
        )
        load_module = _official_module(
            source_dir, "nucleobench.models.bpnet.load_model"
        )
        torch = importlib.import_module("torch")
        raw_model = torch.load(
            prepared.model_artifact,
            weights_only=False,
            map_location=torch.device("cpu"),
        )
        model = model_module.BPNet(
            override_model=load_module.CountWrapper(
                load_module.ControlWrapper(raw_model)
            ),
            **prepared.model_init_args,
        )
    elif case.model_family == "rinalmo":
        module = _official_module(
            source_dir, "nucleobench.models.rna.rinalmo_mrl.model_def"
        )
        model = module.RinalmoMRL(
            override_weights_local_path=str(prepared.model_artifact)
        )
    elif case.model_family == "enformer":
        module = _official_module(
            source_dir, "nucleobench.models.grelu.enformer.model_def"
        )
        torch = importlib.import_module("torch")
        lightning = importlib.import_module("grelu.lightning")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = lightning.LightningModel.load_from_checkpoint(
            prepared.model_artifact,
            map_location=device,
        )
        model = module.Enformer(
            override_model=checkpoint,
            **prepared.model_init_args,
        )
    else:  # pragma: no cover - catalog validation makes this unreachable.
        raise ValueError(f"unsupported model family: {case.model_family}")

    return OfficialAPI(
        model=RecordingSequenceModel(model, runtime),
        run_loop=docker_entrypoint.run_loop,
        parsed_args_type=argparse_lib.ParsedArgs,
    )


def _official_module(source_dir: Path, name: str) -> Any:
    module = importlib.import_module(name)
    module_path = Path(str(module.__file__)).resolve()
    if not module_path.is_relative_to(source_dir):
        raise RuntimeError(
            f"official module {name!r} was imported outside the pinned source: "
            f"{module_path}"
        )
    return module


def _validate_starts(payload: Any, sequence_length: int) -> None:
    if not isinstance(payload, list) or len(payload) != OFFICIAL_START_COUNT:
        raise ValueError(
            f"prepared start set must contain {OFFICIAL_START_COUNT} sequences"
        )
    if any(
        not isinstance(sequence, str)
        or len(sequence) != sequence_length
        or not set(sequence).issubset({"A", "C", "G", "T"})
        for sequence in payload
    ):
        raise ValueError("prepared start set contains an invalid sequence")
    if len(set(payload)) != OFFICIAL_START_COUNT:
        raise ValueError("prepared start sequences must be unique")


def _positions_for_start(
    payload: Any,
    *,
    start_index: int,
    sequence_length: int,
    expected_count: int,
) -> list[int]:
    if not isinstance(payload, list):
        raise TypeError("prepared editable positions must be a JSON array")
    if payload and isinstance(payload[0], list):
        if len(payload) != OFFICIAL_START_COUNT:
            raise ValueError(
                f"per-start editable positions must contain {OFFICIAL_START_COUNT} masks"
            )
        positions = payload[start_index]
    else:
        positions = payload
    return list(
        normalize_editable_positions(
            positions,
            sequence_length=sequence_length,
            expected_count=expected_count,
        )
    )


def _finite_energies(raw: Any, expected: int) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("official model output must be a numeric batch") from exc
    if len(values) != expected:
        raise ValueError("official model output count does not match the input batch")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("official model output must contain finite energies")
    return values


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"required prepared file does not exist: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"prepared file is not valid JSON: {path}") from exc


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"prepared manifest field {key!r} must be an object")
    return value


def _require_file_digest(path: Path, info: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise ValueError(f"prepared artifact does not exist: {path}")
    if file_digest(path) != info.get("file_sha256", info.get("sha256")):
        raise ValueError(f"prepared artifact digest mismatch: {path}")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


__all__ = [
    "OfficialAPI",
    "PreparedCase",
    "RecordingSequenceModel",
    "load_official_case",
    "load_prepared_case",
]
