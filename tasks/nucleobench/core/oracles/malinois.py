"""Pinned Malinois loading and auditable model calls."""

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
from tasks.nucleobench.core.candidate import MutationContext
from tasks.nucleobench.core.cases import get_case
from tasks.nucleobench.core.constants import OFFICIAL_START_COUNT, UPSTREAM_COMMIT
from tasks.nucleobench.core.source import require_clean_revision


@dataclass(frozen=True)
class PreparedMalinoisCase:
    context: MutationContext
    model_artifact: Path
    model_init_args: dict[str, Any]


@dataclass(frozen=True)
class OfficialMalinoisAPI:
    model: RecordingSequenceModel
    run_loop: Callable[..., Any]
    parsed_args_type: type


class RecordingSequenceModel:
    """Preserve official model output while recording digest-only call traces."""

    def __init__(self, model: Callable[[list[str]], Any], runtime: CampaignRuntime):
        self.model = model
        self.runtime = runtime
        self.call_count = int(runtime.budget.counters.get("official_model_calls", 0))
        self.sequence_count = int(
            runtime.budget.counters.get("official_model_sequences", 0)
        )

    def __call__(self, sequences: Sequence[str]) -> Any:
        if isinstance(sequences, str):
            raise ValueError("official model input must be a sequence batch")
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


def load_prepared_malinois(
    prepared_dir: Path,
    *,
    start_index: int,
) -> PreparedMalinoisCase:
    case = get_case("malinois_k562")
    if (
        isinstance(start_index, bool)
        or not isinstance(start_index, int)
        or not 0 <= start_index < OFFICIAL_START_COUNT
    ):
        raise ValueError(
            f"start_index must be in [0, {OFFICIAL_START_COUNT - 1}]"
        )
    prepared_dir = Path(prepared_dir).resolve()
    manifest = _load_json(prepared_dir / "prepared_manifest.json")
    if manifest.get("case_id") != "malinois_k562":
        raise ValueError("prepared manifest is not for malinois_k562")
    source = manifest.get("benchmark_source")
    if not isinstance(source, Mapping) or source.get("revision") != UPSTREAM_COMMIT:
        raise ValueError("prepared manifest benchmark revision is not pinned")

    starts_info = _mapping(manifest, "start_set")
    starts = _load_json(prepared_dir / str(starts_info["path"]))
    if not isinstance(starts, list) or len(starts) != OFFICIAL_START_COUNT:
        raise ValueError(
            f"prepared start set must contain {OFFICIAL_START_COUNT} sequences"
        )
    if any(
        not isinstance(sequence, str)
        or len(sequence) != case.sequence_length
        or not set(sequence).issubset({"A", "C", "G", "T"})
        for sequence in starts
    ):
        raise ValueError("prepared start set contains an invalid sequence")
    start_digest = hashlib.sha256(
        json.dumps(starts, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if start_digest != starts_info.get("sha256"):
        raise ValueError("prepared start set digest mismatch")

    positions_info = _mapping(manifest, "editable_positions")
    positions = _load_json(prepared_dir / str(positions_info["path"]))
    if not isinstance(positions, list):
        raise ValueError("prepared editable positions must be a JSON array")

    model_info = _mapping(manifest, "model_artifact")
    model_artifact = Path(str(model_info["path"]))
    if not model_artifact.is_absolute():
        model_artifact = prepared_dir / model_artifact
    model_artifact = model_artifact.resolve()
    if _sha256_file(model_artifact) != model_info.get("sha256"):
        raise ValueError("prepared model artifact digest mismatch")
    model_init_args = dict(_mapping(manifest, "model_init_args"))
    context = MutationContext(
        case=case,
        start_set_digest=start_digest,
        start_index=start_index,
        start_sequence=starts[start_index],
        editable_positions=tuple(positions),
    )
    return PreparedMalinoisCase(context, model_artifact, model_init_args)


def load_official_malinois(
    source_dir: Path,
    prepared: PreparedMalinoisCase,
    runtime: CampaignRuntime,
) -> OfficialMalinoisAPI:
    source_dir = Path(source_dir).resolve()
    require_clean_revision(source_dir, UPSTREAM_COMMIT)
    source_text = str(source_dir)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    docker_entrypoint = importlib.import_module("docker_entrypoint")
    argparse_lib = importlib.import_module("nucleobench.common.argparse_lib")
    model_module = importlib.import_module("nucleobench.models.malinois.model_def")
    for module in (docker_entrypoint, argparse_lib, model_module):
        module_path = Path(str(module.__file__)).resolve()
        if not module_path.is_relative_to(source_dir):
            raise RuntimeError(
                "official module was imported outside the pinned source: "
                f"{module_path}"
            )
    model = model_module.Malinois(
        model_artifact=str(prepared.model_artifact),
        **prepared.model_init_args,
    )
    return OfficialMalinoisAPI(
        model=RecordingSequenceModel(model, runtime),
        run_loop=docker_entrypoint.run_loop,
        parsed_args_type=argparse_lib.ParsedArgs,
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
        raise ValueError(f"prepared manifest field {key!r} must be an object")
    return value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"prepared model artifact does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "OfficialMalinoisAPI",
    "PreparedMalinoisCase",
    "RecordingSequenceModel",
    "load_official_malinois",
    "load_prepared_malinois",
]
