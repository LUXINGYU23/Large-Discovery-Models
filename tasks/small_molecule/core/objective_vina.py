"""AutoDock Vina scoring for small-molecule campaigns."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

from tasks.small_molecule.core.docking import (
    DockingResult,
    ExtractedCompound,
    ReceptorConfig,
    canonicalize_smiles,
    command_error_tail,
    hash_text,
    model_to_plain_dict,
    parse_vina_score_from_pose,
    parse_vina_score_from_text,
    prepare_ligand,
    prepare_receptor,
    require_production_receptor,
    run_logged_command,
    work_dir_for_receptor,
)


LOGGER = logging.getLogger(__name__)
VINA_PREFLIGHT_TIMEOUT_SECONDS = 10


__all__ = [
    "VinaScorerConfig",
    "VinaScorer",
]


# ---------------------------------------------------------------------------
# Vina binary resolution
# ---------------------------------------------------------------------------


def _resolve_vina_bin(explicit: Optional[str]) -> str:
    """Resolve the AutoDock Vina executable path.

    Resolution order:

    1. ``explicit`` (validated, resolved to absolute).
    2. ``$VINA_BIN`` environment variable.
    3. ``shutil.which("vina")`` (system ``PATH``).
    4. :class:`FileNotFoundError`.
    """
    if explicit:
        path: Optional[Path] = Path(explicit).expanduser().resolve()
    elif os.environ.get("VINA_BIN"):
        path = Path(os.environ["VINA_BIN"]).expanduser().resolve()
    else:
        which = shutil.which("vina")
        path = Path(which).resolve() if which else None

    if path is None or not path.exists() or not os.access(path, os.X_OK):
        raise FileNotFoundError(
            "AutoDock Vina executable not found. Pass vina_bin, set $VINA_BIN, "
            "or add 'vina' to PATH. "
            f"(resolved: {path or '<none>'})"
        )
    _validate_vina_executable(path)
    return str(path)


def _validate_vina_executable(path: Path) -> None:
    try:
        proc = subprocess.run(
            [str(path), "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=VINA_PREFLIGHT_TIMEOUT_SECONDS,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(
            f"AutoDock Vina executable cannot be run: {path} ({exc})"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"AutoDock Vina executable preflight timed out after "
            f"{VINA_PREFLIGHT_TIMEOUT_SECONDS}s: {path}"
        ) from exc
    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            f"AutoDock Vina executable cannot be run: {path} "
            f"(exit={proc.returncode}; {message[-500:]})"
        )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class VinaScorerConfig:
    """Configuration for a :class:`VinaScorer` instance.

    ``cache_dir`` is the single source of truth for all interim files
    (receptor PDBQT + sidecar JSON, ligand PDBQT, docking result
    cache, pose files). The directory is created on instantiation
    and never cleaned up automatically — the user manages it with
    ``rm -rf cache_dir`` or equivalent.

    ``vina_bin=None`` defers to ``$VINA_BIN`` then ``PATH``; pass an
    explicit path (e.g. ``"../bin/vina"``) to bypass both. ``vina_bin``
    is resolved once at ``VinaScorer.__init__`` time (for an explicit
    path) or lazily on the first call (for env / PATH fallback).
    """

    pdb_id: str = "8UN5"
    chain_id: str = "A"
    ligand_resname: Optional[str] = None
    cache_dir: Path = Path("runs/docking")
    allow_zero_charge_fallback: bool = False
    allow_debug_receptor: bool = False

    vina_bin: Optional[str] = None

    exhaustiveness: int = 4
    n_poses: int = 3
    seed: int = 42
    max_workers: int = 1
    use_cache: bool = True


# ---------------------------------------------------------------------------
# Local pass-through docking primitives
#
# These replicate the dispatch shell of ``extract_and_dock.dock_one`` and
# ``extract_and_dock.dock_batch`` with one difference: ``vina_bin`` is a
# required first-class argument used directly when constructing the
# ``vina`` subprocess command (no ``find_local_tool`` lookup, no
# ``$VINA_BIN`` env-var indirection).
# ---------------------------------------------------------------------------


def vina_dock_one(
    smiles: str,
    receptor: ReceptorConfig,
    *,
    vina_bin: str,
    exhaustiveness: int = 16,
    n_poses: int = 5,
    seed: int = 42,
    compound_id: str = "",
    allow_debug_receptor: bool = False,
    timeout: int = 3600,
) -> DockingResult:
    """Dock a single SMILES with AutoDock Vina. Local replica of
    ``extract_and_dock.dock_one`` with ``vina_bin`` as a real argument.
    """
    require_production_receptor(receptor, allow_debug_receptor=allow_debug_receptor)
    canonical = canonicalize_smiles(smiles)
    result_id = compound_id or hash_text(canonical, 12)
    work_path = work_dir_for_receptor(receptor)
    if not canonical:
        return DockingResult(
            compound_id=result_id,
            canonical_smiles="",
            score=None,
            pose_ref=None,
            status="prep_failed",
            message="Empty SMILES.",
        )
    ligand_pdbqt = prepare_ligand(canonical, work_dir=work_path, seed=seed)
    if not ligand_pdbqt:
        return DockingResult(
            compound_id=result_id,
            canonical_smiles=canonical,
            score=None,
            pose_ref=None,
            status="prep_failed",
            message="Ligand PDBQT preparation failed.",
        )

    pose_dir = work_path / "poses"
    pose_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hash_text(
        json.dumps(
            {
                "smiles": canonical,
                "receptor": str(Path(receptor.receptor_pdbqt).resolve()),
                "box_center": receptor.box_center,
                "box_size": receptor.box_size,
                "prep_method": receptor.prep_method,
                "exhaustiveness": exhaustiveness,
                "n_poses": n_poses,
                "seed": seed,
            },
            sort_keys=True,
        ),
        20,
    )
    pose_path = pose_dir / f"{result_id}_{cache_key}.pdbqt"
    log_path = pose_dir / f"{result_id}_{cache_key}.log"
    cmd = [
        vina_bin,
        "--receptor",
        receptor.receptor_pdbqt,
        "--ligand",
        ligand_pdbqt,
        "--center_x",
        f"{receptor.box_center[0]:.3f}",
        "--center_y",
        f"{receptor.box_center[1]:.3f}",
        "--center_z",
        f"{receptor.box_center[2]:.3f}",
        "--size_x",
        f"{receptor.box_size[0]:.3f}",
        "--size_y",
        f"{receptor.box_size[1]:.3f}",
        "--size_z",
        f"{receptor.box_size[2]:.3f}",
        "--exhaustiveness",
        str(exhaustiveness),
        "--num_modes",
        str(n_poses),
        "--seed",
        str(seed),
        "--out",
        str(pose_path),
    ]
    try:
        proc = run_logged_command(cmd, log_path, timeout=timeout)
    except Exception as exc:
        return DockingResult(
            compound_id=result_id,
            canonical_smiles=canonical,
            score=None,
            pose_ref=str(log_path),
            status="dock_failed",
            message=f"Vina execution failed: {exc}",
        )
    if proc.returncode != 0:
        return DockingResult(
            compound_id=result_id,
            canonical_smiles=canonical,
            score=None,
            pose_ref=str(log_path),
            status="dock_failed",
            message=command_error_tail(proc),
        )
    score = parse_vina_score_from_text(proc.stdout or "")
    if score is None:
        score = parse_vina_score_from_pose(pose_path)
    if score is None:
        return DockingResult(
            compound_id=result_id,
            canonical_smiles=canonical,
            score=None,
            pose_ref=str(pose_path) if pose_path.exists() else str(log_path),
            status="dock_failed",
            message="Vina finished but no score could be parsed.",
        )
    return DockingResult(
        compound_id=result_id,
        canonical_smiles=canonical,
        score=score,
        pose_ref=str(pose_path),
        status="ok",
    )


def vina_dock_batch(
    compounds: list[ExtractedCompound],
    receptor: ReceptorConfig,
    *,
    vina_bin: str,
    exhaustiveness: int = 16,
    n_poses: int = 5,
    seed: int = 42,
    use_cache: bool = True,
    allow_debug_receptor: bool = False,
    max_workers: int = 1,
    timeout: int = 3600,
) -> list[DockingResult]:
    """Dock a batch of compounds. Local replica of
    ``extract_and_dock.dock_batch`` with ``vina_bin`` as a real argument.

    Per-compound cache key matches ``extract_and_dock.dock_batch``
    (canonical SMILES + receptor + box + prep_method + docking params)
    so cache files written by either implementation are interoperable.
    """
    require_production_receptor(receptor, allow_debug_receptor=allow_debug_receptor)
    work_path = work_dir_for_receptor(receptor)
    cache_dir = work_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def dock_or_read_cache(compound: ExtractedCompound) -> DockingResult:
        canonical = canonicalize_smiles(compound.full_smiles or "")
        if not canonical:
            return DockingResult(
                compound_id=compound.compound_id,
                canonical_smiles="",
                score=None,
                pose_ref=None,
                status="prep_failed",
                message="Compound has no SMILES.",
            )
        cache_key = hash_text(
            json.dumps(
                {
                    "engine": "vina",
                    "smiles": canonical,
                    "receptor": str(Path(receptor.receptor_pdbqt).resolve()),
                    "box_center": receptor.box_center,
                    "box_size": receptor.box_size,
                    "prep_method": receptor.prep_method,
                    "exhaustiveness": exhaustiveness,
                    "n_poses": n_poses,
                    "seed": seed,
                },
                sort_keys=True,
            ),
            20,
        )
        cache_path = cache_dir / f"docking_{cache_key}.json"
        if use_cache and cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                result = DockingResult(**cached)
                if result.status == "ok" and result.pose_ref and Path(result.pose_ref).exists():
                    result.compound_id = compound.compound_id
                    result.cached = True
                    return result
            except Exception:
                pass
        result = vina_dock_one(
            canonical,
            receptor,
            vina_bin=vina_bin,
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
            seed=seed,
            compound_id=compound.compound_id,
            allow_debug_receptor=allow_debug_receptor,
            timeout=timeout,
        )
        if result.status == "ok":
            cache_path.write_text(
                json.dumps(model_to_plain_dict(result), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return result

    if max_workers <= 1 or len(compounds) <= 1:
        return [dock_or_read_cache(compound) for compound in compounds]

    results: list[Optional[DockingResult]] = [None] * len(compounds)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(dock_or_read_cache, compound): (index, compound)
            for index, compound in enumerate(compounds)
        }
        for future in as_completed(futures):
            index, compound = futures[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                results[index] = DockingResult(
                    compound_id=compound.compound_id,
                    canonical_smiles=canonicalize_smiles(compound.full_smiles or ""),
                    score=None,
                    pose_ref=None,
                    status="dock_failed",
                    message=f"Docking worker failed: {exc}",
                )
    return [result for result in results if result is not None]


# ---------------------------------------------------------------------------
# VinaScorer
# ---------------------------------------------------------------------------


_RECEPTOR_META_FIELDS = (
    "pdb_id",
    "chain_id",
    "ligand_resname",
    "ligand_resseq",
    "ligand_chain",
    "box_center",
    "box_size",
    "prep_method",
    "prep_command",
    "prep_log",
    "warnings",
)


def _receptor_cache_key(pdb_id: str, chain_id: Optional[str], ligand_resname: Optional[str]) -> str:
    """Stable cache key for a receptor config.

    Matches the ``suffix`` used by ``extract_and_dock.prepare_receptor``
    (uppercased pdb_id, chain or ``"all"``, resname or ``"auto"``).
    The cache works best when ``ligand_resname`` is supplied; when
    it's ``None`` the key uses ``"auto"`` and the lookup may miss
    if ``prepare_receptor`` auto-detects a different resname.
    """
    return f"{pdb_id.upper()}_{chain_id or 'all'}_{ligand_resname or 'auto'}"


def _receptor_cache_key_from_path(receptor_pdbqt: str) -> str:
    """Extract the cache key from an actual receptor PDBQT path.

    ``prepare_receptor`` writes the file as ``<key>_receptor.pdbqt``;
    we strip the trailing ``_receptor`` to recover ``<key>``. Used
    after a fresh ``prepare_receptor`` call to write the sidecar at
    the path matching the actual file (even when our config-derived
    key didn't match).
    """
    stem = Path(receptor_pdbqt).stem
    suffix = "_receptor"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def _receptor_pdbqt_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / "receptors" / f"{key}_receptor.pdbqt"


def _receptor_meta_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / "receptors" / f"{key}_receptor.meta.json"


def _read_receptor_meta(path: Path) -> Optional[dict[str, Any]]:
    """Read and validate the receptor sidecar JSON. Returns None on any failure."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    if not all(field in data for field in _RECEPTOR_META_FIELDS):
        return None
    return data


def _write_receptor_meta(path: Path, receptor: ReceptorConfig) -> None:
    """Atomically write the receptor sidecar JSON next to the receptor file.

    Uses the standard write-tmp-then-rename pattern: a crash mid-write
    leaves the old sidecar intact (or no sidecar if first write) and an
    orphan ``.tmp`` that the next write overwrites.
    """
    payload = {field: getattr(receptor, field) for field in _RECEPTOR_META_FIELDS}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class VinaScorer:
    """Callable Vina scoring interface.

    ``scorer(smiles_list)`` returns a ``list[float]`` of the same length:
    the i-th element is the Vina score (kcal/mol) of the i-th SMILES, or
    ``float("nan")`` for any docking failure. ``vina_bin`` is fixed at
    construction time via :class:`VinaScorerConfig` and cannot be
    overridden per call.

    The receptor is cached on disk under ``config.cache_dir/receptors/``
    (sidecar JSON next to the PDBQT). On the first call the cache is
    checked; on miss ``prepare_receptor`` runs and the sidecar is
    rewritten atomically. Ligand prep, docking results, and pose files
    all inherit ``cache_dir`` via the receptor path.
    """

    def __init__(self, config: VinaScorerConfig) -> None:
        self.config = config
        self.cache_dir: Path = Path(config.cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ("receptors", "ligands", "cache", "poses"):
            (self.cache_dir / subdir).mkdir(exist_ok=True)
        self._receptor: Optional[ReceptorConfig] = None
        self.last_results: list[dict[str, Any]] = []
        # Resolve the explicit path once; None means "defer to env / PATH
        # at use time" (preserves the default behavior for callers that
        # do not configure a Vina path).
        self._explicit_vina_bin: Optional[str] = (
            _resolve_vina_bin(config.vina_bin) if config.vina_bin else None
        )

    @property
    def _vina_bin(self) -> str:
        """Resolve the Vina binary path. Falls back to env / PATH at use time."""
        if self._explicit_vina_bin is not None:
            return self._explicit_vina_bin
        return _resolve_vina_bin(None)

    def _get_receptor(self) -> ReceptorConfig:
        if self._receptor is not None:
            return self._receptor
        key = _receptor_cache_key(
            self.config.pdb_id,
            self.config.chain_id,
            self.config.ligand_resname,
        )
        pdbqt_path = _receptor_pdbqt_path(self.cache_dir, key)
        meta_path = _receptor_meta_path(self.cache_dir, key)
        meta = self._try_load_cached_receptor(pdbqt_path, meta_path)
        if meta is not None:
            self._receptor = ReceptorConfig(
                receptor_pdbqt=str(pdbqt_path),
                **{field: meta[field] for field in _RECEPTOR_META_FIELDS if field != "receptor_pdbqt"},
            )
            return self._receptor
        receptor = prepare_receptor(
            self.config.pdb_id,
            chain_id=self.config.chain_id,
            ligand_resname=self.config.ligand_resname,
            work_dir=self.cache_dir,
            allow_zero_charge_fallback=self.config.allow_zero_charge_fallback,
        )
        actual_meta_path = _receptor_meta_path(self.cache_dir, _receptor_cache_key_from_path(receptor.receptor_pdbqt))
        _write_receptor_meta(actual_meta_path, receptor)
        self._receptor = receptor
        return self._receptor

    @staticmethod
    def _try_load_cached_receptor(
        pdbqt_path: Path, meta_path: Path
    ) -> Optional[dict[str, Any]]:
        """Return the sidecar dict on a valid cache hit, else None.

        Validity = (1) PDBQT exists and is non-empty, (2) meta JSON
        exists, parses, and has all required fields. Any failure
        returns None so the caller falls through to ``prepare_receptor``
        and heals the cache.
        """
        if not pdbqt_path.exists() or pdbqt_path.stat().st_size == 0:
            return None
        if not meta_path.exists():
            return None
        return _read_receptor_meta(meta_path)

    @staticmethod
    def _align(name: str, values: Optional[Iterable[Any]], length: int) -> list[Any]:
        if values is None:
            return [None] * length
        values_list = list(values)
        if len(values_list) != length:
            raise ValueError(
                f"{name} length ({len(values_list)}) does not match smiles_list length ({length})."
            )
        return values_list

    def _build_compounds(
        self,
        smiles_list: list[str],
        *,
        compound_ids: Optional[Iterable[str]] = None,
        parent_smiles: Optional[Iterable[str]] = None,
        reasyn_scores: Optional[Iterable[Optional[float]]] = None,
        activity_nM: Optional[Iterable[Optional[float]]] = None,
    ) -> list[ExtractedCompound]:
        ids = self._align("compound_ids", compound_ids, len(smiles_list))
        parents = self._align("parent_smiles", parent_smiles, len(smiles_list))
        reasyn = self._align("reasyn_scores", reasyn_scores, len(smiles_list))
        activity = self._align("activity_nM", activity_nM, len(smiles_list))
        compounds: list[ExtractedCompound] = []
        for index, smiles in enumerate(smiles_list):
            compounds.append(
                ExtractedCompound(
                    compound_id=str(ids[index] or f"smi_{index:04d}"),
                    full_smiles=smiles,
                    seed_smiles=str(parents[index] or ""),
                    reasyn_score=reasyn[index] if reasyn[index] is not None else None,
                    activity_nM=activity[index] if activity[index] is not None else None,
                )
            )
        return compounds

    def _dock_smiles(
        self,
        smiles_list: list[str],
    ) -> list[DockingResult]:
        """Dock a batch of SMILES. Internal helper; ``__call__`` is the public API."""
        if not smiles_list:
            return []
        compounds = self._build_compounds(smiles_list)
        return vina_dock_batch(
            compounds,
            self._get_receptor(),
            vina_bin=self._vina_bin,
            exhaustiveness=self.config.exhaustiveness,
            n_poses=self.config.n_poses,
            seed=self.config.seed,
            use_cache=self.config.use_cache,
            allow_debug_receptor=self.config.allow_debug_receptor,
            max_workers=self.config.max_workers,
        )

    def __call__(self, smiles_list: Iterable[str]) -> list[float]:
        """Score a batch of SMILES with Vina.

        The i-th element of the returned list is the Vina score
        (kcal/mol) of the i-th SMILES in ``smiles_list``, or
        ``float("nan")`` for any docking failure (prep failure,
        binary error, unparseable score). Length always equals
        ``len(smiles_list)``.
        """
        smiles_list = list(smiles_list)
        if not smiles_list:
            self.last_results = []
            return []
        try:
            results = self._dock_smiles(smiles_list)
        except Exception as exc:
            self.last_results = [
                {
                    "input_smiles": smiles,
                    "status": "scorer_exception",
                    "score": None,
                    "message": str(exc),
                }
                for smiles in smiles_list
            ]
            raise
        self.last_results = [
            {
                **model_to_plain_dict(result),
                "input_smiles": smiles,
            }
            for smiles, result in zip(smiles_list, results)
        ]
        out: list[float] = []
        for r in results:
            if (
                r.status == "ok"
                and r.score is not None
                and np.isfinite(float(r.score))
            ):
                out.append(float(r.score))
            else:
                out.append(float("nan"))
        return out
