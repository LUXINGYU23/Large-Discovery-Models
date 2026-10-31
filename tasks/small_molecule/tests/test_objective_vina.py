"""Tests for ``tasks.small_molecule.core.objective_vina``.

Convention follows ``tests/test_analog.py``: ``unittest`` + ``tempfile``
+ ``unittest.mock`` patching. Vina is mocked everywhere; no real
AutoDock Vina is invoked. Receptor prep is mocked so no PDB download
or Meeko call happens.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional
from unittest import mock

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tasks.small_molecule.core.docking import (  # noqa: E402
    DockingResult,
    ExtractedCompound,
    ReceptorConfig,
)

from tasks.small_molecule.core import (  # noqa: E402
    Scorer,
    VinaScorer,
    VinaScorerConfig,
)
from tasks.small_molecule.core import objective_vina as objective_module  # noqa: E402
from tasks.small_molecule.core.objective_vina import (  # noqa: E402
    _resolve_vina_bin,
    vina_dock_batch,
    vina_dock_one,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_receptor(tmp_path: Path, pdb_id: str = "8UN5") -> ReceptorConfig:
    receptor_pdbqt = tmp_path / "receptors" / f"{pdb_id}_A_LIG.pdbqt"
    receptor_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    receptor_pdbqt.write_text("")
    return ReceptorConfig(
        pdb_id=pdb_id,
        receptor_pdbqt=str(receptor_pdbqt),
        box_center=(0.0, 0.0, 0.0),
        box_size=(20.0, 20.0, 20.0),
        chain_id="A",
        ligand_resname="LIG",
        ligand_resseq="1",
        ligand_chain="A",
        prep_method="meeko",
    )


def _make_fake_receptor_with_proper_naming(
    cache_dir: Path, pdb_id: str = "8UN5", ligand_resname: str = "LIG"
) -> ReceptorConfig:
    """Fake receptor that follows prepare_receptor's file naming convention.

    The file is named ``<pdb_id>_<chain>_<lig>_receptor.pdbqt`` so
    the cache key (and sidecar path) can be derived from it.
    """
    receptor_pdbqt = (
        cache_dir / "receptors" / f"{pdb_id}_A_{ligand_resname}_receptor.pdbqt"
    )
    receptor_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    receptor_pdbqt.write_text("")
    return ReceptorConfig(
        pdb_id=pdb_id,
        receptor_pdbqt=str(receptor_pdbqt),
        box_center=(0.0, 0.0, 0.0),
        box_size=(20.0, 20.0, 20.0),
        chain_id="A",
        ligand_resname=ligand_resname,
        ligand_resseq="1",
        ligand_chain="A",
        prep_method="meeko",
    )


def _make_fake_vina(tmp_path: Path) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "fake_vina"
    path.write_text("#!/bin/sh\necho '   1  -7.5  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0'\n")
    path.chmod(0o755)
    return str(path)


def _ok_result(
    compound_id: str, smiles: str, score: float = -7.5
) -> DockingResult:
    return DockingResult(
        compound_id=compound_id,
        canonical_smiles=smiles,
        score=score,
        pose_ref="/fake/pose.pdbqt",
        status="ok",
    )


def _fail_result(compound_id: str, smiles: str, status: str = "dock_failed") -> DockingResult:
    return DockingResult(
        compound_id=compound_id,
        canonical_smiles=smiles,
        score=None,
        pose_ref=None,
        status=status,
        message="boom",
    )


# ---------------------------------------------------------------------------
# Class 1: VinaScorerConfig defaults
# ---------------------------------------------------------------------------


class VinaScorerConfigTests(unittest.TestCase):
    def test_defaults_match_reasyn_optuna_optimization(self) -> None:
        cfg = VinaScorerConfig()
        self.assertEqual(cfg.exhaustiveness, 4)
        self.assertEqual(cfg.n_poses, 3)
        self.assertEqual(cfg.max_workers, 1)
        self.assertIsNone(cfg.vina_bin)

    def test_default_cache_dir(self) -> None:
        cfg = VinaScorerConfig()
        self.assertEqual(cfg.cache_dir, Path("runs/docking"))

# ---------------------------------------------------------------------------
# Class 2: _resolve_vina_bin
# ---------------------------------------------------------------------------


class ResolveVinaBinTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="vina_resolve_"))
        self._saved_vina_env = os.environ.get("VINA_BIN")
        self._saved_path = os.environ.get("PATH")

    def tearDown(self) -> None:
        if self._saved_vina_env is None:
            os.environ.pop("VINA_BIN", None)
        else:
            os.environ["VINA_BIN"] = self._saved_vina_env
        if self._saved_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = self._saved_path

    def test_explicit_path_wins_over_env(self) -> None:
        explicit = _make_fake_vina(self.tmp)
        os.environ["VINA_BIN"] = "/nonexistent/vina"
        resolved = _resolve_vina_bin(explicit)
        self.assertEqual(resolved, str(Path(explicit).resolve()))

    def test_env_var_used_when_no_explicit(self) -> None:
        env_path = _make_fake_vina(self.tmp)
        os.environ["VINA_BIN"] = env_path
        resolved = _resolve_vina_bin(None)
        self.assertEqual(resolved, str(Path(env_path).resolve()))

    def test_shutil_which_used_when_no_explicit_and_no_env(self) -> None:
        env_path = _make_fake_vina(self.tmp)
        os.environ.pop("VINA_BIN", None)
        # shutil.which does its own PATH search; force a deterministic
        # resolution by prepending a directory containing "vina".
        os.environ["PATH"] = f"{self.tmp}{os.pathsep}{self._saved_path or ''}"
        # The fake file we wrote is named "fake_vina"; create a symlink "vina"
        # so shutil.which finds it.
        link = self.tmp / "vina"
        if not link.exists():
            link.symlink_to(env_path)
        resolved = _resolve_vina_bin(None)
        self.assertTrue(resolved.endswith("/vina") or resolved.endswith("vina"))

    def test_raises_when_nothing_available(self) -> None:
        os.environ.pop("VINA_BIN", None)
        # Patch shutil.which to return None directly so we are not at the
        # mercy of whatever defaults Python's path-search uses.
        with mock.patch.object(objective_module.shutil, "which", return_value=None):
            with self.assertRaises(FileNotFoundError) as ctx:
                _resolve_vina_bin(None)
        self.assertIn("AutoDock Vina executable not found", str(ctx.exception))

    def test_raises_when_explicit_path_missing(self) -> None:
        with self.assertRaises(FileNotFoundError):
            _resolve_vina_bin(str(self.tmp / "does_not_exist"))

    def test_raises_when_explicit_path_not_executable(self) -> None:
        not_exec = self.tmp / "not_exec"
        not_exec.write_text("hi")
        # chmod 0o644 (no exec bit)
        not_exec.chmod(0o644)
        with self.assertRaises(FileNotFoundError):
            _resolve_vina_bin(str(not_exec))

    def test_raises_when_explicit_path_cannot_run(self) -> None:
        wrong_format = self.tmp / "wrong_format_vina"
        wrong_format.write_text("not a linux executable")
        wrong_format.chmod(0o755)
        with self.assertRaises(RuntimeError) as ctx:
            _resolve_vina_bin(str(wrong_format))
        self.assertIn("AutoDock Vina executable cannot be run", str(ctx.exception))

    def test_resolves_relative_path_to_absolute(self) -> None:
        fake = _make_fake_vina(self.tmp)
        rel_path = "fake_vina"
        cwd = os.getcwd()
        try:
            os.chdir(self.tmp)
            resolved = _resolve_vina_bin(rel_path)
        finally:
            os.chdir(cwd)
        self.assertTrue(os.path.isabs(resolved))
        self.assertEqual(resolved, str(Path(fake).resolve()))


# ---------------------------------------------------------------------------
# Class 3: VinaScorer lifecycle
# ---------------------------------------------------------------------------


class VinaScorerLifecycleTests(unittest.TestCase):
    def test_init_resolves_explicit_vina_bin(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="vina_scorer_"))
        fake = _make_fake_vina(tmp)
        cfg = VinaScorerConfig(vina_bin=fake)
        scorer = VinaScorer(cfg)
        self.assertEqual(scorer._explicit_vina_bin, str(Path(fake).resolve()))

    def test_init_does_not_resolve_when_vina_bin_none(self) -> None:
        cfg = VinaScorerConfig()
        with mock.patch.object(objective_module, "_resolve_vina_bin") as mocked:
            VinaScorer(cfg)
        mocked.assert_not_called()

    def test_get_receptor_is_lazy_and_cached(self) -> None:
        cfg = VinaScorerConfig()
        scorer = VinaScorer(cfg)
        tmp = Path(tempfile.mkdtemp(prefix="vina_lifecycle_"))
        fake_receptor = _make_fake_receptor(tmp)
        with mock.patch.object(
            objective_module, "prepare_receptor", return_value=fake_receptor
        ) as mocked:
            r1 = scorer._get_receptor()
            r2 = scorer._get_receptor()
        self.assertIs(r1, r2)
        self.assertIs(r1, fake_receptor)
        self.assertEqual(mocked.call_count, 1)


# ---------------------------------------------------------------------------
# Class 4: VinaScorer receptor cache
# ---------------------------------------------------------------------------


def _write_receptor_cache_files(
    cache_dir: Path,
    key: str,
    *,
    pdbqt_content: str = "REMARK fake\nEND\n",
    receptor: Optional[ReceptorConfig] = None,
    box_center: tuple = (0.0, 0.0, 0.0),
    box_size: tuple = (20.0, 20.0, 20.0),
    chain_id: str = "A",
    ligand_resname: str = "LIG",
    prep_method: str = "meeko",
    omit_fields: tuple = (),
) -> Path:
    """Pre-populate the receptor PDBQT + sidecar JSON in the cache."""
    pdbqt_path = objective_module._receptor_pdbqt_path(cache_dir, key)
    meta_path = objective_module._receptor_meta_path(cache_dir, key)
    pdbqt_path.parent.mkdir(parents=True, exist_ok=True)
    pdbqt_path.write_text(pdbqt_content)
    meta: dict = {
        "pdb_id": "8UN5",
        "chain_id": chain_id,
        "ligand_resname": ligand_resname,
        "ligand_resseq": "1",
        "ligand_chain": "A",
        "box_center": list(box_center),
        "box_size": list(box_size),
        "prep_method": prep_method,
        "prep_command": "fake",
        "prep_log": "",
        "warnings": [],
    }
    for field in omit_fields:
        meta.pop(field, None)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return pdbqt_path


class VinaScorerCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="vina_scorer_cache_"))
        self.cfg = VinaScorerConfig(
            cache_dir=self.tmp / "cache",
            ligand_resname="LIG",
        )
        self.scorer = VinaScorer(self.cfg)

    def test_init_creates_cache_dir_and_subdirs(self) -> None:
        # setUp already created it; verify the four subdirs exist.
        for subdir in ("receptors", "ligands", "cache", "poses"):
            self.assertTrue((self.tmp / "cache" / subdir).is_dir())

    def test_init_creates_missing_cache_dir(self) -> None:
        nested = self.tmp / "deeply" / "nested" / "cache"
        cfg = VinaScorerConfig(cache_dir=nested)
        scorer = VinaScorer(cfg)
        self.assertTrue(nested.is_dir())
        self.assertTrue((nested / "receptors").is_dir())

    def test_get_receptor_cache_hit(self) -> None:
        _write_receptor_cache_files(self.cfg.cache_dir, "8UN5_A_LIG")
        with mock.patch.object(
            objective_module, "prepare_receptor"
        ) as mocked:
            receptor = self.scorer._get_receptor()
        mocked.assert_not_called()
        self.assertEqual(receptor.chain_id, "A")
        self.assertEqual(receptor.ligand_resname, "LIG")
        self.assertEqual(receptor.box_center, (0.0, 0.0, 0.0))
        self.assertEqual(receptor.box_size, (20.0, 20.0, 20.0))
        self.assertEqual(receptor.prep_method, "meeko")

    def test_get_receptor_cache_miss_missing_pdbqt(self) -> None:
        # Pre-populate meta only, no PDBQT.
        key = "8UN5_A_LIG"
        meta_path = objective_module._receptor_meta_path(self.cfg.cache_dir, key)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(
                {
                    "pdb_id": "8UN5", "chain_id": "A", "ligand_resname": "LIG",
                    "ligand_resseq": "1", "ligand_chain": "A",
                    "box_center": [0, 0, 0], "box_size": [20, 20, 20],
                    "prep_method": "meeko", "prep_command": "", "prep_log": "",
                    "warnings": [],
                }
            )
        )
        with mock.patch.object(
            objective_module, "prepare_receptor",
            return_value=_make_fake_receptor(self.tmp),
        ) as mocked:
            self.scorer._get_receptor()
        mocked.assert_called_once()

    def test_get_receptor_cache_miss_missing_meta(self) -> None:
        # Pre-populate PDBQT only, no meta.
        key = "8UN5_A_LIG"
        pdbqt_path = objective_module._receptor_pdbqt_path(self.cfg.cache_dir, key)
        pdbqt_path.parent.mkdir(parents=True, exist_ok=True)
        pdbqt_path.write_text("REMARK fake\nEND\n")
        with mock.patch.object(
            objective_module, "prepare_receptor",
            return_value=_make_fake_receptor(self.tmp),
        ) as mocked:
            self.scorer._get_receptor()
        mocked.assert_called_once()

    def test_get_receptor_cache_miss_empty_pdbqt(self) -> None:
        _write_receptor_cache_files(
            self.cfg.cache_dir, "8UN5_A_LIG", pdbqt_content=""
        )
        with mock.patch.object(
            objective_module, "prepare_receptor",
            return_value=_make_fake_receptor(self.tmp),
        ) as mocked:
            self.scorer._get_receptor()
        mocked.assert_called_once()

    def test_get_receptor_cache_miss_malformed_json(self) -> None:
        key = "8UN5_A_LIG"
        pdbqt_path = objective_module._receptor_pdbqt_path(self.cfg.cache_dir, key)
        meta_path = objective_module._receptor_meta_path(self.cfg.cache_dir, key)
        pdbqt_path.parent.mkdir(parents=True, exist_ok=True)
        pdbqt_path.write_text("REMARK fake\nEND\n")
        meta_path.write_text("{not valid json", encoding="utf-8")
        with mock.patch.object(
            objective_module, "prepare_receptor",
            return_value=_make_fake_receptor(self.tmp),
        ) as mocked:
            self.scorer._get_receptor()
        mocked.assert_called_once()

    def test_get_receptor_cache_miss_missing_key(self) -> None:
        _write_receptor_cache_files(
            self.cfg.cache_dir, "8UN5_A_LIG", omit_fields=("box_center",)
        )
        with mock.patch.object(
            objective_module, "prepare_receptor",
            return_value=_make_fake_receptor(self.tmp),
        ) as mocked:
            self.scorer._get_receptor()
        mocked.assert_called_once()

    def test_get_receptor_writes_meta_with_expected_fields(self) -> None:
        # Build a fake receptor that lives under the scorer's cache_dir so
        # the sidecar is written at the expected path.
        receptor = _make_fake_receptor_with_proper_naming(self.cfg.cache_dir)
        with mock.patch.object(
            objective_module, "prepare_receptor", return_value=receptor
        ):
            self.scorer._get_receptor()
        meta_path = Path(receptor.receptor_pdbqt).with_suffix(".meta.json")
        self.assertTrue(meta_path.exists())
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for field in objective_module._RECEPTOR_META_FIELDS:
            self.assertIn(field, meta)
        self.assertEqual(meta["chain_id"], receptor.chain_id)
        self.assertEqual(meta["box_size"], list(receptor.box_size))

    def test_atomic_meta_write_replaces_tmp(self) -> None:
        # After a successful write, the .tmp file should be gone.
        receptor = _make_fake_receptor_with_proper_naming(self.cfg.cache_dir)
        with mock.patch.object(
            objective_module, "prepare_receptor", return_value=receptor
        ):
            self.scorer._get_receptor()
        meta_path = Path(receptor.receptor_pdbqt).with_suffix(".meta.json")
        self.assertTrue(meta_path.exists())
        self.assertFalse(meta_path.with_suffix(".meta.json.tmp").exists())

    def test_meta_write_overwrites_stale_meta(self) -> None:
        # Stale (garbage) meta at the same path; fresh prep should replace it.
        key = "8UN5_A_LIG"
        meta_path = objective_module._receptor_meta_path(self.cfg.cache_dir, key)
        pdbqt_path = objective_module._receptor_pdbqt_path(self.cfg.cache_dir, key)
        pdbqt_path.parent.mkdir(parents=True, exist_ok=True)
        pdbqt_path.write_text("REMARK stale\nEND\n")
        meta_path.write_text("stale garbage", encoding="utf-8")
        receptor = _make_fake_receptor(self.tmp)
        # Make the fake receptor's path match the cache key so the sidecar
        # is written at the same path the stale one is at.
        receptor_pdbqt = str(pdbqt_path)
        receptor = receptor.model_copy(update={"receptor_pdbqt": receptor_pdbqt})
        with mock.patch.object(
            objective_module, "prepare_receptor", return_value=receptor
        ):
            self.scorer._get_receptor()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertEqual(meta["chain_id"], "A")
        self.assertNotEqual(meta, "stale garbage")

    def test_cache_key_matches_extract_and_dock_format(self) -> None:
        # The cache key format must match prepare_receptor's suffix.
        key = objective_module._receptor_cache_key("8un5", "A", "XQ6")
        self.assertEqual(key, "8UN5_A_XQ6")
        key = objective_module._receptor_cache_key("8un5", None, "XQ6")
        self.assertEqual(key, "8UN5_all_XQ6")
        key = objective_module._receptor_cache_key("8un5", "A", None)
        self.assertEqual(key, "8UN5_A_auto")

    def test_receptor_cache_key_from_path_strips_receptor(self) -> None:
        self.assertEqual(
            objective_module._receptor_cache_key_from_path("/x/receptors/8UN5_A_LIG_receptor.pdbqt"),
            "8UN5_A_LIG",
        )


# ---------------------------------------------------------------------------
# Class 4: VinaScorer.__call__ (single canonical scoring interface)
# ---------------------------------------------------------------------------


class VinaScorerCallTests(unittest.TestCase):
    """``VinaScorer.__call__(smiles) -> list[float]`` is the only public
    scoring interface. The i-th output is the i-th SMILES's score, or
    ``float('nan')`` on any docking failure (prep / dock error /
    unparseable score)."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="vina_call_"))
        self.cfg = VinaScorerConfig(vina_bin=_make_fake_vina(self.tmp))
        self.scorer = VinaScorer(self.cfg)
        self.scorer._receptor = _make_fake_receptor(self.tmp)

    def test_call_returns_scores_in_input_order(self) -> None:
        def fake_batch(compounds, receptor, *, vina_bin, **kwargs):
            # Echo back a distinct score per compound.
            return [
                _ok_result(c.compound_id, c.full_smiles or "", score=-7.0 - index)
                for index, c in enumerate(compounds)
            ]

        with mock.patch.object(
            objective_module, "vina_dock_batch", side_effect=fake_batch
        ):
            out = self.scorer(["CCO", "CCN", "CCC"])
        self.assertEqual(out, [-7.0, -8.0, -9.0])

    def test_call_returns_nan_for_dock_failed(self) -> None:
        def fake_batch(compounds, receptor, *, vina_bin, **kwargs):
            return [
                _ok_result("a", "CCO", -7.0),
                _fail_result("b", "CCN", status="dock_failed"),
                _ok_result("c", "CCC", -6.0),
            ]

        with mock.patch.object(
            objective_module, "vina_dock_batch", side_effect=fake_batch
        ):
            out = self.scorer(["CCO", "CCN", "CCC"])
        self.assertEqual(out[0], -7.0)
        self.assertTrue(np.isnan(out[1]))
        self.assertEqual(out[2], -6.0)
        self.assertEqual(self.scorer.last_results[1]["input_smiles"], "CCN")
        self.assertEqual(self.scorer.last_results[1]["status"], "dock_failed")

    def test_call_returns_nan_for_prep_failed(self) -> None:
        def fake_batch(compounds, receptor, *, vina_bin, **kwargs):
            return [
                _fail_result("a", "CCO", status="prep_failed"),
                _ok_result("b", "CCN", -7.5),
            ]

        with mock.patch.object(
            objective_module, "vina_dock_batch", side_effect=fake_batch
        ):
            out = self.scorer(["CCO", "CCN"])
        self.assertTrue(np.isnan(out[0]))
        self.assertEqual(out[1], -7.5)

    def test_call_returns_nan_for_non_finite_score(self) -> None:
        def fake_batch(compounds, receptor, *, vina_bin, **kwargs):
            return [
                DockingResult(
                    compound_id="a",
                    canonical_smiles="CCO",
                    score=float("inf"),   # unparseable / overflow score
                    pose_ref="/fake/pose.pdbqt",
                    status="ok",
                ),
                _ok_result("b", "CCN", -7.0),
            ]

        with mock.patch.object(
            objective_module, "vina_dock_batch", side_effect=fake_batch
        ):
            out = self.scorer(["CCO", "CCN"])
        self.assertTrue(np.isnan(out[0]))
        self.assertEqual(out[1], -7.0)

    def test_call_returns_nan_for_none_score(self) -> None:
        def fake_batch(compounds, receptor, *, vina_bin, **kwargs):
            return [
                DockingResult(
                    compound_id="a",
                    canonical_smiles="CCO",
                    score=None,
                    pose_ref=None,
                    status="ok",   # status ok but score missing
                ),
            ]

        with mock.patch.object(
            objective_module, "vina_dock_batch", side_effect=fake_batch
        ):
            out = self.scorer(["CCO"])
        self.assertTrue(np.isnan(out[0]))

    def test_call_returns_empty_list_for_empty_input(self) -> None:
        with mock.patch.object(objective_module, "vina_dock_batch") as mocked:
            out = self.scorer([])
        self.assertEqual(out, [])
        mocked.assert_not_called()

    def test_call_length_always_matches_input(self) -> None:
        def fake_batch(compounds, receptor, *, vina_bin, **kwargs):
            return [_ok_result(c.compound_id, "") for c in compounds]

        with mock.patch.object(
            objective_module, "vina_dock_batch", side_effect=fake_batch
        ):
            for n in (1, 2, 5, 10):
                out = self.scorer(["CCO"] * n)
                self.assertEqual(len(out), n)

    def test_call_iterable_input_accepted(self) -> None:
        """The parameter type is Iterable[str]; tuples and generators work too."""
        def fake_batch(compounds, receptor, *, vina_bin, **kwargs):
            return [
                _ok_result(c.compound_id, c.full_smiles or "", score=-5.0)
                for c in compounds
            ]

        with mock.patch.object(
            objective_module, "vina_dock_batch", side_effect=fake_batch
        ):
            out = self.scorer(("CCO", "CCN"))  # tuple, not list
        self.assertEqual(out, [-5.0, -5.0])

    def test_satisfies_scorer_type_alias(self) -> None:
        """A VinaScorer instance must be usable wherever a Scorer is expected."""
        # VinaScorer is callable.
        self.assertTrue(callable(self.scorer))
        # The Scorer type alias is importable from the package level.
        from tasks.small_molecule.core import Scorer as TopLevelScorer
        self.assertIs(TopLevelScorer, Scorer)

    def test_call_does_not_read_vina_bin_env(self) -> None:
        """``vina_bin`` is fixed at ``__init__``; the call must not fall
        through to ``$VINA_BIN`` per invocation."""
        captured: dict = {}

        def fake_batch(compounds, receptor, *, vina_bin, **kwargs):
            captured["vina_bin"] = vina_bin
            return [_ok_result(c.compound_id, "") for c in compounds]

        with mock.patch.object(os, "environ", new=os.environ.copy()) as mocked_environ:
            mocked_environ.pop("VINA_BIN", None)
            with mock.patch.object(
                objective_module, "vina_dock_batch", side_effect=fake_batch
            ):
                self.scorer(["CCO"])
        # The path passed through is the one resolved at __init__.
        self.assertEqual(captured["vina_bin"], self.scorer._explicit_vina_bin)
        # The env var was not consulted (we deleted it; resolution still succeeded).


# ---------------------------------------------------------------------------
# Class 5: vina_dock_one / vina_dock_batch direct
# ---------------------------------------------------------------------------


def _mock_run_logged_command(
    returncode: int = 0,
    stdout: str = "   1  -7.5  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0\n",
    stderr: str = "",
):
    def _fake(cmd, log_path, timeout=1800):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"COMMAND:\n{' '.join(cmd)}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args=cmd, returncode=returncode, stdout=stdout, stderr=stderr)
    return _fake


class VinaDockOneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="vina_dock_one_"))
        self.vina_bin = _make_fake_vina(self.tmp)
        self.receptor = _make_fake_receptor(self.tmp)

    def test_cmd_uses_supplied_vina_bin(self) -> None:
        captured: dict = {}

        def fake(cmd, log_path, timeout=1800):
            captured["cmd"] = list(cmd)
            return _mock_run_logged_command()(cmd, log_path, timeout)

        with mock.patch.multiple(
            objective_module,
            canonicalize_smiles=mock.DEFAULT,
            prepare_ligand=mock.DEFAULT,
            run_logged_command=fake,
            parse_vina_score_from_text=lambda text: -7.5,
        ):
            with mock.patch.object(
                objective_module,
                "canonicalize_smiles",
                return_value="CCO",
            ):
                with mock.patch.object(
                    objective_module,
                    "prepare_ligand",
                    return_value="/tmp/lig.pdbqt",
                ):
                    result = vina_dock_one(
                        "CCO",
                        self.receptor,
                        vina_bin=self.vina_bin,
                    )
        self.assertEqual(captured["cmd"][0], self.vina_bin)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.score, -7.5)
        self.assertEqual(result.canonical_smiles, "CCO")

    def test_returns_prep_failed_when_smiles_empty(self) -> None:
        result = vina_dock_one(
            "",
            self.receptor,
            vina_bin=self.vina_bin,
        )
        self.assertEqual(result.status, "prep_failed")
        self.assertIsNone(result.score)

    def test_returns_prep_failed_when_prepare_ligand_fails(self) -> None:
        with mock.patch.object(
            objective_module, "canonicalize_smiles", return_value="CCO"
        ):
            with mock.patch.object(
                objective_module, "prepare_ligand", return_value=None
            ):
                result = vina_dock_one(
                    "CCO",
                    self.receptor,
                    vina_bin=self.vina_bin,
                )
        self.assertEqual(result.status, "prep_failed")

    def test_returns_dock_failed_on_nonzero_returncode(self) -> None:
        with mock.patch.object(
            objective_module, "canonicalize_smiles", return_value="CCO"
        ):
            with mock.patch.object(
                objective_module, "prepare_ligand", return_value="/tmp/lig.pdbqt"
            ):
                with mock.patch.object(
                    objective_module,
                    "run_logged_command",
                    return_value=subprocess.CompletedProcess(
                        args=["x"], returncode=1, stdout="", stderr="boom"
                    ),
                ):
                    result = vina_dock_one(
                        "CCO",
                        self.receptor,
                        vina_bin=self.vina_bin,
                    )
        self.assertEqual(result.status, "dock_failed")
        self.assertIn("boom", result.message)

    def test_returns_dock_failed_when_score_unparseable(self) -> None:
        with mock.patch.object(
            objective_module, "canonicalize_smiles", return_value="CCO"
        ):
            with mock.patch.object(
                objective_module, "prepare_ligand", return_value="/tmp/lig.pdbqt"
            ):
                with mock.patch.object(
                    objective_module,
                    "run_logged_command",
                    return_value=subprocess.CompletedProcess(
                        args=["x"], returncode=0, stdout="", stderr=""
                    ),
                ):
                    with mock.patch.object(
                        objective_module, "parse_vina_score_from_text", return_value=None
                    ):
                        with mock.patch.object(
                            objective_module, "parse_vina_score_from_pose", return_value=None
                        ):
                            result = vina_dock_one(
                                "CCO",
                                self.receptor,
                                vina_bin=self.vina_bin,
                            )
        self.assertEqual(result.status, "dock_failed")
        self.assertIn("no score", result.message)


class VinaDockBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="vina_dock_batch_"))
        self.vina_bin = _make_fake_vina(self.tmp)
        self.receptor = _make_fake_receptor(self.tmp)
        self.receptor_pdbqt_path = Path(self.receptor.receptor_pdbqt)

    def test_passes_vina_bin_through_to_each_dock_one(self) -> None:
        compounds = [
            ExtractedCompound(compound_id="a", full_smiles="CCO"),
            ExtractedCompound(compound_id="b", full_smiles="CCN"),
        ]
        captured_bins: list[str] = []

        def fake_dock_one(smiles, receptor, *, vina_bin, **kwargs):
            captured_bins.append(vina_bin)
            return _ok_result(kwargs.get("compound_id", ""), smiles, -7.0)

        with mock.patch.object(
            objective_module, "vina_dock_one", side_effect=fake_dock_one
        ):
            with mock.patch.object(objective_module, "canonicalize_smiles", side_effect=lambda s: s):
                results = vina_dock_batch(
                    compounds, self.receptor, vina_bin=self.vina_bin
                )
        self.assertEqual(captured_bins, [self.vina_bin, self.vina_bin])
        self.assertEqual(len(results), 2)

    def test_serial_for_max_workers_1(self) -> None:
        compounds = [
            ExtractedCompound(compound_id=f"a{i}", full_smiles="CCO")
            for i in range(3)
        ]
        with mock.patch.object(
            objective_module,
            "vina_dock_one",
            return_value=_ok_result("a", "CCO", -7.0),
        ) as mocked:
            with mock.patch.object(objective_module, "canonicalize_smiles", side_effect=lambda s: s):
                with mock.patch.object(
                    objective_module, "ThreadPoolExecutor", wraps=ThreadPoolExecutor
                ) as tpe_mock:
                    vina_dock_batch(
                        compounds,
                        self.receptor,
                        vina_bin=self.vina_bin,
                        max_workers=1,
                    )
        mocked.assert_called()
        # ThreadPoolExecutor is bypassed on the max_workers<=1 path.
        tpe_mock.assert_not_called()

    def test_uses_threadpool_for_max_workers_gt_1(self) -> None:
        compounds = [
            ExtractedCompound(compound_id=f"a{i}", full_smiles="CCO")
            for i in range(3)
        ]
        with mock.patch.object(
            objective_module,
            "vina_dock_one",
            return_value=_ok_result("a", "CCO", -7.0),
        ):
            with mock.patch.object(objective_module, "canonicalize_smiles", side_effect=lambda s: s):
                with mock.patch.object(
                    objective_module, "ThreadPoolExecutor", wraps=ThreadPoolExecutor
                ) as tpe_mock:
                    vina_dock_batch(
                        compounds,
                        self.receptor,
                        vina_bin=self.vina_bin,
                        max_workers=2,
                    )
        tpe_mock.assert_called_once_with(max_workers=2)

    def test_cache_round_trip(self) -> None:
        # Create a real pose file so the cache hit's pose_ref.exists() check passes.
        pose_file = self.tmp / "fake_pose.pdbqt"
        pose_file.write_text("")

        def fake_dock_one(smiles, receptor, *, vina_bin, **kwargs):
            return DockingResult(
                compound_id=kwargs.get("compound_id", ""),
                canonical_smiles=smiles,
                score=-7.0,
                pose_ref=str(pose_file),
                status="ok",
            )

        compounds = [ExtractedCompound(compound_id="a", full_smiles="CCO")]
        with mock.patch.object(
            objective_module, "vina_dock_one", side_effect=fake_dock_one
        ) as mocked:
            with mock.patch.object(objective_module, "canonicalize_smiles", side_effect=lambda s: s):
                # First call: writes cache.
                first = vina_dock_batch(
                    compounds,
                    self.receptor,
                    vina_bin=self.vina_bin,
                )
                self.assertEqual(mocked.call_count, 1)
                # Second call: should hit the cache.
                second = vina_dock_batch(
                    compounds,
                    self.receptor,
                    vina_bin=self.vina_bin,
                )
                self.assertEqual(mocked.call_count, 1)
        self.assertEqual(first[0].score, -7.0)
        self.assertEqual(second[0].score, -7.0)
        self.assertTrue(second[0].cached)

    def test_handles_empty_smiles_in_compound(self) -> None:
        compounds = [ExtractedCompound(compound_id="a", full_smiles="")]
        with mock.patch.object(objective_module, "canonicalize_smiles", return_value=""):
            with mock.patch.object(
                objective_module, "vina_dock_one"
            ) as mocked:
                results = vina_dock_batch(
                    compounds,
                    self.receptor,
                    vina_bin=self.vina_bin,
                )
        mocked.assert_not_called()
        self.assertEqual(results[0].status, "prep_failed")


# ---------------------------------------------------------------------------
# Class 10: __init__ re-exports
# ---------------------------------------------------------------------------


class PackageExportsTests(unittest.TestCase):
    def test_init_reexports_core_symbols(self) -> None:
        from tasks.small_molecule.core import (
            Scorer,
            VinaScorer,
            VinaScorerConfig,
        )
        self.assertIs(VinaScorer, VinaScorer)
        self.assertIs(VinaScorerConfig, VinaScorerConfig)
        self.assertIs(Scorer, Scorer)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
