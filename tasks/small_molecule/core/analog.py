"""In-process / subprocess bridge to ReaSyn analog generation.

The intended call site is `tasks.small_molecule.core` callers that already hold a seed SMILES
and want synthesizable analogs plus a pathway back as a DataFrame, without
writing intermediate CSV/manifest/log files. By default the call runs in
process and re-uses the calling Python interpreter. Passing ``python_bin``
(or letting the fallback chain discover ``REASYN_PYTHON`` / a sibling
``.venv/bin/python``) switches to a subprocess launcher that talks to ReaSyn
through three short-lived tempfiles (input SMILES, output DataFrame pickle,
launcher script) and cleans them up automatically.

Notes:
* ReaSyn has no ``__init__.py``; we rely on ``sys.path.insert(0, repo)``.
* Default ``devices=[1, 2]`` to avoid occupying card 0 on shared clusters.
* Setting ``CUDA_VISIBLE_DEVICES`` in either mode remaps ReaSyn's relative
  ``cuda:0`` indices to the requested physical device IDs.
"""

from __future__ import annotations

import logging
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import pandas as pd

LOGGER = logging.getLogger(__name__)


_REASYN_ENTRYPOINT_REL = Path("reasyn") / "sampler" / "parallel.py"
_DEFAULT_MODEL_PATHS = (
    "data/trained_model/nv-reasyn-ar-166m-v2.ckpt",
    "data/trained_model/nv-reasyn-eb-174m-v2.ckpt",
)


@dataclass
class ReasynConfig:
    """Configuration for one ``generate_analogs`` call.

    Required:
        ``model_path``: comma-separated string ``"ar.ckpt,eb.ckpt"`` or a
            two-element list. Relative paths are resolved under
            ``reasyn_repo`` once it is known.

    Optional with automatic fallback:
        ``reasyn_repo``: explicit ReaSyn checkout path. Falls back to
            ``REASYN_HOME`` then ``REASYN_REPO``. No convention/sibling
            lookup. A missing or invalid repo raises ``RuntimeError``.
        ``python_bin``: explicit interpreter. Falls back to ``REASYN_PYTHON``,
            then ``REASYN_BIN``, then ``<reasyn_repo>/.venv/bin/python``.
            ``None`` (or matching ``sys.executable``) selects in-process mode.
        ``temp_dir``: where the subprocess launcher drops its three temp
            files. ``None`` selects ``tempfile.gettempdir()``.

    GPU:
        ``devices``: physical CUDA device IDs to use. Default ``[1, 2]`` to
            avoid occupying card 0 on shared clusters. Overridable. The IDs
            must be visible via ``nvidia-smi``; otherwise the call fails.

    Sampling defaults are **deliberately conservative** for shared GPU
    clusters. They run a single worker per GPU with a 2-minute per-molecule
    budget and a small active-state set, which is enough for an API called
    repeatedly inside a search loop without starving co-tenants. For the
    more aggressive ReaSyn README hit-expansion recipe (assumes a dedicated
    H100 node), override explicitly:

        search_width=12, exhaustiveness=128, num_cycles=12,
        num_editflow_samples=100, num_editflow_steps=100, time_limit=10000,
        num_workers_per_gpu=8
    """

    model_path: Union[str, list[str]]

    reasyn_repo: Optional[str] = None
    python_bin: Optional[str] = None
    temp_dir: Optional[Union[str, Path]] = None
    devices: list[int] = field(default_factory=lambda: [1, 2])

    # Shared-cluster-friendly defaults (vs. ReaSyn README's aggressive recipe).
    search_width: int = 6
    exhaustiveness: int = 16
    num_cycles: int = 4
    num_editflow_samples: int = 20
    num_editflow_steps: int = 100
    time_limit: int = 120
    num_workers_per_gpu: int = 1
    task_qsize: int = 0
    result_qsize: int = 0
    filter_sim: float = 0.8
    mols_to_filter: Optional[str] = None
    add_bb_path: Optional[str] = None
    no_exact_break: bool = True

    canonicalize: bool = True
    verbose: bool = True


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def _resolve_reasyn_repo(explicit: Optional[str]) -> Path:
    """Pick the ReaSyn checkout via explicit arg + ``REASYN_HOME`` / ``REASYN_REPO``."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    for env_name in ("REASYN_HOME", "REASYN_REPO"):
        env_val = os.environ.get(env_name)
        if env_val:
            candidates.append(Path(env_val).expanduser())
    tried = [str(c) for c in candidates]
    for candidate in candidates:
        if (candidate / _REASYN_ENTRYPOINT_REL).exists():
            return candidate.resolve()
    raise RuntimeError(
        "Cannot locate ReaSyn repo. Pass `ReasynConfig.reasyn_repo` or set "
        "REASYN_HOME / REASYN_REPO. Tried: " + (", ".join(tried) if tried else "<none>")
    )


def _resolve_python_bin(explicit: Optional[str], reasyn_repo: Path) -> Optional[str]:
    """Return an absolute interpreter path or ``None`` to stay in-process.

    Important: do NOT call ``Path.resolve()`` on the result, because that
    follows symlinks and breaks venv detection. The venv under
    ``<repo>/.venv/bin/python`` is usually a symlink into a uv-managed
    base interpreter; resolving it strips the adjacent ``pyvenv.cfg`` and
    leaves the subprocess Python without the venv's ``site-packages``.
    """
    if explicit:
        return _normalize_python_bin(Path(explicit).expanduser(), "python_bin")
    for env_name in ("REASYN_PYTHON", "REASYN_BIN"):
        env_val = os.environ.get(env_name)
        if env_val:
            return _normalize_python_bin(Path(env_val).expanduser(), env_name)
    venv_bin = reasyn_repo / ".venv" / "bin"
    if venv_bin.exists():
        return _normalize_python_bin(venv_bin, "repo .venv/bin")
    return None


def _normalize_python_bin(path: Path, label: str) -> str:
    """Accept a Python executable, or a venv ``bin`` directory containing one."""
    if not path.exists():
        if path.name == "python":
            fallback = _first_existing_python(path.parent)
            if fallback is not None:
                return str(fallback.absolute())
        if path.is_symlink():
            raise RuntimeError(
                f"{label} is a broken symlink: {path} -> {os.readlink(path)}. "
                f"Use an existing Python executable such as {path.parent / 'python3'}."
            )
        raise RuntimeError(f"{label} not found: {path}")
    if path.is_dir():
        candidate = _first_existing_python(path)
        if candidate is not None:
            return str(candidate.absolute())
        raise RuntimeError(
            f"{label} points to a directory, not a Python executable: {path}. "
            "Expected one of python, python3, or python3.* inside it."
        )
    return str(path.absolute())


def _first_existing_python(bin_dir: Path) -> Path | None:
    for name in ("python", "python3"):
        candidate = bin_dir / name
        if candidate.exists():
            return candidate
    versioned = sorted(
        candidate
        for candidate in bin_dir.glob("python3.*")
        if candidate.exists() and not candidate.is_dir()
    )
    return versioned[0] if versioned else None


def _query_visible_gpu_ids() -> set[int]:
    """Return the set of physical CUDA device IDs reported by ``nvidia-smi``."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return set()
    ids: set[int] = set()
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.isdigit():
            ids.add(int(stripped))
    return ids


def _resolve_devices(devices: list[int]) -> list[int]:
    """Validate that every requested device ID is visible via ``nvidia-smi``."""
    if not isinstance(devices, (list, tuple)) or not devices:
        raise RuntimeError(f"devices must be a non-empty list[int], got {devices!r}")
    if not all(isinstance(d, int) and d >= 0 for d in devices):
        raise RuntimeError(f"devices must be non-negative list[int], got {devices!r}")
    visible = _query_visible_gpu_ids()
    if not visible:
        # Without nvidia-smi we cannot validate; trust the caller and warn.
        LOGGER.warning(
            "nvidia-smi unavailable; skipping device visibility check for %s", devices
        )
        return list(devices)
    missing = [d for d in devices if d not in visible]
    if missing:
        raise RuntimeError(
            f"devices={list(devices)} requested but GPU(s) {missing} not visible via nvidia-smi "
            f"(visible: {sorted(visible)}). Check CUDA_VISIBLE_DEVICES / nvidia-smi."
        )
    return list(devices)


def _parse_model_paths(value: Union[str, list[str]], reasyn_repo: Path) -> list[Path]:
    """Resolve the AR/EB checkpoint paths into exactly two existing files."""
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    else:
        items = [str(item).strip() for item in value if str(item).strip()]
    if len(items) != 2:
        raise RuntimeError(
            f"model_path must resolve to exactly 2 checkpoints (AR + Edit Bridge), got {len(items)}: {items}"
        )
    paths: list[Path] = []
    for item in items:
        path = Path(item).expanduser()
        if not path.is_absolute():
            path = reasyn_repo / item
        paths.append(path.resolve())
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(
            f"ReaSyn inference requires AR and Edit Bridge checkpoints. Missing files: {missing}"
        )
    return paths


# ---------------------------------------------------------------------------
# Molecule helpers
# ---------------------------------------------------------------------------


def _ensure_reasyn_importable(repo: Path) -> None:
    repo_str = str(repo.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _make_mols(smiles_list: Iterable[str]) -> list[Any]:
    """Build ``Molecule`` objects, dropping any that RDKit cannot parse."""
    from reasyn.chem.mol import Molecule

    valid: list[Any] = []
    for raw in smiles_list:
        text = str(raw or "").strip()
        if not text:
            continue
        mol = Molecule(text)
        if mol.is_valid:
            valid.append(mol)
        else:
            print(f"[tasks.small_molecule.core.analog] dropping invalid SMILES: {text!r}", file=sys.stderr)
    return valid


def _canonicalize_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the post-processing that ``run_parallel_sampling_return_smiles`` skips.

    Best-effort: requires ``reasyn.chem.mol`` (and therefore rdkit) on the
    current ``sys.path``. If the in-memory API was run via subprocess and the
    caller process has no rdkit, the data is already canonical (the launcher
    canonicalized in the venv Python) and we return it unchanged.
    """
    try:
        from reasyn.chem.mol import Molecule
    except ImportError:
        LOGGER.debug(
            "reasyn.chem.mol unavailable in calling process; skipping canonicalize"
        )
        return df.reset_index(drop=True)

    out = df.copy()
    out["target"] = out["target"].apply(lambda s: Molecule(str(s)).csmiles)
    out["smiles"] = out["smiles"].apply(lambda s: Molecule(str(s)).csmiles)
    out = out.drop_duplicates(subset=["target", "smiles"])
    if "score" in out.columns:
        out = out.sort_values(["target", "score"], ascending=[True, False])
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Launcher script generation
# ---------------------------------------------------------------------------


def _build_launcher_source(
    *,
    python_bin: str,
    reasyn_repo: str,
    ar_ckpt: str,
    eb_ckpt: str,
    in_path: str,
    out_path: str,
    devices: list[int],
    cuda_visible_devices: str,
    search_width: int,
    exhaustiveness: int,
    num_cycles: int,
    num_editflow_samples: int,
    num_editflow_steps: int,
    time_limit: int,
    num_workers_per_gpu: int,
    task_qsize: int,
    result_qsize: int,
    filter_sim: float,
    add_bb_path: Optional[str],
    mols_to_filter: Optional[str],
    no_exact_break: bool,
) -> str:
    """Build a standalone Python source string that runs ReaSyn in subprocess.

    Note: ReaSyn's in-memory ``run_sampling_one`` /
    ``run_parallel_sampling_return_smiles`` both hardcode ``exact_break=True``
    in their Worker / Sampler construction (only the CLI ``run_parallel_sampling``
    honors it). The ``no_exact_break`` flag is therefore a no-op for the
    in-memory API exposed here; callers that need the ``--no_exact_break``
    behavior must invoke ``scripts/sample.py`` directly.
    """
    _ = no_exact_break  # see docstring; in-memory API cannot honor this flag
    num_gpus = len(devices)
    template = textwrap.dedent(
        """
        # Auto-generated by tasks.small_molecule.core.analog._run_via_subprocess. Do not edit by hand.
        # IMPORTANT: the body is wrapped in `if __name__ == "__main__"` because
        # ``run_parallel_sampling_return_smiles`` uses ``mp.spawn`` which
        # re-executes this script in each worker; without the guard the workers
        # would re-enter the sampling call and ``multiprocessing`` would raise
        # ``RuntimeError: ... before the current process has finished its
        # bootstrapping phase.``
        def _main():
            import os
            import pathlib
            import pickle
            import sys
            import tempfile

            os.environ["CUDA_VISIBLE_DEVICES"] = {cuda_visible_devices!r}

            # Compatibility shim for ReaSyn's fpindex.pkl, which was serialized
            # against an older sklearn that exposed ``ManhattanDistance64`` as a
            # module attribute. Newer sklearn only ships ``ManhattanDistance``
            # and ``ManhattanDistance32``; without this alias, unpickling the
            # fpindex raises AttributeError. The shim fires in BOTH places:
            #
            # 1. Inline at launcher startup: ``run_sampling_one`` pickles the
            #    fpindex in this process, so we patch the module attribute.
            # 2. As a sitecustomize.py on PYTHONPATH: ``mp.set_start_method
            #    ('spawn', force=True)`` makes worker subprocesses re-import
            #    ``reasyn.chem.fpindex`` in a fresh interpreter; Python's
            #    startup hook auto-imports ``sitecustomize`` from any
            #    directory on PYTHONPATH, so the same patch lands in workers.
            try:
                import sklearn.metrics._dist_metrics as _skdist_inline
                if not hasattr(_skdist_inline, "ManhattanDistance64"):
                    _skdist_inline.ManhattanDistance64 = _skdist_inline.ManhattanDistance
            except ImportError:
                pass

            _SITE_DIR = tempfile.mkdtemp(prefix="reasyn_site_")
            pathlib.Path(_SITE_DIR, "sitecustomize.py").write_text(
                "import sklearn.metrics._dist_metrics as _m\\n"
                "if not hasattr(_m, 'ManhattanDistance64'):\\n"
                "    _m.ManhattanDistance64 = _m.ManhattanDistance\\n",
                encoding="utf-8",
            )
            os.environ["PYTHONPATH"] = _SITE_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")

            _REASYN_REPO = pathlib.Path({reasyn_repo!r}).resolve()
            if str(_REASYN_REPO) not in sys.path:
                sys.path.insert(0, str(_REASYN_REPO))

            from reasyn.chem.mol import Molecule
            from reasyn.sampler.parallel import run_sampling_one, run_parallel_sampling_return_smiles

            _IN_PATH = pathlib.Path({in_path!r})
            _OUT_PATH = pathlib.Path({out_path!r})
            _MODEL_PATH = [pathlib.Path({ar_ckpt!r}), pathlib.Path({eb_ckpt!r})]

            _SMILES = _IN_PATH.read_text(encoding="utf-8").splitlines()
            if _SMILES and _SMILES[0].strip().lower() == "smiles":
                _SMILES = _SMILES[1:]
            _MOLS = [Molecule(s) for s in _SMILES if s.strip()]

            _KW = dict(
                model_path=_MODEL_PATH,
                search_width={search_width},
                exhaustiveness={exhaustiveness},
                num_cycles={num_cycles},
                num_editflow_samples={num_editflow_samples},
                num_editflow_steps={num_editflow_steps},
                time_limit={time_limit},
                filter_sim={filter_sim},
            )
            if {add_bb_path!r}:
                _KW["add_bb_path"] = {add_bb_path!r}
            if {mols_to_filter!r}:
                _KW["mols_to_filter"] = {mols_to_filter!r}

            if len(_MOLS) == 1:
                _DF = run_sampling_one(
                    input=_MOLS[0],
                    device="cuda:0",
                    **{{k: v for k, v in _KW.items() if v is not None}},
                )
            else:
                _DF = run_parallel_sampling_return_smiles(
                    input=_MOLS,
                    num_gpus={num_gpus},
                    num_workers_per_gpu={num_workers_per_gpu},
                    task_qsize={task_qsize},
                    result_qsize={result_qsize},
                    **{{k: v for k, v in _KW.items() if v is not None}},
                )

            pickle.dump(_DF, open(_OUT_PATH, "wb"))


        if __name__ == "__main__":
            _main()
        """
    ).strip()
    return template.format(
        cuda_visible_devices=cuda_visible_devices,
        reasyn_repo=reasyn_repo,
        ar_ckpt=ar_ckpt,
        eb_ckpt=eb_ckpt,
        in_path=in_path,
        out_path=out_path,
        devices=devices,
        search_width=search_width,
        exhaustiveness=exhaustiveness,
        num_cycles=num_cycles,
        num_editflow_samples=num_editflow_samples,
        num_editflow_steps=num_editflow_steps,
        time_limit=time_limit,
        num_gpus=num_gpus,
        num_workers_per_gpu=num_workers_per_gpu,
        task_qsize=task_qsize,
        result_qsize=result_qsize,
        filter_sim=filter_sim,
        add_bb_path=add_bb_path,
        mols_to_filter=mols_to_filter,
    )


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def _run_in_process(
    smiles_list: list[str],
    paths: list[Path],
    reasyn_repo: Path,
    gpu_ids: list[int],
    config: ReasynConfig,
) -> pd.DataFrame:
    """Run ReaSyn directly in the calling Python interpreter."""
    _ensure_reasyn_importable(reasyn_repo)
    # Set CUDA_VISIBLE_DEVICES BEFORE ReaSyn / torch initialize their CUDA context.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in gpu_ids)

    from reasyn.sampler.parallel import run_parallel_sampling_return_smiles, run_sampling_one

    mols = _make_mols(smiles_list)
    if not mols:
        return pd.DataFrame(columns=["target", "smiles", "score", "synthesis", "num_steps"])

    common_kwargs: dict[str, Any] = dict(
        model_path=paths,
        search_width=config.search_width,
        exhaustiveness=config.exhaustiveness,
        num_cycles=config.num_cycles,
        num_editflow_samples=config.num_editflow_samples,
        num_editflow_steps=config.num_editflow_steps,
        time_limit=config.time_limit,
        filter_sim=config.filter_sim,
        verbose=config.verbose,
    )
    if config.add_bb_path:
        common_kwargs["add_bb_path"] = config.add_bb_path
    if config.mols_to_filter:
        common_kwargs["mols_to_filter"] = config.mols_to_filter

    if len(mols) == 1:
        return run_sampling_one(
            input=mols[0],
            device="cuda:0",
            **common_kwargs,
        )
    return run_parallel_sampling_return_smiles(
        input=mols,
        num_gpus=len(gpu_ids),
        num_workers_per_gpu=config.num_workers_per_gpu,
        task_qsize=config.task_qsize,
        result_qsize=config.result_qsize,
        **common_kwargs,
    )


def _run_via_subprocess(
    smiles_list: list[str],
    paths: list[Path],
    reasyn_repo: Path,
    python_bin: str,
    config: ReasynConfig,
) -> pd.DataFrame:
    """Run ReaSyn under ``python_bin`` and pick up the pickled DataFrame."""
    temp_root = Path(config.temp_dir) if config.temp_dir else Path(tempfile.gettempdir())
    temp_root.mkdir(parents=True, exist_ok=True)

    pid = os.getpid()
    stamp = int(time.time() * 1000)
    in_path = temp_root / f"reasyn_in_{pid}_{stamp}.smi"
    out_path = temp_root / f"reasyn_out_{pid}_{stamp}.pkl"
    script_path = temp_root / f"reasyn_launcher_{pid}_{stamp}.py"

    in_path.write_text("SMILES\n" + "\n".join(smiles_list), encoding="utf-8")
    script_path.write_text(
        _build_launcher_source(
            python_bin=python_bin,
            reasyn_repo=str(reasyn_repo.resolve()),
            ar_ckpt=str(paths[0]),
            eb_ckpt=str(paths[1]),
            in_path=str(in_path),
            out_path=str(out_path),
            devices=list(config.devices),
            cuda_visible_devices=",".join(str(d) for d in config.devices),
            search_width=config.search_width,
            exhaustiveness=config.exhaustiveness,
            num_cycles=config.num_cycles,
            num_editflow_samples=config.num_editflow_samples,
            num_editflow_steps=config.num_editflow_steps,
            time_limit=config.time_limit,
            num_workers_per_gpu=config.num_workers_per_gpu,
            task_qsize=config.task_qsize,
            result_qsize=config.result_qsize,
            filter_sim=config.filter_sim,
            add_bb_path=config.add_bb_path,
            mols_to_filter=config.mols_to_filter,
            no_exact_break=config.no_exact_break,
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    # Pin the requested physical devices before launch; the launcher also sets
    # this defensively, but having it in the parent env means subprocess.run
    # observers can see the constraint too.
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in config.devices)

    timeout = max(config.time_limit + 10, 60) * max(1, len(smiles_list))
    try:
        proc = _run_launcher_process(
            [python_bin, str(script_path)],
            env=env,
            timeout=timeout,
            cwd=str(reasyn_repo.resolve()),
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()
            tail = tail[-3000:] if tail else f"exit code {proc.returncode}"
            raise RuntimeError(
                f"ReaSyn subprocess failed (rc={proc.returncode}). Tail:\n{tail}"
            )
        if config.verbose and proc.stdout:
            print(proc.stdout, end="")
        try:
            return pickle.loads(out_path.read_bytes())
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"ReaSyn subprocess did not write expected output pickle at {out_path}"
            ) from exc
    finally:
        for tmp_file in (in_path, out_path, script_path):
            try:
                tmp_file.unlink()
            except OSError:
                pass


def _run_launcher_process(
    cmd: list[str],
    *,
    env: dict[str, str],
    timeout: int,
    cwd: str,
) -> subprocess.CompletedProcess[str]:
    popen_kwargs: dict[str, Any] = {}
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    elif hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        **popen_kwargs,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(proc)
        raise RuntimeError(f"ReaSyn subprocess timed out after {timeout} seconds") from exc
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def _terminate_process_group(proc: subprocess.Popen[str]) -> None:
    if proc.returncode is not None:
        return

    if os.name == "posix" and hasattr(os, "killpg"):
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10)
            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
            return
        except OSError:
            pass

    proc.kill()
    proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_analogs(
    input_smiles: Union[str, list[str]],
    config: ReasynConfig,
) -> pd.DataFrame:
    """Generate ReaSyn analogs for ``input_smiles`` and return a DataFrame.

    The DataFrame columns mirror ReaSyn's own ``Sampler.get_dataframe()``
    output: ``target``, ``smiles``, ``score``, ``synthesis``, ``num_steps``,
    plus optional ``scf_sim``, ``pharm2d_sim``, ``rdkit_sim`` for the top
    entries, and ``time`` per target. When ``config.canonicalize`` is True
    (default) the result is canonicalized and deduplicated before being
    returned.
    """
    smiles_list = [input_smiles] if isinstance(input_smiles, str) else list(input_smiles)
    if not smiles_list:
        return pd.DataFrame(columns=["target", "smiles", "score", "synthesis", "num_steps"])

    reasyn_repo = _resolve_reasyn_repo(config.reasyn_repo)
    python_bin = _resolve_python_bin(config.python_bin, reasyn_repo)
    paths = _parse_model_paths(config.model_path, reasyn_repo)
    gpu_ids = _resolve_devices(list(config.devices))

    if python_bin is None or python_bin == sys.executable:
        df = _run_in_process(smiles_list, paths, reasyn_repo, gpu_ids, config)
    else:
        df = _run_via_subprocess(smiles_list, paths, reasyn_repo, python_bin, config)

    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["target", "smiles", "score", "synthesis", "num_steps"])

    if config.canonicalize:
        # Need Molecule on sys.path even in subprocess mode to canonicalize here.
        _ensure_reasyn_importable(reasyn_repo)
        df = _canonicalize_dedup(df)
    return df.reset_index(drop=True)


__all__ = ["ReasynConfig", "generate_analogs"]


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    print(
        "tasks.small_molecule.core.analog exposes ReasynConfig + generate_analogs. "
        "Import this module rather than executing it directly."
    )
    sys.exit(0)
