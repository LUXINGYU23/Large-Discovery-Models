"""Attest the code and Python environment used by physical Oracle workers."""

from hashlib import sha256
from importlib import metadata, util
from pathlib import Path
import sys

from .protocol import digest

QLIB_FFO_SOURCES = {
    "ffo/utils/utils.py": "6fa58d3c69d03c0c8d27be7e707eb68def968355558b2fbab96debf1a6d43146",
    "ffo/backtest/qlib/single_alpha_backtest.py": "b4bff215a9c648665772285cd95cb5b0fd546bdc4cf05e9d7335f104f320daf4",
}
TASK_SOURCES = (
    "oracle_identity.py", "oracle_worker.py", "oracle_service.py",
    "assay_adapter.py", "assay_contract.py", "qlib_operators.py", "grammar.py",
)


def _sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _backend_sources(package):
    spec = util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        raise ValueError("Oracle backend package is not installed: " + package)
    root = Path(next(iter(spec.submodule_search_locations))).resolve()
    files = {}
    for path in sorted(root.rglob("*")):
        if path.suffix not in {".py", ".so"}:
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Oracle backend source escapes its package: " + str(path))
        files[path.relative_to(root).as_posix()] = _sha(path)
    if not files:
        raise ValueError("Oracle backend package has no verifiable source")
    return files


def oracle_environment_digest(config):
    backend = config["backend"]
    if backend not in {"qlib", "assay"}:
        raise ValueError("unknown Oracle backend")
    upstream = {}
    if backend == "qlib":
        root = Path(config["upstream_root"]).resolve()
        for name, expected in QLIB_FFO_SOURCES.items():
            path = root / name
            if path.is_symlink() or not path.resolve().is_relative_to(root) or _sha(path) != expected:
                raise ValueError("pinned AlphaBench Oracle source changed: " + name)
            upstream[name] = expected
    packages = sorted(
        (dist.metadata["Name"].lower(), dist.version)
        for dist in metadata.distributions()
        if dist.metadata.get("Name")
    )
    task_root = Path(__file__).resolve().parent
    return digest({
        "backend": backend,
        "python": list(sys.version_info[:3]),
        "prefix": str(Path(sys.prefix).resolve()),
        "packages": packages,
        "upstream": upstream,
        "backend_source": _backend_sources(backend),
        "task_source": {name: _sha(task_root / name) for name in TASK_SOURCES},
    })
