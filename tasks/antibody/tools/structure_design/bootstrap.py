"""Install the pinned Linux runtimes and public assets in a project workspace.

Usage: python -m tasks.antibody.tools.structure_design.bootstrap --root /path/to/workspace
No system packages, drivers, existing AntBO environment, or user .env are changed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from urllib.request import Request, urlopen

MPNN_COMMIT = "8907e6671bfbfc92303b5f79c4b5e6ce47cdef57"
RFD3_URL = "https://files.ipd.uw.edu/pub/rfd3/rfd3_foundry_2025_12_01_remapped.ckpt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def download(url: str, path: Path) -> dict:
    """Atomic download; existing assets require their matching integrity record."""
    manifest = path.with_name(path.name + ".source.json")
    if path.exists():
        if not manifest.exists():
            raise RuntimeError(f"Unverified existing asset, refusing overwrite: {path}")
        record = json.loads(manifest.read_text())
        if record["url"] != url or record["sha256"] != sha256(path):
            raise RuntimeError(f"Asset checksum/source mismatch: {path}")
        return record
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    print(f"Downloading {url} -> {path}", flush=True)
    with urlopen(Request(url, method="HEAD"), timeout=60) as response:
        expected = response.headers.get("Content-Length")
        ranges = response.headers.get("Accept-Ranges", "").lower() == "bytes"
    if expected and int(expected) > 64 * 1024 * 1024 and ranges:
        size = int(expected)
        parts = path.with_name(path.name + ".parts")
        parts.mkdir(exist_ok=True)
        chunk_size = (size + 7) // 8
        if partial.exists() and not any(parts.iterdir()):
            with partial.open("rb") as previous:
                remaining = partial.stat().st_size
                for index in range(8):
                    take = min(remaining, chunk_size, size - index * chunk_size)
                    if take <= 0:
                        break
                    with (parts / str(index)).open("wb") as piece:
                        left = take
                        while left:
                            block = previous.read(min(left, 8 * 1024 * 1024))
                            if not block:
                                raise RuntimeError("Unexpected truncated prior download")
                            piece.write(block)
                            left -= len(block)
                    remaining -= take

        def fetch(index):
            start, end = index * chunk_size, min(size, (index + 1) * chunk_size) - 1
            piece = parts / str(index)
            present = piece.stat().st_size if piece.exists() else 0
            if present > end - start + 1:
                raise RuntimeError(f"Invalid partial range file: {piece}")
            if present == end - start + 1:
                return
            request = Request(url, headers={"Range": f"bytes={start + present}-{end}"})
            with urlopen(request, timeout=120) as response, piece.open("ab") as dest:
                if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start + present}-{end}/{size}":
                    raise RuntimeError(f"Server did not honor requested range: {url}")
                shutil.copyfileobj(response, dest, length=1024 * 1024)
            if piece.stat().st_size != end - start + 1:
                raise RuntimeError(f"Incomplete range: {piece}")

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(fetch, range(8)))
        with partial.open("wb") as dest:
            for index in range(8):
                with (parts / str(index)).open("rb") as piece:
                    shutil.copyfileobj(piece, dest, length=8 * 1024 * 1024)
        for piece in parts.iterdir():
            piece.unlink()
        parts.rmdir()
    else:
        with urlopen(url, timeout=120) as response, partial.open("wb") as dest:
            shutil.copyfileobj(response, dest, length=8 * 1024 * 1024)
    if expected is not None and partial.stat().st_size != int(expected):
        raise RuntimeError(f"Incomplete download: {path}")
    digest = sha256(partial)
    record = {"url": url, "downloaded_at": datetime.now(timezone.utc).isoformat(),
              "sha256": digest, "bytes": partial.stat().st_size,
              "integrity": "locally recorded SHA256; upstream supplies no trusted digest"}
    # Sidecar first: an interrupted promotion can be retried without trusting a partial file.
    save_json(manifest, record)
    partial.replace(path)
    return record


def export_requirements(python: Path) -> str:
    """Return portable exact versions, without PEP 610 machine-local wheel URLs."""
    query = ("import importlib.metadata as m; "
             "print('\\n'.join(sorted(d.metadata['Name']+'=='+d.version for d in m.distributions() "
             "if d.metadata['Name'].lower() != 'pip')))")
    return subprocess.check_output([str(python), "-c", query], text=True)


def install_env(root: Path, name: str, requirements: list[str], wheels: list[Path]) -> None:
    env_path = root / "env" / name
    python = env_path / "bin/python"
    if not python.exists():
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(env_path)], check=True)
    # System pip's --python supports a venv without ensurepip (Ubuntu minimal images).
    pip = [sys.executable, "-m", "pip", "--python", str(python)]
    subprocess.run(pip + ["install", "pip==25.0.1"], check=True)
    # An equal-version index candidate can win over --find-links. Install the
    # verified large wheels by exact pathname, then resolve their small deps.
    subprocess.run([str(python), "-m", "pip", "install", "--no-deps",
                    *[str(path) for path in wheels]], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "torch==2.6.0",
                    ], check=True)
    lock = Path(__file__).parent / "requirements" / (name + ".lock.txt")
    install_args = ["--requirement", str(lock)] if lock.exists() else requirements
    subprocess.run([str(python), "-m", "pip", "install", *install_args], check=True)
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    frozen = export_requirements(python)
    (env_path / "requirements.lock.txt").write_text(frozen)


def prefetch_cuda_wheels(root: Path):
    """PyPI torch 2.6 Linux wheels use CUDA 12.4; cache large wheels with ranged IO."""
    from pip._vendor.packaging.tags import sys_tags
    from pip._vendor.packaging.utils import parse_wheel_filename
    tags = set(sys_tags())
    with urlopen("https://pypi.org/pypi/torch/2.6.0/json", timeout=30) as stream:
        torch = json.load(stream)
    packages = [("torch", "2.6.0"), ("triton", "3.2.0")]
    import re
    for item in torch["info"]["requires_dist"]:
        match = re.match(r"(nvidia-[\w-]+)==([\d.]+)", item)
        if match:
            packages.append(match.groups())
    paths = []
    for name, version in packages:
        with urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=30) as stream:
            info = json.load(stream)
        compatible = [w for w in info["urls"] if w["filename"].endswith(".whl") and
                      set(parse_wheel_filename(w["filename"])[3]) & tags]
        wheel = compatible[0]
        path = root / "cache/wheels" / wheel["filename"]
        record = download(wheel["url"], path)
        if record["sha256"] != wheel["digests"]["sha256"]:
            raise RuntimeError(f"Wheel differs from published PyPI SHA256: {path}")
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=["all", "env", "assets"], default="all")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12) or sys.platform != "linux":
        parser.error("bootstrap requires Linux Python 3.12")
    root = args.root.resolve()
    for directory in ["env", "cache/pip", "cache/torch", "cache/atomworks", "logs", "tmp", "runs"]:
        (root / directory).mkdir(parents=True, exist_ok=True)
    if not (root / "archive").is_dir():
        parser.error("root/archive must point to an existing project archive")
    os.environ.update(PIP_CACHE_DIR=str(root / "cache/pip"),
                      XDG_CACHE_HOME=str(root / "cache"), TORCH_HOME=str(root / "cache/torch"),
                      TMPDIR=str(root / "tmp"))
    if args.stage in ("all", "assets"):
        download(RFD3_URL, root / "archive/models/rfd3/2025-12-01/rfd3.ckpt")
        source = root / "env/sources/ProteinMPNN"
        files = ["protein_mpnn_run.py", "protein_mpnn_utils.py", "helper_scripts/parse_multiple_chains.py", "LICENSE"]
        if not (source / "source-version.json").exists():
            for relative in files:
                download(f"https://raw.githubusercontent.com/dauparas/ProteinMPNN/{MPNN_COMMIT}/{relative}", source / relative)
            save_json(source / "source-version.json", {"commit": MPNN_COMMIT,
                      "files": {p: sha256(source / p) for p in files}})
        version = json.loads((source / "source-version.json").read_text())
        if version["commit"] != MPNN_COMMIT:
            raise RuntimeError("Existing ProteinMPNN source is a different version")
        download(f"https://raw.githubusercontent.com/dauparas/ProteinMPNN/{MPNN_COMMIT}/vanilla_model_weights/v_48_020.pt",
                 root / f"archive/models/proteinmpnn/{MPNN_COMMIT}/v_48_020.pt")
        download(f"https://raw.githubusercontent.com/dauparas/ProteinMPNN/{MPNN_COMMIT}/soluble_model_weights/v_48_020.pt",
                 root / f"archive/models/proteinmpnn/{MPNN_COMMIT}/soluble/v_48_020.pt")
        download("https://files.rcsb.org/download/1N8Z.pdb",
                 root / "archive/data/structure_design/1N8Z.pdb")
    if args.stage in ("all", "env"):
        wheels = prefetch_cuda_wheels(root)
        install_env(root, "rfd3", ["torch==2.6.0", "numpy==1.26.4", "rc-foundry[rfd3]==0.2.0",
                                   "atomworks[ml]==2.1.1"], wheels)
        install_env(root, "proteinmpnn", ["torch==2.6.0", "numpy==1.26.4"], wheels)
    print(json.dumps({"status": "success", "stage": args.stage, "root": str(root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
