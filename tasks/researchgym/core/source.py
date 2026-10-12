"""Pinned ResearchGym source verification and per-candidate workspace assembly."""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from .cases import RESOURCE_ROOT, CaseSpec, render

CONTRACT_PATH = RESOURCE_ROOT / "upstream_contract.json"
CIFAR100_ARCHIVE_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"


def modified_files(root: Path, digests: dict[str, str]) -> list[str]:
    """Workspace files whose content no longer matches what the evaluator wrote."""
    root = Path(root)
    return sorted(relative for relative, expected in digests.items()
                  if not (root / relative).is_file() or (root / relative).is_symlink()
                  or _sha256((root / relative).read_bytes()) != expected)


class SourceMismatchError(ValueError):
    """The supplied checkout is not the pinned ResearchGym source."""


@lru_cache(maxsize=1)
def upstream_contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text())


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class UpstreamCase:
    """One case directory of a checkout, verified file by file against the pin."""

    def __init__(self, upstream_root: Path, case: CaseSpec, *, data_root: Path | None = None) -> None:
        contract = upstream_contract()
        self.case = case
        self.commit = contract["commit"]
        self.source_url = contract["source_url"]
        self.pinned = contract["cases"][case.case_id]
        self.root = Path(upstream_root).expanduser().resolve() / self.pinned["task_path"]
        self.data_root = Path(data_root).expanduser().resolve() if data_root else self.root

    def verify(self) -> dict[str, Any]:
        """Reject missing or modified tracked files; untracked extras are never copied."""
        mismatched, missing = [], []
        for relative, expected in self.pinned["files"].items():
            path = self.root / relative
            if not path.is_file():
                missing.append(relative)
            elif _sha256(path.read_bytes()) != expected:
                mismatched.append(relative)
        if missing or mismatched:
            raise SourceMismatchError(
                f"{self.root} does not match ResearchGym {self.commit[:12]} "
                f"({len(missing)} missing, {len(mismatched)} modified; first: "
                f"{(missing + mismatched)[:3]}). Use a clean checkout of the pinned commit."
            )
        for link in self.case.raw["links"]:
            source = Path(render(link["source"], {"data_root": self.data_root}))
            if not source.exists():
                raise FileNotFoundError(
                    f"{self.case.case_id} requires {link['workspace']} data at {source}; "
                    "see tasks/researchgym/README.md for case data preparation"
                )
        data_artifacts = _verify_case_data_artifacts(self.case, self.data_root)
        return {"source_url": self.source_url, "commit": self.commit,
                "task_path": self.pinned["task_path"], "tree_sha256": self.pinned["tree_sha256"],
                "verified_files": len(self.pinned["files"]), "data_artifacts": data_artifacts}

    def reference_sources(self) -> dict[str, str]:
        """Public baseline code quoted to research sessions, from pinned files only."""
        result = {}
        for relative in self.case.raw["public_context"]["references"]:
            data = (self.root / relative).read_bytes()
            if _sha256(data) != self.pinned["files"][relative]:
                raise SourceMismatchError(f"reference file changed after verification: {relative}")
            result[relative] = data.decode("utf-8")
        return result

    def build_workspace(self, destination: Path, *, slot: str, program: str) -> dict[str, Any]:
        """Copy pinned files, inject the slot, install support files, and write the program."""
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError(f"evaluation workspace already exists: {destination}")
        destination.mkdir(parents=True)
        written = {}
        for relative, expected in self.pinned["files"].items():
            data = (self.root / relative).read_bytes()
            if _sha256(data) != expected:
                raise SourceMismatchError(f"pinned file changed during workspace assembly: {relative}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            written[relative] = expected
        binding = {"slot": slot, "module": self.case.module_name}
        for injection in self.case.raw["injections"]:
            target = destination / injection["path"]
            text = target.read_text(encoding="utf-8")
            if text.count(injection["anchor"]) != 1:
                raise SourceMismatchError(f"injection anchor is not unique in {injection['path']}")
            block = render(injection["block"], binding)
            replacement = injection["anchor"] + block if injection["where"] == "after" else block + injection["anchor"]
            target.write_text(text.replace(injection["anchor"], replacement, 1), encoding="utf-8")
            written[injection["path"]] = _sha256(target.read_bytes())
        for support in self.case.raw["support_files"]:
            target = destination / support["dest"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((RESOURCE_ROOT / "cases" / support["source"]).read_bytes())
            written[support["dest"]] = _sha256(target.read_bytes())
        for link in self.case.raw["links"]:
            source = Path(render(link["source"], {"data_root": self.data_root}))
            target = destination / link["workspace"]
            if target.exists() or target.is_symlink():
                raise SourceMismatchError(f"data link would shadow a pinned path: {link['workspace']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(source, target)
        for relative in self.case.raw["prepare_dirs"]:
            (destination / render(relative, binding)).mkdir(parents=True, exist_ok=True)
        candidate = destination / self.case.candidate_path
        if candidate.exists():
            raise SourceMismatchError(f"candidate path collides with a pinned file: {self.case.candidate_path}")
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text(program, encoding="utf-8")
        written[self.case.candidate_path] = _sha256(program.encode("utf-8"))
        return {"workspace_files": len(self.pinned["files"]), "candidate_path": self.case.candidate_path,
                "candidate_sha256": written[self.case.candidate_path], "file_digests": written}


def _verify_case_data_artifacts(case: CaseSpec, data_root: Path) -> dict[str, dict[str, str]]:
    """Reject hidden downloader inputs before a real evaluator begins.

    The upstream continual-learning loader re-downloads CIFAR-100 when its
    original archive is absent or has the wrong digest, even if an extracted
    directory happens to be present. Treat that archive as a pinned deployment
    input, so an official-grader job never silently fetches data through a
    linked source tree.
    """
    if case.case_id != "continual_learning" or not case.raw.get("links"):
        return {}
    archive = Path(data_root) / "data" / "cifar-100-python.tar.gz"
    if not archive.is_file():
        raise FileNotFoundError(
            f"continual_learning requires the official CIFAR-100 archive at {archive}; "
            "run tasks/researchgym/scripts/prepare_case_data.py cl --data-root DATA_ROOT first"
        )
    digest = _md5(archive)
    if digest != CIFAR100_ARCHIVE_MD5:
        raise SourceMismatchError(
            f"{archive} is not the official CIFAR-100 archive (md5 {digest}, expected {CIFAR100_ARCHIVE_MD5}); "
            "prepare a separate case-data root instead of allowing the evaluator to download it"
        )
    return {"cifar100_archive": {"path": str(archive), "md5": digest}}


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
