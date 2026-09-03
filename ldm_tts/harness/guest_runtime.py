"""Task-local guest-image identities used by persistent Harness sessions."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


_SHA256 = re.compile(r"[a-f0-9]{64}")
_IMAGE_ID = re.compile(r"[a-z][a-z0-9-]*")
_ROOTFS_SIZE = re.compile(r"[1-9][0-9]*[KMGT]")
_TASK_ID = re.compile(r"[a-z][a-z0-9_]*")
_INSTALL_POLICY = "session_overlay"
_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HarnessGuestRuntime:
    """Immutable, task-owned Gondolin guest selected for one campaign."""

    image_ref: str
    recipe_sha256: str
    rootfs_size: str
    install_policy: str

    def __post_init__(self) -> None:
        if re.fullmatch(r"ldm/[a-z][a-z0-9-]*:[a-f0-9]{12}", self.image_ref) is None:
            raise ValueError("harness guest image_ref must be a logical ldm image ref")
        if _SHA256.fullmatch(self.recipe_sha256) is None:
            raise ValueError("harness guest recipe_sha256 must be a lowercase SHA-256 digest")
        if _ROOTFS_SIZE.fullmatch(self.rootfs_size) is None:
            raise ValueError("harness guest rootfs_size must use a positive K, M, G, or T size")
        if self.install_policy != _INSTALL_POLICY:
            raise ValueError("unsupported harness guest install policy")

    def to_dict(self) -> dict[str, str]:
        return {
            "imageRef": self.image_ref,
            "recipeSha256": self.recipe_sha256,
            "rootfsSize": self.rootfs_size,
            "installPolicy": self.install_policy,
        }


def load_harness_guest_runtime(task_id: str, image_directory: Path) -> HarnessGuestRuntime:
    """Load a task recipe and derive its deterministic local image identity."""

    if _TASK_ID.fullmatch(task_id) is None:
        raise ValueError("harness task_id must be a lowercase identifier")
    descriptor_path = image_directory / "guest-image.json"
    try:
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing harness guest descriptor: {descriptor_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid harness guest descriptor: {descriptor_path}") from exc
    if not isinstance(descriptor, dict):
        raise ValueError("harness guest descriptor must be an object")
    expected = {
        "schemaVersion",
        "imageId",
        "rootfsSize",
        "installPolicy",
        "recipeFiles",
        "smokeScript",
    }
    if set(descriptor) != expected:
        raise ValueError("harness guest descriptor has unexpected or missing fields")
    if descriptor["schemaVersion"] != _SCHEMA_VERSION:
        raise ValueError("unsupported harness guest descriptor schema version")
    image_id = _required_string(descriptor["imageId"], "imageId")
    if _IMAGE_ID.fullmatch(image_id) is None:
        raise ValueError("harness guest imageId must be a lowercase hyphenated identifier")
    rootfs_size = _required_string(descriptor["rootfsSize"], "rootfsSize")
    if _ROOTFS_SIZE.fullmatch(rootfs_size) is None:
        raise ValueError("harness guest rootfsSize must use a positive K, M, G, or T size")
    install_policy = _required_string(descriptor["installPolicy"], "installPolicy")
    if install_policy != _INSTALL_POLICY:
        raise ValueError("unsupported harness guest install policy")
    smoke_script = _relative_recipe_path(_required_string(descriptor["smokeScript"], "smokeScript"))
    recipe_files = _recipe_files(descriptor["recipeFiles"], image_directory)
    if smoke_script not in recipe_files:
        raise ValueError("harness guest smokeScript must be listed in recipeFiles")
    recipe_sha256 = _recipe_sha256(
        image_id=image_id,
        rootfs_size=rootfs_size,
        install_policy=install_policy,
        smoke_script=smoke_script,
        recipe_files=recipe_files,
        image_directory=image_directory,
    )
    return HarnessGuestRuntime(
        image_ref=f"ldm/{image_id}:{recipe_sha256[:12]}",
        recipe_sha256=recipe_sha256,
        rootfs_size=rootfs_size,
        install_policy=install_policy,
    )


def _recipe_files(value: Any, image_directory: Path) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        raise ValueError("harness guest recipeFiles must be a non-empty string array")
    normalized = tuple(sorted(_relative_recipe_path(item) for item in value))
    if len(set(normalized)) != len(normalized):
        raise ValueError("harness guest recipeFiles must be unique")
    if "Dockerfile" not in normalized:
        raise ValueError("harness guest recipeFiles must include Dockerfile")
    for relative_path in normalized:
        _recipe_source(image_directory, relative_path)
    return normalized


def _relative_recipe_path(value: str) -> str:
    if not value or "\\" in value:
        raise ValueError("harness guest recipe paths must use non-empty POSIX relative paths")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("harness guest recipe paths must stay inside the image directory")
    return path.as_posix()


def _recipe_sha256(
    *,
    image_id: str,
    rootfs_size: str,
    install_policy: str,
    smoke_script: str,
    recipe_files: tuple[str, ...],
    image_directory: Path,
) -> str:
    files = []
    for relative_path in recipe_files:
        source = _recipe_source(image_directory, relative_path)
        files.append({
            "path": relative_path,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        })
    value = {
        "imageId": image_id,
        "installPolicy": install_policy,
        "recipeFiles": files,
        "rootfsSize": rootfs_size,
        "schemaVersion": _SCHEMA_VERSION,
        "smokeScript": smoke_script,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"harness guest {name} must be a non-empty string")
    return value


def _recipe_source(image_directory: Path, relative_path: str) -> Path:
    root = image_directory.resolve()
    source = image_directory.joinpath(*PurePosixPath(relative_path).parts)
    try:
        source.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError(f"harness guest recipe file escapes image directory: {relative_path}") from exc
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"harness guest recipe file is not a regular file: {relative_path}")
    return source


__all__ = ["HarnessGuestRuntime", "load_harness_guest_runtime"]
