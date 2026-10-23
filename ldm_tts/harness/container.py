"""Host-side Docker launch configuration for Pi Harness workloads."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path


def resolve_container_user(override: str | None, docker_host: str = "") -> str:
    if override:
        return override
    if docker_host:
        return ""
    getuid = getattr(os, "getuid", None)
    getgid = getattr(os, "getgid", None)
    if getuid is None or getgid is None:
        return ""
    return f"{getuid()}:{getgid()}"


def docker_identity_args(container_user: str, docker_host: str = "") -> tuple[str, ...]:
    if not container_user:
        return ()
    args = ["--user", container_user]
    if docker_host:
        return tuple(args)
    try:
        kvm_group = os.stat("/dev/kvm").st_gid
    except OSError:
        return tuple(args)
    args.extend(("--group-add", str(kvm_group)))
    return tuple(args)


def pi_sidecar_command(
    artifact_root: Path,
    image: str,
    *,
    cache_root: Path | None = None,
    docker_host: str = "",
    container_user: str | None = None,
    environment: Mapping[str, str] | None = None,
    mounts: Sequence[tuple[Path, str, bool]] = (),
) -> list[str]:
    cache = (cache_root or Path.home() / ".cache" / "ldm-gondolin").expanduser().resolve()
    (cache / "runtime-overlays").mkdir(parents=True, exist_ok=True)
    command = ["docker"]
    if docker_host:
        command.extend(("--host", docker_host))
    command.extend(("run", "--rm", "-i"))
    user = resolve_container_user(container_user, docker_host)
    command.extend(docker_identity_args(user, docker_host))
    command.extend(("--device", "/dev/kvm"))
    runtime_environment = {
        "HOME": "/runtime-home",
        "XDG_CACHE_HOME": "/runtime-home/.cache",
        "GONDOLIN_IMAGE_STORE": "/runtime-home/.cache/gondolin/images",
        "GONDOLIN_SESSIONS_DIR": "/runtime-home/.cache/gondolin/sessions",
        "LDM_HARNESS_CACHE_ROOT": "/runtime-home/.cache/gondolin",
        "TMPDIR": "/runtime-home/.cache/gondolin/runtime-overlays",
        **(environment or {}),
    }
    for name, value in sorted(runtime_environment.items()):
        command.extend(("--env", f"{name}={value}"))
    for source, destination, readonly in (
        (artifact_root, "/artifacts", False),
        (cache, "/runtime-home/.cache/gondolin", False),
        *mounts,
    ):
        suffix = ",readonly" if readonly else ""
        command.extend(("--mount", f"type=bind,src={source.resolve()},dst={destination}{suffix}"))
    command.append(image)
    return command


__all__ = ["docker_identity_args", "pi_sidecar_command", "resolve_container_user"]
