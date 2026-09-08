"""Container identity handling for Harness workloads."""

from __future__ import annotations

import os


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


__all__ = ["docker_identity_args", "resolve_container_user"]
