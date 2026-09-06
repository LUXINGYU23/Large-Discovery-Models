from __future__ import annotations

from types import SimpleNamespace

import ldm_tts.harness.container as container


def test_local_identity_includes_the_kvm_group(monkeypatch) -> None:
    monkeypatch.setattr(container.os, "getuid", lambda: 1001, raising=False)
    monkeypatch.setattr(container.os, "getgid", lambda: 1002, raising=False)
    monkeypatch.setattr(
        container.os,
        "stat",
        lambda _: SimpleNamespace(st_gid=108),
    )

    user = container.resolve_container_user(None)

    assert user == "1001:1002"
    assert container.docker_identity_args(user) == (
        "--user",
        "1001:1002",
        "--group-add",
        "108",
    )


def test_remote_docker_does_not_use_local_identity_or_kvm_group(monkeypatch) -> None:
    monkeypatch.setattr(container.os, "getuid", lambda: 1001, raising=False)
    monkeypatch.setattr(container.os, "getgid", lambda: 1002, raising=False)
    docker_host = "tcp://docker.example.test:2376"

    assert container.resolve_container_user(None, docker_host) == ""
    assert container.docker_identity_args("2001:2002", docker_host) == (
        "--user",
        "2001:2002",
    )


def test_identity_works_without_kvm(monkeypatch) -> None:
    def missing_kvm(_):
        raise FileNotFoundError

    monkeypatch.setattr(container.os, "stat", missing_kvm)

    assert container.docker_identity_args("1001:1002") == (
        "--user",
        "1001:1002",
    )
