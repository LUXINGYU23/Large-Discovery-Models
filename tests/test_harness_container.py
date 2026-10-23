from __future__ import annotations

from types import SimpleNamespace

import pytest

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


def test_pi_sidecar_keeps_runtime_mounts_separate_from_task_resources(tmp_path):
    artifacts = tmp_path / "sample artifacts"
    cache = tmp_path / "cache"
    resources = tmp_path / "public resources"
    socket = tmp_path / "host socket"
    command = container.pi_sidecar_command(
        artifacts, "pi-fixture:1", cache_root=cache,
        environment={"LDM_TASK_CONTEXT": "/artifacts/context.json"},
        mounts=((resources, "/resources", True), (socket, "/task-host", False)),
    )
    mounts = [command[i + 1] for i, arg in enumerate(command) if arg == "--mount"]
    assert mounts == [
        f"type=bind,src={artifacts},dst=/artifacts",
        f"type=bind,src={cache},dst=/runtime-home/.cache/gondolin",
        f"type=bind,src={resources},dst=/resources,readonly",
        f"type=bind,src={socket},dst=/task-host",
    ]
    env = dict(command[i + 1].split("=", 1) for i, arg in enumerate(command) if arg == "--env")
    assert env == {
        "HOME": "/runtime-home",
        "XDG_CACHE_HOME": "/runtime-home/.cache",
        "GONDOLIN_IMAGE_STORE": "/runtime-home/.cache/gondolin/images",
        "GONDOLIN_SESSIONS_DIR": "/runtime-home/.cache/gondolin/sessions",
        "LDM_HARNESS_CACHE_ROOT": "/runtime-home/.cache/gondolin",
        "TMPDIR": "/runtime-home/.cache/gondolin/runtime-overlays",
        "LDM_TASK_CONTEXT": "/artifacts/context.json",
    }
    assert (cache / "runtime-overlays").is_dir()
    assert command[:4] == ["docker", "run", "--rm", "-i"]
    assert command[command.index("--device") + 1] == "/dev/kvm"
    assert command[-1] == "pi-fixture:1"


@pytest.mark.parametrize("user", [None, "2001:2002"])
def test_pi_sidecar_remote_launch_uses_remote_identity(tmp_path, user):
    host = "tcp://docker.example.test:2376"
    command = container.pi_sidecar_command(
        tmp_path / "artifacts", "pi-fixture:1", cache_root=tmp_path / "cache",
        docker_host=host, container_user=user,
    )
    assert command[:6] == ["docker", "--host", host, "run", "--rm", "-i"]
    assert "--group-add" not in command
    assert ("--user" in command) is (user is not None)
    if user is not None:
        assert command[command.index("--user") + 1] == user
