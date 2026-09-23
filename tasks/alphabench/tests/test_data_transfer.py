import io
import hashlib
import json
import subprocess
import sys
import tarfile

import pytest

from tasks.alphabench.core import data
from tasks.alphabench.core.protocol import T3Protocol


def test_verified_offline_transfer_is_idempotent_and_excludes_temporary_files(tmp_path, monkeypatch):
    source, target = tmp_path / "local", tmp_path / "server"
    raw = source / "raw"
    raw.mkdir(parents=True)
    (raw / "prices.json").write_text('{"price": 1}')
    (raw / "prices.json.part").write_text("incomplete")
    bundle = tmp_path / "bundle.tar.gz"
    packed = data.pack_bundle(source, bundle)
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("offline import attempted network access"))
    assert data.install_bundle(bundle, target, packed["sha256"])["installed_files"] == 1
    assert (target / "raw/prices.json").read_bytes() == (raw / "prices.json").read_bytes()
    assert not (target / "raw/prices.json.part").exists()
    assert data.install_bundle(bundle, target, packed["sha256"])["installed_files"] == 1
    (target / "raw/prices.json").write_text("different")
    with pytest.raises(ValueError, match="destination differs"):
        data.install_bundle(bundle, target, packed["sha256"])


def test_corrupt_member_prevents_all_install_writes(tmp_path):
    bundle = tmp_path / "corrupt.tar.gz"
    manifest = {"source_manifest_digest": data.digest(data.SOURCES), "files": {"raw/good": "0" * 64}}
    with tarfile.open(bundle, "w:gz") as archive:
        for name, body in (("bundle-manifest.json", json.dumps(manifest).encode()), ("raw/good", b"corrupt")):
            info = tarfile.TarInfo(name); info.size = len(body)
            archive.addfile(info, io.BytesIO(body))
    target = tmp_path / "server"
    with pytest.raises(ValueError, match="member hash mismatch"):
        data.install_bundle(bundle, target, data.sha256(bundle))
    assert not target.exists()


def test_failed_sql_rows_are_not_reusable_data(tmp_path):
    query = "SELECT * FROM prices AS OF 'snapshot'"
    path = tmp_path / (data.digest(query) + ".json")
    path.write_text(json.dumps({"query_execution_status": "Error", "sql_query": query,
                               "query_execution_message": "timeout", "rows": [{"partial": 1}]}))
    with pytest.raises(ValueError, match="incomplete SQL"):
        data.sql_page(tmp_path, {"url": "https://example.invalid"}, query, offline=True)
    assert not path.exists()
    assert len(list(tmp_path.glob("*.failed-*.json"))) == 1


def test_missing_offline_page_never_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "download", lambda *a, **kw: pytest.fail("offline verification attempted a download"))
    with pytest.raises(FileNotFoundError):
        data.sql_page(tmp_path, data.SOURCES["us"], "SELECT 1", offline=True)


def test_truncated_http_body_is_retried_before_publishing_download(tmp_path, monkeypatch):
    calls = []
    def response(*args, **kwargs):
        calls.append(1)
        stream = io.BytesIO(b"cut" if len(calls) == 1 else b"complete")
        stream.headers = {"Content-Length": "8"}
        return stream
    monkeypatch.setattr(data.urllib.request, "urlopen", response)
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    target = tmp_path / "prices.json"
    data.download("https://example.invalid/prices", target, hashlib.sha256(b"complete").hexdigest())
    assert target.read_bytes() == b"complete"
    assert len(calls) == 2


@pytest.mark.parametrize("backend,market", [("qlib", "csi300"), ("assay", "nasdaq100")])
@pytest.mark.parametrize("qualification,data_policy", [("qualified", "qualified_only"),
                                                      ("blocked", "partial_comparison")])
def test_real_run_rejects_changed_data_file_before_creating_run(tmp_path, backend, market,
                                                                 qualification, data_policy):
    if sys.platform == "win32":
        pytest.skip("Oracle service process requires POSIX fcntl")
    from tasks.alphabench.core.oracle_service import OracleService

    root = tmp_path / "market"
    root.mkdir()
    manifest = {"backend": backend, "market": market, "qualification": qualification,
                "source": "frozen-fixture", "archive_sha256": "a" * 64,
                "start": "2015-01-01", "end": "2025-02-01", "benchmark": "INDEX",
                "adjustment": "split", "fields": ["close"], "historical_universe": True}
    if qualification == "blocked":
        manifest["issues"] = [{"code": "unresolved_price_gaps", "symbols": 1}]
    if backend == "qlib":
        files = {}
        for relative in ("calendars/day.txt", f"instruments/{market}.txt",
                         "instruments/all.txt", "features/stock/close.day.bin"):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"DATA")
            files[relative] = data.sha256(target)
        manifest.update(data_root=str(root), files_sha256=data.digest({"features/stock/close.day.bin": files["features/stock/close.day.bin"]}),
                        calendar_sha256=files["calendars/day.txt"],
                        universe_sha256=files[f"instruments/{market}.txt"],
                        all_instruments_sha256=files["instruments/all.txt"])
        manifest_path = tmp_path / f"qlib_{market}.json"
        manifest_path.with_suffix(".files.json").write_text(json.dumps({
            "features/stock/close.day.bin": files["features/stock/close.day.bin"]}))
        altered = root / "features/stock/close.day.bin"
    else:
        assets = {}
        for name in ("calendar", "prices", "events", "membership", "groups", "execution", "benchmark"):
            target = root / (name + ".parquet")
            target.write_bytes(b"DATA")
            assets[name] = {"path": target.relative_to(tmp_path).as_posix(), "sha256": data.sha256(target)}
        manifest.update(assets=assets,
                        files_sha256=data.digest({record["path"]: record["sha256"] for record in assets.values()}),
                        calendar_sha256=assets["calendar"]["sha256"],
                        universe_sha256=assets["membership"]["sha256"])
        manifest_path = tmp_path / f"assay_{market}.json"
        altered = root / "prices.parquet"
    manifest_path.write_text(json.dumps(manifest))
    protocol = T3Protocol(backend=backend, market=market, data_digest=data.digest(manifest),
                          data_policy=data_policy)
    if qualification == "blocked":
        assert protocol.identity != T3Protocol(backend=backend, market=market,
                                               data_digest=protocol.data_digest).identity
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol.to_dict()))
    assert data.verify_data_manifest(manifest_path, protocol) == manifest
    if qualification == "blocked":
        with pytest.raises(ValueError, match="qualification is incomplete"):
            data.verify_data_manifest(manifest_path, T3Protocol(
                backend=backend, market=market, data_digest=protocol.data_digest))
    config = {"backend": backend, "market": market, "data_digest": protocol.data_digest,
              "environment_digest": protocol.environment_digest, "data_manifest": str(manifest_path)}
    if backend == "qlib":
        config.update(data_root=str(root), benchmark="INDEX", upstream_root=str(tmp_path))
    config_path = tmp_path / "oracle-config.json"
    config_path.write_text(json.dumps(config))
    service = OracleService(config_path, tmp_path / "oracle")
    assert service.health()["data_digest"] == protocol.data_digest
    if backend == "qlib":
        config_path.write_text(json.dumps(config | {"data_root": str(tmp_path)}))
        with pytest.raises(ValueError, match="data root or benchmark"):
            OracleService(config_path, tmp_path / "oracle")
        config_path.write_text(json.dumps(config | {"upstream_root": str(root)}))
        drift = service.execute({"protocol": protocol.to_dict(), "request_id": "b" * 64,
                                 "job_permits": 1, "operation": "check", "expression": "$close"})
        assert drift["pause_status"] == "paused_data_integrity"
        config_path.write_text(json.dumps(config))

    altered.write_bytes(b"EDIT")
    if backend == "qlib":
        request_path = tmp_path / "worker-request.json"
        request_path.write_text(json.dumps({"protocol": protocol.to_dict(), "request_id": "a" * 64,
                                            "operation": "check", "expression": "$close"}))
        output = tmp_path / "worker-response.json"
        worker = subprocess.run([sys.executable, "-m", "tasks.alphabench.core.oracle_worker",
                                 str(request_path), str(config_path), str(output),
                                 "--config-digest", service.config_digest],
                                capture_output=True, text=True, timeout=30)
        assert worker.returncode == 0, worker.stderr
        assert json.loads(output.read_text())["pause_status"] == "paused_data_integrity"
    run_dir = tmp_path / "run"
    attempt = subprocess.run([sys.executable, "-m", "tasks.alphabench.ldm_task.procedure",
                              "--protocol-file", str(protocol_path), "--data-manifest", str(manifest_path),
                              "--out-dir", str(run_dir)], capture_output=True, text=True, timeout=30)
    assert attempt.returncode != 0 and "data asset content mismatch" in attempt.stderr
    assert not run_dir.exists()
    run_dir.mkdir()
    sentinel = run_dir / "existing-receipt"
    sentinel.write_text("paid")
    resumed = subprocess.run([sys.executable, "-m", "tasks.alphabench.ldm_task.procedure",
                              "--protocol-file", str(protocol_path), "--data-manifest", str(manifest_path),
                              "--resume-run", str(run_dir)], capture_output=True, text=True, timeout=30)
    assert resumed.returncode != 0 and "data asset content mismatch" in resumed.stderr
    assert sentinel.read_text() == "paid" and list(run_dir.iterdir()) == [sentinel]
    if backend == "qlib":
        altered.write_bytes(b"DATA")
        manifest_path.with_suffix(".files.json").write_text(json.dumps({
            "features/stock/close.day.bin": "0" * 64}))
        with pytest.raises(ValueError, match="file inventory differs"):
            data.verify_data_manifest(manifest_path, protocol)
        manifest_path.with_suffix(".files.json").write_text(json.dumps({
            "features/stock/close.day.bin": data.sha256(altered)}))
        alternate = root / "alternate.bin"
        alternate.write_bytes(b"DATA")
        altered.unlink()
        altered.symlink_to(alternate)
        with pytest.raises(ValueError, match="data asset content mismatch"):
            data.verify_data_manifest(manifest_path, protocol)
