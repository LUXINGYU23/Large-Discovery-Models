import io
import hashlib
import json
import tarfile

import pytest

from tasks.alphabench.core import data


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
