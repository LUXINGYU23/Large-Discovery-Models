"""Archive a finished, reconciled run (contracts §10); verify an archive anywhere.

A run is archived only when its engine finished (or recorded a stop reason), the
public and private slot budgets agree, the blind audit passes and every branch
brief rebuilds from its ledger prefix. The public run directory and the private
service and selection directories go into two separate packages. The archive
directory describes itself: `archive-manifest.json` holds each package's hash,
file count and bytes plus the run manifest and protocol, so it can be verified
after it has been moved off the active area. Raw model requests and responses
are archived like every other file; nothing is dropped to save space.

    python -m tasks.copper_inversion.experiments.rsi.archive create --root R --archive-root A \
        --protocol-id P --experiment M1 --run ldm4geo-s101 [--protocol protocol.json] [--private-uid U]
    python -m tasks.copper_inversion.experiments.rsi.archive verify --archive A/P/M1/ldm4geo-s101 [--root R]

The active copy may be deleted only after `verify --root` passes.
"""

import argparse
import getpass
import hashlib
import io
import json
import os
import socket
import stat
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from ldm_tts.engine.run_store import atomic_json_write
from tasks.copper_inversion.experiments.rsi.freeze import layout

SCHEMA = "rsi-archive-v1"
PACKAGES = {"public": "public.tar.gz", "private": "private.tar.gz"}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def conditions(run, service, *, key_file=None, require_requests=True):
    """contracts §10 archive conditions; raises on the first that fails."""
    from tasks.copper_inversion.rsi.audit import reconcile_budget, scan_blind, verify_briefs
    run, service = Path(run), Path(service)
    result = json.loads((run / "result.json").read_text())
    if result.get("status") not in ("completed", "stopped") or (
            result["status"] == "stopped" and not result.get("stop_reason")):
        raise ValueError("run neither completed nor recorded a stop reason")
    if not (service / "training-sealed.json").is_file():
        raise ValueError("training is not sealed")
    blind = scan_blind(run, service, key_file=key_file, require_requests=require_requests)
    return {"engine": {"status": result["status"], "stop_reason": result.get("stop_reason")},
            "budget": reconcile_budget(run, service),
            "blind": {k: blind[k] for k in ("status", "files_checked", "marker_hits", "iou_hits_by_kind", "leaks")},
            "briefs": verify_briefs(run)}


def _entries(sources):
    """(archive name, path) for every file, link and directory, in a fixed order."""
    rows = []
    for prefix, root in sources:
        root = Path(root)
        rows.append((prefix, root))
        for base, dirs, files in os.walk(root, followlinks=False):
            dirs.sort()
            for name in sorted(dirs) + sorted(files):
                path = Path(base) / name
                rows.append((f"{prefix}/{path.relative_to(root)}", path))
    return sorted(rows, key=lambda row: row[0])


def _pack(target, sources):
    """Write a tar.gz and return its listing; each file is hashed from the bytes packed."""
    temporary = target.with_name("." + target.name + ".partial")
    listing = []
    with tarfile.open(temporary, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for name, path in _entries(sources):
            info = archive.gettarinfo(str(path), arcname=name)
            info.uname = info.gname = ""
            row = {"path": name, "mode": stat.S_IMODE(info.mode), "uid": info.uid, "mtime": info.mtime}
            if info.isreg():
                body = path.read_bytes()
                info.size = len(body)
                archive.addfile(info, io.BytesIO(body))
                row.update(type="file", size=len(body), sha256=hashlib.sha256(body).hexdigest())
            elif info.issym():
                archive.addfile(info)
                row.update(type="symlink", target=info.linkname)
            elif info.isdir():
                archive.addfile(info)
                row.update(type="dir")
            else:
                raise ValueError("unsupported file type in run directory: " + name)
            listing.append(row)
    os.replace(temporary, target)
    return listing


def _summary(target, listing):
    files = [row for row in listing if row["type"] == "file"]
    return {"file": target.name, "sha256": _sha(target), "bytes": target.stat().st_size, "files": len(files),
            "content_bytes": sum(row["size"] for row in files), "entries": len(listing)}


def create(root, archive_root, protocol_id, experiment, run_id, *, protocol=None, private_uid=None,
           key_file=None, require_requests=True):
    paths = layout(root, protocol_id, experiment, run_id)
    run, service, selection = paths["run"], paths["service"], paths["run_selection"]
    manifest = json.loads((run / "manifest.json").read_text())
    if (manifest.get("run_id"), manifest.get("experiment"), manifest.get("protocol_id")) != (run_id, experiment, protocol_id):
        raise ValueError("run manifest identity differs from its archive location")
    if not selection.is_dir():
        raise ValueError("archive only a frozen and validated run")
    target = Path(archive_root) / protocol_id / experiment / run_id
    if (target / "archive-manifest.json").exists():
        return verify(target, root=root)
    checked = conditions(run, service, key_file=key_file, require_requests=require_requests)
    target.mkdir(parents=True, exist_ok=True)
    public = _pack(target / PACKAGES["public"], [(run_id, run)])
    private = _pack(target / PACKAGES["private"], [("service", service), ("selection", selection)])
    # Private listings name private files; they stay with the private package.
    lists = {"public": target / "public-files.json", "private": target / "private-files.json"}
    atomic_json_write(lists["public"], public)
    atomic_json_write(lists["private"], private)
    for name in ("private",):
        for path in (target / PACKAGES[name], lists[name]):
            if private_uid is not None:
                os.chown(path, private_uid, private_uid)
            path.chmod(0o600 if private_uid is not None else 0o640)
    protocol_body = Path(protocol).read_bytes() if protocol is not None else None
    record = {
        "schema": SCHEMA,
        "contract": "contracts.md §10",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(), "operator": getpass.getuser(),
        "protocol_id": protocol_id, "experiment": experiment, "run_id": run_id,
        "code_commit": manifest.get("code_commit"),
        "archive_tool_sha256": _sha(__file__),
        "active_paths": {"run": str(run), "service": str(service), "selection": str(selection)},
        "layout": {"public": f"{PACKAGES['public']}: the public run directory as {run_id}/ (contracts §1.2)",
                   "private": f"{PACKAGES['private']}: service/ (evaluator service directory) and "
                              "selection/ (frozen, validated and possibly tested prefixes); evaluator only"},
        "packages": {name: {**_summary(target / PACKAGES[name], rows), "listing": lists[name].name,
                            "listing_sha256": _sha(lists[name])}
                     for name, rows in (("public", public), ("private", private))},
        "run_manifest": manifest,
        "protocol_sha256": None if protocol_body is None else hashlib.sha256(protocol_body).hexdigest(),
        "protocol": None if protocol_body is None else json.loads(protocol_body),
        "conditions": checked,
        "verify": "python -m tasks.copper_inversion.experiments.rsi.archive verify --archive <this directory>",
    }
    atomic_json_write(target / "archive-manifest.json", record)
    (target / "archive-manifest.json").chmod(0o644)
    return verify(target, root=root)


def _listing_of_tar(path):
    rows = {}
    with tarfile.open(path, "r:gz") as archive:
        for info in archive:
            row = {"path": info.name}
            if info.isreg():
                body = archive.extractfile(info).read()
                row.update(type="file", size=len(body), sha256=hashlib.sha256(body).hexdigest())
            elif info.issym():
                row.update(type="symlink", target=info.linkname)
            elif info.isdir():
                row.update(type="dir")
            else:
                row.update(type="other")
            rows[info.name] = row
    return rows


def _comparable(row):
    return {k: row[k] for k in ("path", "type", "size", "sha256", "target") if k in row}


def verify(target, root=None, packages=("public", "private")):
    """Package hashes, listings and contents agree; with root, the active copy agrees too."""
    target = Path(target)
    record = json.loads((target / "archive-manifest.json").read_text())
    if record.get("schema") != SCHEMA or set(record["packages"]) != set(PACKAGES):
        raise ValueError("not an rsi archive manifest")
    report = {"status": "passed", "archive": str(target), "packages": {}}
    for name in packages:
        summary = record["packages"][name]
        package, listing_path = target / summary["file"], target / summary["listing"]
        if _sha(package) != summary["sha256"] or package.stat().st_size != summary["bytes"]:
            raise ValueError(f"{name} package differs from the archive manifest")
        if _sha(listing_path) != summary["listing_sha256"]:
            raise ValueError(f"{name} file listing differs from the archive manifest")
        listing = json.loads(listing_path.read_text())
        files = [row for row in listing if row["type"] == "file"]
        if (len(files), sum(r["size"] for r in files), len(listing)) != (
                summary["files"], summary["content_bytes"], summary["entries"]):
            raise ValueError(f"{name} file listing is incomplete")
        packed = _listing_of_tar(package)
        if sorted(packed) != sorted(row["path"] for row in listing) or any(
                _comparable(packed[row["path"]]) != _comparable(row) for row in listing):
            raise ValueError(f"{name} package contents differ from its listing")
        active = None
        if root is not None:
            paths = layout(root, record["protocol_id"], record["experiment"], record["run_id"])
            sources = ([(record["run_id"], paths["run"])] if name == "public" else
                       [("service", paths["service"]), ("selection", paths["run_selection"])])
            current = {}
            for arcname, path in _entries(sources):
                row = {"path": arcname}
                if path.is_symlink():
                    row.update(type="symlink", target=os.readlink(path))
                elif path.is_dir():
                    row.update(type="dir")
                else:
                    row.update(type="file", size=path.stat().st_size, sha256=_sha(path))
                current[arcname] = row
            if sorted(current) != sorted(packed) or any(current[k] != _comparable(packed[k]) for k in current):
                raise ValueError(f"active {name} copy differs from the archive")
            active = "identical"
        report["packages"][name] = {"sha256": summary["sha256"], "files": summary["files"],
                                    "bytes": summary["bytes"], "active_copy": active}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("create")
    make.add_argument("--root", type=Path, required=True)
    make.add_argument("--archive-root", type=Path, required=True)
    make.add_argument("--protocol-id", required=True)
    make.add_argument("--experiment", required=True)
    make.add_argument("--run", required=True, action="append", dest="runs")
    make.add_argument("--protocol", type=Path)
    make.add_argument("--private-uid", type=int, help="evaluator account owning the private package")
    make.add_argument("--key-file", type=Path, help="also search the public side for this API key")
    check = sub.add_parser("verify")
    check.add_argument("--archive", type=Path, required=True)
    check.add_argument("--root", type=Path, help="active area; compare the active copy too")
    check.add_argument("--public-only", action="store_true", help="for a reader without the private package")
    args = parser.parse_args(argv)
    if args.command == "create":
        for run_id in args.runs:
            print(json.dumps(create(args.root, args.archive_root, args.protocol_id, args.experiment, run_id,
                                    protocol=args.protocol, private_uid=args.private_uid, key_file=args.key_file),
                             sort_keys=True), flush=True)
    else:
        print(json.dumps(verify(args.archive, root=args.root,
                                packages=("public",) if args.public_only else tuple(PACKAGES)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
