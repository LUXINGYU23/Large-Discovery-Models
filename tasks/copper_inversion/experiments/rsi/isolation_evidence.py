"""Assemble scoped isolation evidence from existing probes and live host checks (root only)."""
import argparse
from datetime import datetime, timezone
import grp
import hashlib
import json
import os
from pathlib import Path
import pwd
import stat
import subprocess

SCOPE = ("Account separation, guest isolation probe, RPC whitelist probe, and blind-scan "
         "positive control; does not cover an adversarial guest attack suite.")


def assemble(*, guest, audit, rpc, positive, private_roots, release_file, socket_dirs, client, client_group, operator, root):
    if os.geteuid() != 0:
        raise PermissionError("run as root")
    checks = {}

    def record(name, passed, reason, path=None):
        entry = {"status": "passed" if passed else "failed", "reason": "" if passed else reason}
        if path is not None:
            p = Path(path).absolute()
            entry.update(evidence_path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None)
        checks[name] = entry

    def evidence(name, path, predicate):
        try:
            data = json.loads(Path(path).read_text())
            ok = predicate(data)
            record(name, ok, "required evidence fields did not pass", path)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            record(name, False, str(exc), path)

    evidence("guest_probe", guest, lambda d: d.get("status") == "passed" and all(d.get(k) is True for k in
        ("own_root_paths_absent", "sibling_paths_absent", "private_paths_absent", "host_credentials_absent", "http_egress_probe_failed"))
        and isinstance(d.get("read_tool_escape_denied"), dict) and len(d["read_tool_escape_denied"]) == 5
        and all(v is True for v in d["read_tool_escape_denied"].values()))
    evidence("access_audit", audit, lambda d: d.get("uid") == 991 and d.get("account_can_read") is True and
        d.get("research_host_user_direct_read_denied") is True and d.get("account") == pwd.getpwuid(991).pw_name)
    evidence("rpc_probe", rpc, lambda d: d.get("status") == "passed" and all(d.get(k) is True for k in
        ("replay_identical", "private_mounts_absent", "request_path_override_rejected", "invalid_output_rejected", "feedback_cap_enforced"))
        and d.get("final_used_slots") == 2)
    evidence("positive_control", positive, lambda d: d.get("status") == "control_detected" and
        d.get("marker_hits", {}).get("sentinel", 0) > 0 and d.get("marker_hits", {}).get("private_value", 0) > 0 and
        d.get("iou_hits_by_kind", {}).get("unexplained", 0) > 0)
    try:
        evaluator, user = pwd.getpwuid(991), pwd.getpwnam(client)
        record("accounts", user.pw_uid != 991, "client and evaluator must be distinct")
    except KeyError as exc:
        evaluator = user = None
        record("accounts", False, str(exc))
    private_base = (Path(root).resolve() / "private")
    record("coverage", bool(private_roots and socket_dirs and operator.strip()), "private roots, socket dirs and operator are required")
    for index, path in enumerate(private_roots):
        try:
            p = Path(path); st = p.lstat()
            ok = p.resolve().is_relative_to(private_base) and stat.S_ISDIR(st.st_mode) and st.st_uid == 991 and stat.S_IMODE(st.st_mode) == 0o700
            record(f"private_root_{index}", ok, "expected directory owned by uid 991 with mode 0700", p)
        except OSError as exc:
            record(f"private_root_{index}", False, str(exc), path)
    try:
        p = Path(release_file)
        if not p.is_file() or not p.resolve().is_relative_to(private_base):
            raise ValueError("private release file missing")
        result = subprocess.run(["runuser", "-u", client, "--", "test", "-r", str(p)], capture_output=True)
        record("client_read_denied", result.returncode == 1 and not result.stderr, "client read was not demonstrably denied", p)
    except (OSError, ValueError) as exc:
        record("client_read_denied", False, str(exc), release_file)
    try:
        group = grp.getgrnam(client_group)
    except KeyError:
        group = None
    for index, path in enumerate(socket_dirs):
        try:
            p = Path(path); st = p.lstat()
            ok = group is not None and stat.S_ISDIR(st.st_mode) and st.st_uid == 991 and st.st_gid == group.gr_gid and stat.S_IMODE(st.st_mode) == 0o750
            record(f"socket_dir_{index}", ok, "expected directory 0750 owned by evaluator:client group", p)
        except OSError as exc:
            record(f"socket_dir_{index}", False, str(exc), path)
    return {"schema": "rsi-isolation-evidence-v1", "status": "passed" if all(c["status"] == "passed" for c in checks.values()) else "failed",
            "operator": operator, "timestamp": datetime.now(timezone.utc).isoformat(), "scope": SCOPE, "checks": checks}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("guest", "audit", "rpc", "positive", "release-file", "client", "client-group", "operator", "root", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--private-root", action="append", required=True, dest="private_roots")
    p.add_argument("--socket-dir", action="append", required=True, dest="socket_dirs")
    a = vars(p.parse_args(argv)); output = Path(a.pop("output"))
    result = assemble(**a)
    with output.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True); f.write("\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
