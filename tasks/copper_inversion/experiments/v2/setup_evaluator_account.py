"""Root-run provisioning of the private evaluator filesystem on the H20 host.

Creates no network service, grants no Docker/sudo rights and publishes no data.
The trusted RPC/execution integration must be qualified separately.
"""
import json
import os
import pwd
import subprocess
from pathlib import Path

ACCOUNT="ldm-copper-eval"
BASE=Path("/Aidata/shared/ldm/copper-v2-evaluator")


def main():
    if os.geteuid()!=0:raise PermissionError("run provisioning as root on the authorized evaluation host")
    try:entry=pwd.getpwnam(ACCOUNT)
    except KeyError:
        subprocess.run(["useradd","--system","--user-group","--no-create-home","--home-dir",str(BASE),
            "--shell","/usr/sbin/nologin",ACCOUNT],check=True)
        entry=pwd.getpwnam(ACCOUNT)
    if entry.pw_shell!="/usr/sbin/nologin" or entry.pw_dir!=str(BASE):raise ValueError("existing account differs; do not modify it")
    if BASE.is_symlink():raise ValueError("evaluator base cannot be a symlink")
    BASE.mkdir(parents=True,exist_ok=True);os.chown(BASE,0,entry.pw_gid);BASE.chmod(0o750)
    for name in ("datasets","runs","inbox"):
        directory=BASE/name
        if directory.is_symlink():raise ValueError("private directory cannot be a symlink")
        directory.mkdir(exist_ok=True);os.chown(directory,entry.pw_uid,entry.pw_gid);directory.chmod(0o700)
    code=BASE/"code"
    if code.is_symlink():raise ValueError("trusted code path cannot be a symlink")
    code.mkdir(exist_ok=True);os.chown(code,0,entry.pw_gid);code.chmod(0o750)
    canary=BASE/"datasets/access-canary.txt"
    if canary.is_symlink():raise ValueError("canary cannot be a symlink")
    canary.write_text("private evaluator permission probe\n");os.chown(canary,entry.pw_uid,entry.pw_gid);canary.chmod(0o400)
    read=lambda user:subprocess.run(["runuser","-u",user,"--","cat",str(canary)],capture_output=True).returncode==0
    allowed,denied=read(ACCOUNT),not read("zsgpu")
    if not allowed or not denied:raise RuntimeError("private account access audit failed")
    report=dict(account=ACCOUNT,uid=entry.pw_uid,base=str(BASE),account_can_read=allowed,
        research_host_user_direct_read_denied=denied,docker_or_sudo_granted=False,
        limitations=["zsgpu is an authorized sudo administrator and can override permissions",
                    "research guests must never receive host credentials or sudo authority",
                    "no RPC or formal dataset isolation qualification yet"])
    (BASE/"access-audit.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=="__main__":main()
