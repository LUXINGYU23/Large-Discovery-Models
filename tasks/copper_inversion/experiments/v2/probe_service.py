"""Development-only two-slot probe of the private evaluator socket."""
import argparse
import json
import socket
from pathlib import Path


def request(path,payload):
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(180)
        connection.connect(str(path))
        connection.sendall(json.dumps(payload).encode()+b"\n")
        with connection.makefile("rb") as stream:return json.loads(stream.readline(1024*1024))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--socket",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);a=parser.parse_args()
    info=request(a.socket,{"operation":"info"});assert info["status"]=="ready" and info["used_slots"]==0
    rejected=request(a.socket,{"operation":"evaluate","source":"x","manifest":"/truth/truth.npy"})
    assert rejected=={"status":"rejected_request"}
    source="""import os
from tasks.copper_inversion.v2.baselines import initialize
def solve(problem,budget,seed):
    for path in ('/truth/truth.npy','/var/run/docker.sock','/Aidata/shared/ldm/copper-v2-evaluator/datasets/access-canary.txt'):
        assert not os.path.exists(path), 'unexpected privileged mount'
    return initialize(problem)
"""
    first=request(a.socket,{"operation":"evaluate","source":source,"contract_digest":info["contract_digest"]})
    if first.get("status")!="succeeded":raise RuntimeError("private evaluation failed: "+str(first))
    replay=request(a.socket,{"operation":"evaluate","source":source,"contract_digest":info["contract_digest"]})
    assert replay==first and request(a.socket,{"operation":"info"})["used_slots"]==1
    illegal="import numpy as np\ndef solve(problem,budget,seed):\n    return np.ones(problem['shape'])\n"
    invalid=request(a.socket,{"operation":"evaluate","source":illegal,"contract_digest":info["contract_digest"]})
    assert invalid["status"]=="invalid_interface"
    exhausted=request(a.socket,{"operation":"evaluate","source":source+"# third source\n","contract_digest":info["contract_digest"]})
    assert exhausted=={"status":"budget_exhausted"}
    final=request(a.socket,{"operation":"info"});assert final["used_slots"]==2
    report=dict(status="passed",contract_digest=info["contract_digest"],evaluation=first,
        replay_identical=True,private_mounts_absent=True,request_path_override_rejected=True,
        invalid_output_rejected=True,feedback_cap_enforced=True,final_used_slots=2,
        limitations="Development socket probe only; no guest attack suite or formal qualification")
    with a.output.open("x") as f:json.dump(report,f,indent=2)
    print(json.dumps(report))


if __name__=="__main__":main()
