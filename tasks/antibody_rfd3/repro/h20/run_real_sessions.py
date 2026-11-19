"""Run the two real-oracle research sessions using the deployed H20 models."""
import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import time
import urllib.request

from qwen_responses_bridge import start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/Aidata/shared/ldm/antibody-poc"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cpus = ",".join(map(str, sorted(os.sched_getaffinity(0))[:2]))
    usage = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu),
                                    "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
    if int(usage.strip()) < 55000:
        raise RuntimeError("Need at least 55 GB free for Qwen and one design worker")
    bridge = "172.17.0.1"
    with socket.socket() as sock:
        sock.bind((bridge, 0))
        port = sock.getsockname()[1]
    url = f"http://{bridge}:{port}/v1"
    key = secrets.token_hex(32)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", VLLM_API_KEY=key,
               LLM_API_KEY=key, LLM_MODEL_NAME="Qwen3-8B", HF_HUB_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", NO_PROXY=bridge+",localhost,127.0.0.1",
               no_proxy=bridge+",localhost,127.0.0.1", LDM_GONDOLIN_CACHE=str(root/"cache/gondolin"),
               LDM_PI_IMAGE="ldm-copper-pi:main-3580273b", VLLM_NO_USAGE_STATS="1",
               VLLM_WORKER_MULTIPROC_METHOD="spawn", VLLM_CACHE_ROOT=str(root/"cache/vllm"))
    command = ["taskset", "-c", cpus, "/home/zsgpu/vllm-env/bin/vllm", "serve",
               "/Aidata/shared/rsi-models/Qwen3-8B", "--host", bridge, "--port", str(port),
               "--served-model-name", "Qwen3-8B", "--max-model-len", "16384",
               "--max-num-seqs", "1", "--max-num-batched-tokens", "1024",
               "--gpu-memory-utilization", "0.27", "--enforce-eager",
               "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
               "--reasoning-parser", "qwen3", "--generation-config", "vllm"]
    results = {}
    listener = None
    with (output/"model.log").open("w") as log:
        server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            for _ in range(150):
                if server.poll() is not None:
                    raise RuntimeError("vLLM exited during startup")
                try:
                    req = urllib.request.Request(url+"/models", headers={"Authorization":"Bearer "+key})
                    with opener.open(req, timeout=3) as response:
                        print("Live model", json.load(response)["data"][0]["id"], flush=True)
                    break
                except OSError:
                    time.sleep(2)
            else:
                raise RuntimeError("Model startup timed out")
            listener = start(bridge, key, url)
            env["LLM_BASE_URL"] = f"http://{bridge}:{listener.server_port}/v1"
            for target in ("her2", "pd1"):
                command = ["taskset", "-c", cpus, str(root/"env/antibody_campaign/bin/python"),
                           "scripts/run_ldm_tts.py", f"config/antibody_rfd3/{target}_real.yaml",
                           "--set", "args.share-gpu=true", "--set", f"args.gpu={args.gpu}",
                           "--set", "args.out-dir="+str(output/target)]
                before = time.monotonic()
                with (output/(target+".log")).open("w") as campaign_log:
                    run = subprocess.run(command, cwd=root/"code", env=env,
                                         stdout=campaign_log, stderr=subprocess.STDOUT)
                results[target] = {"exit_code": run.returncode, "seconds": time.monotonic()-before}
                (output/"launcher-results.json").write_text(json.dumps(results, indent=2)+"\n")
                print(target, results[target], flush=True)
        finally:
            if listener:
                listener.shutdown()
                listener.server_close()
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()
    return int(any(r["exit_code"] for r in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
