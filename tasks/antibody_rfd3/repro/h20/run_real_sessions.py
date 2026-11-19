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


def run_campaigns(methods, args, root, output, cpus, env, results):
    for method in methods:
        for target in ("her2", "pd1"):
            legacy = args.methods == ["ldm_harness"]
            destination = output / target if legacy else output / method / target
            name = target if legacy else method + "-" + target
            config = target + "_real" if method == "ldm_harness" else target + "_" + method + "_real"
            interpreter = "antbo_native" if method == "antbo" else "antibody_campaign"
            command = ["taskset", "-c", cpus, str(root / "env" / interpreter / "bin/python"),
                       "scripts/run_ldm_tts.py", f"config/antibody_rfd3/{config}.yaml",
                       "--set", "args.share-gpu=true", "--set", f"args.gpu={args.gpu}",
                       "--set", "args.out-dir=" + str(destination)]
            if method != "ldm_harness" or args.initialization_root:
                initialization = args.initialization_root or root / "runs/antibody_rfd3/real-sessions-20261119-02"
                command.extend(["--set", "args.initialization-run=" + str(initialization.resolve() / target)])
            before = time.monotonic()
            with (output / (name + ".log")).open("w") as log:
                run = subprocess.run(command, cwd=root / "code", env=env,
                                     stdout=log, stderr=subprocess.STDOUT)
            results[name] = {"exit_code": run.returncode, "seconds": time.monotonic() - before,
                             "run": str(destination)}
            (output / "launcher-results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(name, results[name], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/Aidata/shared/ldm/antibody-poc"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--methods", nargs="+", choices=("antbo", "fixed", "harness", "ldm_harness"),
                        default=["ldm_harness"])
    parser.add_argument("--initialization-root", type=Path)
    args = parser.parse_args()
    if len(set(args.methods)) != len(args.methods):
        parser.error("methods must be distinct")
    root = args.root.resolve()
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cpus = ",".join(map(str, sorted(os.sched_getaffinity(0))[:2]))
    usage = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu),
                                    "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
    pi_methods = [m for m in args.methods if m in ("harness", "ldm_harness")]
    required_memory = 55000 if pi_methods else 24000
    if int(usage.strip()) < required_memory:
        raise RuntimeError(f"Need at least {required_memory} MiB free for this run")
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
    env["ANTBO_HOME"] = str(root / "env/sources/HEBO-77636146/AntBO")
    results = {}
    run_campaigns([m for m in args.methods if m not in pi_methods], args, root, output, cpus, env, results)
    if not pi_methods:
        return int(any(r["exit_code"] for r in results.values()))
    command = ["taskset", "-c", cpus, "/home/zsgpu/vllm-env/bin/vllm", "serve",
               "/Aidata/shared/rsi-models/Qwen3-8B", "--host", bridge, "--port", str(port),
               "--served-model-name", "Qwen3-8B", "--max-model-len", "16384",
               "--max-num-seqs", "1", "--max-num-batched-tokens", "1024",
               "--gpu-memory-utilization", "0.27", "--enforce-eager",
               "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
               "--reasoning-parser", "qwen3", "--generation-config", "vllm"]
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
            run_campaigns(pi_methods, args, root, output, cpus, env, results)
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
