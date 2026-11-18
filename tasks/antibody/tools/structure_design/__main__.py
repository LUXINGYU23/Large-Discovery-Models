"""Standalone, subprocess-isolated structural design CLI (no Pi or oracle)."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

from . import __version__
from .bootstrap import MPNN_COMMIT
from .common import AA, extract_h3, load_json, save_json, sha256, validate_config

HERE = Path(__file__).resolve().parent


def resolve_config(path, root_override=None):
    path = Path(path).resolve()
    config = load_json(path)
    root = Path(root_override or config.get("workspace_root", ".")).expanduser()
    if not root.is_absolute():
        root = path.parent / root
    root = root.resolve()
    config["workspace_root"] = str(root)
    defaults = {
        "rfd3_python": "env/rfd3/bin/python", "proteinmpnn_python": "env/proteinmpnn/bin/python",
        "proteinmpnn_source": "env/sources/ProteinMPNN",
        "rfd3_checkpoint": "archive/models/rfd3/2025-12-01/rfd3.ckpt",
        "proteinmpnn_checkpoint": f"archive/models/proteinmpnn/{MPNN_COMMIT}/v_48_020.pt",
    }
    for name, default in defaults.items():
        value = Path(config.get(name, default)).expanduser()
        absolute = value if value.is_absolute() else root / value
        # venv interpreters are normally symlinks. Resolving one would execute
        # the system Python and silently bypass the isolated environment.
        config[name] = os.path.abspath(absolute) if name.endswith("_python") else str(absolute.resolve())
    value = Path(config["input_structure"]).expanduser()
    config["input_structure"] = str(value.resolve() if value.is_absolute() else (root / value).resolve())
    validate_config(config)
    return config


def runtime_env(config):
    root = Path(config["workspace_root"])
    env = dict(os.environ)
    env.update(XDG_CACHE_HOME=str(root / "cache"), TORCH_HOME=str(root / "cache/torch"),
               TORCHINDUCTOR_CACHE_DIR=str(root / "cache/torchinductor"),
               TRITON_CACHE_DIR=str(root / "cache/triton"),
               PIP_CACHE_DIR=str(root / "cache/pip"), TMPDIR=str(root / "tmp"),
               WANDB_MODE="disabled", PYTHONUNBUFFERED="1")
    for name in ["XDG_CACHE_HOME", "TORCH_HOME", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TMPDIR"]:
        Path(env[name]).mkdir(parents=True, exist_ok=True)
    return env


def run_command(command, directory, name, env):
    """Keep child stdout in logs; sample process GPU memory rather than device totals."""
    log = directory / (name + ".log")
    started = time.monotonic()
    peak = [None]
    stop = threading.Event()
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen([str(c) for c in command], cwd=directory, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT)

        def monitor():
            while not stop.is_set():
                try:
                    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                                             "--format=csv,noheader,nounits"], capture_output=True,
                                            text=True, timeout=5)
                    for row in result.stdout.splitlines():
                        pid, memory = [v.strip() for v in row.split(",")]
                        if pid == str(process.pid):
                            value = int(memory)
                            peak[0] = max(peak[0] or 0, value)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    pass
                stop.wait(1)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            stop.set()
            thread.join(timeout=6)
    record = {"command": [str(c) for c in command], "returncode": code,
              "elapsed_seconds": time.monotonic() - started, "log": str(log),
              "sampled_peak_process_gpu_memory_mib": peak[0], "memory_sampling_interval_seconds": 1}
    save_json(directory / (name + ".execution.json"), record)
    if code:
        raise RuntimeError(f"{name} failed (exit {code}); see {log}")
    return record


def model_command(config, command, *, module=None):
    """Sharing is opt-in; the wrapper leaves upstream scientific code unchanged."""
    if not config.get("allow_gpu_sharing", False):
        return command
    prefix = [config["rfd3_python"] if module else command[0], HERE / "gpu_runner.py",
              "--limit-gib", config["gpu_memory_limit_gib"]]
    if module:
        return [*prefix, "--module", module, "--", *command[1:]]
    return [*prefix, "--script", command[1], "--", *command[2:]]


@contextmanager
def gpu_slot(config, requested=None):
    import fcntl  # Deployment and GPU execution are Linux-only.
    root = Path(config["workspace_root"])
    deadline = time.monotonic() + config.get("gpu_wait_seconds", 600)
    preferred = [int(requested)] if requested is not None else [6, 7]
    while True:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.free,utilization.gpu",
                                 "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
        rows = {int(v[0]): v for v in ([x.strip() for x in row.split(",")]
                                     for row in result.stdout.splitlines())}
        for index in preferred:
            if index not in rows:
                raise ValueError(f"GPU {index} does not exist")
            row = rows[index]
            shared = config.get("allow_gpu_sharing", False)
            required_free = (config["gpu_memory_limit_gib"] + 2) * 1024 if shared else int(row[1]) * 0.95
            if int(row[2]) < required_free or (not shared and int(row[3]) != 0):
                continue
            lock = (root / "tmp" / f"structure-design-gpu{index}.lock").open("a")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            try:
                yield index
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
                lock.close()
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("No idle requested GPU within gpu_wait_seconds; no job launched")
        time.sleep(5)


def provenance(config):
    result = {"tool_version": __version__, "backend": "local RFD3 / original ProteinMPNN",
              "seed": config.get("seed", 42), "parameters": config,
              "input_structure": {"path": config["input_structure"], "sha256": sha256(config["input_structure"])},
              "tool_source_sha256": {p.name: sha256(p) for p in HERE.glob("*.py")}}
    for key in ["rfd3_checkpoint", "proteinmpnn_checkpoint"]:
        path = Path(config[key])
        record = load_json(str(path) + ".source.json")
        if record["sha256"] != sha256(path):
            raise ValueError(f"Weight SHA256 mismatch: {path}")
        result[key] = dict(record, path=str(path))
    source = Path(config["proteinmpnn_source"])
    version = load_json(source / "source-version.json")
    if version["commit"] != MPNN_COMMIT:
        raise ValueError("ProteinMPNN source commit differs from pinned version")
    for relative, expected in version["files"].items():
        if sha256(source / relative) != expected:
            raise ValueError(f"Modified ProteinMPNN source: {relative}")
    result["proteinmpnn_source"] = version
    for key in ["rfd3_python", "proteinmpnn_python"]:
        python = config[key]
        query = "import importlib.metadata as m,json; print(json.dumps({k:m.version(k) for k in ['torch','numpy']}))"
        packages = json.loads(subprocess.check_output([python, "-c", query], text=True))
        if not packages["torch"].startswith("2.6.0") or packages["numpy"] != "1.26.4":
            raise ValueError(f"Unexpected runtime versions: {key}: {packages}")
        if key == "rfd3_python":
            packages.update(json.loads(subprocess.check_output([python, "-c",
                "import importlib.metadata as m,json; print(json.dumps({k:m.version(k) for k in ['rc-foundry','atomworks']}))"], text=True)))
            if packages["rc-foundry"] != "0.2.0" or packages["atomworks"] != "2.1.1":
                raise ValueError(f"Unexpected Foundry/AtomWorks versions: {packages}")
        result[key] = {"executable": python, "packages": packages}
    return result


def prepare(config, directory, env):
    target = directory / "input"
    target.mkdir()
    run_command([config["rfd3_python"], HERE / "worker.py", "prepare", "--config",
                 directory / "config.resolved.json", "--out-dir", target], directory, "prepare", env)
    return load_json(target / "input.json")


def generate(config, directory, env, origin):
    prepare(config, directory, env)
    raw = directory / "rfd3"
    raw.mkdir()
    command = [str(Path(config["rfd3_python"]).parent / "rfd3"), "design",
               f"inputs={directory / 'input/specification.json'}", f"out_dir={raw}",
               f"ckpt_path={config['rfd3_checkpoint']}", "diffusion_batch_size=1",
               f"n_batches={config.get('num_designs', 1)}", f"seed={config.get('seed', 42)}",
               f"inference_sampler.num_timesteps={config.get('num_timesteps', 200)}",
               "dump_trajectories=False", "prevalidate_inputs=True", "skip_existing=False"]
    execution = run_command(model_command(config, command, module="rfd3.cli"), directory, "rfd3", env)
    structures = sorted(raw.glob("*.cif.gz"))
    if len(structures) != config.get("num_designs", 1):
        raise ValueError(f"Expected {config.get('num_designs', 1)} structures, got {len(structures)}")
    candidates = []
    for i, structure in enumerate(structures):
        output = directory / f"generated-{i:04d}"
        output.mkdir()
        run_command([config["rfd3_python"], HERE / "worker.py", "convert", "--config",
                     directory / "config.resolved.json", "--input-manifest", directory / "input/input.json",
                     "--structure", structure, "--out-dir", output], directory, f"convert-{i:04d}", env)
        candidate = load_json(output / "candidate.json")
        candidate.update(candidate_id=f"{directory.name}/rfd3-{i:04d}", parent_design="input",
                         rfd3_metadata=load_json(str(structure).replace(".cif.gz", ".json")),
                         provenance=origin, generation_execution=execution,
                         structure_semantics="original RFD3 structure; MPNN input is backbone-only")
        if config.get("extract_h3", False):
            candidate["h3"] = extract_h3(candidate, config.get("required_h3_length"))
        save_json(output / "candidate.json", candidate)
        candidates.append(candidate)
    save_json(directory / "generation.json", {"status": "success", "candidates": candidates})
    return candidates


def read_fasta(path):
    records = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            records.append([line[1:], ""])
        elif line.strip():
            if not records:
                raise ValueError("Invalid FASTA without a header")
            records[-1][1] += line.strip()
    return records


def sequence(config, directory, env, origin, candidates):
    source = Path(config["proteinmpnn_source"])
    results = []
    for i, candidate in enumerate(candidates):
        target = directory / f"mpnn-{i:04d}"
        target.mkdir()
        pdb_dir = target / "pdb"
        pdb_dir.mkdir()
        backbone = Path(candidate["backbone_path"])
        # Copy rather than mutate the upstream input.
        import shutil
        shutil.copyfile(backbone, pdb_dir / "backbone.pdb")
        run_command([config["proteinmpnn_python"], source / "helper_scripts/parse_multiple_chains.py",
                     "--input_path", pdb_dir, "--output_path", target / "parsed.jsonl"],
                    directory, f"parse-mpnn-{i:04d}", env)
        all_chains = list(candidate["full_sequences"])
        designed_chains = sorted({m["sequence_chain"] for m in candidate["residue_mapping"] if m["designed"]})
        fixed_chains = sorted(set(all_chains) - set(designed_chains))
        if not designed_chains:
            raise ValueError("ProteinMPNN requires at least one designed residue")
        save_json(target / "chains.jsonl", {"backbone": [designed_chains, fixed_chains]})
        save_json(target / "fixed.jsonl", {"backbone": {
            c: [m["sequence_position"] for m in candidate["residue_mapping"]
                if m["sequence_chain"] == c and not m["designed"]] for c in all_chains}})
        command = [config["proteinmpnn_python"], source / "protein_mpnn_run.py",
            "--jsonl_path", target / "parsed.jsonl", "--chain_id_jsonl", target / "chains.jsonl",
            "--fixed_positions_jsonl", target / "fixed.jsonl", "--out_folder", target / "output",
            "--path_to_model_weights", Path(config["proteinmpnn_checkpoint"]).parent,
            "--model_name", "v_48_020", "--seed", config.get("seed", 42) + i,
            "--num_seq_per_target", config.get("sequences_per_design", 1), "--batch_size", 1,
            "--sampling_temp", config.get("sampling_temperature", 0.1), "--backbone_noise", 0.0]
        execution = run_command(model_command(config, command), directory, f"proteinmpnn-{i:04d}", env)
        fasta = read_fasta(target / "output/seqs/backbone.fa")
        sampled = [record for record in fasta if "sample=" in record[0]]
        if len(sampled) != config.get("sequences_per_design", 1):
            raise ValueError("ProteinMPNN returned an unexpected sequence count")
        for sample, (header, value) in enumerate(sampled):
            parts = value.split("/")
            if len(parts) != len(designed_chains):
                raise ValueError("ProteinMPNN output does not match designed-chain order")
            full = dict(candidate["full_sequences"])
            full.update(zip(designed_chains, parts))
            for chain, seq in full.items():
                if len(seq) != len(candidate["full_sequences"][chain]) or any(aa not in AA for aa in seq):
                    raise ValueError("ProteinMPNN sequence has invalid length/alphabet")
            for mapping in candidate["residue_mapping"]:
                if not mapping["designed"] and full[mapping["sequence_chain"]][mapping["sequence_position"] - 1] != mapping["input_aa"]:
                    raise ValueError("ProteinMPNN changed a fixed residue")
            result = dict(candidate, candidate_id=f"{directory.name}/mpnn-{i:04d}-{sample:04d}",
                          parent_design=candidate["candidate_id"], full_sequences=full,
                          sequence_seed=config.get("seed", 42) + i,
                          provenance=origin, sequence_execution=execution, proteinmpnn_header=header,
                          sequence_structure_semantics="new sequence on supplied backbone; sidechains have NOT been repacked or relaxed")
            h3 = extract_h3(result, config.get("required_h3_length"))
            result.pop("h3", None)  # Never retain an upstream H3 after MPNN changes the sequence.
            if config.get("extract_h3", False):
                result["h3"] = h3
            results.append(result)
    save_json(directory / "designs.json", {"status": "success", "candidates": results})
    with (directory / "designs.fasta").open("w") as stream:
        for candidate in results:
            for chain, seq in candidate["full_sequences"].items():
                stream.write(f">{candidate['candidate_id']} chain={chain}\n{seq}\n")
    return results


def doctor(config, directory, env, origin):
    prepare(config, directory, env)
    for key in ["rfd3_python", "proteinmpnn_python"]:
        imports = ("import rfd3.engine; " if key == "rfd3_python" else
                   f"import sys; sys.path.insert(0, {config['proteinmpnn_source']!r}); import protein_mpnn_utils; ")
        query = imports + "import torch,json; assert torch.version.cuda == '12.4', 'Expected CUDA 12.4 PyTorch wheel'; "
        if config.get("device", "cuda") == "cuda":
            query += "assert torch.cuda.is_available(), 'CUDA unavailable'; "
        query += ("print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,"
                  "'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))")
        run_command([config[key], "-c", query], directory, key.replace("_python", "-runtime"), env)
    if not os.access(Path(config["workspace_root"]) / "archive/models", os.R_OK):
        raise ValueError("Archive model directory is not readable")
    save_json(directory / "doctor.json", {"status": "success", "provenance": origin,
                                         "input": load_json(directory / "input/input.json")})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["doctor", "generate", "sequence", "extract-h3", "pipeline"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--generation", type=Path, help="Successful generation.json, for sequence")
    parser.add_argument("--designs", type=Path, help="Candidate manifest, for extract-h3")
    parser.add_argument("--gpu", type=int, help="Physical GPU index; default try idle 6, then 7")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Default: cuda; CPU is explicit, never an automatic fallback")
    parser.add_argument("--allow-gpu-sharing", action="store_true", help="Explicitly allow a busy selected GPU")
    parser.add_argument("--gpu-memory-limit-gib", type=float, default=24,
                        help="PyTorch allocator cap for opt-in sharing (default: 24 GiB)")
    parser.add_argument("--extract-h3", action="store_true", default=None)
    args = parser.parse_args(argv)
    directory = args.out_dir.expanduser().resolve()
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        parser.error("out-dir must be new or empty; existing runs are never overwritten")
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        config = resolve_config(args.config, args.root)
        config["device"] = args.device or config.get("device", "cuda")
        if args.allow_gpu_sharing:
            config["allow_gpu_sharing"] = True
            config["gpu_memory_limit_gib"] = args.gpu_memory_limit_gib
            validate_config(config)
        if config.get("allow_gpu_sharing", False) and args.gpu is None and args.command != "extract-h3":
            raise ValueError("GPU sharing requires an explicit --gpu")
        if config["device"] == "cpu" and (config.get("allow_gpu_sharing", False) or args.gpu is not None):
            raise ValueError("CPU execution cannot also select/share a GPU")
        if args.extract_h3:
            config["extract_h3"] = True
        save_json(directory / "config.resolved.json", config)
        if args.command == "extract-h3":
            if not args.designs:
                raise ValueError("extract-h3 requires --designs")
            manifest = load_json(args.designs)
            if manifest.get("status") != "success":
                raise ValueError("Input manifest is not successful")
            values = [{"candidate_id": c["candidate_id"], "h3": extract_h3(c, config.get("required_h3_length"))}
                      for c in manifest["candidates"]]
            save_json(directory / "h3.json", {"status": "success", "candidates": values})
        else:
            env = runtime_env(config)
            origin = provenance(config)
            save_json(directory / "provenance.json", origin)
            cpu = config["device"] == "cpu"
            with nullcontext(None) if cpu else gpu_slot(config, args.gpu) as gpu:
                env["CUDA_VISIBLE_DEVICES"] = "" if cpu else str(gpu)
                if cpu:
                    threads = str(config.get("cpu_threads", 8))
                    env.update(OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads, MKL_NUM_THREADS=threads)
                else:
                    save_json(directory / "gpu.json", {"physical_gpu_index": gpu})
                save_json(directory / "device.json", {"device": config["device"], "physical_gpu_index": gpu})
                if args.command == "doctor":
                    doctor(config, directory, env, origin)
                elif args.command in ["generate", "pipeline"]:
                    candidates = generate(config, directory, env, origin)
                    if args.command == "pipeline":
                        sequence(config, directory, env, origin, candidates)
                else:
                    if args.generation:
                        manifest = load_json(args.generation)
                        if manifest.get("status") != "success":
                            raise ValueError("Generation manifest is not successful")
                        candidates = manifest["candidates"]
                    else:
                        candidate = prepare(config, directory, env)
                        candidate.update(structure_path=candidate["backbone_path"], provenance=origin)
                        candidates = [candidate]
                    sequence(config, directory, env, origin, candidates)
        save_json(directory / "run.json", {"status": "success", "command": args.command,
                  "device": "none" if args.command == "extract-h3" else config["device"],
                  "started_at_utc": started_at,
                  "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                  "elapsed_seconds": time.monotonic() - started})
        print(json.dumps({"status": "success", "command": args.command, "out_dir": str(directory)}))
        return 0
    except Exception as exc:
        save_json(directory / "run.json", {"status": "failed", "command": args.command,
                  "error": str(exc), "elapsed_seconds": time.monotonic() - started})
        (directory / "error.log").write_text(traceback.format_exc())
        print(json.dumps({"status": "failed", "error": str(exc), "out_dir": str(directory)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
