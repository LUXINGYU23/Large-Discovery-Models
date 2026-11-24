"""Run-local design ledger and single-slot Linux model worker."""
from __future__ import annotations

import json
import math
import os
import random
import signal
import shutil
import subprocess
import sys
from pathlib import Path

from ldm_tts.engine.run_store import atomic_json_write
from tasks.antibody.core.engine_adapters import _sequence_violations
from tasks.antibody.tools.structure_design.common import extract_h3
from ..sequence_design import mutation_options, mutate_sequences, validate_sequence
from ..optional_tools import OptionalProteinTools

TEMPLATE = Path(__file__).resolve().parents[3] / "antibody/tools/structure_design/examples/1n8z_vh.json"
REPO = TEMPLATE.parents[5]
FIXED_MAX_PROPOSAL_BATCHES = 3


class DesignWorker(OptionalProteinTools):
    def __init__(self, run_dir, args, *, seed_backbones=None, execution_gpu=None, runtime_cache=None,
                 prelaunch_retry=None):
        self.jobs = Path(run_dir).resolve() / "design_jobs"
        self.jobs.mkdir(exist_ok=True, parents=True)
        self.args = args
        self.execution_gpu = args.gpu if execution_gpu is None else execution_gpu
        self.runtime_cache = runtime_cache
        self.prelaunch_retry = prelaunch_retry
        self.seed_backbones = dict(seed_backbones or {})
        self.template = json.loads(TEMPLATE.read_text())
        self.round = 0
        self.runtime = None
        self.evaluated = set()

    def records(self):
        for path in sorted(self.jobs.glob("job-*/receipt.json")):
            yield json.loads(path.read_text())

    def candidates(self):
        return {row["design_id"]: row for job in self.records()
                for row in job.get("candidates", [])}

    def remaining(self):
        jobs = [j for j in self.records() if j["round"] == self.round]
        if self.prelaunch_retry and self.prelaunch_retry["round"] == self.round:
            # The failed preflight left no generated candidates. Keep its original
            # receipt and aggregate cost, while restoring that call's round allowance.
            jobs.remove(self.prelaunch_retry)
        # Fixed can resample after an empty batch. Receipts and the campaign-wide
        # generation limits still charge every call, including before a resume.
        batches = FIXED_MAX_PROPOSAL_BATCHES if self.args.method == "fixed" else 1
        return {
            "backbones": batches * self.args.backbones_per_round - sum(j.get("backbones_requested", int(j["action"] == "design_h3")) for j in jobs),
            "sequences": batches * self.args.sequences_per_round - sum(j["count"] for j in jobs),
        }

    def eligible(self, rows):
        seen = set(self.evaluated)
        available = []
        for row in rows:
            if not row["violations"] and row["sequence"] not in seen:
                available.append(row)
                seen.add(row["sequence"])
        return available

    def execute(self, action, count=None, temperature=None, backbone_id=None, seed_sequence=None,
                mutable_positions=None, min_mutations=1, max_mutations=2, allowed_amino_acids=None,
                sequences=None, rationale=None, call_id=None, profile_id=None):
        if action == "sequence_h3_soluble" and not getattr(self.args, "enable_protein_tools", False):
            raise ValueError("Optional protein tools must be enabled by the host")
        boltzgen = action == "design_h3_boltzgen"
        if boltzgen:
            if not getattr(self.args, "enable_boltzgen", False):
                raise ValueError("BoltzGen must be enabled by the host")
            if temperature is not None:
                raise ValueError("BoltzGen uses native diffusion settings, without MPNN temperature")
        elif temperature is None:
            temperature = 0.5
        if action not in ("design_h3", "sequence_h3", "sequence_h3_soluble", "design_h3_boltzgen", "mutate_h3", "register_h3"):
            raise ValueError("Unknown generation action")
        directed = action in ("mutate_h3", "register_h3")
        if directed:
            if backbone_id is not None or not isinstance(rationale, str) or not rationale.strip():
                raise ValueError("Sequence proposals require a rationale and cannot claim a generated backbone")
            if action == "register_h3":
                if mutable_positions is not None or allowed_amino_acids is not None or min_mutations != 1 or max_mutations != 2:
                    raise ValueError("Mutation constraints apply to mutate_h3")
                if not isinstance(sequences, list) or not sequences:
                    raise ValueError("register_h3 requires a nonempty sequences list")
                for sequence in sequences:
                    validate_sequence(sequence)
                if len(set(sequences)) != len(sequences):
                    raise ValueError("Explicit sequences must be unique")
                if seed_sequence is not None:
                    validate_sequence(seed_sequence)
                if count is not None and (type(count) is not int or count != len(sequences)):
                    raise ValueError("count must match the explicit sequences list")
                count = len(sequences)
            else:
                if sequences is not None:
                    raise ValueError("Explicit sequences apply to register_h3")
                options, groups = mutation_options(seed_sequence, mutable_positions, min_mutations,
                                                   max_mutations, allowed_amino_acids)
        elif any(value is not None for value in (
                seed_sequence, mutable_positions, allowed_amino_acids, sequences, rationale)):
            raise ValueError("Sequence design parameters require mutate_h3 or register_h3")
        params = dict(action=action, count=count, temperature=temperature, backbone_id=backbone_id,
                      seed_sequence=seed_sequence, mutable_positions=mutable_positions,
                      min_mutations=min_mutations, max_mutations=max_mutations,
                      allowed_amino_acids=allowed_amino_acids, sequences=sequences, rationale=rationale)
        if call_id is not None:
            matches = [j for j in self.records() if j.get("call_id") == call_id and j["round"] == self.round]
            if matches:
                receipt = matches[0]
                if receipt.get("parameters") != params or receipt["status"] != "succeeded":
                    raise RuntimeError("Generation call cannot be replayed safely")
                return receipt["candidates"]
        if type(count) is not int or count < 1:
            raise ValueError("count must be a positive integer")
        if not boltzgen and (type(temperature) not in (int, float) or not math.isfinite(temperature) or not 0.01 <= temperature <= 1):
            raise ValueError("temperature must be in [0.01,1]")
        if (action in ("sequence_h3", "sequence_h3_soluble")) != bool(backbone_id):
            raise ValueError("sequence_h3 requires a generated backbone ID")
        if self.round == 0 and not self.args.seed_bank and (action != "design_h3" or temperature != self.args.temperature or
                                count != min(self.args.samples_per_call, self.remaining()["sequences"])):
            raise ValueError(f"Shared initialization uses design_h3, count={self.args.samples_per_call}, temperature={self.args.temperature}")
        known = {r["backbone_id"] for r in self.candidates().values()}
        if backbone_id and backbone_id not in known and backbone_id not in self.seed_backbones:
            raise ValueError("Unknown backbone ID in this campaign")
        if backbone_id and backbone_id.endswith("/boltzgen"):
            raise ValueError("BoltzGen backbones are not compatible with ProteinMPNN resequencing")
        backbone_count = int(action == "design_h3" or boltzgen)
        remaining = self.remaining()
        if count > remaining["sequences"] or backbone_count > remaining["backbones"]:
            raise ValueError(f"Generation budget exceeded. Remaining {remaining}")
        index = len(list(self.jobs.glob("job-*")))
        job = self.jobs / f"job-{index:05d}"
        job.mkdir()
        backbone = None if directed else f"{job.name}/boltzgen" if boltzgen else backbone_id or job.name
        seed = self.args.seed + index
        receipt = dict(round=self.round, action=action, count=count, temperature=temperature,
                       backbone_id=backbone, seed=seed, mock_generator=self.args.mock,
                       backbones_requested=backbone_count,
                       status="started", call_id=call_id, parameters=params)
        path = job / "receipt.json"
        atomic_json_write(path, receipt)
        try:
            if self.runtime:
                self.runtime.consume_many({"backbones_requested": backbone_count,
                                           "sequences_requested": count}, usage_key=job.name)
            if action == "register_h3":
                sequences = list(sequences)
            elif action == "mutate_h3":
                sequences = mutate_sequences(seed_sequence, count, options, groups, seed)
            elif self.args.mock:
                rng = random.Random(seed)
                sequences = ["".join(rng.choices("AGSPDTYN", k=11)) for _ in range(count)]
            elif self.round == 0 and self.args.initialization_run:
                source = self.args.initialization_run.resolve() / "design_jobs" / job.name
                sequences = self.replay_initialization(job, source, receipt)
                receipt["initialization_source"] = str(source)
            elif boltzgen:
                sequences = self.run_boltzgen(job, count, seed)
            elif action == "sequence_h3_soluble":
                sequences = self.run_models(job, backbone_id, count, temperature, seed, soluble=True)
            else:
                sequences = self.run_models(job, backbone_id, count, temperature, seed)
            rows = [dict(design_id=f"{job.name}/{i}", backbone_id=backbone, sequence=h3,
                         artifact=(job / ("receipt.json" if self.args.mock or directed else "output/candidates.json" if boltzgen else "output/designs.json")).relative_to(self.jobs.parent).as_posix(),
                         index=i, violations=_sequence_violations(h3, 11),
                         generation_action=action, parent_sequence=seed_sequence, rationale=rationale)
                    for i, h3 in enumerate(sequences)]
            if action in ("sequence_h3_soluble", "design_h3_boltzgen"):
                for row in rows:
                    row["generation_action"] = action
            receipt.update(status="succeeded", candidates=rows)
            return rows
        except Exception as exc:
            receipt.update(status="failed", error=str(exc))
            raise
        finally:
            atomic_json_write(path, receipt)

    def replay_initialization(self, job, source, receipt):
        original = json.loads((source / "receipt.json").read_text())
        fields = ("round", "action", "count", "temperature", "seed", "mock_generator")
        if original["status"] != "succeeded" or any(original[k] != receipt[k] for k in fields):
            raise ValueError("Shared initialization differs in generator, seed or sampling settings")
        old = source / "output"
        config = json.loads((old / "config.resolved.json").read_text())
        for key in ("regions", "heavy_chain", "h3_residues", "expected_input_h3", "required_h3_length", "num_timesteps"):
            if config[key] != self.template[key]:
                raise ValueError(f"Shared initialization differs in template field {key}")
        target = job / "output"
        shutil.copytree(old, target)

        def relocate(value):
            if isinstance(value, str) and value.startswith(str(old) + os.sep):
                return str(target / Path(value).relative_to(old))
            if isinstance(value, dict):
                return {k: relocate(v) for k, v in value.items()}
            if isinstance(value, list):
                return [relocate(v) for v in value]
            return value

        for name in ("generation.json", "designs.json"):
            document = json.loads((target / name).read_text())
            if document["status"] != "success":
                raise ValueError("Invalid initialization manifest")
            atomic_json_write(target / name, relocate(document))
        sequences = [extract_h3(r, 11) for r in document["candidates"]]
        if len(sequences) != receipt["count"] or sequences != [r["sequence"] for r in original["candidates"]]:
            raise ValueError("Initialization receipt differs from mapped full VH sequences")
        return sequences

    def run_models(self, job, backbone_id, count, temperature, seed, *, soluble=False):
        import fcntl

        root = self.args.root.resolve()
        config = dict(self.template, workspace_root=str(root), seed=seed, num_designs=1,
                      sequences_per_design=count, sampling_temperature=temperature)
        if self.runtime_cache:
            assets = json.loads((self.runtime_cache / "manifest.json").read_text())
            for key in ("input_structure", "rfd3_checkpoint", "proteinmpnn_checkpoint"):
                config[key] = str(self.runtime_cache / assets[key]["file"])
        if soluble:
            config["proteinmpnn_variant"] = "soluble"
            config.pop("proteinmpnn_checkpoint", None)
            checkpoint = getattr(self.args, "solublempnn_checkpoint", None)
            if checkpoint:
                config["proteinmpnn_checkpoint"] = str(checkpoint.expanduser().resolve())
        atomic_json_write(job / "config.json", config)
        cpus = sorted(os.sched_getaffinity(0))[:self.args.cpu_limit]
        env = dict(os.environ, PYTHONPATH=str(REPO))
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = str(len(cpus))
        cmd = ["taskset", "-c", ",".join(map(str, cpus)), sys.executable, "-m",
               "tasks.antibody.tools.structure_design", "sequence" if backbone_id else "pipeline",
               "--config", str(job / "config.json"), "--out-dir", str(job / "output"),
               "--extract-h3", "--gpu", str(self.execution_gpu)]
        if self.args.share_gpu:
            cmd.append("--allow-gpu-sharing")
        if backbone_id:
            generation = self.seed_backbones.get(backbone_id, self.jobs / backbone_id / "output/generation.json")
            cmd.extend(["--generation", str(generation)])
        with (root / "tmp/antibody-rfd3-worker.lock").open("a") as lock, (job / "worker.log").open("w") as log:
            fcntl.flock(lock, fcntl.LOCK_EX)
            print(f"Starting structure worker on GPU {self.execution_gpu}", file=log, flush=True)
            process = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, pass_fds=(lock.fileno(),))
            try:
                if process.wait(timeout=900):
                    raise RuntimeError(f"Structure design failed; see {job.name}/worker.log")
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        manifest = json.loads((job / "output/designs.json").read_text())
        if manifest["status"] != "success" or len(manifest["candidates"]) != count:
            raise ValueError("Incomplete ProteinMPNN result")
        return [extract_h3(row, 11) for row in manifest["candidates"]]
