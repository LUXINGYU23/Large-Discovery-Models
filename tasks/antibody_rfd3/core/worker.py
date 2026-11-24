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
from .sequence_design import mutation_options, mutate_sequences, validate_sequence
from .optional_tools import OptionalProteinTools

TEMPLATE = Path(__file__).resolve().parents[2] / "antibody/tools/structure_design/examples/1n8z_vh.json"
REPO = TEMPLATE.parents[5]


class DesignWorker(OptionalProteinTools):
    def __init__(self, run_dir, args, *, seed_backbones=None):
        self.jobs = Path(run_dir).resolve() / "design_jobs"
        self.jobs.mkdir(exist_ok=True, parents=True)
        self.args = args
        self.seed_backbones = dict(seed_backbones or {})
        self.template = json.loads(TEMPLATE.read_text())
        self.round = 0
        self.runtime = None
        self.evaluated = set()
        self.history_log = None

    def records(self):
        for path in sorted(self.jobs.glob("job-*/receipt.json")):
            yield json.loads(path.read_text())

    def candidates(self):
        return {row["design_id"]: row for job in self.records()
                for row in job.get("candidates", [])}

    def remaining(self):
        jobs = [j for j in self.records() if j["round"] == self.round]
        return {"backbones": self.args.backbones_per_round - sum(j.get("backbones_requested", int(j["action"] == "design_h3")) for j in jobs),
                "sequences": self.args.sequences_per_round - sum(j["count"] for j in jobs)}

    def role_budget(self, profile_id):
        """A panel member's share of the same per-generation budget."""
        if self.args.method != "ldm_harness":
            return {"backbones": self.args.backbones_per_round, "sequences": self.args.sequences_per_round}
        if profile_id not in ("explorer", "refiner", "evidence_reviewer"):
            raise ValueError("Unknown panel role")
        if self.round == 0 and not getattr(self.args, "research_candidates_per_role", 0):
            limits = {"backbones": 1, "sequences": self.args.samples_per_call} if profile_id == "explorer" else {
                "backbones": 0, "sequences": 0}
        elif profile_id == "evidence_reviewer":
            limits = {"backbones": 0, "sequences": 0}
        else:
            extra = 1 if profile_id == "explorer" else 0
            limits = {"backbones": (self.args.backbones_per_round + extra) // 2,
                      "sequences": (self.args.sequences_per_round + extra) // 2}
        return limits

    def role_remaining(self, profile_id):
        limits = self.role_budget(profile_id)
        jobs = [j for j in self.records() if j["round"] == self.round and j.get("profile_id") == profile_id]
        return {"backbones": limits["backbones"] - sum(j.get("backbones_requested", int(j["action"] == "design_h3")) for j in jobs),
                "sequences": limits["sequences"] - sum(j["count"] for j in jobs)}

    def eligible(self, rows):
        seen = set(self.evaluated)
        if self.history_log is not None:
            current = self.history_log.view()
            seen.update(current.measured_keys | current.pending_keys)
        available = []
        for row in rows:
            if not row["violations"] and row["sequence"] not in seen:
                available.append(row)
                seen.add(row["sequence"])
        return available

    def execute(self, action, count=None, temperature=None, backbone_id=None, call_id=None, profile_id=None,
                seed_sequence=None, mutable_positions=None, min_mutations=1, max_mutations=2,
                allowed_amino_acids=None, sequences=None, rationale=None):
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
        parameters = None
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
            parameters = dict(seed_sequence=seed_sequence, mutable_positions=mutable_positions,
                              min_mutations=min_mutations, max_mutations=max_mutations,
                              allowed_amino_acids=allowed_amino_acids, sequences=sequences, rationale=rationale)
        elif any(v is not None for v in (seed_sequence, mutable_positions, allowed_amino_acids, sequences, rationale)):
            raise ValueError("Sequence design parameters require mutate_h3 or register_h3")
        if type(count) is not int or count < 1:
            raise ValueError("count must be a positive integer")
        if action == "mutate_h3" and count > sum(size for _, size in groups):
            raise ValueError("Requested count exceeds the unique sequence capacity of these mutation constraints")
        if not boltzgen and (type(temperature) not in (int, float) or not math.isfinite(temperature) or not 0.01 <= temperature <= 1):
            raise ValueError("temperature must be in [0.01,1]")
        if (action in ("sequence_h3", "sequence_h3_soluble")) != bool(backbone_id):
            raise ValueError("sequence_h3 requires a generated backbone ID")
        if call_id is not None and (not isinstance(call_id, str) or not call_id):
            raise ValueError("call_id must be a non-empty string")
        jobs = [row for row in self.records() if row["round"] == self.round]
        if call_id is not None:
            matches = [row for row in jobs if row.get("call_id") == call_id and row.get("profile_id") == profile_id]
            if len(matches) > 1:
                raise RuntimeError("Repeated generation call ID in design receipts")
            if matches:
                receipt = matches[0]
                fields = {"action": action, "count": count, "temperature": temperature,
                          "requested_backbone_id": backbone_id}
                if directed:
                    fields["sequence_design"] = parameters
                if any(receipt.get(key) != value for key, value in fields.items()):
                    raise RuntimeError("Generation call ID was reused with different parameters")
                if receipt["status"] != "succeeded":
                    raise RuntimeError("Generation outcome is not safely replayable; inspect its receipt")
                return receipt["candidates"]
            if any(row["status"] != "succeeded" for row in jobs):
                raise RuntimeError("Generation has an unresolved or failed receipt; inspect it before retrying")
        if self.round == 0 and not getattr(self.args, "research_candidates_per_role", 0) and (
                                action != "design_h3" or temperature != self.args.temperature or
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
        role_remaining = self.role_remaining(profile_id)
        if count > role_remaining["sequences"] or backbone_count > role_remaining["backbones"]:
            raise ValueError(f"Role generation budget exceeded. Remaining {role_remaining}")
        index = len(list(self.jobs.glob("job-*")))
        job = self.jobs / f"job-{index:05d}"
        job.mkdir()
        backbone = None if directed else f"{job.name}/boltzgen" if boltzgen else backbone_id or job.name
        seed = self.args.seed + index
        receipt = dict(round=self.round, action=action, count=count, temperature=temperature, profile_id=profile_id,
                       call_id=call_id, requested_backbone_id=backbone_id,
                       backbone_id=backbone, seed=seed, mock_generator=self.args.mock,
                       backbones_requested=backbone_count, status="started")
        if directed:
            receipt["sequence_design"] = parameters
            receipt["parent_history"] = [entry.to_dict() for entry in self.history_log.view().results
                if entry.key == seed_sequence and entry.status == "succeeded"] if self.history_log is not None else []
        path = job / "receipt.json"
        atomic_json_write(path, receipt)
        try:
            if self.runtime:
                self.runtime.consume_many({"backbones_requested": backbone_count,
                                           "sequences_requested": count}, usage_key=job.name)
            if action == "register_h3":
                proposed = list(sequences)
            elif action == "mutate_h3":
                proposed = mutate_sequences(seed_sequence, count, options, groups, seed)
            elif self.args.mock:
                rng = random.Random(seed)
                proposed = ["".join(rng.choices("AGSPDTYN", k=11)) for _ in range(count)]
            elif self.round == 0 and self.args.initialization_run:
                source = self.args.initialization_run.resolve() / "design_jobs" / job.name
                proposed = self.replay_initialization(job, source, receipt)
                receipt["initialization_source"] = str(source)
            elif boltzgen:
                proposed = self.run_boltzgen(job, count, seed)
            elif action == "sequence_h3_soluble":
                proposed = self.run_models(job, backbone_id, count, temperature, seed, soluble=True)
            else:
                proposed = self.run_models(job, backbone_id, count, temperature, seed)
            rows = [dict(design_id=f"{job.name}/{i}", backbone_id=backbone, sequence=h3,
                         artifact=(job / ("receipt.json" if self.args.mock or directed else "output/candidates.json" if boltzgen else "output/designs.json")).relative_to(self.jobs.parent).as_posix(),
                         index=i, violations=_sequence_violations(h3, 11))
                    for i, h3 in enumerate(proposed)]
            if directed:
                for row in rows:
                    row.update(generation_action=action, parent_sequence=seed_sequence, rationale=rationale,
                               parent_history_seqs=[entry["seq"] for entry in receipt["parent_history"]])
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
        for name in ("input_structure", "rfd3_checkpoint", "proteinmpnn_checkpoint"):
            path = getattr(self.args, name)
            if path is not None:
                config[name] = str(path.expanduser().resolve())
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
               "--extract-h3", "--gpu", str(self.args.gpu)]
        if self.args.share_gpu:
            cmd.append("--allow-gpu-sharing")
        if backbone_id:
            generation = self.seed_backbones.get(backbone_id, self.jobs / backbone_id / "output/generation.json")
            cmd.extend(["--generation", str(generation)])
        with (root / f"tmp/antibody-rfd3-worker-gpu{self.args.gpu}.lock").open("a") as lock, \
                (job / "worker.log").open("w") as log:
            fcntl.flock(lock, fcntl.LOCK_EX)
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
