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

TEMPLATE = Path(__file__).resolve().parents[2] / "antibody/tools/structure_design/examples/1n8z_vh.json"
REPO = TEMPLATE.parents[5]


class DesignWorker:
    def __init__(self, run_dir, args):
        self.jobs = Path(run_dir).resolve() / "design_jobs"
        self.jobs.mkdir(exist_ok=True, parents=True)
        self.args = args
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
        return {"backbones": self.args.backbones_per_round - sum(j.get("backbones_requested", int(j["action"] == "design_h3")) for j in jobs),
                "sequences": self.args.sequences_per_round - sum(j["count"] for j in jobs)}

    def eligible(self, rows):
        seen = set(self.evaluated)
        available = []
        for row in rows:
            if not row["violations"] and row["sequence"] not in seen:
                available.append(row)
                seen.add(row["sequence"])
        return available

    def execute(self, action, count, temperature=None, backbone_id=None):
        boltzgen = action == "design_h3_boltzgen"
        if action not in ("design_h3", "sequence_h3", "sequence_h3_soluble", "design_h3_boltzgen"):
            raise ValueError("Unknown generation action")
        if boltzgen and not getattr(self.args, "enable_boltzgen", False):
            raise ValueError("BoltzGen is optional and must be enabled by the host")
        if type(count) is not int or count < 1:
            raise ValueError("count must be a positive integer")
        if boltzgen and temperature is not None:
            raise ValueError("BoltzGen uses its native diffusion settings, not ProteinMPNN temperature")
        if not boltzgen and (type(temperature) not in (int, float) or not math.isfinite(temperature) or not 0.01 <= temperature <= 1):
            raise ValueError("temperature must be in [0.01,1]")
        if (action in ("sequence_h3", "sequence_h3_soluble")) != bool(backbone_id):
            raise ValueError("sequence_h3 requires a generated backbone ID")
        if self.round == 0 and (action != "design_h3" or temperature != self.args.temperature or
                                count != min(self.args.samples_per_call, self.remaining()["sequences"])):
            raise ValueError(f"Shared initialization uses design_h3, count={self.args.samples_per_call}, temperature={self.args.temperature}")
        known = {r["backbone_id"] for r in self.candidates().values()}
        if backbone_id and backbone_id not in known:
            raise ValueError("Unknown backbone ID in this campaign")
        if backbone_id and "/" in backbone_id:
            raise ValueError("BoltzGen designs cannot be passed to the RFD3 ProteinMPNN resequencing route")
        remaining = self.remaining()
        backbone_count = int(boltzgen or action == "design_h3")
        if count > remaining["sequences"] or backbone_count > remaining["backbones"]:
            raise ValueError(f"Generation budget exceeded. Remaining {remaining}")
        index = len(list(self.jobs.glob("job-*")))
        job = self.jobs / f"job-{index:05d}"
        job.mkdir()
        backbone = backbone_id or job.name
        seed = self.args.seed + index
        receipt = dict(round=self.round, action=action, count=count, temperature=temperature,
                       backbone_id=backbone, seed=seed, mock_generator=self.args.mock,
                       backbones_requested=backbone_count,
                       status="started")
        path = job / "receipt.json"
        atomic_json_write(path, receipt)
        try:
            if self.runtime:
                self.runtime.consume_many({"backbones_requested": backbone_count,
                                           "sequences_requested": count}, usage_key=job.name)
            if self.args.mock:
                rng = random.Random(seed)
                sequences = ["".join(rng.choices("AGSPDTYN", k=11)) for _ in range(count)]
            elif self.round == 0 and self.args.initialization_run:
                source = self.args.initialization_run.resolve() / "design_jobs" / job.name
                sequences = self.replay_initialization(job, source, receipt)
                receipt["initialization_source"] = str(source)
            elif boltzgen:
                sequences = self.run_boltzgen(job, count, seed)
            else:
                sequences = self.run_models(job, backbone_id, count, temperature, seed, soluble=action == "sequence_h3_soluble")
            artifact = "receipt.json" if self.args.mock else "output/candidates.json" if boltzgen else "output/designs.json"
            rows = [dict(design_id=f"{job.name}/{i}", backbone_id=f"{job.name}/boltzgen" if boltzgen else backbone, sequence=h3,
                         artifact=(job / artifact).relative_to(self.jobs.parent).as_posix(),
                         index=i, violations=_sequence_violations(h3, 11))
                    for i, h3 in enumerate(sequences)]
            receipt.update(status="succeeded", candidates=rows)
            return rows
        except Exception as exc:
            receipt.update(status="failed", error=str(exc))
            raise
        finally:
            atomic_json_write(path, receipt)

    def run_boltzgen(self, job, count, seed):
        import fcntl

        suffix = {"HER2": "HER2", "PD-1": "PD1"}.get(self.args.antigen)
        if suffix is None:
            raise ValueError("No BoltzGen antigen configuration")
        keys = ("BOLTZGEN_PYTHON", "BOLTZGEN_SCAFFOLD", "BOLTZGEN_CHECKPOINT", "BOLTZGEN_INVERSE_FOLD_CHECKPOINT", "BOLTZGEN_MOLDIR", "BOLTZGEN_SPEC_" + suffix)
        settings = {k: os.environ.get(k) for k in keys}
        if not all(settings.values()):
            raise ValueError("BoltzGen host configuration is incomplete: " + ", ".join(k for k, v in settings.items() if not v))
        python = Path(settings["BOLTZGEN_PYTHON"]).absolute()
        cmd = [str(python), "-m", "tasks.antibody.tools.complex_screening.boltzgen_design",
               "--python", str(python), "--spec", settings["BOLTZGEN_SPEC_" + suffix],
               "--scaffold", settings["BOLTZGEN_SCAFFOLD"], "--checkpoint", settings["BOLTZGEN_CHECKPOINT"],
               "--inverse-fold-checkpoint", settings["BOLTZGEN_INVERSE_FOLD_CHECKPOINT"],
               "--moldir", settings["BOLTZGEN_MOLDIR"], "--out", str(job / "output"),
               "--gpu", str(self.args.gpu), "--seed", str(seed), "--samples", str(count), "--timeout", "850"]
        cpus = sorted(os.sched_getaffinity(0))[:self.args.cpu_limit]
        env = dict(os.environ, PYTHONPATH=str(REPO))
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = str(len(cpus))
        cmd = ["taskset", "-c", ",".join(map(str, cpus)), *cmd]
        with (self.args.root / "tmp/antibody-rfd3-worker.lock").open("a") as lock, (job / "worker.log").open("w") as log:
            fcntl.flock(lock, fcntl.LOCK_EX)
            process = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, pass_fds=(lock.fileno(),))
            try:
                if process.wait(timeout=900):
                    raise RuntimeError(f"BoltzGen failed; see {job.name}/worker.log")
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        manifest = json.loads((job / "output/run.json").read_text())
        if manifest["status"] != "succeeded" or len(manifest["candidates"]) != count:
            raise ValueError("Incomplete BoltzGen result")
        return [row["h3"] for row in manifest["candidates"]]

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

    def run_models(self, job, backbone_id, count, temperature, seed, soluble=False):
        import fcntl

        root = self.args.root.resolve()
        config = dict(self.template, workspace_root=str(root), seed=seed, num_designs=1,
                      sequences_per_design=count, sampling_temperature=temperature,
                      proteinmpnn_variant="soluble" if soluble else "vanilla")
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
            cmd.extend(["--generation", str(self.jobs / backbone_id / "output/generation.json")])
        with (root / "tmp/antibody-rfd3-worker.lock").open("a") as lock, (job / "worker.log").open("w") as log:
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
