"""Optional protein tools shared by the asynchronous and synchronous workers."""
import json
import os
from pathlib import Path
import signal
import subprocess

REPO = Path(__file__).resolve().parents[3]


def add_optional_args(parser):
    parser.add_argument("--enable-protein-tools", action="store_true", help="Expose optional SolubleMPNN and host-prepared ipSAE tools")
    parser.add_argument("--enable-boltzgen", action="store_true", help="Expose host-configured native BoltzGen H3 proposals")
    parser.add_argument("--solublempnn-checkpoint", type=Path, help="Explicit soluble ProteinMPNN weights")


def validate_optional_args(parser, args, pi_methods):
    if (args.enable_protein_tools or args.enable_boltzgen) and args.method not in pi_methods:
        parser.error("Optional protein tools require a Pi method")


def active_tools(args, base):
    tools = (*base, "sequence_h3_soluble", "score_interface") if getattr(args, "enable_protein_tools", False) else base
    return (*tools, "design_h3_boltzgen") if getattr(args, "enable_boltzgen", False) else tools


def optional_host_flags(worker, directory):
    for name in ("boltzgen", "protein_tools"):
        if getattr(getattr(worker, "args", None), "enable_" + name, False):
            (Path(directory) / (name.replace("_", "-") + ".enabled")).write_text("enabled\n")


class OptionalProteinTools:
    def run_boltzgen(self, job, count, seed):
        import fcntl

        suffix = {"HER2": "HER2", "1N8Z_C": "HER2", "PD-1": "PD1", "5GGR_Y": "PD1"}.get(self.args.antigen)
        if suffix is None:
            raise ValueError("No BoltzGen antigen configuration")
        keys = ("BOLTZGEN_PYTHON", "BOLTZGEN_SCAFFOLD", "BOLTZGEN_CHECKPOINT", "BOLTZGEN_INVERSE_FOLD_CHECKPOINT", "BOLTZGEN_MOLDIR", "BOLTZGEN_SPEC_" + suffix)
        settings = {k: os.environ.get(k) for k in keys}
        if not all(settings.values()):
            raise ValueError("BoltzGen host configuration is incomplete: " + ", ".join(k for k, v in settings.items() if not v))
        gpu = getattr(self, "execution_gpu", self.args.gpu)
        python = Path(settings["BOLTZGEN_PYTHON"]).absolute()
        cmd = [str(python), "-m", "tasks.antibody.tools.complex_screening.boltzgen_design",
               "--python", str(python), "--spec", settings["BOLTZGEN_SPEC_" + suffix],
               "--scaffold", settings["BOLTZGEN_SCAFFOLD"], "--checkpoint", settings["BOLTZGEN_CHECKPOINT"],
               "--inverse-fold-checkpoint", settings["BOLTZGEN_INVERSE_FOLD_CHECKPOINT"],
               "--moldir", settings["BOLTZGEN_MOLDIR"], "--out", str(job / "output"),
               "--gpu", str(gpu), "--seed", str(seed), "--samples", str(count), "--timeout", "850"]
        if getattr(self.args, "share_gpu", False):
            cmd.extend(["--memory-limit-gib", "24"])
        cpus = sorted(os.sched_getaffinity(0))[:self.args.cpu_limit]
        env = dict(os.environ, PYTHONPATH=str(REPO))
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = str(len(cpus))
        cmd = ["taskset", "-c", ",".join(map(str, cpus)), *cmd]
        lock_name = "antibody-rfd3-worker.lock" if hasattr(self, "execution_gpu") else f"antibody-rfd3-worker-gpu{gpu}.lock"
        with (self.args.root / "tmp" / lock_name).open("a") as lock, (job / "worker.log").open("w") as log:
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
