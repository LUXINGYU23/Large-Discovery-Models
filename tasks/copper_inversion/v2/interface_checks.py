"""Controller-owned public interface gate using the existing isolated executor.

With a batching executor all six fixtures run in one container: the supervisor
runs, times and judges each fixture separately, in fixture order. Otherwise one
container per fixture (the prototype path). Both report the same rejection.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os
import shutil
import threading
import tempfile
from ldm_tts.engine.run_store import atomic_json_write
from ldm_tts.contracts.evaluation import EvaluationPaused
from ..core.execution import digest, freeze_model
from ..core.geometry import read_model
from ..core.runner import input_digest
from .data import save_problem
from .fixtures import fixtures, FIXTURE_VERSION
from .geometry import canonicalize
from .suite import file_hash


class InterfaceChecks:
    fixture_count = 6

    def __init__(self, root, execution, slots=None):
        self.root, self.execution = Path(root).resolve(), execution
        self.slots = slots
        self._input_lock = threading.Lock()
        self._inputs = {}

    def __call__(self, payload):
        try:
            return self._check(payload)
        except OSError as exc:
            raise EvaluationPaused("interface infrastructure I/O failed; no scientific rejection charged") from exc
        except ValueError:
            if self.slots is not None:
                self.slots.reserve(payload["sha256"], interface_rejected=True)
            raise

    def _prepare(self, problems):
        """Fixture inputs are written once, atomically, and only read afterwards.

        This lock is the only admission-wide mutual exclusion: concurrent
        admissions share these read-only inputs (parallelism.md §4.2).
        """
        with self._input_lock:
            for problem in problems:
                fixture_digest = digest(dict(version=FIXTURE_VERSION, problem=input_digest(problem)))
                if fixture_digest in self._inputs:
                    continue
                inputs = self.root / "interface-inputs" / fixture_digest
                if not inputs.is_dir():
                    inputs.parent.mkdir(parents=True, exist_ok=True)
                    staging = Path(tempfile.mkdtemp(dir=inputs.parent, prefix=".staging-"))
                    try:
                        save_problem(problem, staging)
                        atomic_json_write(staging / "budget.json", dict(seed=47, wall_time_seconds=5, cpu_count=1, memory_mb=2048))
                        # Never truncate an NPZ file another container may be reading.
                        os.rename(staging, inputs)
                    finally:
                        shutil.rmtree(staging, ignore_errors=True)
                self._inputs[fixture_digest] = inputs

    def _check(self, payload):
        source = self.root / payload["source_path"]
        if source.is_symlink() or not source.resolve().is_relative_to(self.root):
            raise ValueError("invalid candidate path")
        if file_hash(source) != payload["sha256"]:
            raise ValueError("candidate changed before interface checks")
        problems = tuple(fixtures())
        self._prepare(problems)
        if getattr(self.execution, "merged", False):
            return self._check_batch(payload, source, problems)
        # Queue limits govern actual containers. Consume results in fixture order
        # so parallel completion cannot change the reported rejection.
        with ThreadPoolExecutor(max_workers=self.fixture_count) as pool:
            futures = [pool.submit(self._check_fixture, payload, source, problem) for problem in problems]
            for future in futures:
                future.result()
        return True

    def _fixture(self, payload, problem):
        fixture_digest = digest(dict(version=FIXTURE_VERSION, problem=input_digest(problem)))
        return self._inputs[fixture_digest], dict(source=payload["sha256"], fixture=fixture_digest)

    def _check_batch(self, payload, source, problems):
        fixtures_ = [self._fixture(payload, problem) for problem in problems]
        job, receipt = self.execution.run(dict(source=payload["sha256"], fixtures=[i["fixture"] for _, i in fixtures_]),
            phase="interface", cases=[dict(inputs=inputs) for inputs, _ in fixtures_], batch=True,
            source=source.parent, timeout=5, science_module="v2")
        runs = receipt["runs"]
        if runs is None:
            # The host stopped the container before PID 1 attested any run.
            raise ValueError(f"public interface {problems[0]['case_id']} failed: {receipt['reason'] or receipt['exit_code']}")
        for index, (problem, (inputs, identity)) in enumerate(zip(problems, fixtures_)):
            self._judge(runs[index], job / "output" / str(index) / "model.npy", problem, identity)
        return True

    def _check_fixture(self, payload, source, problem):
        inputs, identity = self._fixture(payload, problem)
        job, receipt = self.execution.run(identity, phase="interface", inputs=inputs,
            source=source.parent, timeout=5, science_module="v2")
        self._judge(receipt, job / "output/model.npy", problem, identity)

    def _judge(self, receipt, model, problem, identity):
        if receipt["exit_code"] or receipt["reason"] or receipt["oom"]:
            raise ValueError(f"public interface {problem['case_id']} failed: {receipt['reason'] or receipt['exit_code']}")
        frozen = self.root / "interface-models" / digest(identity)
        try:
            read_model(model, problem["shape"])
        except (FileNotFoundError, PermissionError) as exc:
            raise ValueError("candidate did not produce a readable model") from exc
        freeze_model(model, frozen, problem["shape"])
        canonicalize(problem, read_model(frozen / "model.npy", problem["shape"]))
